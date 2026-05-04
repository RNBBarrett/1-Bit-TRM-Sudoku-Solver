"""Training loop for 1-Bit HTRM (POC and full-spec).

Auto-detects whether the dataset is curriculum-tagged (writes from
`save_curriculum`) and switches to a curriculum-aware sampler. Supports:

  - **Curriculum staging** (step-based) with three stages: Warmup
    (Easy only) -> Mixed-1 (70/30) -> Mixed-2 (30/50/20) -> Final (uniform).
  - **FP warmup** (BitNet b1.58 recipe): the first `--quantize-after-step`
    optimizer steps run with quantization disabled so shadow weights
    settle in FP32 before being squeezed to ternary.
  - **Halt-loss ramp**: weight on the ACT halt loss linearly increases
    from 0 to `--halt-weight` over `--halt-ramp-steps` steps to avoid
    the degenerate "always say 'wrong'" minimum early on.
  - **Gradient checkpointing** via `--gradient-checkpoint` for tight VRAM.

Example (full spec on Samsung's data):
    python train.py --data data/samsung_train.pt --config configs/htrm_full.yaml \
                    --max-steps 50000 --micro-batch 8 --accum-steps 4 \
                    --quantize-after-step 10000 --halt-ramp-steps 5000 \
                    --gradient-checkpoint --out checkpoints/full
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from htrm.bitlinear import set_quantization_enabled
from htrm.config import HTRMConfig
from htrm.dataset import CurriculumSudokuDataset
from htrm.device import get_device, sync
from htrm.htrm_model import HTRM
from htrm.losses import HTRMLoss


def save_resumable_checkpoint(
    path: Path,
    step: int,
    model: HTRM,
    optim: torch.optim.Optimizer,
    cfg: HTRMConfig,
    log: list[dict],
    metrics: dict | None = None,
) -> None:
    """Save model + optimizer + step + log to disk in a CPU-portable format.

    Tensors are detached and moved to CPU so checkpoints load cleanly on
    any backend (CUDA, DirectML, or CPU-only re-runs).
    """
    blob = {
        "step": step,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optim_state": optim.state_dict(),
        "cfg": cfg.to_dict(),
    }
    if metrics is not None:
        blob["metrics"] = metrics
    torch.save(blob, path)
    log_path = path.parent / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def evaluate(
    model: HTRM, loader: DataLoader, device: torch.device, has_difficulty: bool
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    cell_correct = 0
    cell_total = 0
    per_tier_correct = [0, 0, 0]
    per_tier_total = [0, 0, 0]
    with torch.inference_mode():
        for batch in loader:
            if has_difficulty:
                puzzle, solution, difficulty = batch
            else:
                puzzle, solution = batch
                difficulty = torch.zeros(puzzle.shape[0], dtype=torch.long)
            puzzle = puzzle.to(device)
            solution = solution.to(device)
            out = model(puzzle, training=False)
            preds = out["logits"].argmax(dim=-1)
            cell_correct += (preds == solution).sum().item()
            cell_total += preds.numel()
            full_correct = (preds == solution).all(dim=-1)
            correct += int(full_correct.sum().item())
            total += int(full_correct.numel())
            for t in range(3):
                mask = (difficulty == t).cpu()
                per_tier_total[t] += int(mask.sum().item())
                per_tier_correct[t] += int(full_correct.cpu()[mask].sum().item())
    model.train()
    metrics = {
        "puzzle_acc": correct / max(total, 1),
        "cell_acc": cell_correct / max(cell_total, 1),
    }
    for t, name in enumerate(["easy", "medium", "extreme"]):
        if per_tier_total[t] > 0:
            metrics[f"puzzle_acc_{name}"] = per_tier_correct[t] / per_tier_total[t]
    return metrics


def curriculum_stage(step: int, max_steps: int) -> tuple[str, dict[int, float]]:
    """Step-based curriculum: 5% warmup, 20% mixed-1, 50% mixed-2, 25% final.

    Compressed warmup (was 30%) prevents the model from cycling through the
    small Easy tier (~23k Samsung puzzles) too many times and overfitting.
    Mixed-1 starts at step 2,500 of a 50k run, which is when the rotation of
    Easy puzzles is starting to repeat enough to stop providing fresh signal.
    """
    frac = step / max(max_steps, 1)
    if frac < 0.05:
        return "warmup", {0: 1.0, 1: 0.0, 2: 0.0}
    if frac < 0.25:
        return "mixed-1", {0: 0.7, 1: 0.3, 2: 0.0}
    if frac < 0.75:
        return "mixed-2", {0: 0.3, 1: 0.5, 2: 0.2}
    return "final", {0: 1.0, 1: 1.0, 2: 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/samsung_train.pt")
    ap.add_argument("--config", type=str, default="configs/htrm_full.yaml")
    ap.add_argument("--out", type=str, default="checkpoints/run")
    ap.add_argument("--max-steps", "--steps", type=int, default=50000, dest="max_steps",
                    help="hard cap on optimizer steps (whichever limit hits first stops training)")
    ap.add_argument("--hours", type=float, default=None,
                    help="optional wall-clock time limit in hours")
    ap.add_argument("--status-every-min", type=float, default=5.0,
                    help="print a heartbeat status line every N minutes (default 5)")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="max-norm for gradient clipping (lower = tighter / more stable)")
    ap.add_argument("--bf16", action="store_true",
                    help="enable bf16 mixed-precision autocast (CUDA only; ~2-3x speedup on tensor cores)")
    ap.add_argument("--fp16", action="store_true",
                    help="enable fp16 mixed-precision autocast with GradScaler (CUDA only)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--violation-weight", type=float, default=10.0)
    ap.add_argument("--halt-weight", type=float, default=0.1)
    ap.add_argument("--halt-ramp-steps", type=int, default=10000,
                    help="linearly ramp halt-loss weight from 0 to halt-weight over N steps")
    ap.add_argument("--quantize-after-step", type=int, default=-1,
                    help="step at which to enable BitNet quantization "
                         "(default: -1 = enabled from step 0; recommended 20%% of max-steps)")
    ap.add_argument("--gradient-checkpoint", action="store_true")
    ap.add_argument("--no-curriculum", action="store_true",
                    help="disable curriculum even on tagged datasets")
    ap.add_argument("--resume", type=str, default=None,
                    help="path to a checkpoint .pt to resume from (loads model + optimizer + step)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    cfg = HTRMConfig.from_yaml(args.config)
    print(f"config: {cfg.to_dict()}")
    device = get_device(force_cpu=args.force_cpu)
    print(f"device: {device}")

    full = CurriculumSudokuDataset(args.data)
    has_difficulty = bool((full.difficulty != 0).any().item())
    use_curriculum = has_difficulty and not args.no_curriculum
    print(f"loaded {len(full)} examples; curriculum={use_curriculum}")
    if has_difficulty:
        diffs = full.difficulty
        print(f"  difficulty breakdown: easy={int((diffs==0).sum())} "
              f"medium={int((diffs==1).sum())} extreme={int((diffs==2).sum())}")

    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"train={len(train_ds)} val={len(val_ds)} micro_batch={args.micro_batch} accum={args.accum_steps}")

    model = HTRM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    loss_fn = HTRMLoss(violation_weight=args.violation_weight, halt_weight=1.0).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Mixed-precision autocast setup. bf16 is preferred on Ampere/Ada/Hopper
    # GPUs because it doesn't need a GradScaler (wider exponent range than fp16).
    autocast_dtype: torch.dtype | None = None
    if args.bf16:
        autocast_dtype = torch.bfloat16
        print("Mixed precision: bf16 autocast ON")
    elif args.fp16:
        autocast_dtype = torch.float16
        print("Mixed precision: fp16 autocast ON (with GradScaler)")
    else:
        print("Mixed precision: OFF (fp32)")
    scaler = torch.amp.GradScaler("cuda") if args.fp16 else None
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    log: list[dict] = []
    step = 0
    accum_count = 0
    best_val = -1.0

    if args.resume is not None:
        print(f"resuming from {args.resume}")
        ckpt = torch.load(Path(args.resume), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)
        if "optim_state" in ckpt:
            optim.load_state_dict(ckpt["optim_state"])
            print("  optimizer state restored")
        else:
            print("  WARNING: checkpoint has no optim_state; AdamW running averages will start from zero")
        step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("metrics", {}).get("puzzle_acc", best_val))
        prev_log_path = out_dir / "train_log.json"
        if prev_log_path.exists():
            with open(prev_log_path, "r") as f:
                log = json.load(f)
        print(f"  resumed at step {step}, best_val={best_val:.4f}")

    # Set BitNet quantization state based on (possibly resumed) step.
    if args.quantize_after_step >= 0:
        if step >= args.quantize_after_step:
            set_quantization_enabled(model, enabled=True)
            print(f"FP warmup complete (step {step} >= {args.quantize_after_step}); quantization ENABLED")
        else:
            set_quantization_enabled(model, enabled=False)
            print(f"FP warmup ON: quantization will enable at step {args.quantize_after_step}")
    else:
        print("FP warmup OFF: quantization enabled from step 0")
    current_stage_name = "(uninit)"
    train_iter = None
    train_loader = None
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch, shuffle=False)
    t0 = time.perf_counter()

    train_indices_mask = torch.zeros(len(full), dtype=torch.bool)
    train_indices_mask[list(train_ds.indices)] = True

    def make_train_loader(tier_weights: dict[int, float]) -> DataLoader:
        if use_curriculum:
            # Build per-puzzle weights: tier_weight for train indices, 0 for val.
            tier_lookup = torch.tensor(
                [tier_weights.get(t, 0.0) for t in (0, 1, 2)], dtype=torch.float64,
            )
            weights = tier_lookup[full.difficulty] * train_indices_mask.double()
            from torch.utils.data import WeightedRandomSampler
            gen = torch.Generator().manual_seed(args.seed + step)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=args.micro_batch * args.accum_steps * args.eval_every * 4,
                replacement=True,
                generator=gen,
            )
            return DataLoader(
                full, batch_size=args.micro_batch, sampler=sampler, drop_last=True,
            )
        else:
            return DataLoader(
                train_ds, batch_size=args.micro_batch, shuffle=True, drop_last=True,
            )

    interrupted = False
    last_status_time = t0
    last_ce = float("nan")
    last_heartbeat_ce = float("nan")
    last_batch_cell_acc = float("nan")
    last_batch_puzzle_acc = float("nan")
    last_batch_correct = 0
    last_batch_size = 0
    last_violation = float("nan")
    last_halt = float("nan")
    last_halt_w = 0.0

    # Reference baselines for the current dataset. Random over the full
    # vocab is log(V); the "copy clues + uniform digit guess on blanks"
    # CE baseline depends on how many clues each puzzle has. We compute
    # the data-set-wide baseline once for comparison.
    random_ce = float(torch.tensor(cfg.vocab_size).float().log().item())
    avg_clue_frac = float(((full.puzzles != 0).sum(dim=1).float() / 81).mean().item())
    # Per-cell expected CE for the copy-clues baseline:
    #   clue cells: copied perfectly -> CE = 0
    #   blank cells: uniform over digits 1-9 -> CE = log(9) ≈ 2.197
    copy_baseline_ce = (1.0 - avg_clue_frac) * float(torch.tensor(9.0).log().item())

    def render_heartbeat() -> str:
        now = time.perf_counter()
        elapsed_min = (now - t0) / 60.0
        rate = step / max(now - t0, 1e-6)
        remaining = max(args.max_steps - step, 0)
        eta_hr = (remaining / max(rate, 1e-9)) / 3600.0
        progress_pct = 100.0 * step / max(args.max_steps, 1)

        # ----- Intelligence-level interpretation -----
        # Translate the cross-entropy "loss" number into a plain-English
        # description of how smart the model currently is, by comparing it
        # to two reference strategies any beginner can imagine:
        #   (A) "random guess": pick a digit uniformly at random for every cell
        #   (B) "copy clues":   regurgitate the visible digits, random-guess on blanks
        if last_ce != last_ce:  # NaN
            smarts_line = "no measurement yet (waiting for first batch)"
            trend_line = ""
        elif last_ce > random_ce:
            smarts_line = f"WORSE than just guessing randomly (the model is broken or just initialized)"
            trend_line = ""
        else:
            if last_ce > copy_baseline_ce:
                pct_to_baseline = 100.0 * (random_ce - last_ce) / max(random_ce - copy_baseline_ce, 1e-6)
                smarts_line = (
                    f"{pct_to_baseline:.0f}% of the way from random-guessing to "
                    "merely copying the visible numbers"
                )
            else:
                lift = copy_baseline_ce - last_ce
                smarts_line = (
                    "BEYOND just copying clues -- the model is starting to "
                    f"reason about blanks (lift={lift:.2f})"
                )
            if last_heartbeat_ce == last_heartbeat_ce:  # not NaN
                d = last_heartbeat_ce - last_ce
                if d > 0.005:
                    trend_line = f"GETTING SMARTER (uncertainty fell {d:.3f} since last heartbeat)"
                elif d < -0.005:
                    trend_line = f"GETTING WORSE (uncertainty rose {-d:.3f}) -- watch this closely"
                else:
                    trend_line = f"plateaued (no meaningful change in uncertainty since last heartbeat)"
            else:
                trend_line = "(first heartbeat - no trend yet)"

        # ----- Stage description -----
        stage_desc = {
            "warmup":  "EASY puzzles only -- model learns to copy clues",
            "mixed-1": "EASY + MEDIUM mix -- model starts reasoning about blanks",
            "mixed-2": "EASY + MEDIUM + EXTREME mix -- harder reasoning required",
            "final":   "all difficulty levels mixed uniformly -- final polish",
        }.get(current_stage_name, "?")

        # ----- Live batch quality -----
        if last_batch_size > 0:
            cells_right = int(round(last_batch_cell_acc * 81))
            batch_line = (
                f"On the puzzle it just looked at: got {cells_right}/81 cells right "
                f"({last_batch_cell_acc*100:.1f}%), "
                f"completely solved {last_batch_correct}/{last_batch_size} puzzles in this batch"
            )
        else:
            batch_line = "(no batch processed yet)"

        # ----- Cheating detector -----
        if last_violation < 0.02:
            cheat_line = "OK - model isn't trying to place duplicate digits in the same row/column/box"
        else:
            cheat_line = (
                f"PROBLEM ({last_violation:.4f}) - model is guessing duplicates "
                "in the same row/col/box; rule penalty active"
            )

        # ----- Halt-head ('done' signal) -----
        halt_pct = 100.0 * last_halt_w / max(args.halt_weight, 1e-9)
        if halt_pct < 100:
            halt_line = (
                f"Model is learning when to stop thinking. "
                f"This signal is at {halt_pct:.0f}% strength (it ramps up early in training "
                "to avoid a degenerate shortcut)."
            )
        else:
            halt_line = (
                f"'Done thinking' signal is at full strength ({last_halt:.4f}); "
                "model is being trained to halt only when confident."
            )

        # ----- Quantization phase -----
        if args.quantize_after_step >= 0 and step < args.quantize_after_step:
            quant_line = (
                f"Currently using full-precision weights "
                f"(1-bit BitNet quantization will engage at step {args.quantize_after_step})"
            )
        else:
            quant_line = (
                "Using 1-bit BitNet quantization -- model weights are constrained to {-1, 0, +1} "
                "for the published 1.4MB-deployable footprint"
            )

        # ----- ETA in friendly units -----
        if eta_hr < 1:
            eta_str = f"{eta_hr*60:.0f} more minutes"
        elif eta_hr < 24:
            eta_str = f"{eta_hr:.1f} more hours of GPU time"
        else:
            eta_str = f"{eta_hr/24:.1f} more days of GPU time"

        return (
            "\n" + "=" * 70 + "\n"
            f"TRAINING HEARTBEAT  --  {elapsed_min:.1f} min elapsed, "
            f"step {step}/{args.max_steps} ({progress_pct:.2f}% done)\n"
            f"At {rate:.2f} steps/sec, about {eta_str} until 100%.\n"
            "----------------------------------------------------------------------\n"
            f"WHAT'S HAPPENING NOW:\n"
            f"  Stage:        {current_stage_name}  ({stage_desc})\n"
            f"  Weight mode:  {quant_line}\n\n"
            f"HOW SMART IS THE MODEL?\n"
            f"  {batch_line}\n"
            f"  Skill level:  {smarts_line}\n"
            f"  Trend:        {trend_line}\n\n"
            f"HEALTH CHECKS:\n"
            f"  Cheating:     {cheat_line}\n"
            f"  'Done' head:  {halt_line}\n"
            f"  Raw numbers:  uncertainty(CE)={last_ce if last_ce==last_ce else float('nan'):.4f}  "
            f"random={random_ce:.2f}  copy-clues={copy_baseline_ce:.2f}  perfect=0.00\n"
            + "=" * 70
        )

    try:
      while step < args.max_steps:
        # Heartbeat status at fixed wall-clock intervals (independent of step rate).
        now = time.perf_counter()
        if (now - last_status_time) >= args.status_every_min * 60.0:
            print(render_heartbeat())
            last_heartbeat_ce = last_ce
            last_status_time = now

        stage_name, tier_weights = curriculum_stage(step, args.max_steps)
        if stage_name != current_stage_name:
            current_stage_name = stage_name
            print(f"--- curriculum stage: {stage_name} | tier_weights={tier_weights} ---")
            train_loader = make_train_loader(tier_weights)
            train_iter = iter(train_loader)

        # FP-warmup transition
        if 0 <= args.quantize_after_step == step:
            set_quantization_enabled(model, enabled=True)
            print(f"  >>> step {step}: BitNet quantization ENABLED <<<")

        try:
            assert train_iter is not None
            batch = next(train_iter)
        except StopIteration:
            assert train_loader is not None
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # CurriculumSudokuDataset always returns (puzzle, solution, difficulty);
        # difficulty is unused when curriculum mode is off.
        puzzle, solution, _ = batch
        puzzle = puzzle.to(device)
        solution = solution.to(device)

        # Forward + loss under optional autocast for mixed precision.
        if autocast_dtype is not None:
            ac_ctx = torch.amp.autocast(autocast_device, dtype=autocast_dtype)
        else:
            ac_ctx = nullcontext()
        with ac_ctx:
            out = model(puzzle, training=True, gradient_checkpoint=args.gradient_checkpoint)
            # Halt-loss weight ramp 0 -> args.halt_weight over args.halt_ramp_steps.
            if args.halt_ramp_steps > 0:
                halt_w = min(step / args.halt_ramp_steps, 1.0) * args.halt_weight
            else:
                halt_w = args.halt_weight
            comps = loss_fn(
                out["logits"], solution, out["halts"],
                violation_weight=args.violation_weight,
                halt_weight=halt_w,
            )
            total = comps["total"]
        last_ce = float(comps["ce"].item())
        last_violation = float(comps["violation"].item())
        last_halt = float(comps["halt"].item())
        last_halt_w = halt_w
        # Live batch accuracy stats for the heartbeat (cheap; no extra fwd).
        with torch.no_grad():
            preds = out["logits"].argmax(dim=-1)
            last_batch_cell_acc = (preds == solution).float().mean().item()
            full_correct = (preds == solution).all(dim=-1)
            last_batch_correct = int(full_correct.sum().item())
            last_batch_size = int(full_correct.numel())
            last_batch_puzzle_acc = last_batch_correct / max(last_batch_size, 1)
        loss = total / args.accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accum_count += 1

        if accum_count >= args.accum_steps:
            if scaler is not None:
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)
            accum_count = 0
            step += 1

            # Wall-clock time-budget exit. Saves a checkpoint and returns
            # cleanly so the run can be resumed via --resume.
            if args.hours is not None:
                elapsed_hr = (time.perf_counter() - t0) / 3600.0
                if elapsed_hr >= args.hours:
                    print(f"\n[time limit reached at step {step} after {elapsed_hr:.2f} hours]"
                          f" saving last.pt and exiting cleanly")
                    save_resumable_checkpoint(out_dir / "last.pt", step, model, optim, cfg, log)
                    print(f"resume with: --resume {out_dir / 'last.pt'}")
                    return

            if step % 50 == 0:
                sync()
                dt = time.perf_counter() - t0
                rate = step / max(dt, 1e-6)
                print(
                    f"[{step:6d}/{args.max_steps}] stage={current_stage_name} "
                    f"loss={total.item():.4f} ce={comps['ce'].item():.4f} "
                    f"viol={comps['violation'].item():.4f} halt={comps['halt'].item():.4f} "
                    f"halt_w={halt_w:.4f} | {rate:.2f} step/s"
                )
                log.append({
                    "step": step, "stage": current_stage_name,
                    "loss": float(total.item()),
                    "ce": float(comps["ce"].item()),
                    "violation": float(comps["violation"].item()),
                    "halt": float(comps["halt"].item()),
                    "halt_w": halt_w, "elapsed_s": dt,
                })

            if step % args.eval_every == 0:
                metrics = evaluate(model, val_loader, device, has_difficulty)
                meta = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"  eval@{step}: {meta}")
                log.append({"step": step, "eval": metrics})
                if metrics["puzzle_acc"] > best_val:
                    best_val = metrics["puzzle_acc"]
                    save_resumable_checkpoint(out_dir / "best.pt", step, model, optim, cfg, log, metrics=metrics)

            # Early-checkpoint cadence: save every 200 steps for the first
            # 1,000 steps so NaN explosions cost at most ~10 minutes of compute.
            do_early_ckpt = step <= 1000 and step % 200 == 0
            do_periodic_ckpt = step % args.ckpt_every == 0
            if do_early_ckpt or do_periodic_ckpt:
                save_resumable_checkpoint(out_dir / "last.pt", step, model, optim, cfg, log)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[interrupted at step {step}] saving last.pt and exiting cleanly")
        save_resumable_checkpoint(out_dir / "last.pt", step, model, optim, cfg, log)
        print(f"resume with: --resume {out_dir / 'last.pt'}")
        return

    metrics = evaluate(model, val_loader, device, has_difficulty)
    print(f"final eval: {metrics}")
    save_resumable_checkpoint(out_dir / "final.pt", step, model, optim, cfg, log, metrics=metrics)
    print(f"wrote checkpoint + log to {out_dir}")


if __name__ == "__main__":
    main()
