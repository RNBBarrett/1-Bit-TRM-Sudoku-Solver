"""Tier 3 BitDistill training: 1-bit Samsung TRM student distilled from FP teacher.

Single training run that combines:
  - Samsung's TRM architecture (puzzle_emb, ACT, recursion preserved)
  - BitNet b1.58 weight quantization with v6 stabilization (lambda ramp, learnable
    alpha, median scaling)
  - Knowledge distillation from a frozen FP teacher (Tier 1 step_54684)
  - AdamW + EMA at 0.999

Designed to run on Apple Silicon (MPS) without CUDA-only deps.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Monkey-patch CastedLinear BEFORE importing model code
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from bitnet_layers import BitCastedLinear, set_lambda_q, count_bit_linears  # noqa: E402

# Add Samsung's repo to path
SAMSUNG_REPO = os.environ.get("SAMSUNG_REPO", str(Path.home() / "trm-tier3" / "TinyRecursiveModels"))
sys.path.insert(0, SAMSUNG_REPO)

import models.layers as samsung_layers  # noqa: E402


def build_student_model(cfg: dict) -> nn.Module:
    """Build TRM with BitCastedLinear instead of CastedLinear."""
    original = samsung_layers.CastedLinear
    samsung_layers.CastedLinear = BitCastedLinear  # type: ignore
    try:
        from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
        # also patch in trm module's namespace if it imported CastedLinear directly
        import models.recursive_reasoning.trm as trm_mod
        trm_mod.CastedLinear = BitCastedLinear  # type: ignore
        model = TinyRecursiveReasoningModel_ACTV1(cfg)
    finally:
        samsung_layers.CastedLinear = original
        try:
            trm_mod.CastedLinear = original  # type: ignore
        except Exception:
            pass
    return model


def build_teacher_model(cfg: dict) -> nn.Module:
    """Build TRM with original CastedLinear (FP)."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    return TinyRecursiveReasoningModel_ACTV1(cfg)


def move_carry(carry, device: torch.device):
    """Recursively move all tensors in a carry dataclass to a device."""
    if hasattr(carry, "__dataclass_fields__"):
        from dataclasses import replace, fields
        kwargs = {}
        for f in fields(carry):
            v = getattr(carry, f.name)
            if isinstance(v, torch.Tensor):
                kwargs[f.name] = v.to(device)
            elif hasattr(v, "__dataclass_fields__"):
                kwargs[f.name] = move_carry(v, device)
            elif isinstance(v, dict):
                kwargs[f.name] = {k: (vv.to(device) if isinstance(vv, torch.Tensor) else vv) for k, vv in v.items()}
            else:
                kwargs[f.name] = v
        return replace(carry, **kwargs)
    return carry


def detach_carry(carry):
    """Recursively detach all tensors in a carry dataclass (cuts grad graph)."""
    if hasattr(carry, "__dataclass_fields__"):
        from dataclasses import replace, fields
        kwargs = {}
        for f in fields(carry):
            v = getattr(carry, f.name)
            if isinstance(v, torch.Tensor):
                kwargs[f.name] = v.detach()
            elif hasattr(v, "__dataclass_fields__"):
                kwargs[f.name] = detach_carry(v)
            elif isinstance(v, dict):
                kwargs[f.name] = {k: (vv.detach() if isinstance(vv, torch.Tensor) else vv) for k, vv in v.items()}
            else:
                kwargs[f.name] = v
        return replace(carry, **kwargs)
    return carry


def load_teacher_state(model: nn.Module, ckpt_path: str):
    """Load Tier 1 teacher checkpoint, stripping torch.compile prefix."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    PREFIX = "_orig_mod.model."
    cleaned = {}
    for k, v in sd.items():
        if k.startswith(PREFIX):
            k = k[len(PREFIX):]
        cleaned[k] = v.float() if v.is_floating_point() else v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  teacher: {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"  teacher: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")


# ---------------- Dataset (matches Samsung's PuzzleDataset minimal interface) ----

class SudokuTrainDataset:
    """Wraps Samsung's prebuilt npy dataset. Returns batches of dicts."""
    def __init__(self, root: Path, split: str, batch_size: int):
        self.split_dir = root / split
        self.inputs = np.load(self.split_dir / "all__inputs.npy")
        self.labels = np.load(self.split_dir / "all__labels.npy")
        self.pids = np.load(self.split_dir / "all__puzzle_identifiers.npy")
        self.batch_size = batch_size
        self.n = len(self.inputs)
        self.metadata = json.loads((self.split_dir / "dataset.json").read_text())

    def random_batch(self, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self.n, size=self.batch_size)
        return {
            "inputs": torch.from_numpy(self.inputs[idx].astype(np.int64)),
            "labels": torch.from_numpy(self.labels[idx].astype(np.int64)),
            "puzzle_identifiers": torch.from_numpy(self.pids[idx].astype(np.int32)),
        }

    def sequential_batches(self, max_batches: int | None = None):
        for start in range(0, self.n, self.batch_size):
            end = start + self.batch_size
            if end > self.n:
                # pad to batch_size
                pad = end - self.n
                idx = np.concatenate([np.arange(start, self.n), np.zeros(pad, dtype=np.int64)])
                cur_n = self.n - start
            else:
                idx = np.arange(start, end)
                cur_n = self.batch_size
            yield (
                {
                    "inputs": torch.from_numpy(self.inputs[idx].astype(np.int64)),
                    "labels": torch.from_numpy(self.labels[idx].astype(np.int64)),
                    "puzzle_identifiers": torch.from_numpy(self.pids[idx].astype(np.int32)),
                },
                cur_n,
            )
            if max_batches is not None and start // self.batch_size + 1 >= max_batches:
                return


# ---------------- Loss --------------------------------------------------------

def compute_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    temperature: float = 5.0) -> torch.Tensor:
    """KL(softmax(teacher/T) || softmax(student/T)) * T^2."""
    T = temperature
    s_log_p = F.log_softmax(student_logits.float() / T, dim=-1)
    t_p = F.softmax(teacher_logits.float() / T, dim=-1)
    kl = F.kl_div(
        s_log_p.reshape(-1, s_log_p.shape[-1]),
        t_p.reshape(-1, t_p.shape[-1]),
        reduction="batchmean",
        log_target=False,
    )
    return T * T * kl


def compute_ce_loss(logits: torch.Tensor, labels: torch.Tensor,
                    ignore_label_id: int = 0) -> torch.Tensor:
    """Standard CE on cell positions, masking ignore_label_id."""
    # logits: (B, S, V), labels: (B, S)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    return F.cross_entropy(flat_logits, flat_labels.long(),
                           ignore_index=ignore_label_id, reduction="mean")


# ---------------- EMA --------------------------------------------------------

class EMAState:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        sd = model.state_dict()
        for k in self.shadow:
            self.shadow[k].mul_(self.decay).add_(sd[k].detach(), alpha=1.0 - self.decay)


# ---------------- Training -----------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="Root containing train/ and test/ subdirs")
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--global-batch-size", type=int, default=384,
                    help="Effective batch (ignored if grad_accum=1)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-warmup-steps", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=1.0)
    ap.add_argument("--puzzle-emb-lr", type=float, default=1e-4)
    ap.add_argument("--quant-warmup-steps", type=int, default=2000,
                    help="FP warmup before lambda ramp begins")
    ap.add_argument("--lambda-ramp-steps", type=int, default=1000)
    ap.add_argument("--ce-weight", type=float, default=0.1)
    ap.add_argument("--kd-weight", type=float, default=10.0)
    ap.add_argument("--halt-weight", type=float, default=0.1)
    ap.add_argument("--kd-temperature", type=float, default=5.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-batches", type=int, default=20,
                    help="Number of test batches per eval (subset for speed)")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="Quick smoke test: 100 steps, no eval")
    ap.add_argument("--resume", type=str, default=None,
                    help="Resume from a checkpoint dir (loads step_N + step_N_ema + step_N_opt)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"device: {args.device}")
    device = torch.device(args.device)

    # ----- dataset -----
    data_root = Path(args.data_root)
    train_ds = SudokuTrainDataset(data_root, "train", batch_size=args.batch_size)
    test_ds = SudokuTrainDataset(data_root, "test", batch_size=args.batch_size)
    print(f"train: {train_ds.n} examples; test: {test_ds.n} examples")
    print(f"train metadata: {train_ds.metadata}")

    # ----- model config (matches Tier 1's all_config.yaml exactly) -----
    model_cfg = dict(
        batch_size=args.batch_size,
        seq_len=81,
        puzzle_emb_ndim=512,
        puzzle_emb_len=16,
        num_puzzle_identifiers=train_ds.metadata["num_puzzle_identifiers"],
        vocab_size=11,
        H_cycles=3,
        L_cycles=6,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        expansion=4,
        num_heads=8,
        pos_encodings="none",
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        forward_dtype="float32",  # MPS bf16 still has rough edges; fp32 is safe
        mlp_t=True,
        no_ACT_continue=True,
    )

    # ----- student (BitNet) -----
    student = build_student_model(model_cfg).to(device)
    n_student_params = sum(p.numel() for p in student.parameters())
    n_bit_layers = count_bit_linears(student)
    print(f"student: {n_student_params:,} params, {n_bit_layers} BitLinears")

    # ----- teacher (FP) -----
    teacher = build_teacher_model(model_cfg)
    print(f"loading teacher from {args.teacher_ckpt}")
    load_teacher_state(teacher, args.teacher_ckpt)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"teacher: {sum(p.numel() for p in teacher.parameters()):,} params (frozen)")

    # ----- optimizer + EMA -----
    optim = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    ema = EMAState(student, decay=args.ema_decay)

    # ----- resume from checkpoint -----
    start_step = 0
    if args.resume is not None:
        ckpt_path = Path(args.resume)
        # Parse start_step from filename like "step_15000"
        m = re.search(r"step_(\d+)", ckpt_path.name)
        if not m:
            raise ValueError(f"--resume path must end with step_N: {ckpt_path}")
        start_step = int(m.group(1))
        # Load student weights
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(sd)
        # Load EMA shadow
        ema_path = ckpt_path.parent / f"{ckpt_path.name}_ema"
        if ema_path.exists():
            ema.shadow = torch.load(ema_path, map_location=device, weights_only=False)
            print(f"  loaded EMA shadow from {ema_path.name}")
        else:
            print(f"  WARNING: no EMA file at {ema_path}, EMA reinitialized from current weights")
        # Load optimizer state if available
        opt_path = ckpt_path.parent / f"{ckpt_path.name}_opt"
        if opt_path.exists():
            optim.load_state_dict(torch.load(opt_path, map_location=device, weights_only=False))
            print(f"  loaded optimizer state from {opt_path.name}")
        else:
            print(f"  no optimizer state file (fresh AdamW moments — minor warmup needed)")
        print(f"  resuming at step {start_step + 1}")

    # ----- training loop -----
    log_path = out / "tier3.log"
    log_f = open(log_path, "a")
    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"begin training: max_steps={args.max_steps}, batch={args.batch_size}, "
        f"quant_warmup={args.quant_warmup_steps}, lambda_ramp={args.lambda_ramp_steps}, "
        f"start_step={start_step}")

    max_steps = 100 if args.smoke else args.max_steps
    t0 = time.time()
    running_loss = {"ce": 0.0, "kd": 0.0, "halt": 0.0, "total": 0.0}
    log_interval = 50

    student.train()

    # NOTE: We deliberately do NOT use persistent carry across steps. Reason:
    # student and teacher have different q_halt behaviors so their carries
    # desync (different slots loaded with different batches). Without sync,
    # KD is meaningless. Tradeoff: loses multi-iteration ACT depth during
    # training. We compensate by running multi-iteration ACT at eval time.

    for step in range(start_step + 1, max_steps + 1):
        # LR warmup (no decay since lr_min_ratio=1.0 in Samsung's recipe)
        if step <= args.lr_warmup_steps:
            cur_lr = args.lr * (step / args.lr_warmup_steps)
        else:
            cur_lr = args.lr
        for pg in optim.param_groups:
            pg["lr"] = cur_lr

        # Lambda ramp for student
        if step <= args.quant_warmup_steps:
            lam = 0.0
        elif step <= args.quant_warmup_steps + args.lambda_ramp_steps:
            lam = (step - args.quant_warmup_steps) / args.lambda_ramp_steps
        else:
            lam = 1.0
        set_lambda_q(student, lam)

        # ----- single ACT iteration on a fresh batch -----
        batch = train_ds.random_batch(rng)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Fresh carry every step (initial_carry sets halted=True, so first
        # forward call resets all slots to the current batch's data).
        student_carry = move_carry(student.initial_carry(batch), device)
        teacher_carry = move_carry(teacher.initial_carry(batch), device)

        # Student: ONE ACT iteration on the current batch.
        student_carry, s_outputs = student(student_carry, batch)
        s_logits = s_outputs["logits"]  # (B, 16+81, V)

        # Teacher: same, but no_grad
        with torch.no_grad():
            teacher_carry, t_outputs = teacher(teacher_carry, batch)
            t_logits = t_outputs["logits"]

        # Align: drop puzzle_emb prefix tokens (16) so we score the 81 cells
        s_cell_logits = s_logits[:, -81:, :]
        t_cell_logits = t_logits[:, -81:, :]

        labels = batch["labels"]  # (B, 81)
        ce_loss = compute_ce_loss(s_cell_logits, labels, ignore_label_id=0)
        kd_loss = compute_kd_loss(s_cell_logits, t_cell_logits, temperature=args.kd_temperature)

        # Halt loss: BCE of student q_halt_logits vs whether sequence is exactly correct
        with torch.no_grad():
            preds = s_cell_logits.argmax(dim=-1)
            mask = labels != 0
            seq_correct = ((preds == labels) | ~mask).all(dim=-1).float()
        q_halt = s_outputs.get("q_halt_logits")
        if q_halt is not None:
            halt_loss = F.binary_cross_entropy_with_logits(q_halt, seq_correct)
        else:
            halt_loss = torch.tensor(0.0, device=device)

        total_loss = (
            args.ce_weight * ce_loss
            + args.kd_weight * kd_loss
            + args.halt_weight * halt_loss
        )

        optim.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optim.step()
        ema.update(student)

        running_loss["ce"] += ce_loss.item()
        running_loss["kd"] += kd_loss.item()
        running_loss["halt"] += halt_loss.item()
        running_loss["total"] += total_loss.item()

        if step % log_interval == 0:
            elapsed = time.time() - t0
            rate = step / elapsed
            avg_ce = running_loss["ce"] / log_interval
            avg_kd = running_loss["kd"] / log_interval
            avg_halt = running_loss["halt"] / log_interval
            avg_total = running_loss["total"] / log_interval
            log(f"[{step:>6}/{max_steps}] lam={lam:.3f} lr={cur_lr:.2e} "
                f"ce={avg_ce:.4f} kd={avg_kd:.4f} halt={avg_halt:.4f} total={avg_total:.4f} "
                f"| {rate:.2f} step/s")
            running_loss = {k: 0.0 for k in running_loss}

        if step % args.eval_every == 0 and not args.smoke:
            run_eval(student, ema, test_ds, model_cfg, device, args.eval_batches, step, log)

        if step % args.ckpt_every == 0 and not args.smoke:
            ckpt_path = out / f"step_{step}"
            torch.save(student.state_dict(), ckpt_path)
            torch.save(ema.shadow, out / f"step_{step}_ema")
            torch.save(optim.state_dict(), out / f"step_{step}_opt")
            log(f"saved {ckpt_path.name} + ema + opt")

    log("training complete")
    final = out / "final"
    torch.save(student.state_dict(), final)
    torch.save(ema.shadow, out / "final_ema")
    log(f"saved final to {final}")


@torch.no_grad()
def run_eval(student, ema, test_ds, model_cfg, device, max_batches: int, step: int, log):
    student.eval()
    # Apply EMA to a temporary copy
    saved = {k: v.detach().clone() for k, v in student.state_dict().items()}
    student.load_state_dict({k: ema.shadow.get(k, saved[k]) for k in saved}, strict=False)
    set_lambda_q(student, 1.0)  # full quantization at eval

    cell_correct = cell_total = 0
    puzzle_correct = puzzle_total = 0
    for i, (batch, cur_n) in enumerate(test_ds.sequential_batches(max_batches=max_batches)):
        batch_d = {k: v.to(device) for k, v in batch.items()}
        carry = move_carry(student.initial_carry(batch_d), device)
        for _ in range(model_cfg["halt_max_steps"]):
            carry, outputs = student(carry, batch_d)
            if carry.halted.all():
                break
        s_logits = outputs["logits"][:, -81:, :]
        preds = s_logits.argmax(dim=-1).cpu()
        true = batch["labels"][:cur_n]
        preds = preds[:cur_n]
        mask = true != 0
        cell_correct += ((preds == true) & mask).sum().item()
        cell_total += mask.sum().item()
        puzzle_correct += ((preds == true) | ~mask).all(dim=-1).sum().item()
        puzzle_total += cur_n

    student.load_state_dict(saved)
    student.train()
    cell_acc = cell_correct / max(cell_total, 1)
    puzzle_acc = puzzle_correct / max(puzzle_total, 1)
    log(f"eval@{step}: cell_acc={cell_acc:.4f} puzzle_acc={puzzle_acc:.4f} "
        f"({puzzle_correct}/{puzzle_total} on {max_batches} batches)")


if __name__ == "__main__":
    main()
