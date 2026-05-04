"""Evaluate a trained 1-Bit HTRM checkpoint.

Loads a checkpoint, runs inference over a dataset, reports the four
spec-mandated metrics: accuracy %, avg macro loops, avg micro loops,
inference tokens/sec. For POC scope this evaluates against a held-out
slice of the local dataset; the full spec adds Samsung's
sapientinc/sudoku-extreme test set as a separate code path.

Test-time-compute knob: --max-macro N --max-micro M overrides the
config's K and L respectively, allowing post-hoc compute scaling.

Example:
    python evaluate_extreme.py --ckpt checkpoints/poc/best.pt --data data/poc.pt
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from htrm.config import HTRMConfig
from htrm.dataset import CurriculumSudokuDataset
from htrm.device import get_device, sync
from htrm.htrm_model import HTRM
from htrm.sudoku_rules import count_violations


def evaluate_checkpoint(
    ckpt_path: str | Path,
    data_path: str | Path,
    n_samples: int | None,
    max_macro: int | None,
    max_micro: int | None,
    batch_size: int,
    val_frac: float,
    seed: int,
    force_cpu: bool,
) -> dict[str, float | int]:
    device = get_device(force_cpu=force_cpu)
    # weights_only=False is intentional: our own checkpoints contain a
    # config dict alongside the tensors, and DirectML state_dicts use a
    # rebuilder not in the default safe-globals list.
    blob = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = HTRMConfig(**blob["cfg"])
    model = HTRM(cfg)
    model.load_state_dict(blob["model_state"])
    model = model.to(device)
    model.eval()

    # Auto-pick dataset class. CurriculumSudokuDataset returns 3-tuples;
    # plain SudokuDataset returns 2-tuples. We probe by trying the curriculum
    # one first (it accepts both formats).
    full = CurriculumSudokuDataset(data_path)
    has_difficulty = bool((full.difficulty != 0).any().item())

    if val_frac >= 1.0:
        # Eval on the entire dataset (e.g. held-out test split).
        val_ds = full
    else:
        # Mirror train.py's seeded random_split so the held-out val slice
        # is identical between training and offline eval.
        n_val = max(1, int(len(full) * val_frac))
        n_train = len(full) - n_val
        _, val_ds = torch.utils.data.random_split(
            full, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )
    if n_samples is not None and n_samples < len(val_ds):
        val_ds = Subset(val_ds, list(range(n_samples)))

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    cell_correct = 0
    cell_total = 0
    macro_sum = 0
    micro_sum = 0
    n_batches = 0
    valid_count = 0
    per_tier_correct = [0, 0, 0]
    per_tier_total = [0, 0, 0]

    sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            if has_difficulty:
                puzzle, solution, difficulty = batch
            else:
                puzzle, solution = batch
                difficulty = torch.zeros(puzzle.shape[0], dtype=torch.long)
            puzzle = puzzle.to(device)
            solution = solution.to(device)
            out = model(puzzle, training=False, max_macro=max_macro, max_micro=max_micro)
            preds = out["logits"].argmax(dim=-1)
            cell_correct += (preds == solution).sum().item()
            cell_total += preds.numel()
            full_correct = (preds == solution).all(dim=-1)
            correct += int(full_correct.sum().item())
            total += int(full_correct.numel())
            macro_sum += int(out["macro_used"])
            micro_sum += int(out["micro_used"])
            # Count predictions that, while not necessarily matching the
            # ground truth, are *valid* Sudoku grids (no rule violations
            # on filled cells). Helps separate "wrong digit" from "broken".
            preds_cpu = preds.cpu()
            for i in range(preds_cpu.shape[0]):
                if count_violations(preds_cpu[i]) == 0:
                    valid_count += 1
            for t in range(3):
                mask = (difficulty == t).cpu()
                per_tier_total[t] += int(mask.sum().item())
                per_tier_correct[t] += int(full_correct.cpu()[mask].sum().item())
            n_batches += 1
    sync()
    elapsed = time.perf_counter() - t0

    n_puzzles = total
    metrics = {
        "accuracy": correct / max(total, 1),
        "cell_accuracy": cell_correct / max(cell_total, 1),
        "valid_grid_rate": valid_count / max(total, 1),
        "avg_macro_loops": macro_sum / max(n_batches, 1),
        "avg_micro_loops": micro_sum / max(n_batches, 1),
        "tokens_per_sec": (81 * n_puzzles) / max(elapsed, 1e-6),
        "n_puzzles": n_puzzles,
        "elapsed_s": elapsed,
    }
    for t, name in enumerate(["easy", "medium", "extreme"]):
        if per_tier_total[t] > 0:
            metrics[f"accuracy_{name}"] = per_tier_correct[t] / per_tier_total[t]
            metrics[f"n_{name}"] = per_tier_total[t]
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, default="data/poc.pt")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--max-macro", type=int, default=None)
    ap.add_argument("--max-micro", type=int, default=None)
    ap.add_argument("--ttc-sweep", action="store_true",
                    help="run a sweep over (K, L) ∈ {(8,2),(12,3),(16,4),(24,6)}")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv-out", type=str, default="results.csv")
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args()

    sweep = ([(args.max_macro, args.max_micro)] if not args.ttc_sweep
             else [(8, 2), (12, 3), (16, 4), (24, 6)])

    rows = []
    for k, l in sweep:
        tag = f"K={k},L={l}" if k is not None else "config-default"
        print(f"\n=== eval @ {tag} ===")
        m = evaluate_checkpoint(
            ckpt_path=args.ckpt,
            data_path=args.data,
            n_samples=args.n_samples,
            max_macro=k,
            max_micro=l,
            batch_size=args.batch_size,
            val_frac=args.val_frac,
            seed=args.seed,
            force_cpu=args.force_cpu,
        )
        print(f"  puzzle_accuracy : {m['accuracy']:.4f}")
        print(f"  cell_accuracy   : {m['cell_accuracy']:.4f}")
        print(f"  valid_grid_rate : {m['valid_grid_rate']:.4f}")
        print(f"  avg_macro_loops : {m['avg_macro_loops']:.2f}")
        print(f"  avg_micro_loops : {m['avg_micro_loops']:.2f}")
        print(f"  tokens_per_sec  : {m['tokens_per_sec']:.1f}")
        print(f"  n_puzzles       : {m['n_puzzles']}")
        rows.append({"setting": tag, **m})

    out_path = Path(args.csv_out)
    with open(out_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
