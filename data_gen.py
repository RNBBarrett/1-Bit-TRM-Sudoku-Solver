"""Generate Sudoku (puzzle, solution) pairs and save to a .pt file.

POC scope: easy puzzles only, no trajectories, no curriculum tagging.
Uses dokusan for procedural generation + backtracking solve.

Example:
    python data_gen.py --target-count 5000 --avg-rank 50 --out data/poc.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from htrm.dataset import save_pairs


def _str_to_tensor(s: str) -> torch.Tensor:
    """Convert an 81-char digit string ('0' = empty) to (81,) int8 tensor."""
    return torch.tensor([int(c) for c in s], dtype=torch.int8)


def generate(target_count: int, avg_rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Lazy import so importing this module doesn't pay dokusan's import cost.
    from dokusan import generators, solvers

    import random
    random.seed(seed)
    torch.manual_seed(seed)

    puzzles: list[torch.Tensor] = []
    solutions: list[torch.Tensor] = []
    seen: set[str] = set()

    pbar = tqdm(total=target_count, desc=f"generating @ avg_rank={avg_rank}")
    rejected = 0
    while len(puzzles) < target_count:
        try:
            puzzle = generators.random_sudoku(avg_rank=avg_rank)
            puzzle_str = str(puzzle)
            if puzzle_str in seen:
                rejected += 1
                continue
            solution = solvers.backtrack(puzzle)
            if solution is None:
                rejected += 1
                continue
            solution_str = str(solution)
            seen.add(puzzle_str)
            puzzles.append(_str_to_tensor(puzzle_str))
            solutions.append(_str_to_tensor(solution_str))
            pbar.update(1)
        except Exception:
            rejected += 1
            continue
    pbar.close()
    print(f"  generated={len(puzzles)} rejected={rejected}")
    return torch.stack(puzzles), torch.stack(solutions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-count", type=int, default=5000)
    ap.add_argument("--avg-rank", type=int, default=50,
                    help="dokusan difficulty knob (lower = easier; ~50=Easy, ~150=Medium, ~300+=Hard)")
    ap.add_argument("--out", type=str, default="data/poc.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    puzzles, solutions = generate(args.target_count, args.avg_rank, args.seed)
    t1 = time.perf_counter()

    save_pairs(out, puzzles, solutions)
    print(f"wrote {len(puzzles)} pairs to {out} in {t1 - t0:.1f}s")


if __name__ == "__main__":
    main()
