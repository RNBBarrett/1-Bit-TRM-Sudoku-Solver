"""Download Samsung's sapientinc/sudoku-extreme dataset and convert to our .pt shard format.

Schema (from HF dataset):
  - source: string label
  - question: 81-char puzzle, '.' = empty cell
  - answer:   81-char solution, all digits 1-9
  - rating:   difficulty rating string

Cells in the CSV are wrapped in single-quotes inside the field, which we
strip. Empty markers ('.') are mapped to 0 to match our existing layout
(0 = empty, 1-9 = digits).

Example:
    python data_gen_hf.py --target-count 5000 --out data/hf_poc.pt
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from htrm.dataset import save_pairs, save_curriculum, clue_count_to_difficulty


def _strip_quotes(s: str) -> str:
    """The HF CSV wraps each field's content in literal single-quotes."""
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    return s


def _puzzle_to_tensor(s: str) -> torch.Tensor:
    """Convert an 81-char string ('.'/'0' = empty, '1'-'9' = digit) to int8 (81,)."""
    if len(s) != 81:
        raise ValueError(f"expected 81 chars, got {len(s)}")
    out = torch.zeros(81, dtype=torch.int8)
    for i, c in enumerate(s):
        if c == '.' or c == '0':
            out[i] = 0
        else:
            out[i] = int(c)
    return out


def load_hf_pairs(target_count: int, split: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    fp = hf_hub_download("sapientinc/sudoku-extreme", f"{split}.csv", repo_type="dataset")
    print(f"reading {fp}")
    # Reservoir-sample target_count rows so we don't load the whole 700+ MB file.
    sampled: list[tuple[str, str]] = []
    n_seen = 0
    with open(fp, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = _strip_quotes(row["question"])
            a = _strip_quotes(row["answer"])
            if len(q) != 81 or len(a) != 81:
                continue
            if len(sampled) < target_count:
                sampled.append((q, a))
            else:
                # Reservoir sampling: replace with probability target_count / n_seen
                j = rng.randint(0, n_seen)
                if j < target_count:
                    sampled[j] = (q, a)
            n_seen += 1
            if n_seen % 100000 == 0:
                print(f"  scanned {n_seen} rows, kept {len(sampled)}")
    print(f"  scanned {n_seen} total rows; sampled {len(sampled)}")

    puzzles = torch.stack([_puzzle_to_tensor(q) for q, _ in sampled])
    solutions = torch.stack([_puzzle_to_tensor(a) for _, a in sampled])
    return puzzles, solutions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-count", type=int, default=5000)
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--out", type=str, default="data/hf_poc.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--curriculum", action="store_true",
                    help="emit curriculum-tagged shards (puzzles + solutions + difficulty)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    puzzles, solutions = load_hf_pairs(args.target_count, args.split, args.seed)

    n_clues = (puzzles != 0).sum(dim=1)
    print(f"clue counts: min={int(n_clues.min())} max={int(n_clues.max())} mean={n_clues.float().mean().item():.1f}")

    if args.curriculum:
        difficulty = torch.tensor(
            [clue_count_to_difficulty(int(c)) for c in n_clues.tolist()],
            dtype=torch.int8,
        )
        n_easy = int((difficulty == 0).sum())
        n_med = int((difficulty == 1).sum())
        n_extreme = int((difficulty == 2).sum())
        print(f"difficulty breakdown: easy={n_easy} medium={n_med} extreme={n_extreme}")
        save_curriculum(out, puzzles, solutions, difficulty)
        print(f"wrote {puzzles.shape[0]} curriculum-tagged samples to {out} "
              f"in {time.perf_counter() - t0:.1f}s")
    else:
        save_pairs(out, puzzles, solutions)
        print(f"wrote {puzzles.shape[0]} (puzzle, solution) pairs to {out} "
              f"in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
