"""Sudoku datasets: simple (puzzle, solution) pairs and curriculum-tagged variants.

`SudokuDataset` is the POC-scale dataset used for the warm-up runs.
`CurriculumSudokuDataset` extends it with per-puzzle difficulty tags
(0=Easy, 1=Medium, 2=Extreme by clue count) and a weighted sampler
factory for full-spec curriculum staging.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, WeightedRandomSampler


DIFFICULTY_EASY = 0
DIFFICULTY_MEDIUM = 1
DIFFICULTY_EXTREME = 2


def clue_count_to_difficulty(n_clues: int) -> int:
    """Map a Sudoku puzzle's clue count to a difficulty tier.

    Calibrated for Samsung's sapientinc/sudoku-extreme distribution
    (mean ~25, range 17-35 clues out of 81):
      - Easy:    >= 30 clues  (least reasoning required)
      - Medium:  23-29 clues
      - Extreme: <= 22 clues  (most reasoning required)
    """
    if n_clues >= 30:
        return DIFFICULTY_EASY
    if n_clues >= 23:
        return DIFFICULTY_MEDIUM
    return DIFFICULTY_EXTREME


def save_pairs(path: str | Path, puzzles: torch.Tensor, solutions: torch.Tensor) -> None:
    """Save (N, 81) puzzle and solution tensors to a single .pt file."""
    if puzzles.shape != solutions.shape:
        raise ValueError(f"shape mismatch: {puzzles.shape} vs {solutions.shape}")
    if puzzles.shape[1] != 81:
        raise ValueError(f"expected 81 cells per row, got {puzzles.shape[1]}")
    torch.save(
        {"puzzles": puzzles.to(torch.int8), "solutions": solutions.to(torch.int8)},
        Path(path),
    )


def load_pairs(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (puzzles, solutions) tensors from a .pt file written by save_pairs."""
    blob = torch.load(Path(path), weights_only=True)
    return blob["puzzles"], blob["solutions"]


def save_curriculum(
    path: str | Path,
    puzzles: torch.Tensor,
    solutions: torch.Tensor,
    difficulty: torch.Tensor,
) -> None:
    """Save (puzzles, solutions, difficulty) for curriculum-aware training."""
    if puzzles.shape != solutions.shape:
        raise ValueError(f"shape mismatch: {puzzles.shape} vs {solutions.shape}")
    if difficulty.shape[0] != puzzles.shape[0]:
        raise ValueError(f"difficulty length {difficulty.shape[0]} != N={puzzles.shape[0]}")
    torch.save({
        "puzzles": puzzles.to(torch.int8),
        "solutions": solutions.to(torch.int8),
        "difficulty": difficulty.to(torch.int8),
    }, Path(path))


class SudokuDataset(Dataset):
    """In-memory dataset of (puzzle, solution) int8 pairs cast to int64 on access."""

    def __init__(self, path: str | Path):
        self.puzzles, self.solutions = load_pairs(path)

    def __len__(self) -> int:
        return self.puzzles.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.puzzles[idx].long(), self.solutions[idx].long()


class CurriculumSudokuDataset(Dataset):
    """Dataset of (puzzle, solution, difficulty) triples for curriculum sampling.

    Backward-compatible: also reads files written by `save_pairs` (no
    difficulty key) by treating all puzzles as Easy.
    """

    def __init__(self, path: str | Path):
        blob = torch.load(Path(path), weights_only=True)
        self.puzzles: torch.Tensor = blob["puzzles"]
        self.solutions: torch.Tensor = blob["solutions"]
        if "difficulty" in blob:
            self.difficulty: torch.Tensor = blob["difficulty"].long()
        else:
            self.difficulty = torch.zeros(self.puzzles.shape[0], dtype=torch.long)

    def __len__(self) -> int:
        return self.puzzles.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.puzzles[idx].long(),
            self.solutions[idx].long(),
            self.difficulty[idx],
        )

    def make_weighted_sampler(
        self,
        tier_weights: dict[int, float],
        num_samples: int,
        seed: int | None = None,
    ) -> WeightedRandomSampler:
        """Build a sampler that draws each puzzle with weight = tier_weights[difficulty].

        Within a tier, puzzles are drawn uniformly. Between tiers, draws are
        proportional to (tier_weights[t] * count_in_tier_t).

        Pass `tier_weights={0: 0, 1: 0, 2: 1}` to sample only Extreme;
        pass `tier_weights={0: 1, 1: 1, 2: 1}` for frequency-proportional
        (uniform per puzzle, unequal across tiers if tier counts differ).
        """
        weights = torch.tensor(
            [tier_weights.get(int(d.item()), 0.0) for d in self.difficulty],
            dtype=torch.float64,
        )
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        return WeightedRandomSampler(
            weights=weights, num_samples=num_samples, replacement=True, generator=gen,
        )
