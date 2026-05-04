"""Curriculum-aware dataset tests."""
from pathlib import Path

import torch

from htrm.dataset import (
    CurriculumSudokuDataset,
    save_curriculum,
    DIFFICULTY_EASY,
    DIFFICULTY_MEDIUM,
    DIFFICULTY_EXTREME,
    clue_count_to_difficulty,
)


def test_clue_count_to_difficulty_thresholds():
    # Calibrated for Samsung's sapientinc/sudoku-extreme distribution
    # (mean ~25 clues): Easy >= 30, Medium 23-29, Extreme <= 22.
    assert clue_count_to_difficulty(35) == DIFFICULTY_EASY
    assert clue_count_to_difficulty(30) == DIFFICULTY_EASY
    assert clue_count_to_difficulty(29) == DIFFICULTY_MEDIUM
    assert clue_count_to_difficulty(23) == DIFFICULTY_MEDIUM
    assert clue_count_to_difficulty(22) == DIFFICULTY_EXTREME
    assert clue_count_to_difficulty(17) == DIFFICULTY_EXTREME


def test_save_and_load_curriculum_round_trip(tmp_path: Path):
    puzzles = torch.zeros(4, 81, dtype=torch.int8)
    solutions = torch.zeros(4, 81, dtype=torch.int8)
    difficulty = torch.tensor([0, 1, 2, 0], dtype=torch.int8)
    out = tmp_path / "curric.pt"
    save_curriculum(out, puzzles, solutions, difficulty)
    ds = CurriculumSudokuDataset(out)
    assert len(ds) == 4
    assert torch.equal(ds.difficulty, difficulty.long())


def test_curriculum_dataset_getitem_returns_puzzle_solution_difficulty(tmp_path: Path):
    puzzles = torch.tensor([[i] * 81 for i in range(3)], dtype=torch.int8)
    solutions = torch.tensor([[i + 1] * 81 for i in range(3)], dtype=torch.int8)
    difficulty = torch.tensor([0, 1, 2], dtype=torch.int8)
    out = tmp_path / "c.pt"
    save_curriculum(out, puzzles, solutions, difficulty)
    ds = CurriculumSudokuDataset(out)
    p, s, d = ds[1]
    assert p.dtype == torch.long and p.shape == (81,)
    assert s.dtype == torch.long and s.shape == (81,)
    assert int(d.item()) == 1


def test_make_weighted_sampler_respects_tier_weights(tmp_path: Path):
    # 100 examples, 50 Easy / 30 Medium / 20 Extreme
    n_each = [50, 30, 20]
    diff = torch.cat([
        torch.full((n,), tier, dtype=torch.int8)
        for tier, n in enumerate(n_each)
    ])
    puzzles = torch.zeros(100, 81, dtype=torch.int8)
    solutions = torch.zeros(100, 81, dtype=torch.int8)
    out = tmp_path / "c.pt"
    save_curriculum(out, puzzles, solutions, diff)
    ds = CurriculumSudokuDataset(out)
    # Skewed tier weights: only sample from Extreme.
    sampler = ds.make_weighted_sampler({0: 0.0, 1: 0.0, 2: 1.0}, num_samples=200, seed=0)
    seen_difficulties = set()
    for idx in sampler:
        d = int(ds.difficulty[idx].item())
        seen_difficulties.add(d)
    assert seen_difficulties == {2}


def test_make_weighted_sampler_uniform_tier_weights(tmp_path: Path):
    diff = torch.cat([
        torch.full((50,), 0, dtype=torch.int8),
        torch.full((30,), 1, dtype=torch.int8),
        torch.full((20,), 2, dtype=torch.int8),
    ])
    puzzles = torch.zeros(100, 81, dtype=torch.int8)
    solutions = torch.zeros(100, 81, dtype=torch.int8)
    out = tmp_path / "c.pt"
    save_curriculum(out, puzzles, solutions, diff)
    ds = CurriculumSudokuDataset(out)
    sampler = ds.make_weighted_sampler({0: 1.0, 1: 1.0, 2: 1.0}, num_samples=10000, seed=0)
    counts = [0, 0, 0]
    for idx in sampler:
        counts[int(ds.difficulty[idx].item())] += 1
    # With uniform per-puzzle weights but unequal tier sizes, the sampler
    # should still draw roughly uniformly across puzzles (50/30/20 bias)
    # since uniform weight per puzzle == frequency-proportional sampling.
    total = sum(counts)
    assert abs(counts[0] / total - 0.50) < 0.05
    assert abs(counts[1] / total - 0.30) < 0.05
    assert abs(counts[2] / total - 0.20) < 0.05
