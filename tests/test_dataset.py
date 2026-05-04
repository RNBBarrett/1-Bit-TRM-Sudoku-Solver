import torch
from pathlib import Path

from htrm.dataset import SudokuDataset, save_pairs, load_pairs


def test_save_and_load_round_trip(tmp_path: Path):
    puzzles = torch.tensor([[0, 1, 2] + [0] * 78, [3, 0, 5] + [0] * 78], dtype=torch.int8)
    solutions = torch.tensor([[1, 1, 2] + [0] * 78, [3, 4, 5] + [0] * 78], dtype=torch.int8)
    out = tmp_path / "data.pt"
    save_pairs(out, puzzles, solutions)
    loaded_p, loaded_s = load_pairs(out)
    assert torch.equal(loaded_p, puzzles)
    assert torch.equal(loaded_s, solutions)


def test_dataset_length(tmp_path: Path):
    puzzles = torch.zeros(7, 81, dtype=torch.int8)
    solutions = torch.zeros(7, 81, dtype=torch.int8)
    out = tmp_path / "data.pt"
    save_pairs(out, puzzles, solutions)
    ds = SudokuDataset(out)
    assert len(ds) == 7


def test_dataset_returns_long_tensors_of_correct_shape(tmp_path: Path):
    N = 3
    puzzles = torch.randint(0, 10, (N, 81), dtype=torch.int8)
    solutions = torch.randint(0, 10, (N, 81), dtype=torch.int8)
    out = tmp_path / "data.pt"
    save_pairs(out, puzzles, solutions)
    ds = SudokuDataset(out)
    p, s = ds[0]
    assert p.shape == (81,)
    assert s.shape == (81,)
    assert p.dtype == torch.long
    assert s.dtype == torch.long


def test_dataset_iteration_yields_all_examples(tmp_path: Path):
    puzzles = torch.tensor([[i] * 81 for i in range(5)], dtype=torch.int8)
    solutions = torch.tensor([[i + 1] * 81 for i in range(5)], dtype=torch.int8)
    out = tmp_path / "data.pt"
    save_pairs(out, puzzles, solutions)
    ds = SudokuDataset(out)
    seen = []
    for p, s in ds:
        seen.append(int(p[0].item()))
    assert seen == [0, 1, 2, 3, 4]
