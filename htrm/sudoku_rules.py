"""Sudoku rule structure: precomputed group indices and violation counters.

The 81-cell flat layout numbers cells row-major: cell (r, c) lives at
index r*9 + c. We precompute:
  - ROW_INDICES[r]: 9 cell indices in row r
  - COL_INDICES[c]: 9 cell indices in column c
  - BOX_INDICES[b]: 9 cell indices in 3x3 box b (boxes numbered row-major)

Two violation utilities are provided:
  - count_violations(grid): hard, integer count of duplicate digits per
    row/col/box on a fully-filled (or partially-filled) int grid; zeros
    are treated as empty and ignored.
  - soft_group_violation(p_digits): differentiable surrogate that
    penalizes the soft probability mass exceeding 1.0 per (group, digit).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _build_indices() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.zeros(9, 9, dtype=torch.long)
    cols = torch.zeros(9, 9, dtype=torch.long)
    boxes = torch.zeros(9, 9, dtype=torch.long)
    for r in range(9):
        for c in range(9):
            rows[r, c] = r * 9 + c
            cols[c, r] = r * 9 + c
            b = (r // 3) * 3 + (c // 3)
            # Position inside box: row-within-box * 3 + col-within-box
            slot = (r % 3) * 3 + (c % 3)
            boxes[b, slot] = r * 9 + c
    return rows, cols, boxes


ROW_INDICES, COL_INDICES, BOX_INDICES = _build_indices()


def count_violations(grid: torch.Tensor) -> int:
    """Count Sudoku rule violations on a flat (81,) integer grid.

    A violation is any pair of cells in the same row/col/box that both
    hold the same nonzero digit. Returns the total number of such pairs
    summed across all 27 groups.

    Zero entries (unfilled cells) are ignored.
    """
    if grid.dim() != 1 or grid.numel() != 81:
        raise ValueError(f"expected flat (81,) grid, got shape {grid.shape}")
    total = 0
    for indices in (ROW_INDICES, COL_INDICES, BOX_INDICES):
        for group in indices:
            values = grid[group]
            nonzero = values[values != 0]
            if nonzero.numel() == 0:
                continue
            counts = torch.bincount(nonzero, minlength=10)
            # Each digit that appears k times contributes (k - 1) violations
            total += int((counts - 1).clamp(min=0).sum().item())
    return total


def soft_group_violation(p_digits: torch.Tensor) -> torch.Tensor:
    """Differentiable group-violation penalty on digit probabilities.

    p_digits: (B, 81, 9) — softmax probability that each cell holds each
    of the digits 1..9 (slot 0 = empty cell is dropped before calling).

    For each group of 9 cells, the expected count of digit d is
    sum_cells p[cell, d]. A valid solution has count = 1 for every
    (group, digit) pair. We penalize relu(count - 1)^2 — only excess
    above 1 is bad; absence (< 1) is allowed since incomplete grids have
    zeros. Returns a scalar.
    """
    # Group → (B, 9_groups, 9_cells, 9_digits)
    rows = p_digits[:, ROW_INDICES]   # broadcast-gather
    cols = p_digits[:, COL_INDICES]
    boxes = p_digits[:, BOX_INDICES]

    def excess(p_grouped: torch.Tensor) -> torch.Tensor:
        counts = p_grouped.sum(dim=2)
        return F.relu(counts - 1.0).pow(2).mean()

    return excess(rows) + excess(cols) + excess(boxes)
