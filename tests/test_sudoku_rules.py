import torch

from htrm.sudoku_rules import (
    ROW_INDICES,
    COL_INDICES,
    BOX_INDICES,
    count_violations,
    soft_group_violation,
)


def test_row_indices_shape_and_coverage():
    assert ROW_INDICES.shape == (9, 9)
    # Each row group should contain exactly the 9 cells of that row in the
    # flattened 81-cell layout: row r covers cells [r*9, r*9+1, ..., r*9+8].
    for r in range(9):
        expected = set(range(r * 9, r * 9 + 9))
        assert set(ROW_INDICES[r].tolist()) == expected


def test_col_indices_shape_and_coverage():
    assert COL_INDICES.shape == (9, 9)
    for c in range(9):
        expected = {r * 9 + c for r in range(9)}
        assert set(COL_INDICES[c].tolist()) == expected


def test_box_indices_shape_and_coverage():
    assert BOX_INDICES.shape == (9, 9)
    # Box 0 = top-left 3x3 → flat cells {0,1,2,9,10,11,18,19,20}
    expected_box_0 = {0, 1, 2, 9, 10, 11, 18, 19, 20}
    assert set(BOX_INDICES[0].tolist()) == expected_box_0
    # Box 4 = center 3x3 → rows 3-5 cols 3-5 → flat {30,31,32,39,40,41,48,49,50}
    expected_box_4 = {30, 31, 32, 39, 40, 41, 48, 49, 50}
    assert set(BOX_INDICES[4].tolist()) == expected_box_4


def test_all_indices_partition_81_cells():
    # All rows together must cover every cell exactly once (same for cols, boxes)
    for indices in (ROW_INDICES, COL_INDICES, BOX_INDICES):
        flat = sorted(indices.flatten().tolist())
        assert flat == list(range(81))


def test_count_violations_returns_zero_for_valid_solved_grid():
    # A canonical valid 9x9 Sudoku solution.
    solved = torch.tensor([
        5, 3, 4, 6, 7, 8, 9, 1, 2,
        6, 7, 2, 1, 9, 5, 3, 4, 8,
        1, 9, 8, 3, 4, 2, 5, 6, 7,
        8, 5, 9, 7, 6, 1, 4, 2, 3,
        4, 2, 6, 8, 5, 3, 7, 9, 1,
        7, 1, 3, 9, 2, 4, 8, 5, 6,
        9, 6, 1, 5, 3, 7, 2, 8, 4,
        2, 8, 7, 4, 1, 9, 6, 3, 5,
        3, 4, 5, 2, 8, 6, 1, 7, 9,
    ])
    assert count_violations(solved) == 0


def test_count_violations_detects_row_duplicate():
    solved = torch.tensor([
        5, 3, 4, 6, 7, 8, 9, 1, 2,
        6, 7, 2, 1, 9, 5, 3, 4, 8,
        1, 9, 8, 3, 4, 2, 5, 6, 7,
        8, 5, 9, 7, 6, 1, 4, 2, 3,
        4, 2, 6, 8, 5, 3, 7, 9, 1,
        7, 1, 3, 9, 2, 4, 8, 5, 6,
        9, 6, 1, 5, 3, 7, 2, 8, 4,
        2, 8, 7, 4, 1, 9, 6, 3, 5,
        3, 4, 5, 2, 8, 6, 1, 7, 9,
    ])
    # Inject a duplicate 5 in the first row (replace the 4 at index 2)
    broken = solved.clone()
    broken[2] = 5  # row 0 now has two 5s
    assert count_violations(broken) > 0


def test_count_violations_ignores_zeros():
    # A grid with zeros (unfilled cells) should not generate violations
    # for those zeros — only real digit collisions count.
    grid = torch.zeros(81, dtype=torch.long)
    grid[0] = 5
    grid[1] = 5  # same row, both nonzero, this IS a violation
    assert count_violations(grid) > 0
    grid2 = torch.zeros(81, dtype=torch.long)
    grid2[0] = 5  # only one digit total
    assert count_violations(grid2) == 0


def test_soft_group_violation_zero_for_perfect_one_hot_solution():
    # Build perfect probabilities: one-hot at each cell matching a valid grid.
    solved = torch.tensor([
        5, 3, 4, 6, 7, 8, 9, 1, 2,
        6, 7, 2, 1, 9, 5, 3, 4, 8,
        1, 9, 8, 3, 4, 2, 5, 6, 7,
        8, 5, 9, 7, 6, 1, 4, 2, 3,
        4, 2, 6, 8, 5, 3, 7, 9, 1,
        7, 1, 3, 9, 2, 4, 8, 5, 6,
        9, 6, 1, 5, 3, 7, 2, 8, 4,
        2, 8, 7, 4, 1, 9, 6, 3, 5,
        3, 4, 5, 2, 8, 6, 1, 7, 9,
    ])
    # Vocab: 0 = empty, 1..9 = digits. Build (B=1, 81, 10) one-hot.
    probs = torch.zeros(1, 81, 10)
    probs[0, torch.arange(81), solved] = 1.0
    # soft_group_violation only looks at digit slots 1..9
    p_d = probs[..., 1:10]
    v = soft_group_violation(p_d)
    assert v.item() < 1e-6


def test_soft_group_violation_positive_for_duplicates():
    # All cells assigned digit '1' with probability 1 → massive duplicates.
    probs = torch.zeros(1, 81, 10)
    probs[0, :, 1] = 1.0
    p_d = probs[..., 1:10]
    v = soft_group_violation(p_d)
    assert v.item() > 0.0


def test_soft_group_violation_is_differentiable():
    logits = torch.randn(1, 81, 10, requires_grad=True)
    p_d = torch.softmax(logits, dim=-1)[..., 1:10]
    v = soft_group_violation(p_d)
    v.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
