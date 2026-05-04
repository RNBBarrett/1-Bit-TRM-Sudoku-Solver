import torch

from htrm.losses import HTRMLoss


def _perfect_logits(target: torch.Tensor, vocab_size: int = 11) -> torch.Tensor:
    """Build (B, 81, V) logits that put 100x mass on each target index."""
    B, S = target.shape
    logits = torch.full((B, S, vocab_size), -100.0)
    logits.scatter_(2, target.unsqueeze(-1), 100.0)
    return logits


def test_loss_total_combines_components_with_correct_weights():
    loss_fn = HTRMLoss(violation_weight=10.0, halt_weight=0.1)
    target = torch.randint(0, 10, (2, 81))
    logits = torch.randn(2, 81, 11)
    halts = torch.rand(2, 4, 1)
    out = loss_fn(logits, target, halts)
    expected = out["ce"] + 10.0 * out["violation"] + 0.1 * out["halt"]
    assert torch.allclose(out["total"], expected, atol=1e-5)


def test_loss_zero_when_predictions_perfect_and_solution_valid():
    # Use a known-valid 9x9 Sudoku solution as target so violation==0 too.
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
    ]).unsqueeze(0)  # (1, 81)
    loss_fn = HTRMLoss()
    logits = _perfect_logits(solved, vocab_size=11)
    halts = torch.tensor([[[1.0]]])  # halt head says "correct" — matches actual
    out = loss_fn(logits, solved, halts)
    assert out["ce"].item() < 1e-3
    assert out["violation"].item() < 1e-3


def test_loss_violation_fires_on_predicted_duplicates():
    # Predict digit '1' in every cell — every group has 9 duplicates.
    target = torch.zeros(1, 81, dtype=torch.long)  # (B, 81)
    target[0, :] = 1  # target also all 1s — CE will be 0, violation will spike
    logits = _perfect_logits(target, vocab_size=11)
    loss_fn = HTRMLoss()
    halts = torch.zeros(1, 1, 1)
    out = loss_fn(logits, target, halts)
    assert out["violation"].item() > 0.5


def test_loss_gradient_flows_to_logits():
    target = torch.randint(0, 10, (2, 81))
    logits = torch.randn(2, 81, 11, requires_grad=True)
    halts = torch.rand(2, 3, 1, requires_grad=True)
    loss_fn = HTRMLoss()
    out = loss_fn(logits, target, halts)
    out["total"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_halt_loss_targets_per_example_correctness():
    # Two-example batch: predict perfectly on one, wrongly on the other.
    target = torch.zeros(2, 81, dtype=torch.long)
    logits = torch.full((2, 81, 11), -100.0)
    logits[0].scatter_(1, target[0].unsqueeze(-1), 100.0)  # example 0: correct
    logits[1, :, 5] = 100.0                                # example 1: all 5s, wrong
    # Halt head predicts 0.5 for both — should be pushed high for ex0, low for ex1.
    halts = torch.full((2, 1, 1), 0.5)
    loss_fn = HTRMLoss()
    out = loss_fn(logits, target, halts)
    # Halt loss > 0 because halts (0.5) don't match true correctness (1, 0)
    assert out["halt"].item() > 0
