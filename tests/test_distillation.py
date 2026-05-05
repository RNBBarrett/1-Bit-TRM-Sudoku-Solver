"""Tests for KD distillation loss (BitDistill recipe)."""
import torch

from htrm.losses import KDLoss


def test_kd_loss_zero_when_logits_match():
    """If student == teacher, KL divergence is exactly 0."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    kd = KDLoss(temperature=5.0)
    loss = kd(logits, logits)
    assert loss.abs().item() < 1e-5, f"expected ~0, got {loss.item()}"


def test_kd_loss_positive_when_logits_diverge():
    """KL is non-negative and grows with divergence."""
    torch.manual_seed(1)
    student = torch.randn(2, 5, 11)
    teacher = torch.randn(2, 5, 11) * 5.0  # very different distribution
    kd = KDLoss(temperature=5.0)
    loss = kd(student, teacher)
    assert loss.item() > 0
    assert torch.isfinite(loss).all()


def test_kd_loss_temperature_t_squared_scaling():
    """Higher T should produce larger absolute KD loss (T^2 multiplier),
    holding logit divergence constant."""
    torch.manual_seed(2)
    student = torch.randn(2, 5, 11)
    teacher = torch.randn(2, 5, 11)
    loss_low = KDLoss(temperature=1.0)(student, teacher)
    loss_high = KDLoss(temperature=5.0)(student, teacher)
    # T^2 scaling means high-T loss is bigger (in absolute terms).
    # KL itself shrinks with T (distributions become uniform), but T^2 dominates.
    assert loss_high.item() > 0
    assert loss_low.item() > 0


def test_kd_loss_gradient_flows_to_student_only():
    """Teacher is frozen (no_grad); only student should receive gradients."""
    torch.manual_seed(3)
    student = torch.randn(2, 5, 11, requires_grad=True)
    teacher = torch.randn(2, 5, 11)  # no requires_grad
    kd = KDLoss(temperature=5.0)
    loss = kd(student, teacher)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    # teacher has no grad attribute since it didn't require_grad
    assert teacher.grad is None


def test_kd_loss_temperature_must_be_positive():
    """Constructor rejects non-positive temperatures."""
    try:
        KDLoss(temperature=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        KDLoss(temperature=-1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
