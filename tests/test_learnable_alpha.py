"""Learnable per-layer alpha tests (TTQ-style, BitNet Reloaded).

When `learnable_alpha=True`, BitLinear stores a single scalar `alpha`
parameter that is initialized from the data-driven absmean (or median)
on the first forward, then updated by the optimizer like any other
parameter. This is more robust than recomputing 1/absmean each forward
for sub-100M-param ternary models.
"""
import torch

from htrm.bitlinear import BitLinear


def test_learnable_alpha_creates_parameter():
    layer = BitLinear(16, 8, learnable_alpha=True)
    assert layer.alpha is not None
    assert isinstance(layer.alpha, torch.nn.Parameter)
    assert layer.alpha.requires_grad


def test_default_alpha_is_none():
    """With learnable_alpha=False (default), no alpha parameter exists."""
    layer = BitLinear(16, 8, learnable_alpha=False)
    assert layer.alpha is None


def test_alpha_initialized_lazily_from_weight_stats_on_first_forward():
    torch.manual_seed(0)
    layer = BitLinear(16, 8, learnable_alpha=True)
    # Before forward: alpha is the placeholder 1.0
    assert float(layer.alpha.item()) == 1.0
    # After forward: alpha matches 1 / (W.abs().mean() + eps)
    x = torch.randn(2, 5, 16)
    _ = layer(x)
    expected = 1.0 / (layer.weight.abs().mean().item() + 1e-5)
    assert abs(layer.alpha.item() - expected) < 1e-3


def test_median_init_used_when_use_median_true():
    torch.manual_seed(1)
    layer = BitLinear(16, 8, learnable_alpha=True, use_median=True)
    x = torch.randn(2, 5, 16)
    _ = layer(x)
    expected = 1.0 / (layer.weight.abs().median().item() + 1e-5)
    assert abs(layer.alpha.item() - expected) < 1e-3


def test_quantized_weights_in_ternary_set_when_learnable_alpha():
    """With learnable_alpha=True, weight_quant should still produce values
    in {-1/alpha, 0, +1/alpha} (i.e., the ternary set scaled by 1/alpha)."""
    torch.manual_seed(2)
    layer = BitLinear(32, 16, learnable_alpha=True)
    x = torch.randn(2, 5, 32)
    _ = layer(x)  # initialize alpha
    from htrm.bitlinear import weight_quant
    Wq = weight_quant(layer.weight, alpha=layer.alpha)
    raw = (Wq * layer.alpha).round()
    unique = torch.unique(raw).tolist()
    assert set(unique).issubset({-1.0, 0.0, 1.0})


def test_alpha_receives_gradient():
    """The optimizer should be able to update alpha via backprop."""
    torch.manual_seed(3)
    layer = BitLinear(16, 8, learnable_alpha=True)
    x = torch.randn(2, 5, 16)
    y = layer(x)
    y.sum().backward()
    assert layer.alpha.grad is not None
    assert torch.isfinite(layer.alpha.grad).all()
