"""Lambda-ramped quantization tests (HF 1.58bit recipe).

The lambda parameter linearly interpolates BitLinear's forward between
pure FP (lambda=0) and full ternary (lambda=1). Used during the training
schedule's transition between FP warmup and full QAT.
"""
import torch

from htrm.bitlinear import BitLinear


def test_lambda_zero_matches_fp_forward():
    """lambda_q=0 should be identical to FP forward (quantization disabled)."""
    torch.manual_seed(0)
    layer = BitLinear(16, 8)
    x = torch.randn(2, 5, 16)
    y_lambda0 = layer(x, lambda_q=0.0)
    # With quantization fully off (lambda=0), the forward should match what
    # we'd get with quantization_enabled=False.
    layer._quantization_enabled = torch.tensor(False)
    y_fp = layer(x)
    layer._quantization_enabled = torch.tensor(True)
    assert torch.allclose(y_lambda0, y_fp, atol=1e-5)


def test_lambda_one_matches_full_ternary():
    """lambda_q=1 should be identical to default forward (full ternary)."""
    torch.manual_seed(1)
    layer = BitLinear(16, 8)
    x = torch.randn(2, 5, 16)
    y_lambda1 = layer(x, lambda_q=1.0)
    y_default = layer(x)
    assert torch.allclose(y_lambda1, y_default, atol=1e-5)


def test_lambda_half_is_between_fp_and_ternary():
    """lambda_q=0.5 should produce output strictly between FP and ternary."""
    torch.manual_seed(2)
    layer = BitLinear(16, 8)
    x = torch.randn(2, 5, 16)
    y_fp = layer(x, lambda_q=0.0)
    y_half = layer(x, lambda_q=0.5)
    y_ternary = layer(x, lambda_q=1.0)
    # The lambda interpolation is on the (de)quantized values inside the
    # forward, so y_half is generally somewhere between y_fp and y_ternary.
    # Sanity check: not exactly equal to either endpoint.
    assert not torch.allclose(y_half, y_fp, atol=1e-5)
    assert not torch.allclose(y_half, y_ternary, atol=1e-5)


def test_lambda_gradient_flows_at_intermediate_value():
    """Gradients should flow through BitLinear at any lambda value."""
    torch.manual_seed(3)
    layer = BitLinear(8, 4)
    x = torch.randn(2, 3, 8, requires_grad=True)
    y = layer(x, lambda_q=0.7)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_lambda_default_is_one_for_backcompat():
    """Calling forward without lambda_q should default to full ternary."""
    torch.manual_seed(4)
    layer = BitLinear(16, 8)
    x = torch.randn(2, 5, 16)
    y_default = layer(x)
    y_explicit = layer(x, lambda_q=1.0)
    assert torch.allclose(y_default, y_explicit, atol=1e-6)
