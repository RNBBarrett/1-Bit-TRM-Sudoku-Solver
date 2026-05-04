import torch

from htrm.bitlinear import (
    weight_quant,
    activation_quant,
    ste,
    RMSNorm,
    BitLinear,
)


def test_weight_quant_values_are_ternary_after_rescaling():
    torch.manual_seed(0)
    W = torch.randn(64, 32)
    Wq = weight_quant(W)
    scale = 1.0 / (W.abs().mean() + 1e-5)
    raw = (Wq * scale).round()
    unique = torch.unique(raw).tolist()
    assert set(unique).issubset({-1.0, 0.0, 1.0})


def test_weight_quant_zero_input_returns_zero():
    W = torch.zeros(8, 4)
    Wq = weight_quant(W)
    assert torch.allclose(Wq, torch.zeros_like(Wq))


def test_weight_quant_is_antisymmetric():
    torch.manual_seed(1)
    W = torch.randn(16, 8)
    assert torch.allclose(weight_quant(-W), -weight_quant(W))


def test_activation_quant_per_token_scaling_ranges():
    torch.manual_seed(2)
    X = torch.randn(2, 5, 8) * 3.0
    Xq = activation_quant(X)
    # Each token's quantized values should be within +/- max(|X_token|) bounds.
    max_abs_per_token = X.abs().amax(dim=-1, keepdim=True)
    assert (Xq.abs() <= max_abs_per_token + 1e-4).all()


def test_activation_quant_distinct_levels_per_token_at_most_256():
    torch.manual_seed(3)
    X = torch.randn(1, 1, 1024)
    Xq = activation_quant(X)
    # int8 has 256 distinct integer levels; after dequant, distinct floats <= 256
    assert torch.unique(Xq).numel() <= 256


def test_ste_forward_returns_quantized_value():
    x = torch.tensor([0.3, -0.7, 1.2])
    x_q = torch.tensor([0.0, -1.0, 1.0])
    out = ste(x, x_q)
    assert torch.equal(out, x_q)


def test_ste_backward_passes_gradient_through_as_identity():
    x = torch.tensor([0.3, -0.7, 1.2], requires_grad=True)
    x_q = (x * 0.0).detach()  # quantization that destroys all info
    out = ste(x, x_q)
    out.sum().backward()
    # STE: gradient is dout/dx = 1, so x.grad should be ones
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_rmsnorm_output_has_unit_rms_when_weight_is_one():
    norm = RMSNorm(8)
    x = torch.randn(2, 4, 8) * 5.0
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rmsnorm_weight_is_learnable_parameter():
    norm = RMSNorm(8)
    params = list(norm.parameters())
    assert len(params) == 1
    assert params[0].shape == (8,)
    assert params[0].requires_grad


def test_bitlinear_output_shape_matches_linear():
    layer = BitLinear(16, 32)
    x = torch.randn(4, 5, 16)
    y = layer(x)
    assert y.shape == (4, 5, 32)


def test_bitlinear_gradient_flows_to_weight_shadow():
    layer = BitLinear(8, 4)
    x = torch.randn(2, 3, 8)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.abs().sum() > 0


def test_bitlinear_no_nan_on_forward():
    torch.manual_seed(4)
    layer = BitLinear(64, 64)
    x = torch.randn(2, 16, 64) * 10.0
    y = layer(x)
    assert torch.isfinite(y).all()
