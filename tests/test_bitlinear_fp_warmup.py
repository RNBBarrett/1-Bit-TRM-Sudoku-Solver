"""Tests for BitLinear FP-warmup mode (per BitNet b1.58 recipe).

During the first ~20% of training, BitLinear runs without ternarization
so the shadow weights can warm up in full precision; quantization is
enabled for the remaining steps.
"""
import torch

from htrm.bitlinear import BitLinear, set_quantization_enabled
from htrm.config import HTRMConfig
from htrm.htrm_model import HTRM


def test_bitlinear_fp_mode_skips_quantization():
    layer = BitLinear(8, 4)
    set_quantization_enabled(layer, enabled=False)
    x = torch.randn(2, 3, 8)
    y_fp = layer(x)
    set_quantization_enabled(layer, enabled=True)
    y_ternary = layer(x)
    # FP-mode and ternary-mode should produce different outputs (otherwise
    # ternarization is a no-op, which means the test isn't actually exercising it).
    assert not torch.allclose(y_fp, y_ternary, atol=1e-3)


def test_bitlinear_fp_mode_gradients_flow():
    layer = BitLinear(8, 4)
    set_quantization_enabled(layer, enabled=False)
    x = torch.randn(2, 3, 8, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_set_quantization_enabled_walks_module_tree():
    cfg = HTRMConfig(hidden_dim=64, n_layers_per_block=1, K=2, L=1, P=1, T=1)
    model = HTRM(cfg)
    set_quantization_enabled(model, enabled=False)
    # Every BitLinear in the model should have its flag flipped.
    for m in model.modules():
        if isinstance(m, BitLinear):
            assert m.quantization_enabled is False
    set_quantization_enabled(model, enabled=True)
    for m in model.modules():
        if isinstance(m, BitLinear):
            assert m.quantization_enabled is True


def test_default_quantization_is_enabled():
    layer = BitLinear(8, 4)
    assert layer.quantization_enabled is True
