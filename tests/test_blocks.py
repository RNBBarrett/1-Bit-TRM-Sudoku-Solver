import torch

from htrm.blocks import BitMLPBlock, Strategist, Tactician, HaltingHead


def test_bitmlpblock_preserves_shape():
    block = BitMLPBlock(dim=32, mlp_ratio=4)
    x = torch.randn(2, 7, 32)
    y = block(x)
    assert y.shape == x.shape


def test_bitmlpblock_is_residual():
    # With initialized-to-small weights inside BitLinear, the residual
    # connection should make the output close to the input.
    torch.manual_seed(0)
    block = BitMLPBlock(dim=16, mlp_ratio=4)
    x = torch.randn(1, 5, 16) * 0.1
    y = block(x)
    # Output is x + delta — confirm there's a delta but it's not catastrophic
    assert not torch.allclose(y, x)
    assert torch.isfinite(y).all()


def test_strategist_inner_step_preserves_shape():
    s = Strategist(dim=32, mlp_ratio=4, n_layers=1)
    x = torch.randn(2, 81, 32)
    y = torch.randn(2, 81, 32)
    s_prev = torch.randn(2, 81, 32)
    s_new = s.inner(x, y, s_prev)
    assert s_new.shape == (2, 81, 32)


def test_strategist_emit_produces_z_and_focus_mask():
    s = Strategist(dim=32, mlp_ratio=4, n_layers=1)
    s_final = torch.randn(2, 81, 32)
    z, focus_mask = s.emit(s_final)
    assert z.shape == (2, 81, 32)
    assert focus_mask.shape == (2, 81, 1)
    # focus_mask must be a probability via sigmoid: strictly in (0, 1)
    assert (focus_mask > 0).all()
    assert (focus_mask < 1).all()


def test_strategist_focus_mask_is_differentiable():
    s = Strategist(dim=16, mlp_ratio=4, n_layers=1)
    s_final = torch.randn(1, 81, 16, requires_grad=True)
    _, focus_mask = s.emit(s_final)
    focus_mask.sum().backward()
    assert s_final.grad is not None
    assert torch.isfinite(s_final.grad).all()


def test_tactician_output_shape_matches_y():
    t = Tactician(dim=32, mlp_ratio=4, n_layers=1)
    x = torch.randn(2, 81, 32)
    y = torch.randn(2, 81, 32)
    z = torch.randn(2, 81, 32)
    focus_mask = torch.rand(2, 81, 1)
    y_new = t(x, y, z, focus_mask)
    assert y_new.shape == y.shape


def test_tactician_zero_focus_mask_returns_input_y():
    # If focus_mask is all zero, the gated update reduces to: y_new = y + 0 = y.
    t = Tactician(dim=16, mlp_ratio=4, n_layers=1)
    x = torch.randn(1, 81, 16)
    y = torch.randn(1, 81, 16)
    z = torch.randn(1, 81, 16)
    focus_mask = torch.zeros(1, 81, 1)
    y_new = t(x, y, z, focus_mask)
    assert torch.allclose(y_new, y)


def test_halting_head_output_in_unit_interval():
    h = HaltingHead(dim=32)
    y = torch.randn(4, 81, 32)
    conf = h(y)
    assert conf.shape == (4, 1)
    assert (conf > 0).all() and (conf < 1).all()
