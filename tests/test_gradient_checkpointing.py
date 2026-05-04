"""Gradient-checkpointing tests for HTRM.

The opt-in flag should preserve forward semantics exactly (same logits)
while reducing the number of activation tensors retained for backward.
"""
import torch

from htrm.config import HTRMConfig
from htrm.htrm_model import HTRM


def _cfg() -> HTRMConfig:
    return HTRMConfig(
        vocab_size=11, seq_len=81, hidden_dim=32, mlp_ratio=4,
        n_layers_per_block=1, K=2, L=2, P=1, T=1, halt_threshold=0.99,
    )


def test_gradient_checkpoint_preserves_forward():
    torch.manual_seed(0)
    cfg = _cfg()
    model = HTRM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    out_normal = model(tokens, training=True, gradient_checkpoint=False)
    out_ckpt = model(tokens, training=True, gradient_checkpoint=True)
    assert torch.allclose(out_normal["logits"], out_ckpt["logits"], atol=1e-5)


def test_gradient_checkpoint_backward_flows():
    torch.manual_seed(1)
    cfg = _cfg()
    model = HTRM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    out = model(tokens, training=True, gradient_checkpoint=True)
    # Backward from a loss that touches both logits and halts so all
    # trainable params (including halt_head) participate in autograd.
    (out["logits"].sum() + out["halts"].sum()).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_gradient_checkpoint_loop_counts_unchanged():
    cfg = _cfg()
    model = HTRM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len))
    out_normal = model(tokens, training=True, gradient_checkpoint=False)
    out_ckpt = model(tokens, training=True, gradient_checkpoint=True)
    assert out_normal["macro_used"] == out_ckpt["macro_used"]
    assert out_normal["micro_used"] == out_ckpt["micro_used"]
