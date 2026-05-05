import torch

from htrm.config import HTRMConfig
from htrm.htrm_model import HTRM


def _poc_config(**overrides) -> HTRMConfig:
    base = dict(
        vocab_size=11,
        seq_len=81,
        hidden_dim=64,
        mlp_ratio=4,
        n_layers_per_block=1,
        K=4,
        L=2,
        P=1,
        T=1,
        halt_threshold=0.99,
    )
    base.update(overrides)
    return HTRMConfig(**base)


def test_forward_output_shape_is_batch_seq_vocab():
    cfg = _poc_config()
    model = HTRM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    out = model(tokens, training=True)
    assert out["logits"].shape == (2, cfg.seq_len, cfg.vocab_size)


def test_forward_logits_are_finite():
    torch.manual_seed(0)
    cfg = _poc_config()
    model = HTRM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    out = model(tokens, training=True)
    assert torch.isfinite(out["logits"]).all()


def test_forward_macro_count_matches_K_times_T():
    cfg = _poc_config(K=4, T=1)
    model = HTRM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len))
    out = model(tokens, training=True)
    assert out["macro_used"] == cfg.K * cfg.T


def test_forward_micro_count_matches_K_T_times_P_plus_L():
    cfg = _poc_config(K=4, P=1, L=2, T=1)
    model = HTRM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len))
    out = model(tokens, training=True)
    assert out["micro_used"] == cfg.K * cfg.T * (cfg.P + cfg.L)


def test_forward_backward_flows_to_embedding():
    cfg = _poc_config()
    model = HTRM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len))
    out = model(tokens, training=True)
    out["logits"].sum().backward()
    assert model.tok_embed.weight.grad is not None
    assert model.tok_embed.weight.grad.abs().sum() > 0


def test_inference_halts_early_when_confidence_high():
    # Force halt head to always output ~1 by overriding its proj weight bias.
    cfg = _poc_config(K=8)
    model = HTRM(cfg).eval()
    # Replace halting head with a hook that always returns 1.0
    original_forward = model.halt_head.forward
    model.halt_head.forward = lambda y, lambda_q=1.0: torch.ones(y.shape[0], 1)  # type: ignore
    tokens = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len))
    out = model(tokens, training=False)
    # With halt threshold 0.99 and conf == 1.0, the loop should stop on the
    # very first macro cycle of the first outer pass.
    assert out["macro_used"] == 1
    model.halt_head.forward = original_forward  # type: ignore


def test_param_count_under_two_million_for_poc_size():
    cfg = _poc_config(hidden_dim=192, n_layers_per_block=1)
    model = HTRM(cfg)
    n = sum(p.numel() for p in model.parameters())
    # POC target is "small enough to train fast" — under 2M.
    assert n < 2_000_000, f"POC model has {n} params, want < 2M"
