"""Top-level model entry point.

Re-exports the HTRM module from the htrm/ package so the spec-mandated
`model.py` exists at repo root. Running this file directly prints the
parameter count and verifies a forward pass works on the auto-detected
device.
"""
from __future__ import annotations

import argparse

import torch

from htrm.config import HTRMConfig
from htrm.device import get_device, sync
from htrm.htrm_model import HTRM


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def smoke_forward(cfg: HTRMConfig, device: torch.device) -> None:
    model = HTRM(cfg).to(device)
    n = count_parameters(model)
    print(f"params: {n:,}")
    print(f"device: {device}")
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len), device=device)
    out = model(tokens, training=True)
    sync()
    logits = out["logits"]
    print(f"forward output shape: {tuple(logits.shape)}")
    print(f"macro_used: {out['macro_used']}, micro_used: {out['micro_used']}")
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"
    print("smoke forward: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/htrm_poc.yaml")
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args()
    cfg = HTRMConfig.from_yaml(args.config)
    device = get_device(force_cpu=args.force_cpu)
    smoke_forward(cfg, device)


if __name__ == "__main__":
    main()
