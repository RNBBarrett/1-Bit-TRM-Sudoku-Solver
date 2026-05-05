"""Config dataclass for HTRM."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class HTRMConfig:
    vocab_size: int = 11        # 0=empty, 1..9=digits, 10=reserved/PAD
    seq_len: int = 81           # 9x9 flattened
    hidden_dim: int = 192       # POC default; full spec uses 384
    mlp_ratio: int = 4
    n_layers_per_block: int = 1 # POC default; full spec uses 2
    K: int = 8                  # macro cycles per outer pass
    L: int = 2                  # Tactician inner cycles per macro
    P: int = 1                  # Strategist sub-recursion steps per macro
    T: int = 1                  # outer recursive passes
    halt_threshold: float = 0.99
    samsung_mode: bool = False  # if True, disable focus mask (Tactician unrestricted)
                                # and force P=1; matches Samsung TRM's exact 3-level recursion
                                # while keeping BitNet quantization. Used to isolate whether
                                # the focus-mask + sub-recursion are the source of training
                                # instability vs the 1-bit quantization itself.
    # ---- v6 BitNet stabilization options ----
    learnable_alpha: bool = False     # if True, BitLinear stores a learnable scalar alpha
                                      # per layer instead of recomputing 1/(W.abs().mean())
                                      # each forward (TTQ-style)
    use_median_scale: bool = False    # if True, BitLinear initializes alpha from
                                      # 1/(W.abs().median()+eps) (BitNet Reloaded recommendation
                                      # for sub-100M-param models) instead of mean
    ema_decay: float = 0.0            # if > 0 (e.g. 0.999), train.py maintains an EMA copy of
                                      # FP master weights and evaluates from it (TRM reports
                                      # 79.9% -> 87.4% accuracy gain from this alone)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HTRMConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
