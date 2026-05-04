"""HTRM model blocks: BitMLPBlock, Strategist, Tactician, HaltingHead.

The architecture follows Samsung TRM's MLP-only-no-attention finding for
Sudoku, augmented with the spec's hierarchical recursion: a Strategist
that runs an internal P-step sub-recursion and emits a focus mask, and a
Tactician that updates the answer stream y gated by that mask.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from htrm.bitlinear import BitLinear


class BitMLPBlock(nn.Module):
    """Residual MLP block: x + fc2(GELU(fc1(x))). All Linear -> BitLinear."""

    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = BitLinear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = BitLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class Strategist(nn.Module):
    """Macro reasoner with internal P-step sub-recursion and focus emission.

    inner(x, y, s_prev) -> s_new advances the strategist's latent.
    emit(s_final)       -> (z, focus_mask) projects to the latent z used
                            by the Tactician and a per-cell focus mask in (0, 1).
    """

    def __init__(self, dim: int, mlp_ratio: int = 4, n_layers: int = 1):
        super().__init__()
        self.proj_in = BitLinear(3 * dim, dim)
        self.layers = nn.ModuleList([BitMLPBlock(dim, mlp_ratio) for _ in range(n_layers)])
        self.proj_z = BitLinear(dim, dim)
        self.proj_mask = BitLinear(dim, 1)

    def inner(self, x: torch.Tensor, y: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(torch.cat([x, y, s_prev], dim=-1))
        for layer in self.layers:
            h = layer(h)
        return h

    def emit(self, s_final: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj_z(s_final)
        mask_logits = self.proj_mask(s_final)
        focus_mask = torch.sigmoid(mask_logits)
        return z, focus_mask


class Tactician(nn.Module):
    """Micro reasoner that updates y via focus-mask-gated residual delta."""

    def __init__(self, dim: int, mlp_ratio: int = 4, n_layers: int = 1):
        super().__init__()
        self.proj_in = BitLinear(3 * dim, dim)
        self.layers = nn.ModuleList([BitMLPBlock(dim, mlp_ratio) for _ in range(n_layers)])
        self.proj_out = BitLinear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        focus_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.proj_in(torch.cat([x, y, z], dim=-1))
        for layer in self.layers:
            h = layer(h)
        delta = self.proj_out(h)
        return y + focus_mask * delta


class HaltingHead(nn.Module):
    """Sigmoid confidence head over the pooled answer stream y."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = BitLinear(dim, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        pooled = y.mean(dim=1)            # (B, H)
        logit = self.proj(pooled)         # (B, 1)
        return torch.sigmoid(logit)
