"""HTRM model blocks: BitMLPBlock, Strategist, Tactician, HaltingHead.

The architecture follows Samsung TRM's MLP-only-no-attention finding for
Sudoku, augmented with the spec's hierarchical recursion: a Strategist
that runs an internal P-step sub-recursion and emits a focus mask, and a
Tactician that updates the answer stream y gated by that mask.

All forward methods accept an optional `lambda_q: float = 1.0` parameter
which is propagated to every internal BitLinear call. lambda=1 is the
original ternary behavior; lambda<1 smoothly interpolates toward FP for
the v6 quantization-ramp recipe (HF 1.58bit blog).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from htrm.bitlinear import BitLinear


class BitMLPBlock(nn.Module):
    """Residual MLP block: x + fc2(GELU(fc1(x))). All Linear -> BitLinear."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 4,
        learnable_alpha: bool = False,
        use_median: bool = False,
    ):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = BitLinear(dim, hidden, learnable_alpha=learnable_alpha, use_median=use_median)
        self.act = nn.GELU()
        self.fc2 = BitLinear(hidden, dim, learnable_alpha=learnable_alpha, use_median=use_median)

    def forward(self, x: torch.Tensor, lambda_q: float = 1.0) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x, lambda_q=lambda_q)), lambda_q=lambda_q)


class Strategist(nn.Module):
    """Macro reasoner with internal P-step sub-recursion and focus emission.

    inner(x, y, s_prev) -> s_new advances the strategist's latent.
    emit(s_final)       -> (z, focus_mask) projects to the latent z used
                            by the Tactician and a per-cell focus mask in (0, 1).
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 4,
        n_layers: int = 1,
        learnable_alpha: bool = False,
        use_median: bool = False,
    ):
        super().__init__()
        bl_kwargs = {"learnable_alpha": learnable_alpha, "use_median": use_median}
        self.proj_in = BitLinear(3 * dim, dim, **bl_kwargs)
        self.layers = nn.ModuleList([
            BitMLPBlock(dim, mlp_ratio, learnable_alpha=learnable_alpha, use_median=use_median)
            for _ in range(n_layers)
        ])
        self.proj_z = BitLinear(dim, dim, **bl_kwargs)
        self.proj_mask = BitLinear(dim, 1, **bl_kwargs)

    def inner(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        s_prev: torch.Tensor,
        lambda_q: float = 1.0,
    ) -> torch.Tensor:
        h = self.proj_in(torch.cat([x, y, s_prev], dim=-1), lambda_q=lambda_q)
        for layer in self.layers:
            h = layer(h, lambda_q=lambda_q)
        return h

    def emit(
        self,
        s_final: torch.Tensor,
        lambda_q: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj_z(s_final, lambda_q=lambda_q)
        mask_logits = self.proj_mask(s_final, lambda_q=lambda_q)
        focus_mask = torch.sigmoid(mask_logits)
        return z, focus_mask


class Tactician(nn.Module):
    """Micro reasoner that updates y via focus-mask-gated residual delta."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 4,
        n_layers: int = 1,
        learnable_alpha: bool = False,
        use_median: bool = False,
    ):
        super().__init__()
        bl_kwargs = {"learnable_alpha": learnable_alpha, "use_median": use_median}
        self.proj_in = BitLinear(3 * dim, dim, **bl_kwargs)
        self.layers = nn.ModuleList([
            BitMLPBlock(dim, mlp_ratio, learnable_alpha=learnable_alpha, use_median=use_median)
            for _ in range(n_layers)
        ])
        self.proj_out = BitLinear(dim, dim, **bl_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        focus_mask: torch.Tensor,
        lambda_q: float = 1.0,
    ) -> torch.Tensor:
        h = self.proj_in(torch.cat([x, y, z], dim=-1), lambda_q=lambda_q)
        for layer in self.layers:
            h = layer(h, lambda_q=lambda_q)
        delta = self.proj_out(h, lambda_q=lambda_q)
        return y + focus_mask * delta


class HaltingHead(nn.Module):
    """Sigmoid confidence head over the pooled answer stream y."""

    def __init__(
        self,
        dim: int,
        learnable_alpha: bool = False,
        use_median: bool = False,
    ):
        super().__init__()
        self.proj = BitLinear(dim, 1, learnable_alpha=learnable_alpha, use_median=use_median)

    def forward(self, y: torch.Tensor, lambda_q: float = 1.0) -> torch.Tensor:
        pooled = y.mean(dim=1)
        logit = self.proj(pooled, lambda_q=lambda_q)
        return torch.sigmoid(logit)
