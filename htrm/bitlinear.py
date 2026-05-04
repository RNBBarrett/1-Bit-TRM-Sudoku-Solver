"""BitNet b1.58 ternary primitives.

Implements the quantization formulas from "The Era of 1-bit LLMs"
(arXiv:2402.17764). Weights are constrained to {-1, 0, +1} via per-tensor
absmean scaling; activations are quantized per-token to int8 via absmax;
the straight-through estimator (STE) lets gradients flow through the
non-differentiable rounding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-5


def weight_quant(W: torch.Tensor) -> torch.Tensor:
    """Per-tensor absmean ternary quantization.

    Maps W to a tensor whose values are exactly {-1, 0, +1} times a
    per-tensor scale. The dequantized result is returned (not the int
    representation), so downstream matmul stays the same shape.
    """
    scale = 1.0 / (W.abs().mean() + _EPS)
    return (W * scale).round().clamp(-1, 1) / scale


def activation_quant(X: torch.Tensor) -> torch.Tensor:
    """Per-token absmax int8 quantization.

    Each token (last-dim slice) is symmetrically scaled so its largest
    magnitude maps to 127, then rounded and clamped to [-128, 127]. The
    dequantized float is returned.
    """
    scale = 127.0 / (X.abs().amax(dim=-1, keepdim=True) + _EPS)
    return (X * scale).round().clamp(-128, 127) / scale


def ste(x: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward = x_q, backward = identity in x."""
    return x + (x_q - x).detach()


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learnable per-feature scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class BitLinear(nn.Module):
    """Linear layer with BitNet b1.58 ternary weights and int8 activations.

    Pre-norm (RMSNorm) is baked in so callers can't forget it. During
    training the FP32 shadow weights are stored in self.weight and the
    ternary form is recomputed each forward via STE.

    The `quantization_enabled` flag implements the BitNet b1.58 paper's
    full-precision warmup recipe: the first ~20% of training runs with
    quantization disabled (regular FP32 linear) so the shadow weights
    settle into a productive distribution before they're squeezed to
    ternary. Toggle in bulk via `set_quantization_enabled(model, ...)`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.norm = RMSNorm(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        # Buffer (not Parameter) so it's part of state_dict but not trainable.
        self.register_buffer(
            "_quantization_enabled", torch.tensor(True), persistent=False
        )

    @property
    def quantization_enabled(self) -> bool:
        return bool(self._quantization_enabled.item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.quantization_enabled:
            x = ste(x, activation_quant(x))
            w = ste(self.weight, weight_quant(self.weight))
        else:
            w = self.weight
        return F.linear(x, w, self.bias)


def set_quantization_enabled(module: nn.Module, enabled: bool) -> None:
    """Walk a module tree and flip the FP-warmup flag on every BitLinear."""
    flag = torch.tensor(bool(enabled))
    for m in module.modules():
        if isinstance(m, BitLinear):
            m._quantization_enabled = flag.clone()
