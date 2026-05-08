"""BitLinear drop-in replacement for Samsung's CastedLinear.

Implements the BitNet b1.58 ternary weight + 8-bit activation recipe
with the v6 stabilization improvements:
  - Lambda-ramped quantization (HF blog recipe: x_q = x + lambda * (quant(x) - x).detach())
  - Per-layer learnable alpha scale (TTQ)
  - Median-based weight scaling (BitNet Reloaded)

Usage (monkey-patch before model construction):
    import models.layers
    models.layers.CastedLinear = BitCastedLinear
    model = TinyRecursiveReasoningModel_ACTV1(cfg)
    set_lambda_q(model, 0.0)  # FP warmup
    # ... training loop ...
    set_lambda_q(model, current_lambda)  # ramp 0.0 -> 1.0 over schedule
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _trunc_normal_init(tensor: torch.Tensor, std: float):
    # Truncated normal at +/- 2 std, matches Samsung's trunc_normal_init_.
    with torch.no_grad():
        tensor.normal_(mean=0.0, std=std)
        tensor.clamp_(min=-2 * std, max=2 * std)
    return tensor


class BitCastedLinear(nn.Module):
    """1.58-bit ternary weight + 8-bit activation drop-in for CastedLinear."""

    def __init__(self, in_features: int, out_features: int, bias: bool,
                 use_median_scale: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_median_scale = use_median_scale

        # Master FP32 weight (matches Samsung's trunc_normal LeCun init)
        self.weight = nn.Parameter(
            _trunc_normal_init(
                torch.empty((out_features, in_features)),
                std=1.0 / (in_features ** 0.5),
            )
        )
        # Per-layer learnable alpha (TTQ), populated lazily on first forward
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self._alpha_initialized = False
        # Lambda ramp value (0=FP, 1=full quant). Set externally.
        self.register_buffer("lambda_q", torch.tensor(0.0))

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def _init_alpha(self):
        with torch.no_grad():
            w = self.weight.detach()
            if self.use_median_scale:
                scale = w.abs().median().clamp_min(1e-6)
            else:
                scale = w.abs().mean().clamp_min(1e-6)
            self.alpha.fill_(1.0 / scale.item())
        self._alpha_initialized = True

    @staticmethod
    def _activation_quant(x: torch.Tensor) -> torch.Tensor:
        """Per-token absmax 8-bit symmetric quant with STE."""
        scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        x_q = (x * scale).round().clamp(-128, 127) / scale
        return x + (x_q - x).detach()

    def _weight_quant(self) -> torch.Tensor:
        """Ternary {-1, 0, +1}/alpha weights with STE through alpha."""
        w = self.weight
        # round + clamp to ternary
        w_q = (w * self.alpha).round().clamp(-1, 1) / self.alpha
        # STE: forward = w_q, backward = identity for w; alpha gets gradient too
        return w + (w_q - w).detach() + (w_q - w_q.detach())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._alpha_initialized:
            self._init_alpha()

        x = input
        w = self.weight

        lam = float(self.lambda_q.item())
        if lam > 0.0:
            x_q = self._activation_quant(x)
            w_q = self._weight_quant()
            # Lambda interpolation between FP and quantized
            x = x + lam * (x_q - x).detach()
            w = w + lam * (w_q - w).detach() + lam * (w_q - w_q.detach())

        # cast weights to input dtype (matches Samsung's CastedLinear behavior)
        w = w.to(input.dtype)
        b = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(x, w, bias=b)


def set_lambda_q(model: nn.Module, value: float) -> int:
    """Set lambda_q on all BitCastedLinear modules. Returns count modified."""
    n = 0
    for m in model.modules():
        if isinstance(m, BitCastedLinear):
            m.lambda_q.fill_(value)
            n += 1
    return n


def count_bit_linears(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, BitCastedLinear))
