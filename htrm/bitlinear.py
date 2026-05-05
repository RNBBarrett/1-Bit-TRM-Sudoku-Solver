"""BitNet b1.58 ternary primitives.

Implements the quantization formulas from "The Era of 1-bit LLMs"
(arXiv:2402.17764). Weights are constrained to {-1, 0, +1} via per-tensor
absmean scaling; activations are quantized per-token to int8 via absmax;
the straight-through estimator (STE) lets gradients flow through the
non-differentiable rounding.

v6 additions (May 2026, after the v2-v5 collapse experiments):
  - **Lambda-ramped quantization** (HF blog "Fine-tuning LLMs to 1.58bit"):
    BitLinear.forward accepts a `lambda_q` parameter that linearly
    interpolates between FP and ternary. lambda=0 is pure FP, lambda=1
    is the original ternary behavior. The training loop ramps lambda
    from 0 to 1 over ~1000 steps once the FP-warmup phase ends. This is
    a hard requirement for stable from-scratch ternary training; without
    it, switching abruptly to lambda=1 causes loss to explode to ~13.
  - **Learnable per-layer alpha** (TTQ, arXiv 1612.01064): instead of
    recomputing scale = 1/(W.abs().mean()+eps) every forward, store a
    learnable scalar parameter that the optimizer updates. Initialized
    lazily on the first forward from the data-driven absmean (or median).
  - **Median scaling option** (BitNet Reloaded, arXiv 2407.09527):
    sub-100M-param ternary models benefit from median-based weight scale
    instead of mean (median is more robust to weight outliers).

All three new behaviors are gated by config flags so existing callers
that pass plain `BitLinear(in, out)` get the original behavior.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-5


def weight_quant(W: torch.Tensor, alpha: torch.Tensor | None = None) -> torch.Tensor:
    """Per-tensor ternary quantization.

    With alpha=None (default): recomputes the absmean-based scale every
    forward, matching the original BitNet b1.58 paper.

    With alpha != None: uses the provided learnable scalar (TTQ style).
    The dequantized result is returned (not the int representation).
    """
    if alpha is None:
        scale = 1.0 / (W.abs().mean() + _EPS)
    else:
        scale = alpha
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


def lambda_ste(x: torch.Tensor, x_q: torch.Tensor, lambda_q: float) -> torch.Tensor:
    """Lambda-ramped STE: forward = x + lambda*(x_q - x), backward = identity in x.

    With lambda=0: pure FP. With lambda=1: identical to the standard STE.
    Used by BitLinear during the training schedule's "ramp" phase to
    smoothly introduce quantization, per the HuggingFace 1.58bit blog
    recipe (https://huggingface.co/blog/1_58_llm_extreme_quantization).
    """
    return x + lambda_q * (x_q - x).detach()


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

    The `quantization_enabled` flag implements a coarse FP-warmup phase
    boundary: when False (default during the first ~20% of training), the
    layer is a plain FP linear. When True, ternary is applied with the
    optional `lambda_q` ramp parameter (passed in forward).

    Args:
        in_features, out_features, bias: standard nn.Linear semantics.
        learnable_alpha: if True, store a learnable scalar weight scale
            instead of recomputing 1/absmean each forward (TTQ-style).
        use_median: if True (and learnable_alpha=True), the data-driven
            init of alpha uses median absolute value (BitNet Reloaded
            recommendation for sub-100M-param ternary models).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        learnable_alpha: bool = False,
        use_median: bool = False,
    ):
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
        self.learnable_alpha = learnable_alpha
        self.use_median = use_median
        if learnable_alpha:
            # Initialized lazily on first forward from the actual weight
            # statistics. Storing a Parameter here so the optimizer picks
            # it up; it gets overwritten in-place on the first forward.
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.register_buffer("_alpha_initialized", torch.tensor(False), persistent=False)
        else:
            self.alpha = None  # type: ignore[assignment]
            self.register_buffer("_alpha_initialized", torch.tensor(True), persistent=False)
        # Buffer (not Parameter) so it's part of state_dict but not trainable.
        self.register_buffer(
            "_quantization_enabled", torch.tensor(True), persistent=False
        )

    @property
    def quantization_enabled(self) -> bool:
        return bool(self._quantization_enabled.item())

    def _maybe_init_alpha(self) -> None:
        """Lazy initialization of the learnable alpha from the actual weight stats."""
        if not self.learnable_alpha:
            return
        if bool(self._alpha_initialized.item()):
            return
        with torch.no_grad():
            if self.use_median:
                stat = self.weight.abs().median()
            else:
                stat = self.weight.abs().mean()
            init_alpha = 1.0 / (stat + _EPS)
            self.alpha.data.fill_(float(init_alpha.item()))
            self._alpha_initialized = torch.tensor(True, device=self._alpha_initialized.device)

    def _quantize_weight(self) -> torch.Tensor:
        """Compute the dequantized ternary weight tensor for this forward.

        Two paths:
          - learnable_alpha=False: original BitNet b1.58 behavior. STE
            routes gradient through self.weight; alpha is recomputed each
            forward as 1/(W.abs().mean()+eps).
          - learnable_alpha=True: TTQ-style learnable per-layer scale.
            The forward is structured so alpha receives gradients via
            its natural multiplier path while STE still routes weight
            gradients as identity. The trick is the additive zero-term
            `(w_q - w_q.detach())`, which evaluates to zero in the
            forward (no value change) but reintroduces alpha's gradient.
        """
        if self.alpha is not None:
            # Learnable alpha path: keep alpha in the forward graph for grad.
            w_int = (self.weight * self.alpha).round().clamp(-1, 1).detach()
            w_q = w_int / self.alpha  # has gradient to alpha via division
            # STE for self.weight + zero-term for alpha gradient preservation.
            return (
                self.weight
                + (w_q - self.weight).detach()
                + (w_q - w_q.detach())
            )
        # Original BitNet b1.58 path.
        return ste(self.weight, weight_quant(self.weight))

    def forward(self, x: torch.Tensor, lambda_q: float = 1.0) -> torch.Tensor:
        x = self.norm(x)
        if self.quantization_enabled and lambda_q > 0:
            self._maybe_init_alpha()
            x_q = activation_quant(x)
            w_q = self._quantize_weight()
            if lambda_q < 1.0:
                # Smooth interpolation between FP and ternary (HF recipe):
                # x_eff = x + lambda * (x_q - x).detach(), and similarly w.
                # Note: when learnable_alpha=True, self._quantize_weight()
                # already includes the STE; we still want lambda interp
                # between FP weight and the quantized weight.
                x = lambda_ste(x, x_q, lambda_q)
                w = self.weight + lambda_q * (w_q - self.weight).detach()
                if self.alpha is not None:
                    # Preserve alpha gradient even at lambda < 1.
                    w = w + lambda_q * (w_q - w_q.detach())
            else:
                x = ste(x, x_q)
                w = w_q
        else:
            w = self.weight
        return F.linear(x, w, self.bias)


def set_quantization_enabled(module: nn.Module, enabled: bool) -> None:
    """Walk a module tree and flip the FP-warmup flag on every BitLinear."""
    flag = torch.tensor(bool(enabled))
    for m in module.modules():
        if isinstance(m, BitLinear):
            m._quantization_enabled = flag.clone()
