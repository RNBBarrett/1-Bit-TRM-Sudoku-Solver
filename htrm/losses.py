"""HTRM training loss: CE + rule-violation penalty + ACT halt loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from htrm.sudoku_rules import soft_group_violation


class HTRMLoss(nn.Module):
    """Combined loss for HTRM training.

    Components:
      ce        — categorical cross-entropy over the (B, 81) target
                  using full vocab logits (B, 81, V).
      violation — differentiable soft duplicate penalty over predicted
                  digit probabilities (per claude.md "harsh verifier").
      halt      — BCE between halt-head outputs (B, n_halts, 1) and the
                  per-example target = 1 if argmax matches the full
                  target, 0 otherwise (Adaptive Computation Time).

    Defaults: violation_weight=10.0 (per claude.md spec), halt_weight=0.1.
    """

    def __init__(self, violation_weight: float = 10.0, halt_weight: float = 0.1):
        super().__init__()
        self.violation_weight = violation_weight
        self.halt_weight = halt_weight

    def forward(
        self,
        logits: torch.Tensor,    # (B, 81, V)
        target: torch.Tensor,    # (B, 81)
        halts: torch.Tensor,     # (B, n_halts, 1)
        violation_weight: float | None = None,
        halt_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        B, S, V = logits.shape
        vw = self.violation_weight if violation_weight is None else violation_weight
        hw = self.halt_weight if halt_weight is None else halt_weight

        # Cross-entropy
        ce = F.cross_entropy(logits.reshape(B * S, V), target.reshape(B * S))

        # Soft rule-violation penalty (skip empty-cell slot 0)
        p = F.softmax(logits, dim=-1)
        p_d = p[..., 1:10] if V >= 10 else p[..., 1:]
        violation = soft_group_violation(p_d)

        # ACT halt loss
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            is_correct = (preds == target).all(dim=-1).float()  # (B,)
        if halts.numel() > 0:
            target_halt = is_correct.view(B, 1, 1).expand_as(halts)
            # BCE on sigmoid outputs is unsafe under autocast (fp16/bf16 underflow).
            # Disable autocast and cast inputs to fp32 for this small reduction.
            device_type = "cuda" if logits.is_cuda else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                halt_loss = F.binary_cross_entropy(
                    halts.float().clamp(1e-6, 1 - 1e-6),
                    target_halt.float(),
                )
        else:
            halt_loss = torch.zeros((), device=logits.device)

        total = ce + vw * violation + hw * halt_loss

        # Components are returned undetached so callers can reuse them in
        # custom recombinations (e.g. step-dependent halt-loss ramps).
        return {
            "total": total,
            "ce": ce,
            "violation": violation,
            "halt": halt_loss,
        }
