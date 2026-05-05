"""HTRM training losses: CE + rule-violation penalty + ACT halt + KD distillation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from htrm.sudoku_rules import soft_group_violation


class KDLoss(nn.Module):
    """Knowledge-distillation loss between teacher and student logits.

    Standard Hinton-style temperature-scaled KL divergence:
        L_KD = T^2 * KL( softmax(teacher / T) || softmax(student / T) )

    The T^2 multiplier preserves gradient magnitude as T grows (since the
    softmax gradient scales 1/T; without T^2 the loss vanishes).

    Used for the BitDistill recipe (arXiv 2510.13998) where a frozen FP
    teacher's soft predictions guide a 1-bit ternary student. Distillation
    overcomes the "predict empty" basin that traps naive QAT-from-scratch:
    matching the teacher's digit distributions creates a much larger
    gradient signal away from any constant trivial output.

    Args:
        temperature: T in the formula. Higher T = softer distributions =
            more gradient signal from low-probability classes. BitDistill
            paper recommends T=5.0.
    """

    def __init__(self, temperature: float = 5.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"KD temperature must be > 0, got {temperature}")
        self.T = temperature

    def forward(
        self,
        student_logits: torch.Tensor,     # (B, S, V)
        teacher_logits: torch.Tensor,     # (B, S, V), no_grad
    ) -> torch.Tensor:
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"shape mismatch: student {student_logits.shape} vs "
                f"teacher {teacher_logits.shape}"
            )
        T = self.T
        # Compute KL in fp32 for numerical safety under autocast (KL/log_softmax
        # have known underflow at low precision when distributions are sharp).
        device_type = "cuda" if student_logits.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            s_log_p = F.log_softmax(student_logits.float() / T, dim=-1)
            t_p = F.softmax(teacher_logits.float() / T, dim=-1)
            # KLDivLoss expects log-probs as input; reduction='batchmean' divides
            # by batch size (over leading dim only). We want a scalar per
            # (cell, batch) average, so use 'batchmean' on flattened 2D logits.
            kl = F.kl_div(
                s_log_p.reshape(-1, s_log_p.shape[-1]),
                t_p.reshape(-1, t_p.shape[-1]),
                reduction="batchmean",
                log_target=False,
            )
        return T * T * kl


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
