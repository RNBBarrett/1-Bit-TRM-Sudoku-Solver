"""1-Bit Hierarchical Tiny Recursive Model (HTRM) — top-level module.

Three streams (x = question, y = answer, z = latent) are recursed
through Strategist (with internal P-step sub-recursion + focus-mask
emission) and Tactician blocks. The full schedule per training pass is

    for t in T:                        # outer recursive passes
      for k in K:                      # macro cycles
        s = z; for p in P: s = strategist.inner(x, y, s)
        z, focus_mask = strategist.emit(s)
        for ell in L: y = tactician(x, y, z, focus_mask)
        conf = halt_head(y)
        if not training and conf > halt_threshold: break
    final_logits = out_head(y)

During training the first T-1 passes are wrapped in `torch.no_grad`
(Samsung TRM's memory trick); only the final pass carries gradients.
For T=1 (POC default), all passes are full-grad.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from htrm.bitlinear import BitLinear
from htrm.blocks import Strategist, Tactician, HaltingHead
from htrm.config import HTRMConfig


class HTRM(nn.Module):
    def __init__(self, cfg: HTRMConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim

        self.tok_embed = nn.Embedding(cfg.vocab_size, H)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.seq_len, H))
        self.y_init = nn.Parameter(torch.zeros(1, cfg.seq_len, H))
        self.z_init = nn.Parameter(torch.zeros(1, cfg.seq_len, H))

        self.strategist = Strategist(H, cfg.mlp_ratio, cfg.n_layers_per_block)
        self.tactician = Tactician(H, cfg.mlp_ratio, cfg.n_layers_per_block)
        self.halt_head = HaltingHead(H)
        self.out_head = BitLinear(H, cfg.vocab_size)

        # Reasonable inits: small positional + zero stream initializers.
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.tok_embed.weight, std=0.02)

    def _macro_cycle(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        P: int,
        L: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One macro cycle: P Strategist sub-steps, emit, L Tactician steps, halt-conf.

        In samsung_mode, P is forced to 1 and the focus mask is replaced
        by all-ones (Tactician update is unrestricted) so the recursion
        structure matches Samsung's TRM: T x K x (1 + L) inner passes,
        no focus-mask gating. Used to isolate the BitNet quantization
        as the only delta from Samsung's known-good architecture.

        Extracted as a method so `torch.utils.checkpoint.checkpoint` can
        wrap it during training to trade compute for activation memory.
        """
        if self.cfg.samsung_mode:
            P = 1
        s = z
        for _ in range(P):
            s = self.strategist.inner(x, y, s)
        z, focus_mask = self.strategist.emit(s)
        if self.cfg.samsung_mode:
            focus_mask = torch.ones_like(focus_mask)
        for _ in range(L):
            y = self.tactician(x, y, z, focus_mask)
        conf = self.halt_head(y)
        return y, z, conf

    def forward(
        self,
        puzzle_tokens: torch.Tensor,
        training: bool = True,
        max_macro: int | None = None,
        max_micro: int | None = None,
        gradient_checkpoint: bool = False,
    ) -> dict[str, torch.Tensor | int | list[torch.Tensor]]:
        cfg = self.cfg
        K = max_macro if max_macro is not None else cfg.K
        L = max_micro if max_micro is not None else cfg.L
        P = cfg.P
        T = cfg.T

        B = puzzle_tokens.shape[0]
        x = self.tok_embed(puzzle_tokens) + self.pos_embed
        # `.repeat` returns a new copy (not a view), avoiding inference_mode's
        # version-counter conflict that `.expand(...).clone()` triggers when
        # the source is an nn.Parameter.
        y = self.y_init.repeat(B, 1, 1)
        z = self.z_init.repeat(B, 1, 1)

        halts: list[torch.Tensor] = []
        macro_used = 0
        micro_used = 0
        halted = False

        for t in range(T):
            ctx = torch.no_grad() if (training and t < T - 1) else nullcontext()
            with ctx:
                for _ in range(K):
                    if gradient_checkpoint and torch.is_grad_enabled():
                        from torch.utils.checkpoint import checkpoint
                        y, z, conf = checkpoint(
                            self._macro_cycle, x, y, z, P, L,
                            use_reentrant=False,
                        )
                    else:
                        y, z, conf = self._macro_cycle(x, y, z, P, L)
                    macro_used += 1
                    micro_used += P + L
                    halts.append(conf)
                    if not training and (conf > cfg.halt_threshold).all():
                        halted = True
                        break
            if halted:
                break

        logits = self.out_head(y)

        return {
            "logits": logits,
            "halts": torch.stack(halts, dim=1) if halts else torch.empty(B, 0, 1),
            "macro_used": macro_used,
            "micro_used": micro_used,
            "y": y,
            "z": z,
        }
