"""Local CPU evaluation of a Tier 1 Samsung TRM checkpoint.

Loads a checkpoint produced by the cloud Tier 1 run, runs forward passes
against a 200-puzzle test subset of Sudoku-Extreme, and reports exact
puzzle accuracy + cell accuracy. CPU-only (no CUDA needed).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make Samsung's repo importable
SAMSUNG_REPO = Path("C:/Users/Richard/Documents/code/TinyRecursiveModels")
sys.path.insert(0, str(SAMSUNG_REPO))

from models.recursive_reasoning.trm import (  # noqa: E402
    TinyRecursiveReasoningModel_ACTV1,
)


def load_test_subset(test_dir: Path, n_puzzles: int = 200):
    """Load the first n_puzzles examples from the prebuilt test set."""
    inputs = np.load(test_dir / "all__inputs.npy")
    labels = np.load(test_dir / "all__labels.npy")
    puzzle_ids = np.load(test_dir / "all__puzzle_identifiers.npy")
    return (
        torch.from_numpy(inputs[:n_puzzles].astype(np.int64)),
        torch.from_numpy(labels[:n_puzzles].astype(np.int64)),
        torch.from_numpy(puzzle_ids[:n_puzzles].astype(np.int32)),
    )


def main(ckpt_path: str, n_puzzles: int = 200, batch_size: int = 32):
    test_dir = SAMSUNG_REPO / "data" / "sudoku-extreme-1k-aug-1000" / "test"
    print(f"loading test set from {test_dir}")
    inputs, labels, pids = load_test_subset(test_dir, n_puzzles=n_puzzles)
    print(f"  inputs: {inputs.shape} {inputs.dtype}, labels: {labels.shape}, pids: {pids.shape}")

    # Match training config (from all_config.yaml). num_puzzle_identifiers=1000
    # (train) — at test the loader passes pid=0 for every example.
    model_cfg = dict(
        batch_size=batch_size,
        seq_len=81,
        puzzle_emb_ndim=512,
        puzzle_emb_len=16,
        num_puzzle_identifiers=1,
        vocab_size=11,
        H_cycles=3,
        L_cycles=6,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        expansion=4,
        num_heads=8,
        pos_encodings="none",
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        forward_dtype="float32",  # CPU-friendly
        mlp_t=True,
        no_ACT_continue=True,
    )

    print(f"loading checkpoint {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  state_dict keys: {len(sd)}, sample: {list(sd.keys())[0]}")

    model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
    # Strip torch.compile + ACTLossHead wrapper prefix: "_orig_mod.model."
    PREFIX = "_orig_mod.model."
    cleaned = {}
    for k, v in sd.items():
        if k.startswith(PREFIX):
            cleaned[k[len(PREFIX):]] = v.float() if v.is_floating_point() else v
        else:
            cleaned[k] = v.float() if v.is_floating_point() else v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  WARNING missing keys: {len(missing)}: {missing[:5]}")
    if unexpected:
        print(f"  WARNING unexpected keys: {len(unexpected)}: {unexpected[:5]}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    cell_correct = 0
    cell_total = 0
    puzzle_correct = 0
    puzzle_total = 0

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_puzzles, batch_size):
            end = min(start + batch_size, n_puzzles)
            cur_bs = end - start
            if cur_bs < batch_size:
                # pad to batch_size since model.batch_size is fixed
                pad = batch_size - cur_bs
                bx = torch.cat(
                    [inputs[start:end], torch.zeros(pad, 81, dtype=torch.int64)]
                )
                by = torch.cat(
                    [labels[start:end], torch.zeros(pad, 81, dtype=torch.int64)]
                )
                bp = torch.cat([pids[start:end], torch.zeros(pad, dtype=torch.int32)])
            else:
                bx, by, bp = inputs[start:end], labels[start:end], pids[start:end]

            batch = {"inputs": bx, "labels": by, "puzzle_identifiers": bp}
            carry = model.initial_carry(batch)
            # ACT loop: run up to halt_max_steps
            for _ in range(model_cfg["halt_max_steps"]):
                carry, outputs = model(carry, batch)
                if carry.halted.all():
                    break
            logits = outputs["logits"]  # (B, S, vocab); S = puzzle_emb_len + seq_len
            # Strip puzzle_emb_len prefix tokens to align with the 81 cells
            cell_logits = logits[:, -81:, :]
            preds = cell_logits.argmax(dim=-1)  # (B, 81)

            # Trim padded rows
            preds = preds[:cur_bs]
            true = by[:cur_bs]
            # cell-level: how many of 81 cells correct (ignoring blank cells in label)
            mask = true != 0  # ignore_label_id=0; only score real cells
            cell_correct += ((preds == true) & mask).sum().item()
            cell_total += mask.sum().item()
            # puzzle-level: full 81 cells correct
            puzzle_correct += ((preds == true) | ~mask).all(dim=-1).sum().item()
            puzzle_total += cur_bs

            print(
                f"  [{end}/{n_puzzles}] running cell={cell_correct}/{cell_total} "
                f"({cell_correct/max(cell_total,1):.3%})  "
                f"puzzles={puzzle_correct}/{puzzle_total} "
                f"({puzzle_correct/max(puzzle_total,1):.3%})  "
                f"elapsed={time.time()-t0:.1f}s"
            )

    cell_acc = cell_correct / max(cell_total, 1)
    puzzle_acc = puzzle_correct / max(puzzle_total, 1)
    print(
        f"\nFINAL  cell_acc={cell_acc:.4f}  puzzle_acc={puzzle_acc:.4f}  "
        f"({puzzle_correct}/{puzzle_total})  in {time.time()-t0:.1f}s"
    )
    return cell_acc, puzzle_acc


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else (
        "checkpoints/cloud/tier1/step_54684"
    )
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    main(ckpt, n_puzzles=n, batch_size=8)
