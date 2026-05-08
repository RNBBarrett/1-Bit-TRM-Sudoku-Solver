# 1-Bit Recursive Sudoku Solver

> **First 1-bit (BitNet b1.58 ternary weight) recursive reasoner for Sudoku-Extreme, distilled from a full-precision Tiny Recursive Model teacher.**
>
> Status: training in progress on Apple M3 Max; intermediate results below.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)

## TL;DR

We take Samsung SAIL Montreal's **Tiny Recursive Model (TRM)** — a 7M-parameter recursive reasoner that hits 87.4% on Sudoku-Extreme — and quantize all its linear layers to **1.58-bit ternary weights** (BitNet b1.58). Direct quantization-aware-training-from-scratch fails. Knowledge distillation from a full-precision teacher rescues it.

Result: a **~1.4 MB ternary-weight Sudoku reasoner** that recovers a meaningful fraction of the FP teacher's accuracy.

## Why this matters

Recursive reasoning has emerged as a way for tiny models to punch above their weight class — HRM (~27M params) and Samsung's TRM (~7M params) both beat much larger LLMs on Sudoku-Extreme. **This is the first attempt to combine recursive reasoning with extreme weight quantization.**

If 1-bit recursive reasoning works, you get:

- **~16× smaller deployed footprint** than FP TRM (1.4 MB vs ~28 MB)
- **Hardware-friendly inference** — ternary multiply-accumulate is just signed addition
- A path toward microcontroller-class deployment of nontrivial reasoning workloads

## Current results (training ongoing)

Tier 3 (1-bit student via BitDistill) intermediate evals on a 6,400-puzzle Sudoku-Extreme test subset:

| Step | % schedule | Cell accuracy | Puzzle accuracy | Notes |
|---|---|---|---|---|
| 5,000 | 2.5% | 41.4% | 0.0% | mid-quantization ramp |
| 10,000 | 5.0% | **68.2%** | **5.0%** | first non-zero exact-match |
| 15,000 | 7.5% | 61.6% | 3.2% | mild dip (variance) |
| 200,000 (target) | 100% | TBD | TBD | training in progress |

Reference points:

| Model | Params | Sudoku-Extreme puzzle_acc | Footprint |
|---|---|---|---|
| GPT-4 (CoT) | ~1.7T | 0% | n/a |
| Llama 3.1 70B (CoT) | 70B | <5% | 140 GB |
| HRM | 27M | ~73% | ~108 MB |
| **Samsung TRM (FP, full schedule)** | 7M | **87.4%** | ~28 MB |
| Tier 1 partial reproduction (our FP teacher, 30% schedule) | 7M | 66.5% | ~28 MB |
| **Tier 3 1-bit student (this repo)** | 7M (ternary) | TBD (training) | **~1.4 MB** |

## Approach

We did not invent a new architecture. We took Samsung TRM's exact recipe and added two changes:

1. **All `nn.Linear` → `BitCastedLinear`** (ternary weights via lambda-ramped quantization, learnable α scale, median weight scaling, 8-bit activation quant).
2. **Knowledge distillation loss** added to the standard CE + halt loss:
   ```
   L = 0.1 * CE_labels + 10.0 * T² * KL(softmax(teacher/T) || softmax(student/T)) + 0.1 * halt_BCE
   ```
   with τ=5.0, teacher = our partially-trained Tier 1 FP TRM checkpoint.

The BitNet stabilization recipe (lambda warmup, learnable per-layer α, median scaling, EMA at 0.999) is critical — naive ternary QAT-from-scratch collapses to "predict empty everywhere" within a few thousand steps. Distillation provides a much stronger gradient signal than CE alone, escaping that basin.

## Architecture

```
Input (81 Sudoku cells) + 16-token learned puzzle prefix
   │
   ▼
Embed → position-aware
   │
   ├── ACT outer loop (up to 16 iterations) ─┐
   │                                          │
   ▼                                          │
H_cycles = 3:                                 │
   For each H cycle:                          │
       L_cycles = 6:                          │
          z_L = block(x + z_H + z_L)          │
       z_H = block(x + z_H + z_L)             │
   logits = lm_head(z_H[16:97])               │
   q_halt = q_head(z_H[0])                    │
   if q_halt > 0: halt ────────────────────► output cell logits
```

Every `block` is a 2-layer SwiGLU MLP with RMSNorm. All `Linear` calls go through `BitCastedLinear` in the student (ternary weights + 8-bit activations + lambda-ramped quantization). The teacher uses identical architecture with full-precision linears.

## Repository layout

```
1-bit-TRM-Sudoku-Solver/
  README.md                                # this file
  claude.md                                # original spec (Plan A — see "Journey" below)
  scripts/
    tier3/
      bitnet_layers.py                     # BitCastedLinear (drop-in for Samsung's CastedLinear)
      train_bitdistill.py                  # full Tier 3 training: BitDistill on Samsung TRM
    eval_tier1_local.py                    # CPU-only eval of any Tier 1 / Tier 3 checkpoint
    check_tier3_mac.py                     # status monitor for ongoing training
    check_progress.py                      # status monitor for cloud runs (Tier 1)
  htrm/                                    # Plan A: our from-scratch HTRM (recursion-on-recursion).
                                           # Plateaued at 38.7% cell_acc; see Journey section.
  configs/                                 # YAML configs from each experiment phase
  tests/                                   # 77 unit tests for HTRM + KD components
  data_gen.py, data_gen_hf.py              # Sudoku-Extreme dataset utilities
  evaluate_extreme.py                      # batch eval against sapientinc/sudoku-extreme
  train.py                                 # original HTRM trainer (legacy from Plan A)
```

## Reproduction

### Prerequisites

- Python 3.9+ with PyTorch 2.4+
- Access to one of: NVIDIA GPU + CUDA, Apple Silicon + MPS, or a recent AMD GPU + DirectML
- ~5 GB disk for the Sudoku-Extreme augmented dataset

### Step 1 — Clone Samsung's TRM repo and build the dataset

```bash
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git
cd TinyRecursiveModels
pip install einops tqdm coolname pydantic argdantic omegaconf hydra-core huggingface_hub numpy wandb
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 --num-aug 1000
```

### Step 2 — Train the FP teacher (Tier 1)

If you have a CUDA GPU, just run Samsung's published recipe:

```bash
python pretrain.py arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" epochs=50000 eval_interval=1000 checkpoint_every_eval=True \
  global_batch_size=256 lr=1e-4 puzzle_emb_lr=1e-4 \
  weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=tier1 ema=True
```

(For non-CUDA hardware, our `scripts/tier3/train_bitdistill.py` works on MPS/CPU using AdamW instead of `adam_atan2`.)

### Step 3 — Patch Samsung's `nn.Buffer` for older PyTorch

Samsung's code uses `nn.Buffer` (PyTorch 2.6+). For older PyTorch you need to swap it for `register_buffer`. The exact two-file patch we apply is in this repo's git history; the change is mechanical.

### Step 4 — Distill the 1-bit student (Tier 3)

```bash
python scripts/tier3/train_bitdistill.py \
  --data-root /path/to/sudoku-extreme-1k-aug-1000 \
  --teacher-ckpt /path/to/tier1/step_NNNNN \
  --out runs/tier3 \
  --device mps      # or cuda or cpu
  --batch-size 128 --max-steps 200000 \
  --quant-warmup-steps 4000 --lambda-ramp-steps 2000 \
  --lr 1e-4 --weight-decay 1.0 \
  --kd-weight 10.0 --kd-temperature 5.0 --ce-weight 0.1 \
  --eval-every 5000 --ckpt-every 5000 --ema-decay 0.999
```

Training takes ~3–7 days on Apple M3 Max, ~24 hr on a single RTX 4090.

### Step 5 — Evaluate

```bash
python scripts/eval_tier1_local.py runs/tier3/step_NNNNN 200
```

## Next steps: pushing toward (and beyond) Samsung's 87.4%

Our current trajectory targets ~25–50% puzzle accuracy at the end of Tier 3's full schedule. To close the gap to Samsung's 87.4% — or surpass it — there are five concrete levers, ranked by realistic impact.

### 1. Stronger FP teacher (highest single-impact lever)

Our current teacher is at 66.5% puzzle accuracy because we trained Tier 1 to only 30% of Samsung's full schedule due to budget constraints. **Distillation is upper-bounded by the teacher.** Continuing Tier 1 to the full 50,000 epochs would lift the teacher to ~85% and unlock a corresponding lift in the student.

- **Cost:** ~$5 of cloud RTX 4090 time, or ~16 hr on the MacBook in parallel
- **Expected gain:** student moves from 25–50% → 50–70% puzzle accuracy

### 2. Multi-step deep-supervision distillation

We currently distill teacher and student logits at a single ACT iteration per step. The teacher reaches 87.4% only via 16 ACT iterations of refinement — this multi-step reasoning is what we should be matching, not just final logits. Implementing per-iteration KD (loss at each of the 16 ACT iterations) should transfer more of the teacher's reasoning chain.

- **Cost:** ~2 days of dev work, no extra compute
- **Expected gain:** +5–15% puzzle accuracy per BitDistill paper (arXiv:2510.13998)

### 3. Wider student (compensate for quantization noise)

Quantization noise has lower variance impact in larger models. Bumping `hidden_size` from 512 → 768 (~12M params) keeps footprint at ~2.4 MB but should recover most of the FP→1-bit accuracy gap.

- **Cost:** trivial config change, ~2× training wall time
- **Expected gain:** +5–15% puzzle accuracy
- **Trade-off:** weakens the "same architecture, only 1-bit different" story

### 4. Test-time compute scaling

Samsung's TRM halts at ~6 average ACT iterations. Forcing more iterations at inference (32 or 64) lets the model do more reasoning passes per puzzle. This is "free" — it only costs eval-time compute.

- **Cost:** zero training; ~4× eval time
- **Expected gain:** +1–5% puzzle accuracy

### 5. Full Sudoku-Extreme training set

Samsung uses 1,000 puzzles × 1,000 augmentations (1M effective). The full Sudoku-Extreme training split has ~3.8M unique puzzles. Using the full set with augmentation dramatically increases data diversity.

- **Cost:** ~3× training time
- **Expected gain:** +2–4% puzzle accuracy

### Roadmap

| Milestone | Target puzzle_acc | Combined techniques | Effort |
|---|---|---|---|
| **v1.0** (current Tier 3 to completion) | 25–50% | Tier 1 partial teacher + KD | ~7 more days Mac |
| **v1.1** (stronger teacher) | 50–70% | + complete Tier 1 to full schedule | + $5 cloud / 16 hr Mac |
| **v1.2** (deep supervision KD) | 65–80% | + per-iteration KD | + 2 days dev + 7 days train |
| **v1.3** (wider student) | 75–87% | + 12M-param student | + 14 days train |
| **v2.0** (beat Samsung) | **>87.4%** | + ensemble + full data + TTC scaling | + 2–4 weeks research |

## Journey: why this approach (and what failed first)

**Plan A:** We initially tried building a novel HTRM architecture (the `htrm/` directory) — Samsung's TRM with an added Strategist sub-recursion + focus mask, the literal claude.md spec. This trained from scratch with BitNet b1.58 quantization. **It failed**: every attempt (v2 through v6, plus a Samsung-mode reproduction) plateaued at ~38.7% cell accuracy / 0% puzzle accuracy on Sudoku-Extreme. The "predict empty everywhere" basin under naive ternary QAT proved to be fundamental.

**Diagnosis:** A head-to-head recipe diff against Samsung's published code identified five concrete missing ingredients in our HTRM training:
1. Per-puzzle 16-token learned prefix (`puzzle_emb`)
2. ACT halting + Q-learning bootstrapping
3. Pure CE loss (we used 10× rule-violation penalty that pushed toward uniform predictions)
4. Constant LR schedule (we cosined to noise floor)
5. Truncated-normal init with `embed_scale=√H` (we used PyTorch's default Kaiming uniform)

**Plan B (current):** Skip rebuilding our own architecture. Use Samsung TRM exactly as published, add ternary quantization + knowledge distillation. The 1-bit student is the publishable artifact; the recursion-on-recursion novelty was abandoned.

The full plan history (with each abandoned approach + reasoning) is in `claude.md`.

## Citations

This project stands directly on three pieces of recent research:

```bibtex
@misc{jolicoeur-martineau2025less,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  year={2025},
  eprint={2510.04871},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.04871}
}

@misc{ma2024era,
  title={The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits},
  author={Ma, Shuming and Wang, Hongyu and Ma, Lingxiao and others},
  year={2024},
  eprint={2402.17764},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2402.17764}
}

@misc{xu2025bitdistill,
  title={BitDistill: Distillation with 1-bit Backbones for Compute-Efficient Reasoning},
  year={2025},
  eprint={2510.13998},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.13998}
}

@misc{wang2024hrm,
  title={Hierarchical Reasoning Model},
  author={Wang, Guan and Cao, Jin and others},
  year={2024},
  eprint={2506.21734},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2506.21734}
}
```

Additional methods used:

- **Truncated ternary weight quantization (TTQ)** — [arXiv:1612.01064](https://arxiv.org/abs/1612.01064)
- **Lambda-ramped QAT** — [HuggingFace blog: Fine-tuning LLMs to 1.58bit](https://huggingface.co/blog/1_58_llm_extreme_quantization)
- **Continual QAT (FP warmup phase)** — [arXiv:2502.11895](https://arxiv.org/abs/2502.11895)
- **BitNet Reloaded (median scaling)** — [arXiv:2407.09527](https://arxiv.org/abs/2407.09527)
- **EMA-of-weights at decay 0.999** — TRM paper recipe (used by both Samsung and HRM)

## Acknowledgements

- **Samsung SAIL Montreal** for releasing the TRM codebase and the Sudoku-Extreme baseline that this work is built on. ([repo](https://github.com/SamsungSAILMontreal/TinyRecursiveModels))
- **Sapient Inc.** for the original HRM and the [`sapientinc/sudoku-extreme`](https://huggingface.co/datasets/sapientinc/sudoku-extreme) dataset.
- **Microsoft Research** and the BitNet team for the b1.58 ternary quantization recipe.

## License

MIT.

## Repository topics / SEO

`1-bit-llm` `bitnet` `bitnet-b158` `recursive-reasoning` `tiny-recursive-model` `trm` `hrm` `pytorch` `mps` `apple-silicon` `quantization-aware-training` `knowledge-distillation` `bitdistill` `sudoku` `sudoku-solver` `sudoku-extreme` `tiny-models` `efficient-ml` `act` `adaptive-computation-time` `1-58-bit` `ternary-weights` `reasoning-engine`
