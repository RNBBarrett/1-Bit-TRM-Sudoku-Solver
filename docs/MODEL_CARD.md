# Model Card: 1-Bit Recursive Sudoku Solver (Tier 3 student)

> Hugging Face-format model card for the 1.4 MB ternary-weight student model produced by this repository.
>
> **Status: training in progress.** Final numbers pending; intermediate values shown.

## Model details

- **Developed by:** Richard Barrett, building on Samsung SAIL Montreal's TRM and Microsoft's BitNet b1.58.
- **Model type:** Recursive reasoning network with ACT halting; ternary-weight quantization (BitNet b1.58); 8-bit activation quantization.
- **Language(s):** N/A (Sudoku puzzles, not natural language)
- **License:** MIT
- **Source repository:** https://github.com/RNBBarrett/1-Bit-TRM-Sudoku-Solver
- **Architecture:** Tiny Recursive Model (TRM) with `H_cycles=3, L_cycles=6, hidden=512, mlp_t=True, puzzle_emb_len=16, halt_max_steps=16`. All linear layers replaced with `BitCastedLinear` (ternary weights via lambda-ramped quantization).
- **Parameters:** 5,028,876
- **Footprint:** ~1.4 MB at 1.58 bits per weight
- **Training data:** Sudoku-Extreme via [`sapientinc/sudoku-extreme`](https://huggingface.co/datasets/sapientinc/sudoku-extreme), 1,000 puzzles × 1,000 augmentations (1M effective examples), Samsung's `dataset/build_sudoku_dataset.py` script.

## Intended use

- **Primary:** Demonstrate that recursive reasoning architectures can be aggressively quantized to 1.58 bits without losing all of their capability — a research result.
- **Secondary:** Inference engine for Sudoku-Extreme puzzles on memory- or compute-constrained hardware.
- **Out of scope:** Not for production deployment without further validation. Not for non-Sudoku tasks without retraining.

## Training procedure

### Teacher (Tier 1)

- Architecture: identical to student but with full-precision linear layers
- Training: Samsung's published recipe (lr=1e-4 constant, weight_decay=1.0, AdamATan2, EMA at 0.999) for 30% of full schedule
- Final checkpoint: step_54684, puzzle_acc 66.5% on test subset

### Student (Tier 3)

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, betas=(0.9, 0.95), wd=1.0 |
| Base LR | 1e-4 (constant after 2k-step warmup) |
| Batch size | 128 |
| Total steps | 200,000 (Samsung's full schedule) |
| FP warmup | 4,000 steps |
| Lambda ramp | steps 4,000–6,000 (0 → 1) |
| KD temperature | 5.0 |
| Loss weights | CE 0.1, KD 10.0, halt 0.1 |
| EMA decay | 0.999 |
| Hardware | Apple M3 Max (MPS backend) |

### Loss

```
L = 0.1 * CE(student_logits, labels) +
    10.0 * T² * KL(softmax(teacher/T) || softmax(student/T)) +
    0.1 * BCE(student_q_halt, seq_correct)
```

## Evaluation

| Step | Cell accuracy | Puzzle accuracy | Test set |
|---|---|---|---|
| 5,000 | 41.4% | 0.0% | 6,400 puzzle subset |
| 10,000 | 68.2% | 5.0% | 6,400 puzzle subset |
| 15,000 | 61.6% | 3.2% | 6,400 puzzle subset |
| ... | (training in progress) | | |

Final number pending training completion.

## Limitations

- Sudoku-Extreme is one specific reasoning task; transfer to other tasks requires retraining.
- The student is upper-bounded by the teacher (66.5% puzzle_acc currently). To exceed this, the teacher needs to be trained to full Samsung schedule first.
- Sensitivity to weight precision in deep recursive architectures has not been characterized at this scale; this work is itself contributing initial data.
- Training was conducted on consumer hardware (Apple M3 Max) without distributed training. Larger-scale training was not attempted.

## Bias, risks, ethical considerations

This is a Sudoku solver. The primary "risk" is that a 1-bit reasoner pattern, if it works for Sudoku, will be applied to higher-stakes constraint-satisfaction problems (medical triage, security, autonomous systems) without appropriate domain validation. The README "Real-world applications" section maps potential domains; **none should be deployed without per-domain training, validation, and safety review.**

## Citation

```bibtex
@software{barrett2025_1bit_trm_sudoku,
  author = {Barrett, Richard},
  title = {{1-Bit Recursive Sudoku Solver: BitDistill on Tiny Recursive Models}},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/RNBBarrett/1-Bit-TRM-Sudoku-Solver}
}
```

Plus the source papers cited in the repo's main README.
