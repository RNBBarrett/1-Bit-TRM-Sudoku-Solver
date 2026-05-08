# Social distribution drafts

Templates ready to use once final training numbers land. Edit the bracketed values.

---

## Twitter/X thread (10 tweets)

**Tweet 1 — hook (with benchmark image)**
> 1-bit recursive reasoning works.
>
> First ternary-weight (BitNet b1.58) Sudoku-Extreme reasoner, distilled from a 7M-param TRM teacher into a 1.4 MB student.
>
> [TBD]% puzzle accuracy at 1/16 the footprint of the FP baseline.
>
> Thread + repo:

**Tweet 2 — context**
> Tiny recursive models like Samsung TRM and HRM beat trillion-parameter LLMs on hard reasoning tasks.
>
> The catch: they're still ~28 MB FP weights.
>
> What if you could squeeze a recursive reasoner into 1.4 MB?

**Tweet 3 — what we tried**
> Naive QAT-from-scratch on a 6M recursive model: tried 6 times. Every time it collapsed to "predict empty everywhere" once full ternary engaged.
>
> The "predict empty" basin is real and attractive when you only have hard CE labels.

**Tweet 4 — what worked**
> Knowledge distillation from a full-precision teacher.
>
> Teacher's soft probability distributions over digits 1–9 give the student a ~1000× stronger gradient signal than CE.
>
> The "predict empty" basin produces KD loss ≈ -log(1e-6) ≈ 13.8 per cell. Catastrophically high. The math overwhelms the bad attractor.

**Tweet 5 — recipe (with code snippet image)**
> Recipe:
> • Lambda-ramped ternary QAT (HF blog 2024)
> • Per-layer learnable α (TTQ-style)
> • Median weight scaling (BitNet Reloaded)
> • EMA at 0.999
> • KD with τ=5.0, λ_KD=10, λ_CE=0.1
>
> Plus the BitDistill paper's distillation technique.

**Tweet 6 — results table image**

**Tweet 7 — failure-mode honesty**
> First we tried building a novel "recursion-on-recursion" architecture. It plateaued at 38.7% cell accuracy.
>
> Diagnosis: 5 specific recipe gaps vs Samsung's TRM (puzzle_emb, ACT bootstrapping, loss weights, LR schedule, init).
>
> Negative results matter. Posted the diff in the README.

**Tweet 8 — applications**
> Why a 1.4 MB recursive reasoner matters:
>
> • ICS/SCADA anomaly detection on PLCs
> • ECG arrhythmia detection on smartwatches at <1 mW
> • Mars rover anomaly screening
> • Drug-drug interaction reasoning at point-of-dispensing
> • Drone navigation in GPS-denied environments

**Tweet 9 — what's next**
> Roadmap: full FP teacher → multi-step deep-supervision KD → wider student → ensemble + full Sudoku-Extreme + test-time compute scaling.
>
> Goal: beat Samsung's 87.4% with 1-bit weights.
>
> Estimated 6 weeks + ~$30 cloud.

**Tweet 10 — call to action**
> Repo (MIT license, full reproduction recipe):
> github.com/RNBBarrett/1-Bit-TRM-Sudoku-Solver
>
> Citations to all papers we built on:
> • TRM (Samsung)
> • BitNet b1.58 (Microsoft)
> • BitDistill
> • HRM (Sapient)
>
> Stars + forks welcome. RTs even more so.

---

## Hacker News submission

**Title (≤80 chars):**
> Show HN: 1-bit recursive Sudoku reasoner in 1.4 MB

**Submission body (the post itself just needs the URL; this goes as a top comment):**

> Hi HN — I'm an independent researcher posting this in-progress work.
>
> The premise: Samsung's Tiny Recursive Model (TRM, 2025) is a ~7M-parameter recursive reasoner that hits 87.4% on Sudoku-Extreme — beating GPT-4 (0%) and Llama 3.1 70B (<5%) at 1/10,000 the parameter count.
>
> I quantized all of TRM's linear layers to 1.58-bit ternary weights (BitNet b1.58) using knowledge distillation from a full-precision teacher checkpoint. Direct quantization-aware-training-from-scratch fails — every recursive QAT run I tried (5 of them) collapsed to "predict empty everywhere" within a few thousand steps. Distillation rescues it.
>
> Result so far: a ~1.4 MB ternary-weight Sudoku reasoner that's currently at [TBD]% puzzle accuracy. Training is ongoing on an Apple M3 Max.
>
> The README documents the journey including:
> - Why my original "recursion-on-recursion" architecture failed (5 specific recipe gaps vs Samsung's published code)
> - Full reproduction recipe + commands
> - Concrete roadmap to beating Samsung's 87.4% (~6 weeks, ~$30 cloud)
> - 7 industry verticals where this could deploy (cybersecurity edge, medical wearables, satellite onboard, etc.)
>
> Happy to answer any questions. Would especially love feedback from anyone who's quantized recursive architectures — there's almost no published work in this space.

---

## Reddit r/MachineLearning [P]roject post

**Title:**
> [P] First 1-bit (BitNet b1.58) recursive Sudoku reasoner — distilled from Samsung TRM into a 1.4 MB ternary model

**Body:**

> Hi r/MachineLearning — sharing an in-progress research result.
>
> **Background:** Samsung SAIL Montreal's Tiny Recursive Model (TRM, arXiv:2510.04871) is a 7M-param recursive reasoner that gets 87.4% on Sudoku-Extreme. Microsoft's BitNet b1.58 (arXiv:2402.17764) showed that LLMs can be trained with ternary {-1, 0, +1} weights at minimal accuracy loss. **Nobody has published a 1-bit recursive reasoner.**
>
> **What I did:**
> 1. Cloned Samsung's TRM, replaced every `nn.Linear` with a `BitCastedLinear` (ternary weights, 8-bit activations, lambda-ramped QAT, learnable per-layer α, median scaling).
> 2. Tried QAT from scratch — collapses to "predict empty everywhere" every time. The recursion amplifies quantization noise badly enough that the model finds a low-effort degenerate solution before learning to reason.
> 3. Trained a partial FP teacher (66.5% puzzle accuracy at 30% of Samsung's schedule), then distilled into the ternary student via temperature-scaled KL divergence.
>
> **Recipe:** standard BitDistill (arXiv:2510.13998) approach with the v6 stabilization tricks: lambda warmup over 2k steps, FP warmup phase before quantization phases in, EMA at 0.999, AdamW with constant LR after 2k-step linear warmup.
>
> **Current status:** training is ongoing. Intermediate evals at step 10k show 68.2% cell accuracy and 5% exact-puzzle-match — first non-zero puzzle accuracy for any 1-bit recursive reasoner I'm aware of.
>
> **Honest caveats:**
> - Final numbers TBD (training continues for several more days)
> - The teacher is itself partially-trained; full Samsung schedule would lift the student ceiling
> - This is single-machine work; no distributed training, no extensive hyperparameter sweeps
>
> **Repo:** github.com/RNBBarrett/1-Bit-TRM-Sudoku-Solver
> Full README has the failure modes, the recipe diff vs Samsung, the roadmap to >87.4%, and 7 candidate domain transfers.
>
> Genuinely interested in feedback — particularly from anyone who's quantized recursive or iterative-refinement architectures.

---

## ML newsletter pitch (for TLDR AI / Last Week in AI / etc.)

**Subject:** 1.4 MB ternary-weight Sudoku reasoner — first 1-bit recursive model

**Body:**

> Hi [editor],
>
> [Your name] here. I'd like to submit a research result for consideration.
>
> A few weeks ago I started trying to build a 1-bit version of Samsung's Tiny Recursive Model — the 7M-param recursive reasoner that beats trillion-param LLMs on Sudoku-Extreme. After several failed approaches, I got a working 1-bit student via distillation from a full-precision teacher.
>
> Highlights:
> - First 1-bit (BitNet b1.58) recursive reasoner published anywhere
> - 1.4 MB footprint (vs 28 MB for the FP baseline)
> - [TBD]% puzzle accuracy on Sudoku-Extreme at intermediate training
> - Repo includes failure-mode analysis, full reproduction recipe, and roadmap to beating Samsung's 87.4%
>
> Repo: github.com/RNBBarrett/1-Bit-TRM-Sudoku-Solver
>
> Would be glad to write a 100-word summary if it fits your format.
>
> Thanks,
> [Your name]

---

## Notes on timing

- **Twitter/Reddit:** Tue–Thu, 9-11am US ET = best uptake.
- **Hacker News:** Sun–Tue, 9-11am US ET. Don't post Friday afternoon.
- **Newsletters:** mid-week emails get fastest turnaround.
- **Wait until final numbers land** before any of this. Posting "TBD" results is a credibility hit.
