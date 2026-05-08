# Contributing

Thanks for taking the time to look. This project is research-grade — actively in-progress training, methods are being validated, and results may shift. PRs are welcome but please read the relevant section below first.

## Ways to contribute

### Found a bug
Open an [issue](https://github.com/RNBBarrett/1-Bit-TRM-Sudoku-Solver/issues) with:
- Repro steps (full command line + which checkpoint)
- Hardware (GPU/MPS/CPU; vendor; RAM)
- PyTorch version + Python version
- Full traceback

### Want to extend the work
Particularly welcome:
- **Multi-step deep-supervision distillation** (per-iteration KD, see Phase 2 in README roadmap)
- **Wider-student variants** — `hidden_size=768` or `hidden_size=1024` runs
- **Other quantization schemes** — int4, mixed-precision (8-bit attention + 1-bit MLP)
- **Other reasoning benchmarks** — Maze-Hard, ARC-AGI, KenKen, kakuro
- **Real-world domain transfers** — see "Real-world applications" section in the README; we'd love to see this architecture deployed for any of those domains

### Smaller, easier
- README typos / clarifications
- Improvements to `scripts/check_tier3_mac.py` (the status monitor)
- Reproducibility tickets — tell us what didn't reproduce on your hardware
- Adding test cases

## Development setup

```bash
git clone https://github.com/RNBBarrett/1-Bit-TRM-Sudoku-Solver.git
cd 1-Bit-TRM-Sudoku-Solver
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt    # core deps
pip install -r requirements-cuda.txt  # if you have CUDA
pytest tests/ -q   # 77 tests should pass
```

For Tier 3 distillation work you'll also need Samsung's repo cloned + patched (see README §Reproduction).

## Pull request process

1. **Discuss first.** Open an issue describing what you want to change before writing a large PR. Saves both of us time if the direction needs adjustment.
2. **Keep changes focused.** One concept per PR. Bug fix + refactor + new feature in one PR will be split.
3. **Tests where it matters.** New BitNet variants, KD loss changes, dataset changes — add tests. README typo fixes don't.
4. **No drive-by reformatting.** Don't reformat existing code in a feature PR; submit formatting changes separately if you want them.
5. **Mention the issue number** in the PR description if applicable.

## Code style

- **Python:** PEP 8 with 100-char line limit. We don't use a hard formatter; just be reasonable.
- **Comments:** explain the *why* (especially for non-obvious training-recipe choices). The *what* is in the code.
- **Citations:** if you implement a paper's idea, cite the paper inline as a comment.

## Testing

```bash
pytest tests/ -q                      # all 77 unit tests
pytest tests/test_distillation.py -v  # KD loss tests
pytest tests/test_lambda_ramp.py -v   # BitLinear quantization tests
```

For training-loop changes, also do a smoke test:
```bash
python scripts/tier3/train_bitdistill.py [...] --smoke
```

## License

By submitting a PR you agree your contribution will be licensed under the project's MIT license.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be excellent to each other.
