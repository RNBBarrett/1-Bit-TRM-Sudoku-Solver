# Security Policy

## Reporting a vulnerability

This is a research project — there's no production deployment to attack. But if you find:

- A serious bug that could affect downstream users (e.g., a way for malicious training data or a malformed checkpoint to cause arbitrary code execution)
- A supply-chain concern with our dependencies
- A privacy issue with how we handle the Sudoku-Extreme dataset

…please open a private security advisory via GitHub:

1. Go to https://github.com/RNBBarrett/1-Bit-TRM-Sudoku-Solver/security
2. Click "Report a vulnerability"
3. Fill in the form

We'll acknowledge within 72 hours and aim to resolve within 30 days.

## Out of scope

- "The model gets the wrong answer" — that's a training/research issue; open a regular issue.
- Vulnerabilities in upstream PyTorch / CUDA / Apple MPS — report directly to those projects.
- Vulnerabilities in cloned-and-patched Samsung TRM code — we vendor a small patch on top of upstream; report architectural issues to the [upstream repo](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

## Supported versions

This is research code. No long-term support is provided. The latest commit on `main` is the supported version.
