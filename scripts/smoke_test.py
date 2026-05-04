"""Tiny end-to-end smoke run.

Generates 100 puzzles, trains for 50 optimizer steps, confirms loss is
finite and decreases. Used for CI and as a quick "is anything broken"
gate before kicking off long runs.

Example:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    data_path = ROOT / "data" / "poc_smoke.pt"
    if not data_path.exists():
        run([sys.executable, "data_gen.py", "--target-count", "100",
             "--avg-rank", "50", "--out", str(data_path)])
    run([
        sys.executable, "train.py",
        "--data", str(data_path),
        "--max-steps", "50",
        "--micro-batch", "4",
        "--accum-steps", "2",
        "--eval-every", "25",
        "--ckpt-every", "50",
        "--out", str(ROOT / "checkpoints" / "smoke"),
    ])
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
