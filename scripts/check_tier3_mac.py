"""Standalone progress monitor for the Tier 3 BitDistill training on the Mac.

SSH-fetches the latest training log and prints a human-readable summary.
Run as often as you like — does not consume any cloud or Claude credits.

    python scripts/check_tier3_mac.py
    python scripts/check_tier3_mac.py --watch 60   # auto-refresh every 60s
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

MAC_USER = os.environ.get("MAC_USER", "richard")
MAC_HOST = os.environ.get("MAC_HOST", "192.168.1.188")
LOG_PATH = "/Users/richard/trm-tier3/runs/tier3/tier3.log"


def fetch_log(n_lines: int = 200) -> str:
    cmd = ["ssh", f"{MAC_USER}@{MAC_HOST}", f"tail -{n_lines} {LOG_PATH}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__"
    if r.returncode != 0:
        return f"__ERR__\n{r.stderr}"
    return r.stdout


def parse(log: str) -> dict:
    state = {"step": None, "max_steps": None, "lam": None, "lr": None,
             "ce": None, "kd": None, "halt": None, "total": None, "rate": None,
             "eval_step": None, "eval_cell_acc": None, "eval_puzzle_acc": None,
             "elapsed_min": None}
    if log.startswith("__"):
        state["err"] = log
        return state
    # Parse compact step lines: [    50/200000] lam=... lr=... ce=... kd=... halt=... total=... | 0.57 step/s
    p = re.compile(
        r"\[\s*(\d+)/(\d+)\]\s+lam=([\d.]+)\s+lr=(\S+)\s+ce=([\d.]+)\s+"
        r"kd=([\d.]+)\s+halt=([\d.]+)\s+total=([\d.]+)\s+\|\s+(\S+)\s+step/s",
    )
    for m in p.finditer(log):
        state["step"] = int(m.group(1))
        state["max_steps"] = int(m.group(2))
        state["lam"] = float(m.group(3))
        state["lr"] = m.group(4)
        state["ce"] = float(m.group(5))
        state["kd"] = float(m.group(6))
        state["halt"] = float(m.group(7))
        state["total"] = float(m.group(8))
        try:
            state["rate"] = float(m.group(9))
        except ValueError:
            pass
    # Eval lines: eval@5000: cell_acc=0.4321 puzzle_acc=0.1543 ...
    e = re.compile(r"eval@(\d+):\s+cell_acc=([\d.]+)\s+puzzle_acc=([\d.]+)")
    for m in e.finditer(log):
        state["eval_step"] = int(m.group(1))
        state["eval_cell_acc"] = float(m.group(2))
        state["eval_puzzle_acc"] = float(m.group(3))
    return state


def render(state: dict) -> str:
    lines = []
    width = 70
    lines.append("=" * width)
    lines.append(f"  TIER 3 (1-bit Samsung TRM via distillation)  {time.strftime('%H:%M:%S')}")
    lines.append("=" * width)
    if "err" in state:
        lines.append("  SSH error:")
        lines.append("  " + state["err"][:500])
        return "\n".join(lines)

    if state["step"] is None:
        lines.append("  No step output yet (still loading model?)")
        return "\n".join(lines)

    pct = 100.0 * state["step"] / state["max_steps"]
    if state["rate"] and state["rate"] > 0:
        remaining = (state["max_steps"] - state["step"]) / state["rate"] / 3600
    else:
        remaining = float("nan")

    if state["lam"] == 0:
        phase = "Stage A (FP warmup, no quantization yet)"
    elif state["lam"] < 1.0:
        phase = f"Stage B (lambda ramp at {state['lam']:.2%})"
    else:
        phase = "Stage C (full ternary quantization)"

    lines.append(f"  Phase            | {phase}")
    lines.append(f"  Step             | {state['step']:,} / {state['max_steps']:,}  ({pct:.2f}% done)")
    lines.append(f"  Speed            | {state['rate']:.2f} step/s  (~{remaining:.1f} hr remaining)")
    lines.append(f"  Lambda           | {state['lam']:.3f}     (0=FP, 1=full quant)")
    lines.append("-" * width)
    lines.append(f"  CE loss          | {state['ce']:.4f}     (std cross-entropy)")
    lines.append(f"  KD loss          | {state['kd']:.4f}     (KL to teacher * T^2)")
    lines.append(f"  Halt loss        | {state['halt']:.4f}")
    lines.append("-" * width)
    if state["eval_step"] is not None:
        lines.append(f"  Latest eval      | step {state['eval_step']:,}")
        lines.append(f"    cell_acc       | {state['eval_cell_acc']:.4f}  ({state['eval_cell_acc']*100:.2f}%)")
        lines.append(f"    puzzle_acc     | {state['eval_puzzle_acc']:.4f}  ({state['eval_puzzle_acc']*100:.2f}%)")
        lines.append(f"  Tier 1 reference | cell=88.1%  puzzle=66.5%  (FP teacher)")
    else:
        lines.append("  Eval             | not yet (first eval at step 5000)")
    lines.append("=" * width)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", nargs="?", const=60, type=int, default=None)
    args = ap.parse_args()
    if args.watch is None:
        log = fetch_log()
        print(render(parse(log)))
        return
    while True:
        try:
            log = fetch_log()
            os.system("cls" if os.name == "nt" else "clear")
            print(render(parse(log)))
            print(f"\n  (auto-refresh every {args.watch}s — Ctrl-C to exit)")
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(f"poll error: {e}")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
