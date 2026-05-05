"""Standalone progress monitor for the cloud training run.

SSHes to the pod, reads the training log, parses the latest state, and
prints a human-readable table. Run as often as you like:

    python scripts/check_progress.py

Optional:
    python scripts/check_progress.py --watch     # auto-refresh every 60 sec
    python scripts/check_progress.py --watch 30  # auto-refresh every 30 sec

If the pod IP/port changes (e.g. after a stop/start), edit the constants
at the top of this file or override via env vars POD_IP / POD_PORT.

No external dependencies - uses only stdlib + system `ssh` command.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ----- Pod connection (override via env vars if you redeploy) -----
POD_IP = os.environ.get("POD_IP", "103.196.86.82")
POD_PORT = os.environ.get("POD_PORT", "42367")
SSH_KEY = os.environ.get("SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))
LOG_PATH = os.environ.get(
    "LOG_PATH",
    "/workspace/1-Bit-TRM-Sudoku-Solver/runs/samsung_v6.log",
)

# ----- Cost / budget -----
COST_PER_HOUR = float(os.environ.get("POD_COST_PER_HR", "0.69"))
BUDGET = float(os.environ.get("BUDGET", "20.00"))
# Wall-clock auto-stop (matches the --hours flag passed to train.py).
# Training will gracefully exit at this many hours regardless of step count.
TRAIN_HOURS_BUDGET = float(os.environ.get("TRAIN_HOURS_BUDGET", "24"))

# ----- Reference baselines (Samsung sudoku-extreme distribution) -----
# Random over 11-token vocab; copy-clues = clue cells perfect, blanks uniform digit guess.
RANDOM_CE = math.log(11)        # 2.398
COPY_CLUES_CE = 1.51            # for ~25-clue puzzles


def ssh_fetch_log(n_lines: int = 400, timeout: int = 30) -> str:
    """Return the last N lines of the training log file on the pod."""
    cmd = [
        "ssh",
        "-p", POD_PORT,
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        "-o", "BatchMode=yes",
        f"root@{POD_IP}",
        f"tail -{n_lines} {LOG_PATH} 2>/dev/null && echo '__END__'",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__"
    if result.returncode != 0:
        return f"__SSH_ERROR__\n{result.stderr}"
    return result.stdout


def parse_state(log: str) -> dict:
    """Extract the most recent training state from log text."""
    state = {
        "step": None,
        "max_steps": None,
        "ce": None,
        "rate": None,
        "phase": None,
        "elapsed_min": None,
        "is_nan": False,
        "cell_acc_batch": None,
        "puzzles_solved_batch": None,
        "puzzles_in_batch": None,
        "eval_puzzle_acc": None,
        "eval_cell_acc": None,
        "eval_step": None,
        "ssh_error": None,
    }
    if log.startswith("__SSH_ERROR__"):
        state["ssh_error"] = log
        return state
    if log.startswith("__TIMEOUT__"):
        state["ssh_error"] = "SSH timeout (pod may be unreachable)"
        return state

    # 50-step compact log lines
    for m in re.finditer(
        r"\[\s*(\d+)/(\d+)\]\s+stage=(\w+)\s+loss=(\S+)\s+ce=(\S+)\s+"
        r"viol=\S+\s+halt=\S+\s+halt_w=\S+\s+\|\s+(\S+)\s+step/s",
        log,
    ):
        state["step"] = int(m.group(1))
        state["max_steps"] = int(m.group(2))
        state["phase"] = m.group(3)
        ce_str = m.group(5)
        if ce_str.lower() == "nan":
            state["is_nan"] = True
        else:
            try:
                state["ce"] = float(ce_str)
            except ValueError:
                pass
        try:
            state["rate"] = float(m.group(6))
        except ValueError:
            pass

    # Heartbeat elapsed time
    hb_matches = re.findall(r"TRAINING HEARTBEAT\s+--\s+([\d.]+)\s*min elapsed", log)
    if hb_matches:
        state["elapsed_min"] = float(hb_matches[-1])

    # Per-batch accuracy from the most recent heartbeat
    cell_matches = re.findall(r"got (\d+)/81 cells right", log)
    if cell_matches:
        state["cell_acc_batch"] = int(cell_matches[-1]) / 81.0

    solved_matches = re.findall(r"completely solved (\d+)/(\d+) puzzles", log)
    if solved_matches:
        state["puzzles_solved_batch"] = int(solved_matches[-1][0])
        state["puzzles_in_batch"] = int(solved_matches[-1][1])

    # Most recent eval result
    eval_re = re.compile(
        r"eval@(\d+):\s+puzzle_acc=([\d.]+)\s+cell_acc=([\d.]+)"
    )
    evals = eval_re.findall(log)
    if evals:
        state["eval_step"] = int(evals[-1][0])
        state["eval_puzzle_acc"] = float(evals[-1][1])
        state["eval_cell_acc"] = float(evals[-1][2])

    return state


def skill_level(ce: float | None) -> tuple[str, float]:
    """Return (description, score_1_to_10) given current cross-entropy."""
    if ce is None:
        return ("Not measured yet", 0.0)
    if ce > RANDOM_CE:
        return ("Worse than random - broken or just initialized", 0.5)
    # Normalize position relative to baselines.
    if ce > COPY_CLUES_CE:
        # In the random -> copy-clues range
        pct = 100.0 * (RANDOM_CE - ce) / max(RANDOM_CE - COPY_CLUES_CE, 1e-6)
        score = 0.5 + 1.5 * (pct / 100.0)  # 0.5 .. 2.0
        return (f"Learning to copy visible clues ({pct:.0f}% of the way)", score)
    if ce > 1.0:
        # Beyond copy-clues - into reasoning territory
        depth = (COPY_CLUES_CE - ce) / (COPY_CLUES_CE - 1.0)
        score = 2.0 + 3.0 * depth   # 2.0 .. 5.0
        return ("Reasoning about blanks - past copy-clues baseline", score)
    if ce > 0.5:
        score = 5.0 + 3.0 * (1.0 - ce) / 0.5
        return ("Solving most cells - close to puzzle-level accuracy", score)
    score = 8.0 + 2.0 * (0.5 - ce) / 0.5
    return ("Solving full puzzles consistently", min(score, 10.0))


def smarter_verdict(ce: float | None, is_nan: bool) -> str:
    if is_nan:
        return "STOPPED (NaN - training crashed)"
    if ce is None:
        return "Not measured yet"
    if ce > RANDOM_CE:
        return "No - broken / not learning"
    if ce > 2.0:
        return "Just starting (only a tiny bit smarter than random)"
    if ce > COPY_CLUES_CE:
        return "Yes - learning to copy clues"
    if ce > 1.0:
        return "Yes - past copy-clues, doing real reasoning"
    if ce > 0.5:
        return "Yes - strongly reasoning, close to solving"
    return "Yes - solving puzzles"


def estimate_pod_uptime_min(state: dict, training_log: str) -> float:
    """Best-effort uptime estimate from the elapsed-min field in the log."""
    # Use the last heartbeat's elapsed time. If unavailable, fall back to
    # a rough guess based on (step / rate) seconds.
    if state["elapsed_min"] is not None:
        return state["elapsed_min"]
    if state["step"] and state["rate"] and state["rate"] > 0:
        return (state["step"] / state["rate"]) / 60.0
    return 0.0


def render(state: dict, log: str) -> str:
    """Build the human-readable table."""
    lines = []
    if state["ssh_error"]:
        lines.append("=" * 60)
        lines.append("ERROR connecting to pod:")
        lines.append(state["ssh_error"][:500])
        lines.append("=" * 60)
        return "\n".join(lines)

    smart = smarter_verdict(state["ce"], state["is_nan"])
    skill_desc, score = skill_level(state["ce"])

    if state["puzzles_solved_batch"] is not None:
        solved_str = f"{state['puzzles_solved_batch']}/{state['puzzles_in_batch']} (this batch)"
    else:
        solved_str = "0 (no batch reported yet)"
    if state["eval_puzzle_acc"] is not None:
        # Eval is over the held-out val set, more meaningful
        n_solved = round(state["eval_puzzle_acc"] * 10000)
        solved_str = (
            f"{state['eval_puzzle_acc']*100:.2f}% on val set "
            f"(eval@{state['eval_step']})"
        )

    elapsed_min = estimate_pod_uptime_min(state, log)
    cost_so_far = (elapsed_min / 60.0) * COST_PER_HOUR

    if state["max_steps"] and state["rate"] and state["rate"] > 0:
        # Training stops at MIN(steps_to_finish, --hours budget).
        steps_remaining_min = (state["max_steps"] - (state["step"] or 0)) / max(state["rate"], 1e-6) / 60.0
        hours_budget_remaining_min = max(TRAIN_HOURS_BUDGET * 60.0 - elapsed_min, 0)
        remaining_min = min(steps_remaining_min, hours_budget_remaining_min)
        eta_total_min = elapsed_min + remaining_min
        projected_total = (eta_total_min / 60.0) * COST_PER_HOUR
        if state["max_steps"]:
            steps_pct = 100.0 * (state["step"] or 0) / state["max_steps"]
        else:
            steps_pct = 0.0
        # Note which limit will hit first
        if steps_remaining_min < hours_budget_remaining_min:
            stop_reason = f"step {state['max_steps']}"
        else:
            stop_reason = f"{TRAIN_HOURS_BUDGET:.0f}-hour limit"
    else:
        projected_total = float("nan")
        steps_pct = 0.0
        stop_reason = "(unknown)"

    if state["is_nan"]:
        budget_status = "STOPPED - training crashed"
    elif projected_total <= BUDGET * 0.85:
        budget_status = "On track (>15% buffer)"
    elif projected_total <= BUDGET:
        budget_status = "On track (tight, <15% buffer)"
    else:
        budget_status = f"OVER by ${projected_total - BUDGET:.2f}"

    if state["phase"] == "warmup":
        phase_desc = "Easy puzzles - learning to copy visible numbers"
    elif state["phase"] == "mixed-1":
        phase_desc = "Easy + Medium mix - starting to reason about blanks"
    elif state["phase"] == "mixed-2":
        phase_desc = "Easy + Medium + Extreme mix - harder reasoning"
    elif state["phase"] == "final":
        phase_desc = "All difficulty levels uniform - final polish"
    else:
        phase_desc = state["phase"] or "(unknown)"

    width = 70
    lines.append("=" * width)
    lines.append(f"  TRAINING STATUS  (refreshed: {time.strftime('%H:%M:%S')})")
    lines.append("=" * width)
    lines.append(f"  Smarter?         | {smart}")
    lines.append(f"  Phase            | {phase_desc}")
    lines.append(f"  Skill level      | {skill_desc}")
    lines.append(f"  Score (1-10)     | {score:.1f}")
    lines.append(f"  Puzzles solved   | {solved_str}")
    lines.append("-" * width)
    if state["step"]:
        lines.append(f"  Step             | {state['step']:,} / {state['max_steps']:,}  ({steps_pct:.2f}% done)")
    else:
        lines.append("  Step             | (training hasn't logged a step yet)")
    if state["rate"]:
        lines.append(f"  Speed            | {state['rate']:.2f} steps/sec")
    if state["ce"] is not None:
        lines.append(f"  Uncertainty (CE) | {state['ce']:.4f}   "
                     f"(random={RANDOM_CE:.2f}, copy-clues={COPY_CLUES_CE:.2f}, perfect=0)")
    lines.append("-" * width)
    lines.append(f"  Time training    | {elapsed_min:.1f} min")
    lines.append(f"  Cost so far      | ${cost_so_far:.2f} of ${BUDGET:.2f}")
    if not math.isnan(projected_total):
        lines.append(f"  Projected total  | ${projected_total:.2f}  (stops at {stop_reason})")
    lines.append(f"  Budget status    | {budget_status}")
    lines.append("=" * width)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", nargs="?", const=60, type=int, default=None,
                    help="auto-refresh every N seconds (default 60)")
    args = ap.parse_args()

    if args.watch is None:
        log = ssh_fetch_log()
        state = parse_state(log)
        print(render(state, log))
        return

    while True:
        try:
            log = ssh_fetch_log()
            state = parse_state(log)
            # Clear screen on Windows / Unix
            os.system("cls" if os.name == "nt" else "clear")
            print(render(state, log))
            print(f"\n  (refreshing every {args.watch}s - Ctrl-C to exit)")
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(f"poll error: {e}")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
