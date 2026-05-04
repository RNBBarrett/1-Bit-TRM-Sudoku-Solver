# Training the 1-Bit HTRM on RunPod (Cloud GPU Runbook)

End-to-end instructions for training the full-spec model on a rented
RTX 4090. Estimated total cost: **~$10–15** for one full 50k-step run.
Estimated wall-clock: **~17 hours** for the run + ~30 min of setup.

## Phase 0 — Prerequisites (do once)

1. **A RunPod account** — sign up at <https://runpod.io>. Add a payment
   method (Stripe). $20 credit is plenty for one full run + a couple of
   restarts.
2. **An SSH client on Windows.** Built-in is fine: `ssh` is available
   in PowerShell out of the box on Windows 10+.
3. **GitHub repo access** OR ability to upload the code as a tarball.
   Both paths covered below.

## Phase 1 — Spin up the GPU pod (~5 min)

1. RunPod dashboard → **Pods** → **Deploy**.
2. **GPU**: select **RTX 4090** under Community Cloud (cheapest tier).
   Spot pricing fluctuates; expect **$0.34–0.69/hr**.
3. **Template**: choose `RunPod PyTorch 2.4` (or any "PyTorch 2.x" template).
4. **Disk**: 30 GB container disk + 30 GB volume is enough. Volumes
   persist across pod stops; container disk does not.
5. **Expose ports**: SSH (22) is enabled by default on RunPod. JupyterLab
   (8888) is auto-exposed too if you want a web shell.
6. **Deploy on Spot**. (We have `--resume` so spot interruption is fine.)
7. Wait ~60 seconds for the pod to come online. Note the pod's **public
   IP and SSH port** (something like `ssh root@123.45.67.89 -p 12345`).

## Phase 2 — Connect to the pod (~2 min)

From your local Windows PowerShell:

```powershell
ssh root@<POD_IP> -p <POD_SSH_PORT>
```

Accept the host key fingerprint on first connect. You should land in
`/workspace`, which is the persistent volume.

## Phase 3 — Get the code on the pod (~5 min)

**Option A: via GitHub** (if you've pushed the repo somewhere):

```bash
cd /workspace
git clone https://github.com/<YOUR-USER>/1-bit-TRM-Sudoku-Solver.git
cd 1-bit-TRM-Sudoku-Solver
```

**Option B: via direct upload** (if no GitHub):

On your local Windows PowerShell, from inside the repo directory:

```powershell
# Compress the repo (excluding data and checkpoints)
$exclude = @("data", "checkpoints", "runs", "__pycache__", ".pytest_cache", ".venv")
Compress-Archive -Path * -DestinationPath ..\htrm_repo.zip -Force

# Upload it
scp -P <POD_SSH_PORT> ..\htrm_repo.zip root@<POD_IP>:/workspace/
```

Then on the pod:

```bash
cd /workspace
mkdir 1-bit-TRM-Sudoku-Solver
cd 1-bit-TRM-Sudoku-Solver
unzip /workspace/htrm_repo.zip
```

## Phase 4 — Upload the dataset (~3 min)

The Samsung-converted dataset is ~165 MB. Upload from your Windows machine:

```powershell
scp -P <POD_SSH_PORT> data/samsung_train.pt root@<POD_IP>:/workspace/1-bit-TRM-Sudoku-Solver/data/
scp -P <POD_SSH_PORT> data/samsung_test.pt root@<POD_IP>:/workspace/1-bit-TRM-Sudoku-Solver/data/
```

(Create the `data/` directory on the pod first if it doesn't exist:
`mkdir -p /workspace/1-bit-TRM-Sudoku-Solver/data` over SSH.)

## Phase 5 — Run the cloud setup script (~3 min)

On the pod:

```bash
cd /workspace/1-bit-TRM-Sudoku-Solver
bash scripts/cloud_setup.sh
```

This:
- Verifies CUDA is visible (`nvidia-smi`)
- Confirms PyTorch sees the GPU
- Installs project deps (skips torch if already there)
- Runs the unit-test suite (61 tests, ~5 sec)
- Smoke-tests the full-spec model on the GPU

You should see:
```
=== 1. CUDA visibility ===
GeForce RTX 4090, 24576 MiB, 555.xx, 8.9
...
=== 4. Running unit tests ===
61 passed in 5s
=== 5. Model smoke check ===
params: 6,012,672
device: cuda:0
forward output shape: (2, 81, 11)
macro_used: 48, micro_used: 432
smoke forward: OK
```

If any step fails, **stop and ping me** before launching the actual run.

## Phase 6 — Launch training in a screen/tmux session (~1 min)

The training run is multi-hour, and your SSH connection will likely
drop. Use `tmux` (pre-installed on RunPod) so the run survives
disconnect.

```bash
tmux new -s training
```

Inside the tmux session, launch training:

```bash
python -u train.py \
  --data data/samsung_train.pt \
  --config configs/htrm_full.yaml \
  --max-steps 50000 \
  --hours 24 \
  --status-every-min 15 \
  --micro-batch 4 \
  --accum-steps 8 \
  --eval-every 2000 \
  --ckpt-every 2000 \
  --val-frac 0.01 \
  --halt-ramp-steps 5000 \
  --lr 5e-5 \
  --weight-decay 1.0 \
  --grad-clip 0.5 \
  --gradient-checkpoint \
  --out checkpoints/full_v2 \
  --seed 42 \
  2>&1 | tee runs/cloud_run_$(date +%Y%m%d_%H%M%S).log
```

The `tee` writes the same output to a log file so we have an audit
trail.

**To detach from tmux without killing training**: `Ctrl-b` then `d`.

**To re-attach later**: `tmux attach -t training`.

## Phase 7 — Monitor while it runs (~17 hours)

Every 15 minutes you'll see a plain-English heartbeat block. Look for:
- `Trend: GETTING SMARTER` — uncertainty falling between heartbeats
- `cell_acc=` rising over time
- `position: BEYOND just copying clues` — the meaningful threshold

Every ~2,600 steps an eval lands with real puzzle/cell accuracy stats.

You can SSH back in and `tmux attach -t training` from anywhere to
check progress. Or check the log file:

```bash
tail -f runs/cloud_run_*.log
```

## Phase 8 — Pull results back (~5 min)

When the run finishes (or when you stop it with Ctrl-C in tmux), the
checkpoints are at `checkpoints/full_v2/`. Pull the best one back to
your local Windows machine:

```powershell
scp -P <POD_SSH_PORT> root@<POD_IP>:/workspace/1-bit-TRM-Sudoku-Solver/checkpoints/full_v2/best.pt checkpoints/full_v2/
scp -P <POD_SSH_PORT> root@<POD_IP>:/workspace/1-bit-TRM-Sudoku-Solver/checkpoints/full_v2/last.pt checkpoints/full_v2/
scp -P <POD_SSH_PORT> root@<POD_IP>:/workspace/1-bit-TRM-Sudoku-Solver/checkpoints/full_v2/train_log.json checkpoints/full_v2/
```

Then evaluate locally (your AMD GPU is plenty for inference):

```powershell
python evaluate_extreme.py --ckpt checkpoints/full_v2/best.pt --data data/samsung_test.pt --val-frac 1.0 --batch-size 4 --csv-out cloud_results.csv
```

## Phase 9 — Stop the pod (~2 min)

CRITICAL: pods bill per-second while running. **Always stop the pod
when you're done.**

RunPod dashboard → **Pods** → your training pod → **Stop**. The volume
persists, so if you want to extend training next week you can restart
the same pod and `tmux attach`.

If you're sure you won't extend, you can **Terminate** instead of just
Stop to free the volume too.

## Optional Phase 10 — Resume on cloud

To extend training later:

1. Start the pod (RunPod dashboard → Start)
2. SSH in
3. `tmux attach -t training` (the previous session is gone after
   pod stop, so launch fresh):
4. Re-run the same launch command **with `--resume checkpoints/full_v2/last.pt`** at the end.

## Quick reference: full command (cloud)

```bash
python -u train.py --data data/samsung_train.pt --config configs/htrm_full.yaml --max-steps 50000 --hours 24 --status-every-min 15 --micro-batch 4 --accum-steps 8 --eval-every 2000 --ckpt-every 2000 --val-frac 0.01 --halt-ramp-steps 5000 --lr 5e-5 --weight-decay 1.0 --grad-clip 0.5 --gradient-checkpoint --out checkpoints/full_v2 --seed 42
```

To resume: append `--resume checkpoints/full_v2/last.pt`.

## Cost expectations

| Scenario | Cost |
|---|---|
| Single clean 17-hour run on 4090 spot @ $0.50/hr | $8.50 |
| Run + one restart (24-hour budget × 1.3) | ~$11 |
| Run + multiple restarts (NaN debug) | ~$15–20 |
| Hard ceiling (full 50k steps + buffer) | ~$25 |

Add ~$0.50 for the setup phase (1 hour at $0.50/hr).

## If something goes wrong

| Symptom | Fix |
|---|---|
| `cuda not available` after setup script | wrong template; redeploy with a CUDA template |
| Pod times out / spot preempted | Re-launch pod, `tmux new`, re-run with `--resume` |
| NaN in training | Ctrl-C, escalate mitigation (lower lr, tighter clip), restart fresh `--out` dir |
| OOM | Reduce `--micro-batch` from 4 to 2 |
| `tmux: command not found` | `apt-get install -y tmux` (rare on RunPod) |
| `pip install` fails | Try `pip install --no-cache-dir -r requirements-cuda.txt` |
| SSH connection drops mid-training | Re-SSH, `tmux attach -t training` — training still running |
