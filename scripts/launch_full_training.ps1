# Full-spec 1-Bit HTRM training launch script.
#
# Trains the 6M-param 4-level recursion model on Samsung's curriculum-
# tagged data. Defaults: 50,000 optimizer steps, ~3 weeks elapsed on
# AMD RX 5700 XT via DirectML.
#
# Usage (Windows PowerShell 5.1):
#   powershell -ExecutionPolicy Bypass -File .\scripts\launch_full_training.ps1
# Usage (PowerShell 7+, if installed):
#   pwsh -ExecutionPolicy Bypass -File .\scripts\launch_full_training.ps1
#
# Or call train.py directly (simplest):
#   python train.py --data data/samsung_train.pt --config configs/htrm_full.yaml \
#                   --max-steps 50000 --micro-batch 8 --accum-steps 4 \
#                   --eval-every 2000 --ckpt-every 2000 --val-frac 0.01 \
#                   --quantize-after-step 10000 --halt-ramp-steps 5000 \
#                   --gradient-checkpoint --out checkpoints/full --seed 42
#
# To resume: see 'How to resume' below.

param(
    [int]$MaxSteps = 50000,
    [int]$EvalEvery = 2000,
    [int]$CkptEvery = 2000,
    [int]$MicroBatch = 8,
    [int]$AccumSteps = 4,
    [string]$DataPath = "data/samsung_train.pt",
    [string]$OutDir = "checkpoints/full",
    [int]$Seed = 42,
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"

# Quantize-after-step at 20% of max-steps (BitNet b1.58 paper recipe).
$QuantizeAfterStep = [int]($MaxSteps * 0.20)
# Halt-loss ramp at 10% of max-steps.
$HaltRampSteps = [int]($MaxSteps * 0.10)

Write-Host "===== 1-Bit HTRM Full-Spec Training =====" -ForegroundColor Cyan
Write-Host "max_steps           : $MaxSteps"
Write-Host "quantize_after_step : $QuantizeAfterStep (FP warmup ends here)"
Write-Host "halt_ramp_steps     : $HaltRampSteps"
Write-Host "data                : $DataPath"
Write-Host "out_dir             : $OutDir"
Write-Host "expected wall-clock : ~3 weeks on AMD RX 5700 XT (DirectML)"
Write-Host ""

$args_list = @(
    "-u", "train.py",
    "--data", $DataPath,
    "--config", "configs/htrm_full.yaml",
    "--max-steps", $MaxSteps,
    "--micro-batch", $MicroBatch,
    "--accum-steps", $AccumSteps,
    "--eval-every", $EvalEvery,
    "--ckpt-every", $CkptEvery,
    "--val-frac", "0.01",
    "--quantize-after-step", $QuantizeAfterStep,
    "--halt-ramp-steps", $HaltRampSteps,
    "--gradient-checkpoint",
    "--out", $OutDir,
    "--seed", $Seed
)
if ($Resume -ne "") {
    $args_list += "--resume"
    $args_list += $Resume
    Write-Host "RESUMING from $Resume" -ForegroundColor Yellow
}
python @args_list

# How to stop:
#   Press Ctrl-C — the training loop catches it, saves last.pt + train_log.json,
#   prints the resume command, and exits cleanly. You can also kill the process
#   harshly; you'll lose at most $CkptEvery steps of progress (the last
#   periodic checkpoint is always on disk).
#
# How to resume (Windows PowerShell 5.1):
#   powershell -ExecutionPolicy Bypass -File .\scripts\launch_full_training.ps1 -Resume "checkpoints/full/last.pt"
# Or directly:
#   python train.py --resume checkpoints/full/last.pt --data data/samsung_train.pt \
#                   --config configs/htrm_full.yaml --max-steps 50000 \
#                   --quantize-after-step 10000 --halt-ramp-steps 5000 \
#                   --gradient-checkpoint --out checkpoints/full
# This restores the model, optimizer, step counter, and log, then continues.
# The FP-warmup transition and curriculum stage are auto-derived from step.
#
# How to monitor:
#   - tail $OutDir/train_log.json    # JSON progress log
#   - last 50-step log lines stream to stdout (and the calling shell)
#   - to view interactively, redirect to a file: ... 2>&1 | Tee-Object run.log
