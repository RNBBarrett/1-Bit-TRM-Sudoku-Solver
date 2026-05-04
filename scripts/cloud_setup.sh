#!/usr/bin/env bash
# Cloud setup script for the 1-Bit HTRM training run.
# Designed for RunPod / Vast.ai / Lambda Labs Linux instances with a CUDA GPU.
#
# Usage (on the cloud instance, in the repo root):
#   bash scripts/cloud_setup.sh
#
# What it does:
#   1. Verifies CUDA is visible
#   2. Installs missing Python deps (uses the pre-installed torch if present)
#   3. Runs the unit-test suite to confirm everything composes
#   4. Verifies the model wires up on the cloud GPU

set -euo pipefail

echo "=== 1. CUDA visibility ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo ""

echo "=== 2. PyTorch + CUDA ==="
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo ""

echo "=== 3. Installing project deps (skips torch if already present) ==="
pip install --quiet --upgrade pip
pip install --quiet -r requirements-cuda.txt
echo "  installed."
echo ""

echo "=== 4. Running unit tests ==="
python -m pytest tests/ -q
echo ""

echo "=== 5. Model smoke check ==="
python model.py --config configs/htrm_full.yaml
echo ""

echo "=== Setup complete. ==="
echo "Next: launch training with one of the commands in TRAINING_CLOUD.md"
