#!/usr/bin/env bash
# Sets up a venv, installs dependencies, verifies CUDA is visible to torch,
# and launches the full 40-model SRNN ensemble training for Henon-Heiles.
#
# Usage (from within this folder):
#   ./run_cluster_job.sh                      # defaults: workers=1, ensemble=40
#   WORKERS=4 ENSEMBLE=40 ./run_cluster_job.sh
#   VENV_DIR=asrnns-venv ./run_cluster_job.sh  # reuse an existing venv by name
#
# See RUN_INSTRUCTIONS.md for the full walkthrough, including the SLURM
# submission alternative (submit_slurm.sbatch).
set -euo pipefail

WORKERS="${WORKERS:-1}"
ENSEMBLE="${ENSEMBLE:-40}"
OUT_DIR="${OUT_DIR:-./Baseline_SRNN}"
VENV_DIR="${VENV_DIR:-venv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

if ! python -c "import torch" 2>/dev/null; then
    echo "Installing torch (default PyPI wheel, CUDA-enabled on Linux with an NVIDIA driver present)..."
    pip install torch
else
    echo "torch already installed, skipping (delete $VENV_DIR/ to force a clean reinstall)."
fi

pip install -r requirements.txt

echo "--- CUDA check ---"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
else:
    print("WARNING: no CUDA device visible to torch -- training will fall back to CPU.")
    print("If you expected a GPU here, see the troubleshooting note in RUN_INSTRUCTIONS.md")
    print("about installing a CUDA-matched torch wheel.")
PY
echo "------------------"

echo "Launching training: workers=$WORKERS ensemble=$ENSEMBLE out_dir=$OUT_DIR"
python train_srnn_ensemble.py --workers "$WORKERS" --ensemble "$ENSEMBLE" --out-dir "$OUT_DIR"
