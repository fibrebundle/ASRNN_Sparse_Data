#!/usr/bin/env bash
#SBATCH --job-name=morse_asrnn_verified
#SBATCH --output=morse_asrnn_verified_%j.out
#SBATCH --error=morse_asrnn_verified_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# NOTE: the directives above (partition, gres, time, mem) are placeholders --
# edit them to match your cluster's actual queue names and limits before
# submitting (`sbatch submit_slurm.sbatch`). Add `#SBATCH --partition=...`
# if your cluster requires an explicit partition/queue name.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

WORKERS="${WORKERS:-1}"
ENSEMBLE="${ENSEMBLE:-10}"
OUT_DIR="${OUT_DIR:-./Baseline_ASRNN_Verified}"

if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
python -c "import torch" 2>/dev/null || pip install torch
pip install -r requirements.txt

python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
PY

python train_asrnn_ensemble.py --workers "$WORKERS" --ensemble "$ENSEMBLE" --out-dir "$OUT_DIR"
