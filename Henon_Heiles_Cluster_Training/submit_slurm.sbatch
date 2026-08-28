#!/usr/bin/env bash
#SBATCH --job-name=hh_srnn_ensemble
#SBATCH --output=hh_srnn_ensemble_%j.out
#SBATCH --error=hh_srnn_ensemble_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
# NOTE: the directives above (partition, gres, time, mem) are placeholders --
# edit them to match your cluster's actual queue names and limits before
# submitting (`sbatch submit_slurm.sbatch`). Add `#SBATCH --partition=...`
# if your cluster requires an explicit partition/queue name.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# If your cluster uses environment modules, load them here, e.g.:
#   module load cuda/12.1
#   module load python/3.10

WORKERS="${WORKERS:-1}"      # set to the number of GPUs requested above if using more than one
ENSEMBLE="${ENSEMBLE:-40}"
OUT_DIR="${OUT_DIR:-./Baseline_SRNN}"

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

python train_srnn_ensemble.py --workers "$WORKERS" --ensemble "$ENSEMBLE" --out-dir "$OUT_DIR"
