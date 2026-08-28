# Running the Henon-Heiles SRNN ensemble on the cluster

This folder is self-contained: it does not depend on anything else in the
repository. It trains a fixed-lambda SRNN baseline (Baseline A, review
response) for the Henon-Heiles system at the two headline parameter pairs --
(0.4, 0.6) and (0.5, 0.7) -- across 9 noise conditions, with a 40-model
ensemble per condition (2 x 9 x 40 = **720 models total**), matching the
paper's ASRNN ensemble size.

## What's in this folder

- `helper.py`, `srnn_helper.py` -- model, integrator, and data-generation code
  (unmodified copies of the local `Henon_Heiles_Code/` pipeline).
- `train_srnn_ensemble.py` -- the training script. Auto-detects CUDA and
  round-robins across all visible GPUs; falls back to CPU only if none is
  visible. Resumable (skips any model+losses file pair that already exists)
  and fault-tolerant (a crashed config is logged to `failures.log` and
  skipped, rather than killing the whole job).
- `requirements.txt` -- non-torch dependencies (numpy, scikit-learn, tqdm).
- `run_cluster_job.sh` -- sets up a venv, installs everything, sanity-checks
  that torch sees a GPU, then launches training. Use this for an interactive
  session or inside a `tmux`/`screen`.
- `submit_slurm.sbatch` -- SLURM job template calling the same setup +
  training steps. **Edit the `#SBATCH` directives before submitting** --
  partition name, GPU count, time limit, and memory are placeholders and
  almost certainly need to match your specific cluster.

## 1. Get the code onto the cluster

```bash
git clone <the repository URL>
cd <repository>/Henon_Heiles_Cluster_Training
```

(This assumes this folder has been pushed to the repo, as planned.)

## 2. Check what GPU(s) and CUDA version you have

```bash
nvidia-smi
```

Note the CUDA version shown in the top-right of the output -- you may need it
in step 3 if the default torch install doesn't pick up the GPU.

## 3. Set up the environment and run

**Option A -- plain venv, interactive/tmux session:**

```bash
chmod +x run_cluster_job.sh
WORKERS=1 ENSEMBLE=40 ./run_cluster_job.sh
```

Run this inside `tmux` or `screen` (or with `nohup ... &`) since the full run
will take a long time -- see "How long will this take" below.

`WORKERS` controls how many training processes run concurrently. If you have
more than one GPU visible, set `WORKERS` to (a small multiple of) the number
of GPUs so configs get distributed across them -- e.g. `WORKERS=4` on a
4-GPU node. On a single GPU, multiple concurrent processes can still help
since each individual model is small, but test with `WORKERS=2` or `3` first
and watch `nvidia-smi` for memory/utilization before pushing higher.

**Option B -- SLURM:**

```bash
# edit the #SBATCH lines in submit_slurm.sbatch first (partition, gres, time, mem)
WORKERS=1 ENSEMBLE=40 sbatch submit_slurm.sbatch
```

## 4. Troubleshooting

**torch doesn't see the GPU:** `run_cluster_job.sh` prints a CUDA check after
installing torch. If it says `cuda available: False` on a machine that
clearly has a GPU (per `nvidia-smi`), the plain `pip install torch` likely
resolved a CPU-only or mismatched-CUDA wheel. Fix by installing a
CUDA-matched build instead -- for this cluster's CUDA 12.6:

```bash
source venv/bin/activate
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

(If `cu126` isn't available for some reason, `cu124` or `cu121` wheels are
typically forward-compatible with a 12.6 driver -- see
https://pytorch.org/get-started/locally/ for the current mapping.) Then
re-run `python train_srnn_ensemble.py ...` directly (no need to redo the venv
setup).

**`RuntimeError: Cannot re-initialize CUDA in forked subprocess`:** this was
a real bug in an earlier version of `train_srnn_ensemble.py` -- the main
process calls `torch.cuda.is_available()` to auto-detect GPUs, which
initializes a CUDA context, and the default `fork` start method on Linux then
handed every worker process a copy of that already-initialized context, which
CUDA doesn't support. Fixed by launching the worker pool with an explicit
`multiprocessing.get_context("spawn")`. If you're seeing this error, make
sure you have the current version of this file (`git pull`) rather than
patching around it.

## 5. Resuming an already-completed local run (optional, saves ~25% of the work)

10 models per condition have already been trained locally on this project
(seeds 0-9). If you copy that `Baseline_SRNN/` folder into this directory
before running (matching the `--out-dir` path, default `./Baseline_SRNN`),
the script will detect those 180 already-done (point, noise, seed)
combinations and skip them, training only the remaining 30 seeds per
condition (540 models instead of 720).

## 6. How long will this take

Each model trains for a fixed 500 LBFGS epochs; the paper reports "a few
minutes per model" on an RTX 4070 Ti Super. With `WORKERS=1` on a single
modern GPU, budget roughly 720 x a few minutes (could be 24-48+ hours
depending on the actual GPU); increasing `WORKERS` to use more GPUs or more
concurrent jobs on one GPU should reduce this roughly proportionally as long
as the GPU isn't saturated. The job is resumable, so it's safe to let it run
across multiple queue time-limits/preemptions -- just resubmit and it will
pick up where it left off.

## 7. Bringing the models back

Once done, `Baseline_SRNN/` will contain, per (point, noise condition, seed):

```
Baseline_SRNN/point_<a1>_<a2>/nsr<nsr>_tau<tau>/srnn_..._seed<NN>_model.pt
Baseline_SRNN/point_<a1>_<a2>/nsr<nsr>_tau<tau>/srnn_..._seed<NN>_losses.npz
```

Copy the whole `Baseline_SRNN/` directory back to the local `Henon_Heiles_Code/`
folder (same relative path/name), overwriting the existing 10-model version --
the local analysis scripts (`baseline_srnn_predictions.py`,
`plot_baseline_energy_error.py`) already point at that path and will pick up
all 40 models automatically once you also delete or refresh the cached
`Baseline_SRNN_Predictions/` outputs (the prediction cache keys on model file
names, so stale cached predictions for the old 10-model runs will simply be
joined by new ones for the additional seeds -- run
`baseline_srnn_predictions.py` again afterward to extend the cache and rebuild
the aggregate `noise_analysis` arrays over the full 40-model ensemble).

If `failures.log` exists in the output folder, check it -- any failed config
listed there did not produce a model file and should be re-run (the resume
logic means simply re-running the same command will retry only those).
