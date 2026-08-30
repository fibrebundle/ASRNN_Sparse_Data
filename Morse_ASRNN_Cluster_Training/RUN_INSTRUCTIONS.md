# Running the Morse ASRNN verification ensemble on the cluster

This folder is self-contained. It retrains ASRNN from scratch for the Morse
system using the paper's confirmed data protocol (800 trajectory segments per
training alpha value), across 9 noise conditions with a 10-model ensemble
per condition (90 models total) -- to check whether the pre-existing ASRNN
predictions in `Correlated_noise_new/Morse/` might have been trained on less
data than specified, which Baseline A's finding (SRNN beating those ASRNN
predictions at every noise level for Morse) made us suspicious of.

## What's in this folder

- `helper.py` -- model, integrator, and data-generation code (unmodified copy
  of the local `Morse_Code/helper.py` pipeline).
- `train_asrnn_ensemble.py` -- the training script. Each model is trained
  jointly across all 4 training alpha values {0.5, 1.0, 2.0, 4.0}, exactly as
  Section V.B describes -- this is the real (lambda-aware) `Hamiltonian_MLP_Network`,
  not the fixed-lambda SRNN from the Baseline A folders. Auto-detects CUDA
  and round-robins across all visible GPUs; falls back to CPU only if none is
  visible. Resumable and fault-tolerant, same as the Henon-Heiles cluster
  script (`Henon_Heiles_Cluster_Training/train_srnn_ensemble.py`) -- see that
  folder's RUN_INSTRUCTIONS.md if you hit the `fork`/CUDA multiprocessing
  error; the same fix (spawn context) is already applied here.
- `requirements.txt`, `run_cluster_job.sh`, `submit_slurm.sbatch` -- same
  pattern as the Henon-Heiles cluster folder.

## Steps

Same as `Henon_Heiles_Cluster_Training/RUN_INSTRUCTIONS.md`:

1. `git clone <repo>` and `cd Morse_ASRNN_Cluster_Training`
2. `nvidia-smi` to check GPU/CUDA version
3. `WORKERS=1 ENSEMBLE=10 ./run_cluster_job.sh` (or `sbatch submit_slurm.sbatch`
   after editing its `#SBATCH` directives)
4. If torch doesn't see the GPU, or you hit the fork/CUDA error, see the
   troubleshooting section in `Henon_Heiles_Cluster_Training/RUN_INSTRUCTIONS.md`
   -- both apply identically here.

## How long will this take

Each model is ~4x the data of a single-point SRNN baseline model (it pools
all 4 training alpha values), so expect roughly 4x the per-model time of the
Henon-Heiles/Morse SRNN baselines on the same hardware -- on this project's
CPU-only laptop that benchmarked at ~21 min/model; on a real GPU it should be
dramatically faster (the paper reports "a few minutes per model" for the
full ASRNN training on an RTX 4070 Ti Super, which is comparable in scope).
90 models total. Resumable if the job is preempted or the time limit is hit.

## Bringing the models back

`Baseline_ASRNN_Verified/nsr<nsr>_tau<tau>/asrnn_..._seed<NN>_model.pt` (+
matching `_losses.npz`) per noise condition and seed. Copy the whole
`Baseline_ASRNN_Verified/` folder back to the local `Morse_Code/` directory
(same relative name) so the local analysis pipeline can evaluate each model
at both headline points (alpha=2.0 seen, alpha=1.5 unseen) against the
existing SRNN baseline and the original `Correlated_noise_new` ASRNN
predictions.
