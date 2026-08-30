"""
Cluster training script -- verification run for the Morse ASRNN ensemble
(review response follow-up).

Baseline A's fixed-lambda SRNN comparison found that, for the Morse system, a
plain SRNN outperforms the pre-existing ASRNN predictions
(Correlated_noise_new/Morse/...) at every noise level tested -- the opposite
of what happened for Henon-Heiles. Since the training config/data actually
used to produce those pre-existing ASRNN predictions can't be verified after
the fact, this script retrains ASRNN from scratch using the CONFIRMED-correct
data protocol from the paper (800 trajectory segments per training alpha
value, dt=0.1, N=2 obs/window, T_window=15), to check whether the original
models might simply have been trained on less data than specified.

Each ASRNN model is trained jointly across all 4 training alpha values --
{0.5, 1.0, 2.0, 4.0} -- exactly as Section V.B describes (unlike the SRNN
baseline, which trains one model per single alpha value), using the
unmodified Hamiltonian_MLP_Network + VerletIntegrator from helper.py. A
10-model ensemble is trained per noise condition (9 conditions x 10 = 90
models total), matching the SRNN baseline's ensemble size.

Auto-detects and uses CUDA (round-robins across all visible GPUs); resumable
(skips any model+losses file pair that already exists) and fault-tolerant (a
crashed config is logged to failures.log and skipped, not fatal).

See RUN_INSTRUCTIONS.md in this folder for exact commands to run this on the
cluster (venv setup, requirements, and job submission).
"""
import os

# Must be set before torch is imported anywhere in this process. Irrelevant
# on the actual CUDA cluster this script targets, but if it's ever run (or
# validated) on CPU -- e.g. on macOS -- torch's linear algebra there routes
# through Apple's Accelerate/vecLib BLAS, which manages its own thread pool
# independently of torch.set_num_threads() and reads its thread count from
# this env var at load time. Left unset, each "single-threaded" worker
# oversubscribes the machine by as many threads as there are cores (see
# Henon_Heiles_Code/baseline_srnn_train.py for the full writeup).
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import argparse
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch.optim import LBFGS
from tqdm import tqdm

from helper import F, generate_data, train_test_split, residuals, Hamiltonian_MLP_Network, VerletIntegrator

# --- data-generation protocol, matching Section IV / Section V.B exactly ---
T_WINDOW = 15
SPLITS = 20
IN_CONDS = 4          # 10*IN_CONDS*SPLITS = 800 segments/parameter
L_TOTAL = SPLITS + T_WINDOW - 1
N_SAMPLES = 2
DT = 0.1
COARSEN = 100
VAL_SIZE = 0.25
STEPS = 500

# --- architecture, matching ASRNN's Morse config (Section V.B) exactly ---
KIN_HIDDEN, KIN_LAYERS = 50, 2
POT_HIDDEN, POT_LAYERS = 50, 2

ALPHAS_TRAIN = [0.5, 1.0, 2.0, 4.0]
NOISE_CONDITIONS = [(0.0, 0.0)] + [
    (nsr, tau) for nsr in (0.05, 0.10) for tau in (0.02, 0.1, 0.5, 2.5)
]
N_ENSEMBLE_DEFAULT = 10
BASE_OUT_DEFAULT = "./Baseline_ASRNN_Verified"


def train_one(nsr_sd, tau_c, seed, out_dir, device_str):
    stem = f"asrnn_nsr{nsr_sd:.2f}_tau{tau_c:.3f}_seed{seed:02d}"
    model_out = os.path.join(out_dir, f"{stem}_model.pt")
    losses_out = os.path.join(out_dir, f"{stem}_losses.npz")
    if os.path.exists(model_out) and os.path.exists(losses_out):
        return stem, "skipped", None, 0.0

    t_start = time.time()
    try:
        torch.set_num_threads(1)
        device = torch.device(device_str)
        dtype = torch.float32

        torch.manual_seed(seed)
        alphas = torch.tensor(ALPHAS_TRAIN, dtype=dtype, device=device)
        nsr_var = float(nsr_sd ** 2)
        apply_ou_noise = nsr_sd > 0.0
        theta = float(1.0 / tau_c) if apply_ou_noise else 0.0

        trajectories, params, indices = generate_data(
            alphas, F, L_TOTAL, T_WINDOW, N_SAMPLES,
            dt=DT, in_conds=IN_CONDS, coarsening_factor=COARSEN,
            nsr=nsr_var, theta=theta, burn_in_var=0,
            apply_ou_noise=apply_ou_noise, device=device, dtype=dtype,
            seed_ou=seed,
        )
        (train_traj, train_p, train_idx), (val_traj, val_p, val_idx) = train_test_split(
            trajectories, params, indices, val_size=VAL_SIZE
        )

        model = Hamiltonian_MLP_Network(
            kin_hidden_dim=KIN_HIDDEN, kin_n_hidden=KIN_LAYERS,
            pot_hidden_dim=POT_HIDDEN, pot_n_hidden=POT_LAYERS,
            device=device,
        ).to(device)
        integrator = VerletIntegrator(model=model, dt=DT)

        optimizer = LBFGS(
            model.parameters(), lr=1.0, history_size=10, line_search_fn="strong_wolfe",
            tolerance_grad=1e-32, tolerance_change=1e-32,
        )

        def closure():
            optimizer.zero_grad()
            loss = residuals(train_traj, train_p, train_idx, T_WINDOW, integrator)
            loss.backward()
            return loss

        training_losses, validation_losses = [], []
        for _ in range(STEPS):
            model.train()
            train_loss = float(optimizer.step(closure).item())
            model.eval()
            val_loss = float(residuals(val_traj, val_p, val_idx, T_WINDOW, integrator).detach().cpu())
            training_losses.append(train_loss)
            validation_losses.append(val_loss)

        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), model_out)
        np.savez(
            losses_out,
            training_losses=np.asarray(training_losses, dtype=np.float64),
            validation_losses=np.asarray(validation_losses, dtype=np.float64),
        )
        return stem, "trained", None, time.time() - t_start
    except Exception as e:
        return stem, "failed", "".join(traceback.format_exception(type(e), e, e.__traceback__)), time.time() - t_start


def build_configs(base_out, n_ensemble, devices):
    configs = []
    i = 0
    for (nsr, tau) in NOISE_CONDITIONS:
        tau_tag = f"{tau:.3f}" if nsr > 0 else "none"
        out_dir = os.path.join(base_out, f"nsr{nsr:.2f}_tau{tau_tag}")
        for seed in range(n_ensemble):
            device_str = devices[i % len(devices)]
            configs.append(dict(nsr_sd=nsr, tau_c=tau if nsr > 0 else 1.0,
                                 seed=seed, out_dir=out_dir, device_str=device_str))
            i += 1
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ensemble", type=int, default=N_ENSEMBLE_DEFAULT)
    parser.add_argument("--out-dir", type=str, default=BASE_OUT_DEFAULT)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is not None:
        devices = [args.device]
    elif torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(n_gpus)]
        print(f"Detected {n_gpus} CUDA device(s): {devices}")
    else:
        devices = ["cpu"]
        print("No CUDA device detected -- falling back to CPU.")

    configs = build_configs(args.out_dir, args.ensemble, devices)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Total configs: {len(configs)} (ensemble={args.ensemble}, workers={args.workers})")

    failures = []
    n_remaining_at_start = len(configs)
    train_times = []
    # 'spawn' is required for CUDA -- see Henon_Heiles_Cluster_Training/train_srnn_ensemble.py
    # for the full explanation of why 'fork' (the Linux default) breaks CUDA here.
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx) as ex:
        futures = [ex.submit(train_one, **cfg) for cfg in configs]
        n_trained, n_skipped, n_failed = 0, 0, 0
        pbar = tqdm(as_completed(futures), total=len(futures), desc="ASRNN verified ensemble (cluster)")
        for f in pbar:
            stem, status, err, elapsed = f.result()
            n_remaining_at_start -= 1
            if status == "trained":
                n_trained += 1
                train_times.append(elapsed)
                avg = sum(train_times) / len(train_times)
                eta_hours = avg * n_remaining_at_start / args.workers / 3600
                tqdm.write(
                    f"[{n_trained + n_skipped + n_failed}/{len(configs)}] trained {stem} "
                    f"in {elapsed:.1f}s | avg {avg:.1f}s/model (n={len(train_times)}) | "
                    f"{n_remaining_at_start} configs left | ETA ~{eta_hours:.1f}h"
                )
            elif status == "skipped":
                n_skipped += 1
            else:
                n_failed += 1
                failures.append((stem, err))
                tqdm.write(f"[FAILED] {stem}")
                print(f"[FAILED] {stem}\n{err}", file=sys.stderr)

    print(f"Done. trained={n_trained} skipped={n_skipped} failed={n_failed}")
    if failures:
        fail_log = os.path.join(args.out_dir, "failures.log")
        with open(fail_log, "w") as fh:
            for stem, err in failures:
                fh.write(f"=== {stem} ===\n{err}\n\n")
        print(f"Wrote failure details to {fail_log}")


if __name__ == "__main__":
    main()
