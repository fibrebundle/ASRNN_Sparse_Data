"""
Cluster training script -- Baseline A, full 40-model ensemble, fixed-lambda
SRNN for the Henon-Heiles system (review response).

Trains one SRNN (no parameter-adaptability, V_theta2(q) only) independently at
each of the paper's two headline parameter pairs -- (0.4, 0.6) [seen for
ASRNN] and (0.5, 0.7) [unseen for ASRNN] -- under the identical sparse/noisy
data protocol used for ASRNN (800 trajectory segments, N=2 obs/window,
T_window=15, dt=0.1), across 9 noise conditions (no noise; NSR in {5%,10%} x
tau in {dt/5, dt, 5dt, 25dt}), with a 40-model ensemble per condition to match
the paper's ASRNN ensemble size (2 x 9 x 40 = 720 models total).

This is a standalone, self-contained copy of the local Baseline_SRNN pipeline
(Henon_Heiles_Code/{helper,srnn_helper,baseline_srnn_train}.py), adapted to:
  - auto-detect and use CUDA (round-robins across all visible GPUs if more
    than one is available; falls back to CPU only if no GPU is visible)
  - fail gracefully per-config (a crashed config is logged and skipped rather
    than killing the whole job -- important for a long unattended run)
  - resume from wherever it left off: any (point, noise, seed) whose model +
    losses file already exist on disk is skipped, so a preempted/restarted
    job (or a folder seeded with the 10 models already trained locally) picks
    up cleanly without redoing finished work.

See RUN_INSTRUCTIONS.md in this folder for exact commands to run this on the
cluster (venv setup, requirements, and job submission).
"""
import os
import sys
import time
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch.optim import LBFGS
from tqdm import tqdm

from helper import F, generate_data, train_test_split, residuals
from srnn_helper import SRNN_Network, SRNNVerletIntegrator

# --- fixed data-generation protocol, matching Section IV / the ASRNN runs ---
T_WINDOW = 15
SPLITS = 100          # in_conds * SPLITS = 800 segments/parameter, matching ASRNN's
IN_CONDS = 8          # per-parameter data volume (Section V.A: 800 segments/pair)
L_TOTAL = SPLITS + T_WINDOW - 1
N_SAMPLES = 2
DT = 0.1
COARSEN = 100
VAL_SIZE = 0.25
STEPS = 500           # LBFGS epochs, matching ASRNN training (Section V.A)

# --- architecture, matching ASRNN's Henon-Heiles config (Section V.A) minus lambda ---
KIN_HIDDEN, KIN_LAYERS = 30, 3
POT_HIDDEN, POT_LAYERS = 30, 3

POINTS = [(0.4, 0.6), (0.5, 0.7)]          # seen / unseen headline pairs (Fig. 4/5)
NOISE_CONDITIONS = [(0.0, 0.0)] + [
    (nsr, tau) for nsr in (0.05, 0.10) for tau in (0.02, 0.1, 0.5, 2.5)
]                                            # 1 clean + 8 noisy = 9 conditions
N_ENSEMBLE_DEFAULT = 40
BASE_OUT_DEFAULT = "./Baseline_SRNN"


def resolve_device(device_str):
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def train_one(alpha1, alpha2, nsr_sd, tau_c, seed, out_dir, device_str):
    stem = f"srnn_a1_{alpha1:.2f}_a2_{alpha2:.2f}_nsr{nsr_sd:.2f}_tau{tau_c:.3f}_seed{seed:02d}"
    model_out = os.path.join(out_dir, f"{stem}_model.pt")
    losses_out = os.path.join(out_dir, f"{stem}_losses.npz")
    if os.path.exists(model_out) and os.path.exists(losses_out):
        return stem, "skipped", None, 0.0  # resumable: skip work already done

    t_start = time.time()
    try:
        torch.set_num_threads(1)
        device = torch.device(device_str)
        dtype = torch.float32

        torch.manual_seed(seed)

        alphas = torch.tensor([[alpha1, alpha2]], dtype=dtype, device=device)
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

        model = SRNN_Network(
            kin_hidden_dim=KIN_HIDDEN, kin_n_hidden=KIN_LAYERS,
            pot_hidden_dim=POT_HIDDEN, pot_n_hidden=POT_LAYERS,
            device=device,
        ).to(device)
        integrator = SRNNVerletIntegrator(model=model, dt=DT)

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
    for (a1, a2) in POINTS:
        for (nsr, tau) in NOISE_CONDITIONS:
            tau_tag = f"{tau:.3f}" if nsr > 0 else "none"
            out_dir = os.path.join(base_out, f"point_{a1:.1f}_{a2:.1f}", f"nsr{nsr:.2f}_tau{tau_tag}")
            for seed in range(n_ensemble):
                device_str = devices[i % len(devices)]
                configs.append(dict(alpha1=a1, alpha2=a2, nsr_sd=nsr, tau_c=tau if nsr > 0 else 1.0,
                                     seed=seed, out_dir=out_dir, device_str=device_str))
                i += 1
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1,
                        help="number of concurrent training processes")
    parser.add_argument("--ensemble", type=int, default=N_ENSEMBLE_DEFAULT,
                        help="models per (point, noise) condition")
    parser.add_argument("--out-dir", type=str, default=BASE_OUT_DEFAULT)
    parser.add_argument("--device", type=str, default=None,
                        help="force a single device (e.g. 'cpu', 'cuda:0'); "
                             "default: auto-detect and round-robin all visible GPUs")
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
    train_times = []  # only successful trains -- skips are near-instant and would skew the average
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(train_one, **cfg) for cfg in configs]
        n_trained, n_skipped, n_failed = 0, 0, 0
        pbar = tqdm(as_completed(futures), total=len(futures), desc="HH SRNN ensemble (cluster)")
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
