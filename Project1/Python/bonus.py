#!/usr/bin/env python3

import os
import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from simulate import runsim
from exercise2_3 import get_animal_data, exercise2_3
from cmc_controllers.metrics import (
    compute_mechanical_frequency_amplitude_fft,
    compute_neural_phase_lags,
)

BASE_PATH = "logs/exercise2_3_bonus_gd/"
CSV_LOG_PATH = os.path.join(BASE_PATH, "optimization_log.csv")
ANIMAL_DATA_PATH = "cmc_project_pack/models/a2sw5_cycle_smoothed.csv"

N_JOINT = 8
N_WORKERS = 4
GD_LR = 0.05
GD_N_ITERS = 50
FINITEDIFF_STEP = 0.05

BOUNDS = {
    "drive":       (1.5, 5.0),
    "pl":          (0.1, 1.2),
    "offset_freq": (0.1, 3.0),
    "g_freq":      (0.01, 2.0),
}

loss_history = []
params_history = []


def get_targets():
    f_animal, amp_animal, _, ipl_mean, _, _ = get_animal_data(ANIMAL_DATA_PATH)
    return {
        "freq": float(np.mean(f_animal)),
        "amp":  float(np.mean(amp_animal)),
        "ipl":  float(ipl_mean),
    }


def make_controller(drive, pl, offset_freq, g_freq):
    return {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left':  drive,
            'drive_right': drive,
            'd_low':  1.0,
            'd_high': 5.0,
            'a_rate':      np.ones(8) * 3.0,
            'offset_freq': np.ones(8) * offset_freq,
            'offset_amp':  np.ones(8) * 0.5,
            'G_freq':      np.ones(8) * g_freq,
            'G_amp':       np.ones(8) * 0.25,
            'PL':          np.ones(7) * pl,
            'coupling_weights_rostral': 5.0,
            'coupling_weights_caudal':  5.0,
            'coupling_weights_contra':  10.0,
            'init_phase':  np.zeros(16),
        }
    }


def _sim_worker(drive, pl, offset_freq, g_freq, base_path, hdf5_name):
    controller = make_controller(float(drive), float(pl), float(offset_freq), float(g_freq))
    runsim(
        controller=controller,
        base_path=base_path,
        headless=True,
        simulation_time=8.0,
        hdf5_name=hdf5_name,
    )
    hdf5_path = os.path.join(base_path, hdf5_name)
    with h5py.File(hdf5_path, "r") as f:
        t = f["times"][:]
        j = f["FARMSLISTanimats"]["0"]["sensors"]["joints"]["array"][:, :N_JOINT, 0]

    cut = len(t) // 2
    t_ss, j_ss = t[cut:], j[cut:]
    freqs, amps = compute_mechanical_frequency_amplitude_fft(t_ss, j_ss)
    f_sim = float(np.mean(freqs))
    a_sim = float(np.mean(amps))
    _, ipl_sim = compute_neural_phase_lags(
        t_ss, j_ss, freqs, [[i, i + 1] for i in range(N_JOINT - 1)]
    )
    return f_sim, a_sim, float(ipl_sim)


def _loss(f_sim, a_sim, ipl_sim, targets):
    err_f = ((f_sim - targets["freq"]) / targets["freq"]) ** 2
    err_a = ((a_sim - targets["amp"])  / targets["amp"])  ** 2
    err_p = ((ipl_sim - targets["ipl"]) / targets["ipl"]) ** 2
    return err_f + err_a + err_p, err_f, err_a, err_p


def _run_batch(batch, targets):
    """Run one batch of 4 simulations in parallel and return {name: loss}."""
    losses = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(_sim_worker, d, p, of, gf, BASE_PATH, name): name
            for d, p, of, gf, name in batch
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                total, *_ = _loss(*future.result(), targets)
                losses[name] = float(total)
            except Exception as e:
                print(f"  Worker '{name}' failed: {e}")
                losses[name] = 10.0
    return losses


def batch_gradient_step(drive, pl, offset_freq, g_freq, targets, h):
    """
    2 sequential batches of 4 workers (central finite differences).

    Batch 1 — gradient for drive and pl:
        fd_drive_p/m, fd_pl_p/m

    Batch 2 — gradient for offset_freq and G_freq:
        fd_ofreq_p/m, fd_gfreq_p/m
    """
    batch1 = [
        (drive + h, pl,     offset_freq,     g_freq,     "fd_drive_p.hdf5"),
        (drive - h, pl,     offset_freq,     g_freq,     "fd_drive_m.hdf5"),
        (drive,     pl + h, offset_freq,     g_freq,     "fd_pl_p.hdf5"),
        (drive,     pl - h, offset_freq,     g_freq,     "fd_pl_m.hdf5"),
    ]
    batch2 = [
        (drive, pl, offset_freq + h, g_freq,     "fd_ofreq_p.hdf5"),
        (drive, pl, offset_freq - h, g_freq,     "fd_ofreq_m.hdf5"),
        (drive, pl, offset_freq,     g_freq + h, "fd_gfreq_p.hdf5"),
        (drive, pl, offset_freq,     g_freq - h, "fd_gfreq_m.hdf5"),
    ]

    l1 = _run_batch(batch1, targets)
    l2 = _run_batch(batch2, targets)

    grad = np.array([
        (l1["fd_drive_p.hdf5"]  - l1["fd_drive_m.hdf5"])  / (2.0 * h),
        (l1["fd_pl_p.hdf5"]     - l1["fd_pl_m.hdf5"])     / (2.0 * h),
        (l2["fd_ofreq_p.hdf5"]  - l2["fd_ofreq_m.hdf5"])  / (2.0 * h),
        (l2["fd_gfreq_p.hdf5"]  - l2["fd_gfreq_m.hdf5"])  / (2.0 * h),
    ])
    approx_loss = (sum(l1.values()) + sum(l2.values())) / 8.0

    return grad, approx_loss


def main():
    os.makedirs(BASE_PATH, exist_ok=True)

    if os.path.exists(CSV_LOG_PATH):
        os.remove(CSV_LOG_PATH)
    with open(CSV_LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow([
            "iter", "drive", "pl", "offset_freq", "g_freq",
            "grad_drive", "grad_pl", "grad_ofreq", "grad_gfreq", "approx_loss"
        ])

    targets = get_targets()
    print(
        f"Targets — freq={targets['freq']:.3f}  "
        f"amp={targets['amp']:.3f}  IPL={targets['ipl']:.3f}\n"
    )

    drive, pl, offset_freq, g_freq = 3.0, 0.5, 1.0, 0.5

    for i in range(GD_N_ITERS):
        print(
            f"\n=== GD iter {i+1}/{GD_N_ITERS} | "
            f"drive={drive:.3f}  pl={pl:.3f}  "
            f"offset_freq={offset_freq:.3f}  g_freq={g_freq:.3f} ==="
        )
        print(f"  Batch 1/2: drive & pl  |  Batch 2/2: offset_freq & g_freq")

        grad, approx_loss = batch_gradient_step(
            drive, pl, offset_freq, g_freq, targets, FINITEDIFF_STEP
        )

        g_norm = np.linalg.norm(grad)
        if g_norm > 1.0:
            grad = grad / g_norm

        drive_new       = float(np.clip(drive       - GD_LR * grad[0], *BOUNDS["drive"]))
        pl_new          = float(np.clip(pl          - GD_LR * grad[1], *BOUNDS["pl"]))
        offset_freq_new = float(np.clip(offset_freq - GD_LR * grad[2], *BOUNDS["offset_freq"]))
        g_freq_new      = float(np.clip(g_freq      - GD_LR * grad[3], *BOUNDS["g_freq"]))

        loss_history.append(approx_loss)
        params_history.append([drive, pl, offset_freq, g_freq])

        print(
            f"  grad=[{grad[0]:+.4f}, {grad[1]:+.4f}, "
            f"{grad[2]:+.4f}, {grad[3]:+.4f}]  "
            f"approx_loss={approx_loss:.4f}"
        )
        print(
            f"  → drive={drive_new:.3f}  pl={pl_new:.3f}  "
            f"offset_freq={offset_freq_new:.3f}  g_freq={g_freq_new:.3f}"
        )

        with open(CSV_LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([
                i + 1, drive, pl, offset_freq, g_freq,
                grad[0], grad[1], grad[2], grad[3], approx_loss
            ])

        drive, pl, offset_freq, g_freq = drive_new, pl_new, offset_freq_new, g_freq_new

    print("\n==============================")
    print("GRADIENT DESCENT FINISHED")
    print(f"drive={drive:.4f}  pl={pl:.4f}  offset_freq={offset_freq:.4f}  g_freq={g_freq:.4f}")
    print("==============================")

    print("\nRunning final simulation with best controller…")
    runsim(
        controller=make_controller(drive, pl, offset_freq, g_freq),
        base_path=BASE_PATH,
        headless=True,
        simulation_time=8.0,
        hdf5_name="simulation.hdf5",
    )
    print(f"Saved to {os.path.join(BASE_PATH, 'simulation.hdf5')}")

    evals = np.arange(1, len(loss_history) + 1)
    ph = np.array(params_history)

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(evals, loss_history, 'k-d')
    axes[0].set_title("Approx Loss per GD Iteration")
    axes[0].set_xlabel("GD Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid()

    axes[1].plot(evals, ph[:, 0], 'b-o',  label='drive')
    axes[1].plot(evals, ph[:, 1], 'r-s',  label='pl')
    axes[1].plot(evals, ph[:, 2], 'g-^',  label='offset_freq')
    axes[1].plot(evals, ph[:, 3], 'm-d',  label='g_freq')
    axes[1].set_title("Parameter Evolution")
    axes[1].set_xlabel("GD Iteration")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, "gd_diagnostics.png"), dpi=150)
    plt.show()
    exercise2_3(os.path.join(BASE_PATH, "simulation.hdf5"))


if __name__ == "__main__":
    main()
