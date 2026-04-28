#!/usr/bin/env python3

import os
import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from simulate import runsim
from exercise2_3 import get_animal_data,exercise2_3
from cmc_controllers.metrics import (
    compute_mechanical_frequency_amplitude_fft,
    compute_neural_phase_lags,
)

# ============================================================================
# CONFIG
# ============================================================================

BASE_PATH = "logs/exercise2_3_bonus_gd/"
CSV_LOG_PATH = os.path.join(BASE_PATH, "optimization_log.csv")
ANIMAL_DATA_PATH = "cmc_project_pack/models/a2sw5_cycle_smoothed.csv"

N_JOINT = 8
N_WORKERS = 4
GD_LR = 0.05
GD_N_ITERS = 50
FINITEDIFF_STEP = 0.05

loss_history = []
params_history = []


# ============================================================================
# TARGETS
# ============================================================================

def get_targets():
    f_animal, amp_animal, _, ipl_mean, _, _ = get_animal_data(ANIMAL_DATA_PATH)
    return {
        "freq": float(np.mean(f_animal)),
        "amp":  float(np.mean(amp_animal)),
        "ipl":  float(ipl_mean),
    }


# ============================================================================
# CONTROLLER
# ============================================================================

def make_controller(drive, pl):
    return {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left':  drive,
            'drive_right': drive,
            'd_low':  1.0,
            'd_high': 5.0,
            'a_rate':      np.ones(8) * 3.0,
            'offset_freq': np.ones(8) * 1.0,
            'offset_amp':  np.ones(8) * 0.5,
            'G_freq':      np.ones(8) * 0.5,
            'G_amp':       np.ones(8) * 0.25,
            'PL':          np.ones(7) * pl,
            'coupling_weights_rostral': 5.0,
            'coupling_weights_caudal':  5.0,
            'coupling_weights_contra':  10.0,
            'init_phase':  np.zeros(16),
        }
    }


# ============================================================================
# SIMULATION WORKER  (module-level so it is picklable on Windows spawn)
# ============================================================================

def _sim_worker(drive, pl, base_path, hdf5_name):
    """Run one simulation and return (f_sim, a_sim, ipl_sim). Executed in a subprocess."""
    controller = make_controller(float(drive), float(pl))
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


# ============================================================================
# BATCH GRADIENT STEP  — 4 workers, central finite differences
# ============================================================================

def batch_gradient_step(drive, pl, targets, step_size):
    """
    Launch 4 parallel simulations to estimate ∂L/∂drive and ∂L/∂pl via
    central differences, then return (gradient, approx_loss).

    Batch layout (nworkers = 4):
        fd_drive_p  →  (drive+h, pl)
        fd_drive_m  →  (drive-h, pl)
        fd_pl_p     →  (drive,   pl+h)
        fd_pl_m     →  (drive,   pl-h)
    """
    h = step_size
    batch = [
        (drive + h, pl,     "fd_drive_p.hdf5"),
        (drive - h, pl,     "fd_drive_m.hdf5"),
        (drive,     pl + h, "fd_pl_p.hdf5"),
        (drive,     pl - h, "fd_pl_m.hdf5"),
    ]

    losses = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(_sim_worker, d, p, BASE_PATH, name): name
            for d, p, name in batch
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                metrics = future.result()
                total, *_ = _loss(*metrics, targets)
                losses[name] = float(total)
            except Exception as e:
                print(f"  Worker '{name}' failed: {e}")
                losses[name] = 10.0

    l_dp = losses["fd_drive_p.hdf5"]
    l_dm = losses["fd_drive_m.hdf5"]
    l_pp = losses["fd_pl_p.hdf5"]
    l_pm = losses["fd_pl_m.hdf5"]

    grad_drive = (l_dp - l_dm) / (2.0 * h)
    grad_pl    = (l_pp - l_pm) / (2.0 * h)
    approx_loss = (l_dp + l_dm + l_pp + l_pm) / 4.0

    return np.array([grad_drive, grad_pl]), approx_loss


# ============================================================================
# MAIN
# ============================================================================

def main():
    
    os.makedirs(BASE_PATH, exist_ok=True)

    if os.path.exists(CSV_LOG_PATH):
        os.remove(CSV_LOG_PATH)
    with open(CSV_LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(
            ["iter", "drive", "pl", "grad_drive", "grad_pl", "approx_loss"]
        )

    targets = get_targets()
    print(
        f"Targets — freq={targets['freq']:.3f}  "
        f"amp={targets['amp']:.3f}  IPL={targets['ipl']:.3f}\n"
    )

    drive, pl = 3.0, 0.5

    for i in range(GD_N_ITERS):
        print(f"\n=== GD iter {i + 1}/{GD_N_ITERS}  drive={drive:.4f}  pl={pl:.4f} ===")
        print(f"  Launching {N_WORKERS} parallel simulations (central finite differences)…")

        grad, approx_loss = batch_gradient_step(drive, pl, targets, FINITEDIFF_STEP)

        # gradient clipping to avoid overshooting
        g_norm = np.linalg.norm(grad)
        if g_norm > 1.0:
            grad = grad / g_norm

        drive_new = float(np.clip(drive - GD_LR * grad[0], 1.5, 5.0))
        pl_new    = float(np.clip(pl    - GD_LR * grad[1], 0.1, 1.2))

        loss_history.append(approx_loss)
        params_history.append([drive, pl])

        print(
            f"  grad=[{grad[0]:+.4f}, {grad[1]:+.4f}]  "
            f"approx_loss={approx_loss:.4f}"
        )
        print(f"  → drive={drive_new:.4f}  pl={pl_new:.4f}")

        with open(CSV_LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow(
                [i + 1, drive, pl, grad[0], grad[1], approx_loss]
            )

        drive, pl = drive_new, pl_new

    print("\n==============================")
    print("GRADIENT DESCENT FINISHED")
    print(f"Final params: drive={drive:.4f}  pl={pl:.4f}")
    print("==============================")

    # ========================================================================
    # FINAL SIMULATION with best controller
    # ========================================================================

    print("\nRunning final simulation with best controller…")
    best_controller = make_controller(drive, pl)
    runsim(
        controller=best_controller,
        base_path=BASE_PATH,
        headless=True,
        simulation_time=8.0,
        hdf5_name="simulation.hdf5",
    )
    print(f"Saved to {os.path.join(BASE_PATH, 'simulation.hdf5')}")

    # ========================================================================
    # PLOTS
    # ========================================================================

    evals = np.arange(1, len(loss_history) + 1)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(evals, loss_history, 'k-d')
    axes[0].set_title("Approx Loss per GD Iteration")
    axes[0].set_xlabel("GD Iteration")
    axes[0].set_ylabel("Loss (mean of 4 FD evals)")
    axes[0].grid()

    ax2b = axes[1].twinx()
    line1, = axes[1].plot(evals, [p[0] for p in params_history], 'b-o', label='Drive')
    line2, = ax2b.plot(evals, [p[1] for p in params_history], 'r-s', label='PL')
    axes[1].set_title("Parameter Evolution")
    axes[1].set_xlabel("GD Iteration")
    axes[1].set_ylabel("Drive", color='b')
    ax2b.set_ylabel("PL", color='r')
    axes[1].legend([line1, line2], ['Drive', 'PL'], loc='upper left')
    axes[1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, "gd_diagnostics.png"), dpi=150)
    plt.show()
    exercise2_3(os.path.join(BASE_PATH, "simulation.hdf5"))

if __name__ == "__main__":
    main()
