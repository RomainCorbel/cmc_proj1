#!/usr/bin/env python3

import os
import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from simulate import runsim
from exercise2_3 import get_animal_data, exercise2_3
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

loss_history = []
params_history = []
detailed_loss_history = []


# ============================================================================
# TARGETS
# ============================================================================

def get_targets():
    f_animal, amp_animal, _, ipl_mean, _, _ = get_animal_data(ANIMAL_DATA_PATH)

    return {
        "freq": float(np.mean(f_animal)),
        "amp": float(np.mean(amp_animal)),
        "ipl": float(ipl_mean),
    }


# ============================================================================
# CONTROLLER
# ============================================================================

def make_controller(drive, pl):
    return {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': drive,
            'drive_right': drive,
            'd_low': 1.0,
            'd_high': 5.0,
            'a_rate': np.ones(8) * 3.0,
            'offset_freq': np.ones(8) * 1.0,
            'offset_amp': np.ones(8) * 0.5,
            'G_freq': np.ones(8) * 0.5,
            'G_amp': np.ones(8) * 0.25,
            'PL': np.ones(7) * pl,
            'coupling_weights_rostral': 5.0,
            'coupling_weights_caudal': 5.0,
            'coupling_weights_contra': 10.0,
            'init_phase': np.zeros(16),
        }
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_controller(drive, pl):
    controller = make_controller(drive, pl)

    runsim(
        controller=controller,
        base_path=BASE_PATH,
        headless=True,
        simulation_time=8.0,
    )

    hdf5_path = os.path.join(BASE_PATH, "simulation.hdf5")

    with h5py.File(hdf5_path, "r") as f:
        t = f["times"][:]
        j = f["FARMSLISTanimats"]["0"]["sensors"]["joints"]["array"][:, :N_JOINT, 0]

    cut = len(t) // 2
    t_ss = t[cut:]
    j_ss = j[cut:]

    freqs, amps = compute_mechanical_frequency_amplitude_fft(t_ss, j_ss)

    f_sim = float(np.mean(freqs))
    a_sim = float(np.mean(amps))

    _, ipl_sim = compute_neural_phase_lags(
        t_ss,
        j_ss,
        freqs,
        [[i, i+1] for i in range(N_JOINT - 1)]
    )

    return f_sim, a_sim, float(ipl_sim)


# ============================================================================
# OBJECTIVE
# ============================================================================

def objective_function(x, targets):
    drive = np.clip(x[0], 1.5, 5.0)
    pl    = np.clip(x[1], 0.1, 1.2)

    try:
        f_sim, a_sim, ipl_sim = evaluate_controller(drive, pl)

        err_f = ((f_sim - targets["freq"]) / targets["freq"]) ** 2
        err_a = ((a_sim - targets["amp"]) / targets["amp"]) ** 2
        err_p = ((ipl_sim - targets["ipl"]) / targets["ipl"]) ** 2

        total_loss = err_f + err_a + err_p

    except Exception as e:
        print("Simulation failed:", e)
        total_loss = 10.0
        err_f = err_a = err_p = np.nan

    loss_history.append(total_loss)
    params_history.append([drive, pl])
    detailed_loss_history.append([err_f, err_a, err_p])

    with open(CSV_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            len(loss_history),
            drive,
            pl,
            err_f,
            err_a,
            err_p,
            total_loss
        ])

    print(
        f"Iter {len(loss_history):02d} | "
        f"drive={drive:.3f} | pl={pl:.3f} | "
        f"loss={total_loss:.4f}"
    )

    return total_loss


# ============================================================================
# MAIN
# ============================================================================

def main():

    os.makedirs(BASE_PATH, exist_ok=True)

    if os.path.exists(CSV_LOG_PATH):
        os.remove(CSV_LOG_PATH)

    targets = get_targets()
    print(f"Objectives are to match animal metrics: frequency={targets['freq']:.3f}, amplitude={targets['amp']:.3f}, and IPL={targets['ipl']:.3f} \n\n\n\n")
    res = minimize(
        objective_function,
        x0=[3.0, 0.5],
        args=(targets,),
        method='Nelder-Mead',
        options={
            'maxiter': 25,
            'xatol': 0.02,
            'fatol': 1e-3,
            'disp': True
        }
    )

    best_drive, best_pl = res.x

    print("\n==============================")
    print("OPTIMIZATION FINISHED")
    print(f"Objectives are to match animal metrics: frequency={targets['freq']:.3f}, amplitude={targets['amp']:.3f}, and IPL={targets['ipl']:.3f}")
    print("==============================")
    print(f"Best Drive : {best_drive:.4f}")
    print(f"Best PL    : {best_pl:.4f}")
    print(f"Best Loss  : {res.fun:.4f}")

    # ========================================================================
    # PLOTS
    # ========================================================================

    evals = np.arange(1, len(loss_history)+1)
    det = np.array(detailed_loss_history)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(evals, det[:, 0], 'r-o', label='Freq')
    axes[0].plot(evals, det[:, 1], 'b-s', label='Amp')
    axes[0].plot(evals, det[:, 2], 'g-^', label='IPL')
    axes[0].set_title("Loss Components")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Iterations")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(evals, loss_history, 'k-d')
    axes[1].set_title("Total Loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Iterations")
    axes[1].grid()

    ax2b = axes[2].twinx()
    line1, = axes[2].plot(evals, [p[0] for p in params_history], 'b-o', label='Drive')
    line2, = ax2b.plot(evals, [p[1] for p in params_history], 'r-s', label='PL')

    axes[2].set_title("Parameter Evolution")
    axes[2].grid()
    axes[2].set_ylabel("Drive Value", color='b') 
    ax2b.set_ylabel("PL Value", color='r')
    axes[2].set_xlabel("Iterations")

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    axes[2].legend(lines, labels, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, "optimization_diagnostics.png"), dpi=150)
    plt.show()
    # exercise2_3(sim_result = 'logs/exercise2_3_bonus_gd/simulation.hdf5')

if __name__ == "__main__":
    main()