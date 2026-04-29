#!/usr/bin/env python3

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from cmc_controllers.plot_utils import plot_results_EXO2_3
from farms_core import pylog
from cmc_controllers.metrics import (
    compute_mechanical_frequency_amplitude_fft,
    compute_neural_phase_lags,
)
from cmc_controllers.plot_utils import plot_kinematics_comparison

BASE_PATH       = 'logs/exercise2_3/'
GRID_PATH       = 'logs/exercise2_2/grid1_drive_pl/'
ANIMAL_DATA_PATH = 'cmc_project_pack/models/a2sw5_cycle_smoothed.csv'

N_JOINT    = 8
N_GRID     = 10
DRIVE_VALS = np.linspace(2.0, 4.0, N_GRID)
PL_VALS    = np.linspace(np.pi / (2 * N_JOINT), 3 * np.pi / N_JOINT, N_GRID)


# ──────────────────────────────────────────────────────────────────────────────
# Animal data
# ──────────────────────────────────────────────────────────────────────────────

def get_animal_data(path):
    """Load animal CSV and return scaled metrics + raw kinematics."""
    data         = np.genfromtxt(path, delimiter=',', skip_header=1)
    times        = data[:, 0]
    joint_angles = np.deg2rad(data[:, 1:9])

    freqs, amplitudes = compute_mechanical_frequency_amplitude_fft(times, joint_angles)
    f_animal   = freqs
    amp_animal = amplitudes
    inds_couples = [[i, i + 1] for i in range(N_JOINT - 1)]
    ipl_animal, ipl_animal_mean = compute_neural_phase_lags(
        times=times,
        smooth_signals=joint_angles,
        freqs=freqs,
        inds_couples=inds_couples,
    )

    # Dynamical scaling: frequency scales by sqrt(1/6.5) (Project Ref [4])
    freq_robot_scaled = np.sqrt(1 / 6.5) * f_animal

    return freq_robot_scaled, amp_animal, ipl_animal, ipl_animal_mean, times, joint_angles


# ──────────────────────────────────────────────────────────────────────────────
# Grid scan helpers
# ──────────────────────────────────────────────────────────────────────────────

def _robot_metrics_from_hdf5(hdf5_path):
    """Return (f, amp, ipl, times, joint_pos) from a grid simulation file."""
    with h5py.File(hdf5_path, "r") as f:
        times  = f['times'][:]
        joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    joint_pos = joints[:, :N_JOINT, 0]   # joint angles (rad)

    # Steady-state: last 50%
    cut  = len(times) // 2
    t_ss = times[cut:]
    j_ss = joint_pos[cut:]

    f_arr, a_arr = compute_mechanical_frequency_amplitude_fft(t_ss, j_ss)
    f_sim  = float(np.mean(f_arr))
    a_sim  = float(np.mean(a_arr))

    inds_couples = [[i, i + 1] for i in range(N_JOINT - 1)]
    _, ipl_sim = compute_neural_phase_lags(t_ss, j_ss, f_arr, inds_couples)

    return f_sim, a_sim, float(ipl_sim), times, joint_pos


def _grid1_hdf5_path(drive, pl):
    return GRID_PATH + f"simulation_drive{drive:0.3f}_PLarr{N_JOINT - 1}_{pl:0.3f}.hdf5"


# ──────────────────────────────────────────────────────────────────────────────
# Best imitation search
# ──────────────────────────────────────────────────────────────────────────────

def find_best_imitation(target_f, target_ipl, target_amp):
    """
    Scan all grid1_drive_pl simulations, compute (f, amp, IPL) from steady-state
    joint angles, and find the run whose kinematics best match the animal targets.

    Error = |f - f*|/f*  +  |ipl - ipl*|/|ipl*|  +  |amp - amp*|/amp*
    """
    # Storage for heatmaps
    grid_error = np.full((N_GRID, N_GRID), np.nan)
    grid_f     = np.full((N_GRID, N_GRID), np.nan)
    grid_amp   = np.full((N_GRID, N_GRID), np.nan)
    grid_ipl   = np.full((N_GRID, N_GRID), np.nan)

    best_error  = float('inf')
    best_params = None
    best_hdf5   = None

    print(f"\nTarget  →  f={target_f:.3f} Hz  |  IPL={target_ipl:.3f} rad  |  amp={target_amp:.3f} rad")
    print(f"{'Drive':>6}  {'PL':>6}  {'f_sim':>7}  {'amp_sim':>8}  {'ipl_sim':>8}  {'error':>8}")
    print("-" * 58)

    for i, drive in enumerate(DRIVE_VALS):
        for j, pl in enumerate(PL_VALS):
            fpath = _grid1_hdf5_path(drive, pl)
            if not os.path.exists(fpath):
                print(f"{drive:>6.2f}  {pl:>6.3f}  {'MISSING':>34}")
                continue

            f_sim, a_sim, ipl_sim, _, _ = _robot_metrics_from_hdf5(fpath)

            err = (abs(f_sim   - target_f)   / (target_f         + 1e-9)
                 + abs(ipl_sim - target_ipl)  / (abs(target_ipl)  + 1e-9)
                 + abs(a_sim   - target_amp)  / (target_amp        + 1e-9))

            grid_error[i, j] = err
            grid_f[i, j]     = f_sim
            grid_amp[i, j]   = a_sim
            grid_ipl[i, j]   = ipl_sim

            marker = " <-- best" if err < best_error else ""
            print(f"{drive:>6.2f}  {pl:>6.3f}  {f_sim:>7.3f}  {a_sim:>8.3f}  {ipl_sim:>8.3f}  {err:>8.4f}{marker}")

            if err < best_error:
                best_error  = err
                best_params = {'drive': drive, 'pl': pl,
                               'f': f_sim, 'amp': a_sim, 'ipl': ipl_sim}
                best_hdf5   = fpath

    print("-" * 58)
    if best_params:
        print(f"Best  →  drive={best_params['drive']:.2f}  "
              f"PL={best_params['pl']:.3f}  error={best_error:.4f}\n")

    # ── Heatmaps of error and metrics ────────────────────────────────────────
    pl_labels    = [f'{p:.2f}' for p in PL_VALS]
    drive_labels = [f'{d:.2f}' for d in DRIVE_VALS]

    _, axes = plt.subplots(1, 4, figsize=(18, 4))
    data_list  = [grid_error, grid_f,   grid_amp,  grid_ipl]
    titles     = ['Imitation error', 'Frequency (Hz)', 'Amplitude (rad)', 'IPL (rad)']
    cmaps      = ['YlOrRd_r',        'viridis',        'viridis',          'viridis']

    for ax, data, title, cmap in zip(axes, data_list, titles, cmaps):
        im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(N_GRID))
        ax.set_xticklabels(pl_labels, rotation=45)
        ax.set_yticks(range(N_GRID))
        ax.set_yticklabels(drive_labels)
        ax.set_xlabel('Phase lag (rad)')
        ax.set_ylabel('Drive')
        ax.set_title(title)

    # Mark best cell
    if best_params is not None:
        i_best = np.argmin(np.abs(DRIVE_VALS - best_params['drive']))
        j_best = np.argmin(np.abs(PL_VALS    - best_params['pl']))
        axes[0].add_patch(plt.Rectangle(
            (j_best - 0.5, i_best - 0.5), 1, 1,
            fill=False, edgecolor='cyan', linewidth=2.5, label='best'
        ))

    plt.suptitle('Grid search: imitation of animal swimming', fontsize=12)
    plt.tight_layout()
    os.makedirs(BASE_PATH, exist_ok=True)
    plt.savefig(BASE_PATH + 'imitation_heatmaps.png', dpi=150)
    plt.show()

    return best_params, best_hdf5


# ──────────────────────────────────────────────────────────────────────────────
# Kinematic comparison (bonus)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_one_cycle(times, joint_pos, f_sim):
    """Extract the last full cycle from the steady-state robot data."""
    T       = 1.0 / max(f_sim, 0.05)
    t_start = times[-1] - T
    mask    = times >= t_start
    return times[mask], joint_pos[mask]


# ──────────────────────────────────────────────────────────────────────────────
# Main exercise
# ──────────────────────────────────────────────────────────────────────────────

def exercise2_3(sim_result = 'logs/exercise2_1/simulation.hdf5', plot=True): # added this parameter to be able to use this function in bonus.py without modification
    """
    Analyze animal data, compare with CPG baseline, and (bonus) find best
    imitation from the grid1_drive_pl simulations.
    """
    pylog.set_level('critical')

    f_animal, a_animal, ipl_animal, ipl_animal_mean, animal_times, animal_joints = \
        get_animal_data(ANIMAL_DATA_PATH)

    with h5py.File(sim_result, "r") as f:
        sim_times   = f['times'][:]
        joints_data = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    joint_pos_baseline = joints_data[:, :N_JOINT, 0]

    f_robot, a_robot = compute_mechanical_frequency_amplitude_fft(
        sim_times, joint_pos_baseline)
    ipl_robot , ipl_robot_mean = compute_neural_phase_lags(
        sim_times, joint_pos_baseline, f_robot,
        [[i, i + 1] for i in range(N_JOINT - 1)])
    
    plot_results_EXO2_3(
        f_animal, a_animal, ipl_animal, ipl_animal_mean,
        f_robot, a_robot, ipl_robot, ipl_robot_mean, BASE_PATH, plot=plot
    )
    if plot:
        plt.show()

if __name__ == '__main__':
    exercise2_3(plot=True)
