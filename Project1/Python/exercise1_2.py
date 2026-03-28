#!/usr/bin/env python3
"""Run parameter sweeps for exercise 1.2 and plot metric heatmaps."""

import os
import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from cmc_controllers.plot_utils import plot_gridsearch_heatmaps
from farms_core import pylog

from simulate import run_multiple
from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    compute_neural_phase_lags,
    filter_signals,
)

# Multiprocessing

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

MAX_WORKERS = 8  # adjust based on your hardware capabilities

pylog.set_level('critical')

BASE_PATH = 'logs/exercise1_2/'
PLOT_PATH = 'results'
RECORDING = None  # disable recording for parallel runs


def get_metrics(twl, amp):
    """Compute mechanical metrics for a single parameter set."""
    # Load HDF5
    sim_result = BASE_PATH + \
        f'simulation_twl{twl:0.3f}_amp{amp:0.3f}.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    sensor_data_links_positions = sensor_data_links[:, :, 7:10]

    sensor_data_links_velocities = sensor_data_links[:, :, 14:17]
    sensor_data_joints_velocities = sensor_data_joints[:, :, 1]
    sensor_data_joints_torques = sensor_data_joints[:, :, 2]

    speed_forward, _ = compute_mechanical_speed(
        links_positions=sensor_data_links_positions,
        links_velocities=sensor_data_links_velocities,
    )
    _, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=sensor_data_links_positions,
        joints_torques=sensor_data_joints_torques,
        joints_velocities=sensor_data_joints_velocities,
    )

    # Load Controller
    controller_file = os.path.join(
        BASE_PATH,
        f"controller_twl{twl:0.3f}_amp{amp:0.3f}.pkl",
    )
    with open(controller_file, "rb") as f:
        controller_data = pickle.load(f)

    indices = controller_data["indices"]
    neural_signals = (
        controller_data["state"][:, indices['left_body_idx']]
        - controller_data["state"][:, indices['right_body_idx']]
    )
    neural_signals_smoothed = filter_signals(
        times=sim_times, signals=neural_signals)
    signal_freqs, _, _ = compute_frequency_amplitude_fft(
        times=sim_times,
        smooth_signals=neural_signals_smoothed,
    )
    inds_couples = [[i, i + 1]
                    for i in range(neural_signals_smoothed.shape[1] - 1)]
    _, ipls_mean = compute_neural_phase_lags(
        times=sim_times,
        smooth_signals=neural_signals_smoothed,
        freqs=signal_freqs,
        inds_couples=inds_couples,
    )

    return speed_forward, cot, float(ipls_mean)


def exercise1_2(**kwargs):
    """ex1.2 main"""
    os.makedirs(PLOT_PATH, exist_ok=True)
    base_controller = {
        'loader': 'cmc_controllers.wave_controller.WaveController',
        'config': {
            'freq': 1.5,
            'twl': 1.0,
            'amp': 1.0}}
    # Hint: You don't need to test all combinations of parameters with complexity of O(n^3)
    # You can replace range with list of length 1 to keep some parameters fixed
    # while testing others O(n^2) or O(n)

    twl_range = np.linspace(0.2, 1.5, 10)
    amp_range = np.linspace(1.0, 4.0, 10)

    parameter_grid_example = {
        'twl': twl_range,
        'amp': amp_range,
    }
    n_workers = 8
    """
    run_multiple(
        max_workers=n_workers,
        controller=base_controller,
        base_path=BASE_PATH,
        parameter_grid=parameter_grid_example,
        common_kwargs={'fast': True, 'headless': True},
    )
    """


    plot_gridsearch_heatmaps(
        twl_range=twl_range,
        amp_range=amp_range,
        get_metrics=get_metrics,
        base_path=BASE_PATH,
    )
    
if __name__ == '__main__':
    exercise1_2(plot=True)

