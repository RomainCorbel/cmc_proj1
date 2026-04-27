#!/usr/bin/env python3

import os
import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
from farms_core.utils.profile import profile
from farms_core import pylog
from cmc_controllers.plot_utils import plot_results_EXO1_1
from simulate import runsim
from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_frequency_amplitude_fft,
    compute_mechanical_speed,
    compute_neural_phase_lags,
    filter_signals,
)

BASE_PATH = 'logs/exercise1_1/'
PLOT_PATH = 'results'


def post_processing():
    """Post processing"""
    # Load HDF5
    sim_result = BASE_PATH + 'simulation.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    sensor_data_links_positions = sensor_data_links[:, :, 7:10]
    sensor_data_links_velocities = sensor_data_links[:, :, 14:17]
    sensor_data_joints_positions = sensor_data_joints[:, :, 0]
    sensor_data_joints_velocities = sensor_data_joints[:, :, 1]
    sensor_data_joints_torques = sensor_data_joints[:, :, 2]

    # Load Controller
    with open(BASE_PATH + "controller.pkl", "rb") as f:
        controller_data = pickle.load(f)

    indices = controller_data["indices"]
    neural_signals = (
        controller_data["state"][:, indices['left_body_idx']]
        - controller_data["state"][:, indices['right_body_idx']]
    )
    neural_signals_smoothed = filter_signals(
        times=sim_times, signals=neural_signals)

    # Metrics computation
    pylog.warning("TODO: 1.1: Complete metrics implementation in metrics.py")
    freq, _, amp = compute_frequency_amplitude_fft(
        times=sim_times, smooth_signals=neural_signals_smoothed)

    inds_couples = [[i, i+1]
                    for i in range(neural_signals_smoothed.shape[1] - 1)]
    ipls, ipls_mean = compute_neural_phase_lags(times=sim_times,
                                             smooth_signals=neural_signals_smoothed,
                                             freqs=freq,
                                             inds_couples=inds_couples)

    mech_freq, mech_amp = compute_mechanical_frequency_amplitude_fft(
        times=sim_times,
        signals=sensor_data_joints_positions[:, :8],
    )

    speed_forward, speed_lateral = compute_mechanical_speed(
        links_positions=sensor_data_links_positions,
        links_velocities=sensor_data_links_velocities,
    )

    energy, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=sensor_data_links_positions,
        joints_torques=sensor_data_joints_torques,
        joints_velocities=sensor_data_joints_velocities,
    )
    print("energy: ", energy, "cot: ", cot)

    pylog.warning("TODO: 1.2: Verify the computed metrics are consistent with the expected values")
    print('Estimated neural metrics:')
    print('Frequencies: ', freq, 'Mean frequency: ', np.mean(freq), '\nAmplitudes: ', amp, "Mean amplitude: ", np.mean(amp), 'phase lags (radians): ', ipls,
          '\nMean phase lags (radians): ', ipls_mean)
    print('Estimated mechanical metrics:')
    print(
        'Frequencies: ',
        mech_freq,
        '\nmean frequency: ',
        np.mean(mech_freq),
        '\nAmplitudes: ',
        mech_amp,
        '\nmean amplitude: ',
        np.mean(mech_amp),
        '\nforward speed: ',
        speed_forward,
        '\nlateral speed: ',
        speed_lateral,
        '\nEnergy: ',
        energy,
        '\nCoT: ',
        cot)
    #Plotting results
    plot_results_EXO1_1(
        sim_times=sim_times,
        freq=freq,
        sensor_data_joints_positions=sensor_data_joints_positions,
        sensor_data_links_positions=sensor_data_links_positions,
        base_path=BASE_PATH,
        )  

def main(**kwargs):
    """ex1.1 main"""
    os.makedirs(PLOT_PATH, exist_ok=True)
    controller = {
        'loader': 'cmc_controllers.wave_controller.WaveController',
        'config': {'freq': 2.0,
                   'twl': 1.0,
                   'amp': 2.0}
    }

    runsim(
        controller=controller,
        base_path=BASE_PATH,
        recording='exercise1_1.mp4',
    )

    post_processing()


def exercise1_1(**kwargs):
    """Entry point for exercise 1.1 with optional plotting and runtime flags."""
    profile(function=main, profile_filename='',
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise1_1(plot=True)

