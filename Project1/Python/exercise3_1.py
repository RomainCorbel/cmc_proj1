#!/usr/bin/env python3

import os
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from cmc_controllers.plot_utils import plot_results_EXO2_1

from farms_core import pylog
from farms_core.utils.profile import profile

from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    filter_signals,
)
from simulate import runsim


BASE_PATH = 'logs/exercise3_1/'
PLOT_PATH = 'results'

def post_processing(base_path, sim_name, controller_name, plot = True):
    """Post processing"""
    # Load HDF5
    sim_result = base_path + sim_name
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
    with open(base_path + controller_name, "rb") as f:
        controller_data = pickle.load(f)

    if plot:
        plot_results_EXO2_1(sim_times, sensor_data_joints_positions, sensor_data_links_positions,  base_path, controller_data)
    
    # Metrics computation
    indices = controller_data["indices"]
    neural_signals = (
        controller_data["state"][:, indices['left_body_idx']]
        - controller_data["state"][:, indices['right_body_idx']]
    )
    neural_signals_smoothed = filter_signals(
        times=sim_times, signals=neural_signals)
    
    freq, _, amp = compute_frequency_amplitude_fft(
        times=sim_times, smooth_signals=neural_signals_smoothed)

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
    
    print('Neural frequencies: ', freq)
    print('Neural amplitudes: ', amp)
    print('Forward speed: ', speed_forward)
    print("CoT: ", cot)

def main(**kwargs):
    """Run exercise 3.1 simulations with and without sensory feedback."""
    controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': 3,
            'drive_right': 3,
            'd_low': 1,
            'd_high': 5,
            'a_rate': np.ones(8) * 3,
            'offset_freq': np.ones(8) * 1,
            'offset_amp': np.ones(8) * 0.5,
            'G_freq': np.ones(8) * 0.5,
            'G_amp': np.ones(8) * 0.25,
            'PL': np.ones(7) * np.pi * 2 / 8,
            'coupling_weights_rostral': 5,
            'coupling_weights_caudal': 5,
            'coupling_weights_contra': 10,
            'init_phase': np.random.default_rng(
                seed=42).uniform(
                0.0,
                2 * np.pi,
                size=16),
        },
    }
    w_ipsi = 3.0
    fast = kwargs.pop('fast', False)
    headless = kwargs.pop('headless', False)

    pylog.warning("TODO: 3.1 Simulate with and without sensory feedback")

    pylog.warning("TODO: 3.1 Compare the performance")

    runsim(
        controller=controller,
        base_path=BASE_PATH,
        w_ipsi=w_ipsi,
        recording='animation3_1_with_sf.mp4',
        hdf5_name='simulation_with_sf.hdf5',
        controller_name='controller_with_sf.pkl',
        runtime_n_iterations=5001,
        runtime_buffer_size=5001,
        fast=fast,
        headless=headless,
    )

    runsim(
        controller=controller,
        base_path=BASE_PATH,
        w_ipsi=0,
        recording='animation3_1_without_sf.mp4',
        hdf5_name='simulation_without_sf.hdf5',
        controller_name='controller_without_sf.pkl',
        runtime_n_iterations=5001,
        runtime_buffer_size=5001,
        fast=fast,
        headless=headless,
    )

def exercise3_1(**kwargs):
    """ex3.1 main"""
    profile(function=main, profile_filename='',
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)
    plot = kwargs.pop('plot', False)

    print("\nPost processing - Wipsi = 3.0")
    print("-----------------------------")
    post_processing(
        base_path=BASE_PATH,
        sim_name='simulation_with_sf.hdf5',
        controller_name='controller_with_sf.pkl',
        plot=plot
    )
    print("\nPost processing - Wipsi = 0.0")
    print("-----------------------------")
    post_processing(
        base_path=BASE_PATH,
        sim_name='simulation_without_sf.hdf5',
        controller_name='controller_without_sf.pkl',
        plot=plot
    )


if __name__ == '__main__':
    exercise3_1(plot=True)

