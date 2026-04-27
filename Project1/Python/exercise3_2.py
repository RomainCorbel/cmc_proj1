#!/usr/bin/env python3


import os
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt

from farms_core import pylog

from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    filter_signals,
)
from simulate import run_multiple

# Multiprocessing
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
MAX_WORKERS = 2  # adjust based on your hardware capabilities


BASE_PATH = 'logs/exercise3_2/'
PLOT_PATH = 'results'

# CPG parameters
DRIVE_LEFT = 3
DRIVE_RIGHT = 3
DRIVE_LOW = 1
DRIVE_HIGH = 5
A_RATE = np.ones(8) * 3
OFFSET_FREQ = np.ones(8) * 1
OFFSET_AMP = np.ones(8) * 0.5
G_FREQ = np.ones(8) * 0.5
G_AMP = np.ones(8) * 0.25
PHASELAG = np.ones(7) * np.pi * 2 / 8
COUPLING_WEIGHTS_ROSTRAL = 5
COUPLING_WEIGHTS_CAUDAL = 5
COUPLING_WEIGHTS_CONTRA = 10
# random init phases for 16 oscillators for 8 joints
INIT_PHASE = np.random.default_rng(
    seed=42).uniform(0.0, 2 * np.pi, size=16)

pylog.set_level('warning')
# pylog.set_level('critical') # suppress logging output in multi-processing

def get_data(base_path, sim_name, controller_name):
    """Post processing"""
    # Load HDF5
    sim_result = base_path + sim_name
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    # Load Controller
    with open(base_path + controller_name, "rb") as f:
        controller_data = pickle.load(f)
    
    # Metrics computation - remove transient
    skip_start = 500
    sim_times = sim_times[skip_start:]

    sensor_data_links_positions = sensor_data_links[skip_start:, :, 7:10]
    sensor_data_links_velocities = sensor_data_links[skip_start:, :, 14:17]
    sensor_data_joints_positions = sensor_data_joints[skip_start:, :, 0]
    sensor_data_joints_velocities = sensor_data_joints[skip_start:, :, 1]
    sensor_data_joints_torques = sensor_data_joints[skip_start:, :, 2]

    state = controller_data['state']
    n_osc  = state.shape[1] // 3
    motor      = state[:, 2*n_osc:]
    left_idx   = controller_data['indices']['left_body_idx']
    right_idx  = controller_data['indices']['right_body_idx']

    ML = motor[:, left_idx]
    MR = motor[:, right_idx]
    neural_signals = ML - MR

    neural_signals_smoothed = filter_signals(
        times=sim_times, signals=neural_signals)

    # Metrics computation
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
    
    return [speed_forward, cot, np.mean(freq), np.mean(amp)]

def post_processing(base_path, variable):
    all_files = os.listdir(base_path)
    controllers = {}
    sims = {}

    for i, file in enumerate(all_files):
        print(f"Processing file {i}/{len(all_files)}")
        if "controller" in file:
            value = float(file.split(".pkl")[0].split(variable)[-1])
            controllers[value] = file
        elif "simulation" in file:
            value = float(file.split(".hdf5")[0].split(variable)[-1])
            sims[value] = file

    if controllers.keys() != sims.keys():
        pylog.error("Simulation outputs do not match")
        return
    
    vals = list(controllers.keys())
    data = []

    for i, val in enumerate(vals):
        print(f"Processing simulation {i}/{len(vals)}")
        data.append(get_data(base_path, sims[val], controllers[val]))

    data = np.array(data)
    labels = ["Forward speed [m/s]", "CoT [J/m]", "Neural frequency [Hz]", "Neural amplitude [-]"]

    plt.figure(layout="constrained")
    plt.suptitle("Impacts of sensory feedback strength")

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.scatter(vals, data[:, i])
        plt.xlabel("w_ipsi")
        plt.ylabel(labels[i])

    plt.savefig(BASE_PATH + "wipsi_varying.png")
    plt.show()


def exercise3_2(**kwargs):
    """ex3.2 main"""
    pylog.warning("TODO: 3.2 Explore the effect of stretch feedback on the metrics.")

    w_ipsi_range = np.arange(-3, 17.5, step=0.5)

    controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': DRIVE_LEFT,
            'drive_right': DRIVE_RIGHT,
            'd_low': DRIVE_LOW,
            'd_high': DRIVE_HIGH,
            'a_rate': A_RATE,
            'offset_freq': OFFSET_FREQ,
            'offset_amp': OFFSET_AMP,
            'G_freq': G_FREQ,
            'G_amp': G_AMP,
            'PL': PHASELAG,
            'coupling_weights_rostral': COUPLING_WEIGHTS_ROSTRAL,
            'coupling_weights_caudal': COUPLING_WEIGHTS_CAUDAL,
            'coupling_weights_contra': COUPLING_WEIGHTS_CONTRA,
            'init_phase': INIT_PHASE,
        },
    }
    # run_multiple(
    #     max_workers=MAX_WORKERS,
    #     controller=controller,
    #     base_path=BASE_PATH,
    #     parameter_grid={'w_ipsi': w_ipsi_range},
    #     common_kwargs={
    #         'fast': True,
    #         'headless': True,
    #         'runtime_n_iterations': 5001,
    #         'runtime_buffer_size': 20001,
    #     },
    # )

    plot = kwargs.pop('plot', False)
    if plot:
        post_processing(BASE_PATH, "w_ipsi")


if __name__ == '__main__':
    exercise3_2(plot=True)

