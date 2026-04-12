#!/usr/bin/env python3

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from farms_core import pylog

from cmc_controllers.metrics import (
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
)
from simulate import runsim, run_multiple

pylog.set_level('warning')
# pylog.set_level('critical') # suppress logging output in multi-processing

BASE_PATH = 'logs/exercise3_3/'
PLOT_PATH = 'results'

# CPG parameters
DRIVE_LEFT = 3
DRIVE_RIGHT = 3
DRIVE_LOW = 1
DRIVE_HIGH = 5
A_RATE = np.ones(8) * 3
OFFSET_FREQ = np.ones(8) * 1
OFFSET_AMP = np.ones(8) * 0
G_FREQ = np.ones(8) * 0.5
G_AMP = np.ones(8) * 0.25
PHASELAG = np.ones(7) * np.pi * 2 / 8
COUPLING_WEIGHTS_ROSTRAL = 5
COUPLING_WEIGHTS_CAUDAL = 5
COUPLING_WEIGHTS_CONTRA = 10
# random init phases for 16 oscillators for 8 joints
INIT_PHASE = np.random.default_rng(
    seed=42).uniform(0.0, 2 * np.pi, size=16)
W_IPSI = 10.0

# disruption propabilities
DISRUPTION_P_SENSORS = 0.2
DISRUPTION_P_COUPLINGS = 0.2
RANDOM_SEED = 42
MAX_WORKERS = 1

def load_sim_data(hdf5_path, skip_start=500):
    """Load simulation sensor data and slice out initial transient."""
    with h5py.File(hdf5_path, "r") as f:
        sim_times = f["times"][:]
        sensor_links = f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"][:]
        sensor_joints = f["FARMSLISTanimats"]["0"]["sensors"]["joints"]["array"][:]

    sim_times = sim_times[skip_start:]
    sensor_links_pos = sensor_links[skip_start:, :, 7:10]
    sensor_joints_pos = sensor_joints[skip_start:, :, 0]

    return sim_times, sensor_links_pos, sensor_joints_pos


def exercise3_3(**kwargs):
    """ex3.3 main"""
    pylog.warning("TODO: 3.3 Implement neural disruptions and compare with no disruption.")

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

    # Run original sim
    # runsim(
    #     controller=controller,
    #     base_path=BASE_PATH,
    #     w_ipsi=3.0,
    #     disruption_p_sensors=DISRUPTION_P_SENSORS,
    #     disruption_p_couplings=DISRUPTION_P_COUPLINGS,
    #     random_seed=RANDOM_SEED,
    #     recording='animation3_3.mp4',
    #     hdf5_name='simulation.hdf5',
    #     controller_name='controller.pkl',
    #     runtime_n_iterations=10001,
    #     runtime_buffer_size=10001,
    #     fast=True,
    #     headless=True,
    # )

    # Run with different disruptions
    disr_p_range = np.linspace(0, 0.15, 5)
    run_multiple(
        max_workers=MAX_WORKERS,
        controller=controller,
        base_path=BASE_PATH,
        parameter_grid={
            'disruption_p_sensors': disr_p_range,
            # coupling disruption = 1 is the same as removing all ipsilateral couplings
            'disruption_p_couplings': np.append(disr_p_range, 1), 
            },
        common_kwargs={
            'fast': True,
            'headless': True,
            'runtime_n_iterations': 10001,
            'runtime_buffer_size': 10001,
        },
    )
    

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise3_3(plot=True)

