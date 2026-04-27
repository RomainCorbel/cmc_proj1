#!/usr/bin/env python3
"""Run exercise 2.2 parameter sweeps and generate heatmaps/trajectory plots."""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from farms_core import pylog

from cmc_controllers.metrics import (
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    compute_trajectory_curvature,
)
from cmc_controllers.plot_utils import plot_drive_pl_heatmaps, plot_diff_drive_heatmaps

# Multiprocessing
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from simulate import run_multiple
MAX_WORKERS = 4  # adjust based on your hardware capabilities

# CPG parameters
BASE_PATH = 'logs/exercise2_2/'
PLOT_PATH = 'results'
RECORDING = None

def load_metrics_from_hdf5(hdf5_path):
    """Load speed and CoT metrics from an HDF5 simulation result."""
    with h5py.File(hdf5_path, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    links_positions = sensor_data_links[:, :, 7:10]
    links_velocities = sensor_data_links[:, :, 14:17]
    joints_velocities = sensor_data_joints[:, :, 1]
    joints_torques = sensor_data_joints[:, :, 2]

    speed_forward, speed_lateral = compute_mechanical_speed(
        links_positions=links_positions,
        links_velocities=links_velocities,
    )
    _, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=links_positions,
        joints_torques=joints_torques,
        joints_velocities=joints_velocities,
    )

    return speed_forward, speed_lateral, cot


def get_metrics_drive_pl(drive, pl, base_path):
    """Load speed and CoT for a (drive, pl) pair from grid1_drive_pl."""
    n_joint = 8
    hdf5_path = (
        base_path
        + f"simulation_drive{drive:0.3f}_PLarr{n_joint - 1}_{pl:0.3f}.hdf5"
    )
    if not os.path.exists(hdf5_path):
        return np.nan, np.nan
    with h5py.File(hdf5_path, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    links_positions = sensor_data_links[:, :, 7:10]
    links_velocities = sensor_data_links[:, :, 14:17]
    joints_velocities = sensor_data_joints[:, :, 1]
    joints_torques = sensor_data_joints[:, :, 2]

    speed_forward, _ = compute_mechanical_speed(
        links_positions=links_positions,
        links_velocities=links_velocities,
    )
    _, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=links_positions,
        joints_torques=joints_torques,
        joints_velocities=joints_velocities,
    )
    return speed_forward, cot


def get_metrics_diff_drive(drive_left, drive_right, base_path):
    """Load trajectory curvature for a (drive_left, drive_right) pair."""
    hdf5_path = (
        base_path
        + f"simulation_drive_left{drive_left:0.3f}_drive_right{drive_right:0.3f}.hdf5"
    )
    if not os.path.exists(hdf5_path):
        return np.nan
    with h5py.File(hdf5_path, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]

    links_positions = sensor_data_links[:, :, 7:10]
    trajectory = np.mean(links_positions[:, :, :2], axis=1)
    timestep = float(sim_times[1] - sim_times[0])

    return compute_trajectory_curvature(trajectory=trajectory, timestep=timestep)


def exercise2_2(**kwargs):
    pylog.warning("TODO: 2.2: Explore the effect of drive parameters and body phase bias")
    pylog.set_level('critical')

    plot = kwargs.pop('plot', False)
    run = kwargs.pop('run', False)
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
                size=16)}}
    

    n_joint = 8
    N_GRID = 10  # grid resolution

    # Grid 1: symmetric drive (drive_left = drive_right) × phase lag per joint
    drive_vals = np.linspace(2.0, 4.0, N_GRID)
    pl_vals = np.linspace(np.pi / (2 * n_joint), 3 * np.pi / n_joint, N_GRID)
    # PL is an array of (n_joint-1) equal values — one per inter-joint connection
    pl_arrays = [np.ones(n_joint - 1) * pl for pl in pl_vals]

    # 'drive' is a special alias in runsim that sets both drive_left and drive_right
    grid1_path = BASE_PATH + 'grid1_drive_pl/'

    if run :
        run_multiple(
            max_workers=MAX_WORKERS,
            controller=controller,
            base_path=grid1_path,
            parameter_grid={'drive': drive_vals, 'PL': pl_arrays},
            common_kwargs={'fast': True, 'headless': True},
        )

        run_multiple(
            max_workers=MAX_WORKERS,
            controller=controller,
            base_path=BASE_PATH + 'grid2_diff_drive/',
            parameter_grid={'drive_left': drive_vals, 'drive_right': drive_vals},
            common_kwargs={'fast': True, 'headless': True},
        )

    
   
        
    if plot:
        plot_drive_pl_heatmaps(
            drive_range=drive_vals,
            pl_vals=pl_vals,
            get_metrics=lambda d, pl: get_metrics_drive_pl(d, pl, grid1_path),
            base_path=grid1_path,
        )
        grid2_path = BASE_PATH + 'grid2_diff_drive/'
        plot_diff_drive_heatmaps(
            drive_vals=drive_vals,
            get_metrics=lambda dl, dr: get_metrics_diff_drive(dl, dr, grid2_path),
            base_path=grid2_path,
        )
        plt.show()


if __name__ == '__main__':
    exercise2_2(plot=True,run = False)

