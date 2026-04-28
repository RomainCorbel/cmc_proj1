#!/usr/bin/env python3

import os
import h5py
import numpy as np
import pickle
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
from cmc_controllers.plot_utils import plot_results_EXO2_1
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
MAX_WORKERS = 5

def load_sim_data(hdf5_path, skip_start=500):
    """Load simulation sensor data and slice out initial transient."""
    with h5py.File(hdf5_path, "r") as f:
        sim_times = f["times"][:]
        sensor_data_links = f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"][:]
        sensor_data_joints = f["FARMSLISTanimats"]["0"]["sensors"]["joints"]["array"][:]

    sim_times = sim_times[skip_start:]
    sensor_data_links_positions = sensor_data_links[skip_start:, :, 7:10]
    sensor_data_links_velocities = sensor_data_links[skip_start:, :, 14:17]
    sensor_data_joints_positions = sensor_data_joints[skip_start:, :, 0]
    sensor_data_joints_velocities = sensor_data_joints[skip_start:, :, 1]
    sensor_data_joints_torques = sensor_data_joints[skip_start:, :, 2]
    
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

    return speed_forward, cot

def post_processing(base_path, sim_name, controller_name, plot = True, subfolder = ""):
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

    output_folder = base_path + "/" + subfolder + "/"

    if plot:
        plot_results_EXO2_1(sim_times, sensor_data_joints_positions, sensor_data_links_positions,  output_folder, controller_data)
    

def parse_all_results(base_path):
    all_files = os.listdir(base_path)

    fout = open(base_path + "/parsed_data.csv", "w")
    fout.write("Sensors,Couplings,RandSeed,Speed,CoT\n")

    for i, file in enumerate(all_files):
        print(f"Processing file {i}/{len(all_files)}")
        if "controller" in file:
            continue
        file_mod = file.replace("_disruption_p_sensors", ";")
        file_mod = file_mod.replace("_disruption_p_couplings", ";")
        file_mod = file_mod.replace("_random_seed", ";")

        sections = file_mod.split(";")
        if len(sections) < 4:
            continue
        
        d_sensors = float(sections[1])
        d_cpls = float(sections[2])
        rand = float(sections[3].split(".")[0])

        speed, cot = load_sim_data(base_path + "/" + file)
        fout.write(f"{d_sensors},{d_cpls},{rand},{speed},{cot}\n")

    fout.close()

def plot_parsed_reslts(paths):

    fig_speed, ax_speed = plt.subplots(3, 2, layout='constrained')
    fig_cot, ax_cot = plt.subplots(3, 2, layout='constrained')
    fig_speed.suptitle("Forward Speed [m/s]")
    fig_cot.suptitle("CoT [J/m]")

    ax_speed[0,0].set_title("Combined")
    ax_speed[0,1].set_title("Decoupled")
    ax_speed[0,0].set_ylabel("Muted Sensors")
    ax_speed[1,0].set_ylabel("Removed Couplings")
    ax_speed[2,0].set_ylabel("Mixed")

    ax_speed[0,0].get_xaxis().set_visible(False)
    ax_speed[0,1].get_xaxis().set_visible(False)
    ax_speed[1,0].get_xaxis().set_visible(False)
    ax_speed[1,1].get_xaxis().set_visible(False)
    ax_cot[0,0].get_xaxis().set_visible(False)
    ax_cot[0,1].get_xaxis().set_visible(False)
    ax_cot[1,0].get_xaxis().set_visible(False)
    ax_cot[1,1].get_xaxis().set_visible(False)

    ax_cot[0,0].set_title("Combined")
    ax_cot[0,1].set_title("Decoupled")
    ax_cot[0,0].set_ylabel("Muted Sensors")
    ax_cot[1,0].set_ylabel("Removed Couplings")
    ax_cot[2,0].set_ylabel("Mixed")

    for path in paths:
        data = open(BASE_PATH + "/" + path + "/parsed_data.csv")
        data.readline()

        sensors = []
        couplings = []
        rands = []
        speeds = []
        cots = []

        for line in data:
            linesplit = line.split(",")
            sensors.append(float(linesplit[0]))
            couplings.append(float(linesplit[1]))
            rands.append(float(linesplit[2]))
            speeds.append(float(linesplit[3]))
            cots.append(float(linesplit[4]))

        if 'combined' in path:
            ax2 = 0
        else:
            ax2 = 1
        if 'sensors' in path:
            ax1 = 0
            x_vals = sensors
        elif 'couplings' in path:
            ax1 = 1
            x_vals = couplings
        else:
            ax1 = 2
            x_vals = sensors

        x_plot = list(set(x_vals))
        x_plot.sort()
        speed_plot = []
        speed_errs = [[],[]]
        cot_plot = []
        cot_errs = [[],[]]
        for x in x_plot:
            x_speeds = [speeds[i] for i in range(len(speeds)) if x_vals[i] == x]
            x_cots = [cots[i] for i in range(len(cots)) if x_vals[i] == x]
            
            speed_plot.append(np.mean(x_speeds))
            cot_plot.append(np.mean(x_cots))

            speed_errs[0].append(np.mean(x_speeds) - np.min(x_speeds))
            speed_errs[1].append(np.max(x_speeds) - np.mean(x_speeds))
            cot_errs[0].append(np.mean(x_cots) - np.min(x_cots))
            cot_errs[1].append(np.max(x_cots) - np.mean(x_cots))

        speed_errs = np.round(speed_errs, 4)
        cot_errs = np.round(cot_errs, 4)
        ax_speed[ax1, ax2].errorbar(x_plot, speed_plot, yerr=speed_errs, capsize=3)
        ax_cot[ax1, ax2].errorbar(x_plot, cot_plot, yerr=cot_errs, capsize=3)

    ax_speed[2,0].set_xlabel("Disruption Probability")
    ax_speed[2,1].set_xlabel("Disruption Probability")
    ax_speed[2,0].set_xticks(x_plot)
    ax_speed[2,1].set_xticks(x_plot)

    ax_cot[2,0].set_xlabel("Disruption Probability")
    ax_cot[2,1].set_xlabel("Disruption Probability")
    ax_cot[2,0].set_xticks(x_plot)
    ax_cot[2,1].set_xticks(x_plot)

    fig_speed.savefig(BASE_PATH + '/disruption_speed_plots.png')
    fig_cot.savefig(BASE_PATH + '/disruption_cot_plots.png')
    

def exercise3_3(**kwargs):
    """ex3.3 main"""
    plot = kwargs.pop('plot', False)

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

    # Run sim with base disruptions
    runsim(
        controller=controller,
        base_path=BASE_PATH,
        w_ipsi=W_IPSI,
        disruption_p_sensors=DISRUPTION_P_SENSORS,
        disruption_p_couplings=DISRUPTION_P_COUPLINGS,
        random_seed=RANDOM_SEED,
        recording='animation3_3_with_distruption.mp4',
        hdf5_name='simulation_with_distruption.hdf5',
        controller_name='controller_with_distruption.pkl',
        runtime_n_iterations=10001,
        runtime_buffer_size=20001,
        fast=True,
        headless=True,
    )

    # Rerun sim with no disruptions
    runsim(
        controller=controller,
        base_path=BASE_PATH,
        w_ipsi=W_IPSI,
        disruption_p_sensors=0,
        disruption_p_couplings=0,
        random_seed=RANDOM_SEED,
        recording='animation3_3_no_disruption.mp4',
        hdf5_name='simulation_no_disruption.hdf5',
        controller_name='controller_no_disruption.pkl',
        runtime_n_iterations=10001,
        runtime_buffer_size=20001,
        fast=True,
        headless=True,
    )

    disr_p_range = np.linspace(0, 0.15, 5)
    rand_seeds = np.array([234, 2354, 5432, 2346, 14, 342342, 325, 892])
    
    # Sensors, combined
    run_multiple(
            max_workers=MAX_WORKERS,
            controller=controller,
            base_path=BASE_PATH + "/sensors_combined/",
            parameter_grid={
                'disruption_p_sensors': disr_p_range,
                'disruption_p_couplings': [0], 
                'random_seed': rand_seeds
                },
            common_kwargs={
                'fast': True,
                'headless': True,
                'runtime_n_iterations': 10001,
                'runtime_buffer_size': 20001,
                'w_ipsi': W_IPSI
            },
        )
    
    # Couplings, combined
    run_multiple(
            max_workers=MAX_WORKERS,
            controller=controller,
            base_path=BASE_PATH + "/couplings_combined/",
            parameter_grid={
                'disruption_p_sensors': [0],
                'disruption_p_couplings': disr_p_range, 
                'random_seed': rand_seeds
                },
            common_kwargs={
                'fast': True,
                'headless': True,
                'runtime_n_iterations': 10001,
                'runtime_buffer_size': 20001,
                'w_ipsi': W_IPSI
            },
        )
    
    # Mixed, combined
    for prob in disr_p_range:
        run_multiple(
            max_workers=MAX_WORKERS,
            controller=controller,
            base_path=BASE_PATH + "/mixed_combined/",
            parameter_grid={
                'disruption_p_sensors': [prob],
                'disruption_p_couplings': [prob], 
                'random_seed': rand_seeds
                },
            common_kwargs={
                'fast': True,
                'headless': True,
                'runtime_n_iterations': 10001,
                'runtime_buffer_size': 20001,
                'w_ipsi': W_IPSI
            },
        )

    controller_decoupled = {
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
            'coupling_weights_rostral': 0,
            'coupling_weights_caudal': 0,
            'coupling_weights_contra': COUPLING_WEIGHTS_CONTRA,
            'init_phase': INIT_PHASE,
        },
    }

    # Sensors, decoupled
    run_multiple(
            max_workers=MAX_WORKERS,
            controller=controller_decoupled,
            base_path=BASE_PATH + "/sensors_decoupled/",
            parameter_grid={
                'disruption_p_sensors': disr_p_range,
                'disruption_p_couplings': [0], 
                'random_seed': rand_seeds
                },
            common_kwargs={
                'fast': True,
                'headless': True,
                'runtime_n_iterations': 10001,
                'runtime_buffer_size': 20001,
                'w_ipsi': W_IPSI
            },
        )
    
    # Couplings, decoupled
    run_multiple(
            max_workers=MAX_WORKERS,
            controller=controller_decoupled,
            base_path=BASE_PATH + "/couplings_decoupled/",
            parameter_grid={
                'disruption_p_sensors': [0],
                'disruption_p_couplings': disr_p_range, 
                'random_seed': rand_seeds
                },
            common_kwargs={
                'fast': True,
                'headless': True,
                'runtime_n_iterations': 10001,
                'runtime_buffer_size': 20001,
                'w_ipsi': W_IPSI
            },
        )
    
    # Mixed, decoupled
    for prob in disr_p_range:
        run_multiple(
            max_workers=MAX_WORKERS,
            controller=controller_decoupled,
            base_path=BASE_PATH + "/mixed_decoupled/",
            parameter_grid={
                'disruption_p_sensors': [prob],
                'disruption_p_couplings': [prob], 
                'random_seed': rand_seeds
                },
            common_kwargs={
                'fast': True,
                'headless': True,
                'runtime_n_iterations': 10001,
                'runtime_buffer_size': 20001,
                'w_ipsi': W_IPSI
            },
        )
    
    paths = ['couplings_combined', 'couplings_decoupled',
                 'mixed_combined', 'mixed_decoupled',
                 'sensors_combined', 'sensors_decoupled']
    for path in paths:
        parse_all_results(BASE_PATH + f'/{path}/')
    plot_parsed_reslts(paths)

    if not os.path.exists(BASE_PATH + "/" + "with_disruption"):
        os.mkdir(BASE_PATH + "/" + "with_disruption")
    if not os.path.exists(BASE_PATH + "/" + "no_disruption"):
        os.mkdir(BASE_PATH + "/" + "no_disruption")

    print("\nPost processing - Disruption")
    print("-----------------------------")
    post_processing(
        base_path=BASE_PATH,
        sim_name='simulation_with_distruption.hdf5',
        controller_name='controller_with_distruption.pkl',
        plot=plot,
        subfolder='with_disruption'
    )

    print("\nPost processing - No disruption")
    print("-----------------------------")
    post_processing(
        base_path=BASE_PATH,
        sim_name='simulation_no_disruption.hdf5',
        controller_name='controller_no_disruption.pkl',
        plot=plot,
        subfolder='no_disruption'
    )

    if plot:
        plt.show()


if __name__ == '__main__':
    exercise3_3(plot=True)

