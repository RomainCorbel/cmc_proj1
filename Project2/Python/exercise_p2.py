"""[Project1] Exercise 2: Swimming & Walking with Salamander Robot"""

import os
import shutil
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters

_BODY_DATA = {
    'dlow': 1.0,
    'dhigh': 5.0,
    'cv1': 0.2,
    'cv0': 0.3,
    'cr1': 0.065,
    'cr0': 0.196,
    'vsat': 0.0,
    'Rsat': 0.0, 
    'amp_rate': 20.0,
}
_LIMB_DATA = {
    'dlow': 1.0,
    'dhigh': 3.0,
    'cv1': 0.2,
    'cv0': 0.0,
    'cr1': 0.131,
    'cr0': 0.131,
    'vsat': 0.0,
    'Rsat': 0.0, 
    'amp_rate': 20.0,
}
_COUPLING_WEIGHTS = {
    'body_ips_down': 10.0,  # Towards the tail
    'body_ips_up': 10.0,    # Away from the tail
    'body_contra': 10.0,    # Between left/right body oscillators
    'limb_close_body': 30,  # Between the inner limb joint and the next segment body joints
    'limb_contra': 10.0 ,    # Between the two oscillators for one limb joint
    'limb_ips': 10,         # Along one limb
    'limb_close_lr': 10,    # Between the inner limb joints on the same axis
    'limb_close_fb': 10,    # Between the inner limb joints on the same side of the body
    'other': 0,
}
_PHASE_BIASES = {
    'body_ips_down': -2*np.pi/8,    # Towards the tail
    'body_ips_up': 2*np.pi/8,       # Away from the tail
    'body_contra': np.pi,           # Between left/right body oscillators
    'limb_close_body': np.pi,       # Between the inner limb joint and the next segment body joints
    'limb_contra': np.pi,           # Between the two oscillators for one limb joint
    'limb_ips': 0,                  # Along one limb
    'limb_close_lr': np.pi,         # Between the inner limb joints on the same axis
    'limb_close_fb': np.pi,         # Between the inner limb joints on the same side of the body
    'other': 0,
}

def exercise_walk(timestep):
    "[Project 1] Q2 Walking with fixed drive"
    # Use exercise_example.py for reference
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=2.0,
        body_data=_BODY_DATA,
        limb_data=_LIMB_DATA,
        coupling_weights=_COUPLING_WEIGHTS,
        phase_biases=_PHASE_BIASES,
    )
    log_path = 'logs/exercise2_walk/sim_0'
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        headless=True,
        output=log_path,
        record=True,
        record_path='logs/exercise2_walk/video_walk.mp4',
        record_elevation=-10,
        record_distance=2,
    )
    return


def exercise_ramp_swim(timestep):
    "[Project 1] Q2 Swimming with an increasing (ramp) drive"
    # Use exercise_example.py for reference
    sim_parameters = SimulationParameters(
        duration=40,
        timestep=timestep,
        spawn_position=[0, 0, 0.0],
        spawn_orientation=[0, 0, np.pi/2],
        drive=0.0,
        drive_ramp_start=0.0,
        drive_ramp_end=6.0,
        drive_ramp_duration=40.0,
        body_data=_BODY_DATA,
        limb_data=_LIMB_DATA,
        coupling_weights=_COUPLING_WEIGHTS,
        phase_biases=_PHASE_BIASES,
    )
    log_path = 'logs/exercise2_ramp_swim/sim_0'
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    simulation(
        sim_parameters=sim_parameters,
        arena='water',
        fast=True,
        headless=True,
        output=log_path,
        record=True,
        record_path='logs/exercise2_ramp_swim/video_ramp_swim.mp4',
        record_aziomuth=90,
        record_elevation=-5,
        record_distance=3,
    )
    return


def exercise_ramp_walk(timestep):
    "[Project 1] Q2 Walking with an increasing (ramp) drive"
    # Use exercise_example.py for reference
    sim_parameters = SimulationParameters(
        duration=40,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=0.0,
        drive_ramp_start=0.0,
        drive_ramp_end=6.0,
        drive_ramp_duration=40.0,
        body_data=_BODY_DATA,
        limb_data=_LIMB_DATA,
        coupling_weights=_COUPLING_WEIGHTS,
        phase_biases=_PHASE_BIASES,
    )
    log_path = 'logs/exercise2_ramp_walk/sim_0'
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        headless=True,
        output=log_path,
        record=True,
        record_path='logs/exercise2_ramp_walk/video_ramp_walk.mp4',
        record_aziomuth=90,
        record_elevation=-5,
        record_distance=3,
    )
    return


if __name__ == '__main__':
    exercise_walk(timestep=5e-3)
    exercise_ramp_swim(timestep=5e-3)
    exercise_ramp_walk(timestep=5e-3)
