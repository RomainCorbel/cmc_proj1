"""[Project1] Exercise 2: Swimming & Walking with Salamander Robot"""

import os
import shutil
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters

def exercise_walk(timestep):
    "[Project 1] Q2 Walking with fixed drive"
    # Use exercise_example.py for reference
    sim_parameters = SimulationParameters(
        duration=45,
        timestep=timestep,
        spawn_position=[0, 0, 0.10],
        spawn_orientation=[0, 0, np.pi/2],
        drive=2.0,
    )
    log_path = 'logs/exercise2_walk/sim_0'
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        fast=True,
        headless=False, # True for no live
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
        record_elevation=-10,
        record_distance=2,
    )
    return


def exercise_ramp_walk(timestep):
    "[Project 1] Q2 Walking with an increasing (ramp) drive"
    # Use exercise_example.py for reference
    sim_parameters = SimulationParameters(
        duration=40,
        timestep=timestep,
        spawn_position=[0, 0, 0],
        spawn_orientation=[0, 0, np.pi/2],
        drive=0.0,
        drive_ramp_start=0.0,
        drive_ramp_end=6.0,
        drive_ramp_duration=40.0,
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
        record_elevation=-10,
        record_distance=2,
    )
    return


if __name__ == '__main__':
    exercise_walk(timestep=5e-3)
    # exercise_ramp_swim(timestep=5e-3)
    # exercise_ramp_walk(timestep=5e-3)
