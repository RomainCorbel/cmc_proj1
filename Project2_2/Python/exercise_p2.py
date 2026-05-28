"""[Project1] Exercise 2: Swimming & Walking with Salamander Robot"""

import os
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters

VISUALIZATION = False

def exercise_walk(timestep):
    "[Project 1] Q2 Walking with a fixed drive"
    parameter_set = [
        SimulationParameters(
            duration=15,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=2.0,
            phase_lag_body=None,
            amplitude_gradient=None,
        )
    ]
    os.makedirs('./logs/exercise2_walk/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        simulation(
            sim_parameters=sim_parameters,
            arena='land',
            fast=not VISUALIZATION,
            headless=not VISUALIZATION,
            output=f'logs/exercise2_walk/sim_{simulation_i}',
            record=not VISUALIZATION,
            record_path=f'logs/exercise2_walk/video_walk.mp4',
        )
    return


def exercise_ramp_swim(timestep):
    "[Project 1] Q2 Swimming with an increasing (ramp) drive"
    duration = 15
    parameter_set = [
        SimulationParameters(
            duration=duration,
            timestep=timestep,
            spawn_position=[0, 0, -0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=1.0,
            drive_ramp_start=1.0,
            drive_ramp_end=6.0,
            drive_ramp_duration=duration,
            phase_lag_body=None,
            amplitude_gradient=None,
        )
    ]
    os.makedirs('./logs/exercise2_ramp_swim/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        simulation(
            sim_parameters=sim_parameters,
            arena='water',
            fast=not VISUALIZATION,
            headless=not VISUALIZATION,
            output=f'logs/exercise2_ramp_swim/sim_{simulation_i}',
            record=not VISUALIZATION,
            record_path=f'logs/exercise2_ramp_swim/video_ramp_swim.mp4',
        )
    return


def exercise_ramp_walk(timestep):
    "[Project 1] Q2 Walking with an increasing (ramp) drive"
    duration = 15
    parameter_set = [
        SimulationParameters(
            duration=duration,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=1.0,
            drive_ramp_start=1.0,
            drive_ramp_end=6.0,
            drive_ramp_duration=duration,
            phase_lag_body=None,
            amplitude_gradient=None,
        )
    ]
    os.makedirs('./logs/exercise2_ramp_walk/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        simulation(
            sim_parameters=sim_parameters,
            arena='land',
            fast=not VISUALIZATION,
            headless=not VISUALIZATION,
            output=f'logs/exercise2_ramp_walk/sim_{simulation_i}',
            record=not VISUALIZATION,
            record_path=f'logs/exercise2_ramp_walk/video_ramp_walk.mp4',
        )
    return


def exercise_swim(timestep):
    "[Project 1] Q2 Swimming with a fixed drive"
    parameter_set = [
        SimulationParameters(
            duration=15,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=4.0,
            phase_lag_body=None,
            amplitude_gradient=None,
        )
    ]
    os.makedirs('./logs/exercise2_swim/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        simulation(
            sim_parameters=sim_parameters,
            arena='water',
            fast=not VISUALIZATION,
            headless=not VISUALIZATION,
            output=f'logs/exercise2_swim/sim_{simulation_i}',
            record=not VISUALIZATION,
            record_path=f'logs/exercise2_swim/video_swim.mp4',
        )
    return



if __name__ == '__main__':
    exercise_walk(timestep=5e-3)
    exercise_ramp_swim(timestep=5e-3)
    exercise_ramp_walk(timestep=5e-3)
    exercise_swim(timestep=5e-3)
