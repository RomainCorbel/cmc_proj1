"""Exercise 4: Transitions between swimming and walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
#import farms_pylog as pylog


def exercise_4a_transition(timestep):
    """4a Transitions

    In this exerices, we will implement transitions.
    The salamander robot needs to perform swimming to walking
    and walking to swimming transitions.

    Hint:
        - The handling of the drive update is done in robot_parameters.py
        - Set the  arena to 'amphibious'
        - Use the contacts values to find the point where
          the robot should transition
        - Simulation can be stopped/played in the middle
          by pressing the space bar
        - Printing or debug mode of vscode can be used
          to understand the sensor values

    We recommend using the following in robot_parameters.py::step():

    index = 0 if iteration == 0 else (iteration - 1)
    contacts_all = np.linalg.norm(np.array(
        salamandra_data.sensors.contacts.totals()[index]
    ), axis=1)
    contacts_body = contacts_all[:9]
    contacts_upper_limbs = contacts_all[9:17:2]
    contacts_feet = contacts_all[10:18:2]

    # Use self.update_drive = parameters.update_drive in __init__
    if self.update_drive:
        ...

    """
    os.makedirs('./logs/exercise4a/sim_0', exist_ok=True)
    os.makedirs('./logs/exercise4a/sim_1', exist_ok=True)

    # Swim → Walk: start in water (x < 0), high drive, facing land
    sim_parameters_s2w = SimulationParameters(
        duration=30,
        timestep=timestep,
        spawn_position=[2, 0, 0.1],
        spawn_orientation=[0, 0, 0],
        drive=4.0,
        update_drive='swim2walk',
    )
    
    simulation(
        sim_parameters=sim_parameters_s2w,
        arena='amphibious',
        fast=True,
        record=True,
        record_path='logs/exercise4a/swim2walk.mp4',
        output='logs/exercise4a/sim_0',
    )
    
    # Walk → Swim: start on land (x < 0), low drive, facing water
    sim_parameters_w2s = SimulationParameters(
        duration=30,
        timestep=timestep,
        spawn_position=[-1, 0, 0.1],
        spawn_orientation=[0, 0, np.pi],
        drive=2.0,
        update_drive='walk2swim',
    )

    simulation(
        sim_parameters=sim_parameters_w2s,
        arena='amphibious',
        fast=True,
        record=True,
        record_path='logs/exercise4a/walk2swim.mp4',
        output='logs/exercise4a/sim_1',
    )

    return


if __name__ == '__main__':
    exercise_4a_transition(timestep=5e-3)

