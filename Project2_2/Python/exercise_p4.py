"""Exercise 4: Transitions between swimming and walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog


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
    # Use exercise_example.py for reference
    # Additional hints:
    # sim_parameters = SimulationParameters(
    #     ...,
    #     spawn_position=[4, 0, 0.0],
    #     spawn_orientation=[0, 0, np.pi],
    # )
    # _sim, _data = simulation(
    #     sim_parameters=sim_parameters,
    #     arena='amphibious',
    #     fast=True,
    #     record=True,
    #     record_path='walk2swim',  # or swim2walk
    # )
    pass
    return


if __name__ == '__main__':
    exercise_4a_transition(timestep=5e-3)

