"""Exercise 3: Limb and Spine Coordination while walking"""

import os
import numpy as np
from salamandra_simulation.simulation import simulation, simulation_sweep
from simulation_parameters import SimulationParameters
import farms_pylog as pylog


def exercise_3_disable_limb_spine_coupling(timestep):
    """ Walk with disabled limb-spine limbs """
    # Use exercise_example.py for reference
    pass
    return


def exercise_3_limb_spine_antiphase(timestep):
    """ Walk with limb-spine in anti-phase """
    # Use exercise_example.py for reference
    pass
    return


def exercise_3a_coordination(timestep):
    """Exercise 3a Limb and Spine coordination

    This exercise explores how phase difference between spine and legs
    affects locomotion.

    Run the simulations for different walking drives and phase lag between body
    and limb oscillators.

    """
    # Use exercise_example.py for reference
    pass
    # # For sweeps with many simulations running in parallel
    # parameter_set = [
    #     SimulationParameters(...)
    #     for ... in ...
    #     for ... in ...
    # ]
    # os.makedirs('./logs/sweep_3a/', exist_ok=True)
    # simulation_sweep([
    #     {
    #         'sim_parameters': sim_parameters,
    #         'arena': 'land',
    #         'fast': True,  # For fast mode (not real-time)
    #         'headless': True,  # For headless mode (No GUI, could be faster)
    #         'output': f'logs/ex3a/simulation_{simulation_i}',
    #         'verbose': False,
    #     }
    #     for simulation_i, sim_parameters in enumerate(parameter_set)
    # ], processes=4)  # Adjust based on your hardware
    return


def exercise_3b_coordination(timestep):
    """Exercise 3b Limb and Spine coordination

    This exercise explores how axial and limb amplitudes affect coordination.

    Run the simulations for different axial and limb amplitudes.

    """
    # Use exercise_example.py for reference
    pass
    return


if __name__ == '__main__':
    exercise_3_disable_limb_spine_coupling(timestep=5e-3)
    exercise_3_limb_spine_antiphase(timestep=5e-3)
    exercise_3a_coordination(timestep=5e-3)
    exercise_3b_coordination(timestep=5e-3)

