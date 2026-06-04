"""Exercise 3: Limb and Spine Coordination while walking"""

import os
import numpy as np
from salamandra_simulation.simulation import simulation, simulation_sweep
from simulation_parameters import SimulationParameters
#import farms_pylog as pylog
import plots_ex3

VIDEO = True

def exericse_3_run_normal(timestep):
    """ Walk with normal configuration """
    parameter_set = [
        SimulationParameters(
            duration=20,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=2.0,
            disable_limb_spine_coupling=False,
        )
    ]
    os.makedirs('./logs/exercise3_normal/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        simulation(
            sim_parameters=sim_parameters,
            arena='land',
            fast=False,
            headless=False,
            output=f'logs/exercise3_normal/sim_{simulation_i}',
            record=VIDEO,
            record_path=f'logs/exercise3_normal/exercise3_normal.mp4',
        )
    return


def exercise_3_disable_limb_spine_coupling(timestep):
    """ Walk with disabled limb-spine limbs """
    parameter_set = [
        SimulationParameters(
            duration=20,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=2.0,
            disable_limb_spine_coupling=True,
        )
    ]
    os.makedirs('./logs/exercise3_disable_coupling/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        simulation(
            sim_parameters=sim_parameters,
            arena='land',
            fast=False,
            headless=False,
            output=f'logs/exercise3_disable_coupling/sim_{simulation_i}',
            record=VIDEO,
            record_path=f'logs/exercise3_disable_coupling/exercise3_disable_coupling.mp4',
        )
    return

lag_range = np.linspace(0, 2*np.pi, 10)
drive_range = np.arange(1.2, 3, 0.2)

def exercise_3a_coordination(timestep):
    """Exercise 3a Limb and Spine coordination

    This exercise explores how phase difference between spine and legs
    affects locomotion.

    Run the simulations for different walking drives and phase lag between body
    and limb oscillators.

    """
    # For sweeps with many simulations running in parallel
    parameter_set = [
        SimulationParameters(
            duration=20,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=drive,
            disable_limb_spine_coupling=True,
            phase_lag_body=lag,
        )
        for lag in lag_range
        for drive in drive_range
    ]
    simulation_sweep([
        {
            'sim_parameters': sim_parameters,
            'arena': 'land',
            'fast': True,  # For fast mode (not real-time)
            'headless': True,  # For headless mode (No GUI, could be faster)
            'output': f'logs/ex3_part3/simulation_{simulation_i}',
            'verbose': False,
        }
        for simulation_i, sim_parameters in enumerate(parameter_set)
    ], processes=4)  # Adjust based on your hardware
    return

def exercise_3_optimal(timestep):
    """ Walk with optimal parameters """
    parameter_set = [
        SimulationParameters(
            duration=20,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=2.8,
            disable_limb_spine_coupling=False,
            phase_lag_body=5.585,
        )
    ]
    os.makedirs('./logs/exercise3_optimal/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        simulation(
            sim_parameters=sim_parameters,
            arena='land',
            fast=False,
            headless=False,
            output=f'logs/exercise3_optimal/sim_{simulation_i}',
            record=VIDEO,
            record_path=f'logs/exercise3_optimal/exercise3_optimal.mp4',
        )
    return

def exercise_3_limb_spine_antiphase(timestep):
    """ Walk with limb-spine in anti-phase """
    parameter_set = [
        SimulationParameters(
            duration=20,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=2.8,
            disable_limb_spine_coupling=False,
            phase_lag_body=5.585-np.pi,
        )
    ]
    os.makedirs('./logs/exercise3_antiphase/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        simulation(
            sim_parameters=sim_parameters,
            arena='land',
            fast=False,
            headless=False,
            output=f'logs/exercise3_antiphase/sim_{simulation_i}',
            record=VIDEO,
            record_path=f'logs/exercise3_antiphase/exercise3_antiphase.mp4',
        )
    return

axial_gains = np.arange(0, 3, 0.2)
limb_gains = np.arange(0, 3, 0.2)

def exercise_3b_coordination(timestep):
    """Exercise 3b Limb and Spine coordination

    This exercise explores how axial and limb amplitudes affect coordination.

    Run the simulations for different axial and limb amplitudes.

    """
    parameter_set = [
        SimulationParameters(
            duration=20,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=2.8,
            disable_limb_spine_coupling=True,
            phase_lag_body=5.585,
            axial_amp_gain=gain_a,
            limb_amp_gain=gain_l
        )
        for gain_a in axial_gains
        for gain_l in limb_gains
    ]
    simulation_sweep([
        {
            'sim_parameters': sim_parameters,
            'arena': 'land',
            'fast': True,  # For fast mode (not real-time)
            'headless': True,  # For headless mode (No GUI, could be faster)
            'output': f'logs/ex3_part4/simulation_{simulation_i}',
            'verbose': False,
        }
        for simulation_i, sim_parameters in enumerate(parameter_set)
    ], processes=4)  # Adjust based on your hardware
    return

def exercise_3_optimal_2(timestep):
    """ Walk with optimal parameters """
    parameter_set = [
        SimulationParameters(
            duration=20,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=2.8,
            disable_limb_spine_coupling=False,
            phase_lag_body=5.585,
            axial_amp_gain=1,
            limb_amp_gain=1.6,
        )
    ]
    os.makedirs('./logs/exercise3_optimal_2/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        simulation(
            sim_parameters=sim_parameters,
            arena='land',
            fast=False,
            headless=False,
            output=f'logs/exercise3_optimal_2/sim_{simulation_i}',
            record=VIDEO,
            record_path=f'logs/exercise3_optimal_2/exercise3_optimal_2.mp4',
        )
    return


if __name__ == '__main__':
    PLOT = True

    #exericse_3_run_normal(timestep=5e-3)
    # exercise_3_disable_limb_spine_coupling(timestep=5e-3)
    #exercise_3a_coordination(timestep=5e-3)
    # exercise_3_optimal(timestep=5e-3)
    # exercise_3_limb_spine_antiphase(timestep=5e-3)
   # exercise_3b_coordination(timestep=5e-3)
    exercise_3_optimal_2(timestep=5e-3)

    if PLOT:
        # plots_ex3.main_single('logs/exercise3_normal/sim_{}', 'Normal Operation')
        # plots_ex3.main_single('logs/exercise3_disable_coupling/sim_{}', 'Disabled Couplings')
        # plots_ex3.main_sweep('logs/ex3_part3/simulation_{}', range(len(lag_range)*len(drive_range)), 'Drive/Phase Lag Sweep', case=3)
        # plots_ex3.main_single('logs/exercise3_optimal/sim_{}', 'Ideal Phase Lag')
        # plots_ex3.main_single('logs/exercise3_antiphase/sim_{}', 'Antiphase Behaviour')
        # plots_ex3.main_sweep(
        #     'logs/ex3_part4/simulation_{}', 
        #     range(len(axial_gains)*len(limb_gains)), 
        #     'Oscillator Amplitude Lag Sweep', case=4)
        plots_ex3.main_single('logs/exercise3_optimal_2/sim_{}', 'Ideal Amplitude Gains')

