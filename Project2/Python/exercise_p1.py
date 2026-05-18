"""[Project1] Exercise 1: Implement & run network without MuJoCo"""

import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from farms_core import pylog
from salamandra_simulation.data import SalamandraState
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork


@dataclass
class DataState:
    state: SalamandraState


def run_network(duration, update=False, drive=0, timestep=1e-2):
    """ Run network without MuJoCo and plot results
    Parameters
    ----------
    duration: <float>
        Duration in [s] for which the network should be run
    update: <bool>
        True: use the prescribed drive parameter, False: update the drive during the simulation
    drive: <float/array>
        Central drive to the oscillators
    """
    # Simulation setup
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    initial_drive = drive[0] if hasattr(drive, '__len__') else drive
    sim_parameters = SimulationParameters(
        drive=initial_drive,
        amplitude_gradient=None,
        phase_lag_body=None,
        # Feel free to include parameters
        body_data = {
            'dlow': 1.0,
            'dhigh': 5.0,
            'cv1': 0.2,
            'cv0': 0.3,
            'cr1': 0.065,
            'cr0': 0.196,
            'vsat': 0.0,
            'Rsat': 0.0, 
            'amp_rate': 20.0,
        },
        limb_data = {
            'dlow': 1.0,
            'dhigh': 3.0,
            'cv1': 0.2,
            'cv0': 0.0,
            'cr1': 0.131,
            'cr0': 0.131,
            'vsat': 0.0,
            'Rsat': 0.0, 
            'amp_rate': 20.0,
        },
        coupling_weights = {
            'body_ips_down': 10.0,  # Towards the tail
            'body_ips_up': 10.0,    # Away from the tail
            'body_contra': 10.0,    # Between left/right body oscillators
            'limb_close_body': 30,  # Between the inner limb joint and the next segment body joints
            'limb_contra': 10.0 ,    # Between the two oscillators for one limb joint
            'limb_ips': 10,         # Along one limb
            'limb_close_lr': 10,    # Between the inner limb joints on the same axis
            'limb_close_fb': 10,    # Between the inner limb joints on the same side of the body
            'other': 0,
        },
        phase_biases = {
            'body_ips_down': -2*np.pi/8,    # Towards the tail
            'body_ips_up': 2*np.pi/8,       # Away from the tail
            'body_contra': np.pi,           # Between left/right body oscillators
            'limb_close_body': np.pi,       # Between the inner limb joint and the next segment body joints
            'limb_contra': np.pi,           # Between the two oscillators for one limb joint
            'limb_ips': 0,                  # Along one limb
            'limb_close_lr': np.pi,         # Between the inner limb joints on the same axis
            'limb_close_fb': np.pi,         # Between the inner limb joints on the same side of the body
            'other': 0,
        },
    )
    pylog.warning(
        'Modify the scalar drive to be a vector of length n_iterations. By doing so the drive will be modified to be drive[i] at each time step i.')
    state = SalamandraState.salamandra_robot(n_iterations, n_oscillators=32)
    network = SalamandraNetwork(
        sim_parameters,
        n_iterations,
        DataState(
            state=state))
    osc_left = np.arange(0, 16, 2)
    osc_right = np.arange(1, 16, 2)
    osc_legs = np.arange(16, 32)

    # Logs
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    phases_log[0, :] = network.state.phases(iteration=0)
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)
    freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    freqs_log[0, :] = network.robot_parameters.freqs

    # comment below pass to run file
    # pylog.warning('Remove the pass to run your code!!')
    # pass

    pylog.warning(
        'Implement plots here, try to plot the various logged data to check the implementation')
    # Run network ODE and log data
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        if update:
            current_drive = drive[i+1] if hasattr(drive, '__len__') else drive
            network.robot_parameters.update(
                SimulationParameters(drive=current_drive)
            )
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs
    toc = time.time()

    # Network performance
    pylog.info('Time to run simulation for {} steps: {} [s]'.format(
        n_iterations,
        toc - tic
    ))

    # Motor outputs: x_i = r_i * (1 + cos(phi_i))
    outputs = amplitudes_log * (1 + np.cos(phases_log))

    # Instantaneous frequency via unwrapped phase derivative
    phases_unwrapped = np.unwrap(phases_log, axis=0)
    inst_freq = np.diff(phases_unwrapped, axis=0) / (2 * np.pi * timestep)

    # Drive trajectory for plotting
    drive_plot = drive if hasattr(drive, '__len__') else np.full(n_iterations, drive)

    label = f'ex1_{"ramp" if update else f"drive_{initial_drive}"}'
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, num=label)
    fig.suptitle(f'CPG Network Dynamics (duration={duration}s, drive ramp={update})')

    # A: body left oscillator activations (waterfall, head at top)
    offset = 1.2
    for k, idx in enumerate(osc_left):
        axes[0].plot(times, outputs[:, idx] + k * offset, color='steelblue', linewidth=0.8)
    axes[0].set_ylabel('Body (L)\nactivation')
    axes[0].set_yticks([])

    # B: limb activations — first oscillator of each limb
    for k, idx in enumerate(osc_legs[::4]):
        axes[1].plot(times, outputs[:, idx] + k * offset, color='darkorange', linewidth=0.8)
    axes[1].set_ylabel('Limb\nactivation')
    axes[1].set_yticks([])

    # C: mean instantaneous frequency of left body oscillators
    axes[2].plot(times[1:], np.mean(inst_freq[:, osc_left], axis=1), color='black')
    axes[2].set_ylabel('Freq [Hz]')
    axes[2].set_ylim([0, 2])

    # D: drive signal
    axes[3].plot(times, drive_plot, color='red')
    axes[3].set_ylabel('Drive d')
    axes[3].set_xlabel('Time [s]')

    plt.tight_layout()

    return


def exercise_1a_networks(plot, timestep=1e-2):
    """[Project 1] Exercise 1: """

    # Exercise 1A: fixed drive in walking regime to verify network
    run_network(duration=10, drive=2.0, timestep=timestep)

    # Exercise 1B: linearly increasing drive 0 -> 6 over 20s (Ijspeert 2007 Fig. 2)
    duration_ramp = 20
    drive_ramp = np.linspace(0, 6, int(duration_ramp / timestep))
    run_network(duration=duration_ramp, update=True, drive=drive_ramp, timestep=timestep)

    # Show plots
    if True:
        if plot:
            plt.show()
        else:
            save_figures()
        return


if __name__ == '__main__':
    exercise_1a_networks(plot=not save_plots())

