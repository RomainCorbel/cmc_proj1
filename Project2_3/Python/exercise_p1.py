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
from robot_parameters import BODY, LIMB


@dataclass
class DataState:
    state: SalamandraState


def run_network(duration, update=True, drive=0, timestep=1e-2):
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
    sim_parameters = SimulationParameters(
        drive=drive,
        amplitude_gradient=None,
        phase_lag_body=None,
        # Feel free to include parameters
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

    # Run network ODE and log data
    tic = time.time()
    prev_mode = None
    for i, time0 in enumerate(times[1:]):
        if update:
            current_drive = drive[i+1] if hasattr(drive, '__len__') else drive
            sp = SimulationParameters(drive=current_drive)
            network.robot_parameters.set_frequencies(sp)
            network.robot_parameters.set_nominal_amplitudes(sp)
            current_mode = 'swim' if current_drive > 3.0 else 'walk'
            if current_mode != prev_mode:
                network.robot_parameters.set_phase_bias(sp)
                prev_mode = current_mode
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

    drive_val = drive if not hasattr(drive, '__len__') else 'ramp'
    label = f'{"B" if update else "A"}_network_dynamics_d{drive_val}'
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, num=label)
    part = 'Part B' if update else 'Part A'
    fig.suptitle(f'{part} - CPG Network Dynamics (d={drive_val}, duration={duration}s)')

    # A: body left oscillator activations (waterfall, head at top)
    body_offset = 1.2
    for k, idx in enumerate(reversed(osc_left)):
        axes[0].plot(times, outputs[:, idx] + k * body_offset, color='steelblue', linewidth=0.8)
    axes[0].set_ylabel('Body (L)\nactivation')
    axes[0].set_yticks([])

    # B: limb activations — first oscillator of each limb
    limb_offset = 2.5
    limb_indices = list(osc_legs[::4])
    for k, idx in enumerate(limb_indices):
        axes[1].plot(times, outputs[:, idx] + k * limb_offset, color='darkorange', linewidth=0.8)
    axes[1].set_ylabel('Limb\nactivation')
    axes[1].set_yticks([])
    axes[1].set_ylim([-0.5, limb_offset * (len(limb_indices) - 1) + 3.0])

    # C: instantaneous frequency of all oscillators
    for idx in osc_left:
        axes[2].plot(times[1:], inst_freq[:, idx], color='steelblue', linewidth=0.6, alpha=0.7)
    for idx in osc_right:
        axes[2].plot(times[1:], inst_freq[:, idx], color='cornflowerblue', linewidth=0.6, alpha=0.5)
    for idx in osc_legs:
        axes[2].plot(times[1:], inst_freq[:, idx], color='darkorange', linewidth=0.6, alpha=0.7)
    axes[2].set_ylabel('Freq [Hz]')
    axes[2].set_ylim([0, 2])

    # D: drive signal
    axes[3].plot(times, drive_plot, color='red')
    axes[3].set_ylabel('Drive d')
    axes[3].set_xlabel('Time [s]')

    plt.tight_layout()

    # Oscillator properties figure (Ijspeert Fig. 5 equivalent) — Part B only
    if update:
        bod = BODY
        leg = LIMB

        # Analytical curves: ν(d) and R(d)
        d_vals = np.linspace(0, 6, 300)
        body_freq_d = np.where(
            (d_vals > bod['dlow']) & (d_vals < bod['dhigh']),
            bod['cv1']*d_vals + bod['cv0'], 0.0)
        limb_freq_d = np.where(
            (d_vals > leg['dlow']) & (d_vals < leg['dhigh']),
            leg['cv1']*d_vals + leg['cv0'], 0.0)
        body_amp_d = np.where(
            (d_vals > bod['dlow']) & (d_vals < bod['dhigh']),
            bod['cr1']*d_vals + bod['cr0'], 0.0)
        limb_amp_d = np.where(
            (d_vals > leg['dlow']) & (d_vals < leg['dhigh']),
            leg['cr1']*d_vals + leg['cr0'], 0.0)

        # Time-series: one body and one limb oscillator
        b_idx, l_idx = osc_left[0], osc_legs[0]
        body_freq_t = freqs_log[:, b_idx] / (2*np.pi)
        limb_freq_t = freqs_log[:, l_idx] / (2*np.pi)
        body_amp_t = amplitudes_log[:, b_idx]
        limb_amp_t = amplitudes_log[:, l_idx]
        body_out_t = outputs[:, b_idx]
        limb_out_t = outputs[:, l_idx]

        fig2 = plt.figure(figsize=(18, 8), num='B_osc_properties')
        fig2.suptitle('Part B - Oscillator Properties')
        gs = fig2.add_gridspec(4, 2, hspace=0.55, wspace=0.35)
        ax_A = fig2.add_subplot(gs[0:2, 0])
        ax_B = fig2.add_subplot(gs[2:4, 0])
        ax_C = fig2.add_subplot(gs[0, 1])
        ax_D = fig2.add_subplot(gs[1, 1])
        ax_E = fig2.add_subplot(gs[2, 1])
        ax_F = fig2.add_subplot(gs[3, 1])

        ax_A.plot(d_vals, body_freq_d, 'k-', label='Body')
        ax_A.plot(d_vals, limb_freq_d, 'k--', label='Limb')
        ax_A.set_ylabel('ν [Hz]')
        ax_A.set_xlabel('drive')
        ax_A.legend(fontsize=8)

        ax_B.plot(d_vals, body_amp_d, 'k-', label='Body')
        ax_B.plot(d_vals, limb_amp_d, 'k--', label='Limb')
        ax_B.set_ylabel('R')
        ax_B.set_xlabel('drive')
        ax_B.legend(fontsize=8)

        c_offset = max(limb_out_t.max() - limb_out_t.min(), 0.5)
        ax_C.plot(times, body_out_t + c_offset, 'k-', linewidth=0.5, label='Body')
        ax_C.plot(times, limb_out_t, 'k--', linewidth=0.5, label='Limb')
        ax_C.set_ylabel('x')
        ax_C.set_yticks([])
        ax_C.legend(fontsize=7)

        ax_D.plot(times, body_freq_t, 'k-')
        ax_D.plot(times, limb_freq_t, 'k--')
        ax_D.set_ylabel('Freq [Hz]')

        ax_E.plot(times, body_amp_t, 'k-')
        ax_E.plot(times, limb_amp_t, 'k--')
        ax_E.set_ylabel('r')

        ax_F.plot(times, drive_plot, 'k-')
        ax_F.axhline(y=bod['dhigh'], color='k', linestyle=':', linewidth=0.8)
        ax_F.axhline(y=leg['dhigh'], color='k', linestyle=':', linewidth=0.8)
        ax_F.set_ylabel('d (drive)')
        ax_F.set_xlabel('Time [s]')

    return


def exercise_1a_networks(plot, timestep=1e-2):
    """[Project 1] Exercise 1: """

    # Exercise 1A: fixed drive — walking regime (d=2) then swimming regime (d=4)
    run_network(duration=10, update=False, drive=2.0, timestep=timestep)
    run_network(duration=10, update=False, drive=4.0, timestep=timestep)

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
    exercise_1a_networks(plot= save_plots())


