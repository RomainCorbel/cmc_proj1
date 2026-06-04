"""Plot results"""

import os
import pickle
import numpy as np
from requests import head
from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from salamandra_simulation.data import SalamandraData
from simulation_parameters import SimulationParameters
from farms_amphibious.data.data import AmphibiousExperimentData

from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from network import motor_output
import matplotlib.colors as colors


def load_data(
        log_files: str,
        simulation_i: int,
) -> tuple[AmphibiousExperimentData, SimulationParameters]:
    """Load data"""
    experiment_data_file = os.path.join(
        log_files.format(simulation_i),
        'simulation.hdf5',
    )
    exp_data = AmphibiousExperimentData.from_file(experiment_data_file)
    sim_parameters_file = os.path.join(
        log_files.format(simulation_i),
        'sim_parameters.pickle',
    )
    with open(sim_parameters_file, 'rb') as param_file:
        parameters = pickle.load(param_file)
    return exp_data, parameters


def plot_positions(times, link_data, title):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=['x', 'y', 'z'][i])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.grid(True)
    plt.title(f"Positions - {title}")


def plot_trajectory(link_data, title, label=None, color=None):
    """Plot trajectory"""
    plt.plot(link_data[:, 0], link_data[:, 1], label=label, color=color)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.grid(True)
    plt.title(f'Trajectory - {title}')


def plot_2d(results, labels, title, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xs = sorted(np.unique(results[:, 0]))
    ys = sorted(np.unique(results[:, 1]))

    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}

    grid = np.full((len(ys), len(xs)), np.nan)
    for x, y, z in results:
        grid[y_idx[y], x_idx[x]] = z

    dx = (xs[1] - xs[0]) / 2 if len(xs) > 1 else 0.5
    dy = (ys[1] - ys[0]) / 2 if len(ys) > 1 else 0.5

    imgplot = plt.imshow(
        grid,
        extent=(min(xs) - dx, max(xs) + dx, min(ys) - dy, max(ys) + dy),
        aspect='auto',
        origin='lower',
        interpolation='none',
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.xticks(xs)
    plt.yticks(ys)
    cbar = plt.colorbar()
    cbar.set_label(labels[2])
    plt.title(f'{labels[2]} - {title}')


def max_distance(link_data, nsteps_considered=None):
    """Compute max distance"""
    if not nsteps_considered:
        nsteps_considered = link_data.shape[0]
    com = np.mean(link_data[-nsteps_considered:], axis=1)
    return np.sqrt(
        np.max(np.sum((link_data[:, :]-link_data[0, :])**2, axis=1)))


def compute_speed(links_positions, links_vel, nsteps_considered=200):
    """
    Computes the axial and lateral speed based on the PCA of the links positions
    """

    links_pos_xy = links_positions[-nsteps_considered:, :, :2]
    joints_vel_xy = links_vel[-nsteps_considered:, :, :2]
    time_idx = links_pos_xy.shape[0]

    speed_forward = []
    speed_lateral = []
    com_pos = []

    for idx in range(time_idx):
        x = links_pos_xy[idx, :9, 0]
        y = links_pos_xy[idx, :9, 1]

        pheadtail = links_pos_xy[idx][0]-links_pos_xy[idx][8]  # head - tail
        pcom_xy = np.mean(links_pos_xy[idx, :9, :], axis=0)
        vcom_xy = np.mean(joints_vel_xy[idx], axis=0)

        covmat = np.cov([x, y])
        eig_values, eig_vecs = np.linalg.eig(covmat)
        largest_index = np.argmax(eig_values)
        largest_eig_vec = eig_vecs[:, largest_index]

        ht_direction = np.sign(np.dot(pheadtail, largest_eig_vec))
        largest_eig_vec = ht_direction * largest_eig_vec

        v_com_forward_proj = np.dot(vcom_xy, largest_eig_vec)

        left_pointing_vec = np.cross(
            [0, 0, 1],
            [largest_eig_vec[0], largest_eig_vec[1], 0]
        )[:2]

        v_com_lateral_proj = np.dot(vcom_xy, left_pointing_vec)

        com_pos.append(pcom_xy)
        speed_forward.append(v_com_forward_proj)
        speed_lateral.append(v_com_lateral_proj)

    return np.mean(speed_forward), np.mean(speed_lateral)

def compute_cot(times: np.ndarray,
                                      links_positions: np.ndarray,
                                      joints_torques: np.ndarray,
                                      joints_velocities: np.ndarray,
                                      links_masses
                                      ):
    """
    Compute sum of energy consumptions and CoT.
    Hint:
    Only take POSITIVE values during energy consumption (no energy storing of the active part)
    Compute the integration of traveled distance for the CoM of the robot (useful varibles: LINKS_MASSES, TOTAL_MASS)
    """
    dt = times[1] - times[0]

    power_positive = np.maximum(joints_torques * joints_velocities, 0)
    energy = dt * np.sum(power_positive)

    com_pos = np.average(links_positions, axis=1, weights=links_masses)
    d_fwd = np.linalg.norm(com_pos[-1] - com_pos[0])

    deltas = np.diff(com_pos, axis=0)  # shape: (n_frames-1, 3)
    step_distances = np.linalg.norm(deltas, axis=1)  # shape: (n_frames-1,)
    total_distance = np.sum(step_distances)


    cot = energy / total_distance if total_distance > 1e-9 else np.nan

    return cot


def sum_torques(joints_data):
    """Compute sum of torques

    Example:

    joints_data = data.sensors.joints.motor_torques_all()

    """
    return np.sum(np.abs(joints_data[:, :]))


def plot_simulation_results(times, osc_phases, osc_amplitudes,
                            joints_positions, tail_positions, title):
    """Plot simulation results: spine and limb oscillator activations."""
    # Pre-compute motor outputs: x_i = r_i * (1 + cos(phi_i))
    osc_phases_arr = np.asarray(osc_phases)
    osc_amp_arr = np.asarray(osc_amplitudes)
    outputs = osc_amp_arr * (1 + np.cos(osc_phases_arr))  # [n_iter, n_osc]
    n_osc = outputs.shape[1]
    # Body oscillators: first 16 (or all if fewer); limb oscillators: remainder
    n_body = min(16, n_osc)
    n_limb = n_osc - n_body

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f'Oscillator Activations - {title}', fontsize=14, fontweight='bold')

    osc_left = np.arange(0, 16, 2)
    osc_right = np.arange(1, 16, 2)
    osc_legs = np.arange(16, 32)

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

    fig.tight_layout()
    return fig

    # --- Commented-out subplots (kept for reference) ---
    # # Oscillator phases over time
    # ax.plot(times, osc_phases_arr[:, i], lw=0.8)
    # ax.set_ylabel('Phase [rad]')
    # ax.set_title('Oscillator phases')

    # # Oscillator amplitudes over time
    # ax.plot(times, osc_amp_arr[:, i], lw=0.8)
    # ax.set_ylabel('Amplitude')
    # ax.set_title('Oscillator amplitudes')

    # # Joint positions over time
    # ax.plot(times, joints_pos_arr[:, i], lw=0.8)
    # ax.set_ylabel('Position [rad]')
    # ax.set_title('Joint positions')

    # # Tail positions (x, y, z) over time
    # ax.plot(times, tail_pos_arr[:, i], lw=0.8, label=label)
    # ax.set_ylabel('Position [m]')
    # ax.set_title('Tail positions')

def main_sweep(file_format, sim_is, title, case=3):
    cots = []
    fwd_speeds = []
    lat_speeds = []


    for i in sim_is:
        exp_data, parameters = load_data(file_format, i)
        data = exp_data.animats[0]
        timestep = exp_data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        amplitudes = getattr(parameters, 'amplitudes', None)
        phase_lag_body = getattr(parameters, 'phase_lag_body', None)
        drive = getattr(parameters, 'drive', None)
        axial_amp_gain = getattr(parameters, 'axial_amp_gain', None)
        limb_amp_gain = getattr(parameters, 'limb_amp_gain', None)
        osc_phases = data.state.phases_all()
        osc_amplitudes = data.state.amplitudes_all()
        links_positions = data.sensors.links.urdf_positions()
        links_velocities = data.sensors.links.com_lin_velocities()
        links_masses = data.sensors.links.masses
        #print([m for m in dir(data.sensors.links)])
        # See data.sensors.links.names for finding corresponsing indices
        head_positions = links_positions[:, 0, :]
        tail_positions = links_positions[:, 8, :]
        joints_positions = data.sensors.joints.positions_all()
        joints_velocities = data.sensors.joints.velocities_all()
        joints_torques = data.sensors.joints.motor_torques_all()

        links_positions = np.asarray(links_positions)
        links_velocities = np.asarray(links_velocities)
        joints_torques = np.asarray(joints_torques)
        joints_velocities = np.asarray(joints_velocities)
        links_masses = np.asarray(links_masses)

        v_fwd, v_lat = compute_speed(links_positions, links_velocities, len(links_positions))
        cot = compute_cot(times, links_positions, joints_torques, joints_velocities, links_masses)
        print(cot, v_fwd, v_lat)
        
        if case == 3:
            print(drive, phase_lag_body, cot, v_fwd, v_lat)
            cots.append([drive, phase_lag_body, cot])
            fwd_speeds.append([drive, phase_lag_body, v_fwd])
            lat_speeds.append([drive, phase_lag_body, v_lat])
        elif case == 4:
            print(axial_amp_gain, limb_amp_gain, cot, v_fwd, v_lat)
            cots.append([axial_amp_gain, limb_amp_gain, cot])
            fwd_speeds.append([axial_amp_gain, limb_amp_gain, v_fwd])
            lat_speeds.append([axial_amp_gain, limb_amp_gain, v_lat])

    cots = np.array(cots)
    fwd_speeds = np.array(fwd_speeds)
    lat_speeds = np.array(lat_speeds)
    
    if case == 3:
        plot_2d(cots, ['Drive', 'Phase Lag', 'Cost of Transport [J/m]'], title)
        plt.figure()
        plot_2d(fwd_speeds, ['Drive', 'Phase Lag', 'Forward Speed [m/s]'], title)
        plt.show()
    elif case == 4:
        plot_2d(cots, ['Axial Amplitude Gain', 'Limb Amplitude Gain', 'Cost of Transport [J/m]'], title)
        plt.figure()
        plot_2d(fwd_speeds, ['Axial Amplitude Gain', 'Limb Amplitude Gain', 'Forward Speed [m/s]'], title)
        plt.show()



def main_single(log_files, title, plot=True):
    """Main"""
    simulation_i = 0
    exp_data, parameters = load_data(log_files, simulation_i)
    data = exp_data.animats[0]
    timestep = exp_data.timestep
    n_iterations = np.shape(data.sensors.links.array)[0]
    times = np.arange(
        start=0,
        stop=timestep*n_iterations,
        step=timestep,
    )
    timestep = times[1] - times[0]
    amplitudes = getattr(parameters, 'amplitudes', None)
    phase_lag_body = getattr(parameters, 'phase_lag_body', None)
    osc_phases = data.state.phases_all()
    osc_amplitudes = data.state.amplitudes_all()
    links_positions = data.sensors.links.urdf_positions()
    links_velocities = data.sensors.links.com_lin_velocities()
    links_masses = data.sensors.links.masses
    #print([m for m in dir(data.sensors.links)])
    # See data.sensors.links.names for finding corresponsing indices
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 8, :]
    joints_positions = data.sensors.joints.positions_all()
    joints_velocities = data.sensors.joints.velocities_all()
    joints_torques = data.sensors.joints.motor_torques_all()

    # Notes:
    # For the links arrays: positions[iteration, link_id, xyz]
    # For the positions arrays: positions[iteration, xyz]
    # For the joints arrays: positions[iteration, joint]

    # Plot data
    head_positions = np.asarray(head_positions)
    plt.figure('Positions')
    plot_positions(times, head_positions, title)
    plt.figure('Trajectory')
    plot_trajectory(head_positions, title)

    # Plot simulation results subplot
    plot_simulation_results(
        times=times,
        osc_phases=osc_phases,
        osc_amplitudes=osc_amplitudes,
        joints_positions=joints_positions,
        tail_positions=tail_positions,
        title=title
    )
    # print(np.asarray(data.sensors.links.array[637, 0, :]))

    # print(np.asarray(data.sensors.links.com_lin_velocities()[637, 0, :]))   # 14 15 16
    # print(np.asarray(data.sensors.links.com_positions()[637, 0, :]))        # 0 1 2    
    # print(np.asarray(data.sensors.links.urdf_positions()[637, 0, :]))       # 7 8 9
    # print(np.asarray(data.sensors.links.urdf_orientations()[637, 0, :]))    # 11 12 13 14
    # -> use URDF positions and com_lin velocities
    links_positions = np.asarray(links_positions)
    links_velocities = np.asarray(links_velocities)
    joints_torques = np.asarray(joints_torques)
    joints_velocities = np.asarray(joints_velocities)
    links_masses = np.asarray(links_masses)

    v_fwd, v_lat = compute_speed(links_positions, links_velocities, len(links_positions))
    cot = compute_cot(times, links_positions, joints_torques, joints_velocities, links_masses)
    print(f'Forward speed: {round(v_fwd, 3)}, Lateral speed: {round(v_lat, 3)}, CoT: {round(cot, 3)}')

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main_single(plot=save_plots())