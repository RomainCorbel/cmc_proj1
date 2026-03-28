import matplotlib.pyplot as plt
import numpy as np
from cmc_controllers.metrics import LINKS_MASSES

def plot_results_EXO1_1(sim_times, freq, sensor_data_joints_positions, sensor_data_links_positions, base_path):

    """
    Plots joint angles over a few cycles and the CoM trajectory.
    """
    n_cycles = 6
    T = 1.0 / np.mean(freq)
    t_end = n_cycles * T
    mask = sim_times <= t_end

    # Joint angles
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for j in range(8):
        ax1.plot(sim_times[mask], sensor_data_joints_positions[mask, j], label=f'joint {j}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Joint angle (rad)') 
    ax1.set_title('Joint angles over a few cycles')
    ax1.legend(loc='upper right', fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(base_path + 'joint_angles.png')

    # CoM trajectory
    com_xy = np.average(sensor_data_links_positions[:, :, :2], axis=1, weights=LINKS_MASSES)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(com_xy[:, 0], com_xy[:, 1], linewidth=1.5)
    ax2.scatter(com_xy[0, 0], com_xy[0, 1], color='green', zorder=5, label='start')
    ax2.scatter(com_xy[-1, 0], com_xy[-1, 1], color='red', zorder=5, label='end')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('CoM trajectory (2D)')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(base_path + 'com_trajectory.png')

    plt.show()
    print(f'CoM displacement: {np.linalg.norm(com_xy[-1] - com_xy[0]):.4f} m')

def plot_gridsearch_heatmaps(twl_range, amp_range, get_metrics, base_path):
    """
    Plots heatmaps of forward speed, IPL, and CoT over a grid of TWL and amplitude values.
    """
    n_twl = len(twl_range)
    n_amp = len(amp_range)

    grid_speed = np.zeros((n_twl, n_amp))
    grid_ipl   = np.zeros((n_twl, n_amp))
    grid_cot   = np.zeros((n_twl, n_amp))

    for i, twl in enumerate(twl_range):
        for j, amp in enumerate(amp_range):
            val1, val2, val3 = get_metrics(twl, amp)
            grid_speed[i, j] = val1
            grid_ipl[i, j]   = val3
            grid_cot[i, j]   = val2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    grids  = [grid_speed, grid_ipl, grid_cot]
    titles = ['Forward speed (m/s)', 'IPL (rad)', 'CoT (J/m)']

    for ax, grid, title in zip(axes, grids, titles):
        im = ax.imshow(grid, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n_amp))
        ax.set_xticklabels([f'{a:.2f}' for a in amp_range], rotation=45)
        ax.set_yticks(range(n_twl))
        ax.set_yticklabels([f'{t:.2f}' for t in twl_range])
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('TWL')
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(base_path + 'gridsearch_heatmaps.png', dpi=150)
    plt.show()


def plot_results_EXO2_1(
    sim_times,
    sensor_data_joints_positions,
    sensor_data_links_positions,
    base_path,
    controller_data,
    t_max=5.0
):
    """
    Plot oscillator states, muscle outputs, joint angles, and CoM trajectory.
    Clean version with consistent time handling.
    """


    # ─────────────────────────────────────────────
    # Time handling (use simulation time as reference)
    # ─────────────────────────────────────────────
    sim_mask = sim_times <= t_max
    t_sim = sim_times[sim_mask]


    # ─────────────────────────────────────────────
    # Controller data
    # ─────────────────────────────────────────────
    state_full = controller_data['state']  # (n_iter, 3*n_osc)
    left_idx   = controller_data['indices']['left_body_idx']
    right_idx  = controller_data['indices']['right_body_idx']

    n_ctrl = state_full.shape[0]
    n_osc  = state_full.shape[1] // 3
    n_joints = n_osc // 2

    # Create controller time aligned to sim duration
    t_ctrl = np.linspace(sim_times[0], sim_times[-1], n_ctrl)
    ctrl_mask = t_ctrl <= t_max
    t_ctrl = t_ctrl[ctrl_mask]

    state = state_full[ctrl_mask]

    phases     = state[:, :n_osc]
    amplitudes = state[:, n_osc:2*n_osc]
    motor      = state[:, 2*n_osc:]

    ML = motor[:, left_idx]
    MR = motor[:, right_idx]

    M_sum  = ML + MR
    M_diff = ML - MR

    # ─────────────────────────────────────────────
    # FIG 1: Oscillator states
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for i in range(n_osc):
        axes[0].plot(t_ctrl, phases[:, i], alpha=0.7)
    axes[0].set_ylabel('Phase θ [rad]')
    axes[0].set_title('Oscillator Phases')

    for i in range(n_osc):
        axes[1].plot(t_ctrl, amplitudes[:, i], alpha=0.7)
    axes[1].set_ylabel('Amplitude r')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_title('Oscillator Amplitudes')

    plt.tight_layout()
    plt.savefig(base_path + 'oscillator_states.png', dpi=150)

    # ─────────────────────────────────────────────
    # FIG 2: Muscle outputs
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for i in range(n_joints):
        axes[0].plot(t_ctrl, M_sum[:, i], alpha=0.7)
    axes[0].set_ylabel('ML + MR')
    axes[0].set_title('Muscle Output Sum')

    for i in range(n_joints):
        axes[1].plot(t_ctrl, M_diff[:, i], alpha=0.7)
    axes[1].set_ylabel('ML - MR')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_title('Muscle Output Difference')

    plt.tight_layout()
    plt.savefig(base_path + 'muscle_outputs.png', dpi=150)

    # ─────────────────────────────────────────────
    # FIG 3: Joint angles (simulation)
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))

    for j in range(8):
        ax.plot(t_sim, sensor_data_joints_positions[sim_mask, j], label=f'joint {j}', alpha=0.7)

    ax.set_ylabel('Joint angle [rad]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Joint Angles (Simulation)')
    ax.legend(loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(base_path + 'joint_angles.png', dpi=150)

    # ─────────────────────────────────────────────
    # FIG 4: CoM trajectory
    # ─────────────────────────────────────────────
    com_xy = np.average(
        sensor_data_links_positions[:, :, :2],
        axis=1,
        weights=LINKS_MASSES
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(com_xy[:, 0], com_xy[:, 1], linewidth=1.5)
    ax.scatter(com_xy[0, 0], com_xy[0, 1], color='green', label='start')
    ax.scatter(com_xy[-1, 0], com_xy[-1, 1], color='red', label='end')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('CoM Trajectory (2D)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(base_path + 'com_trajectory.png', dpi=150)

    # ─────────────────────────────────────────────
    # Final
    # ─────────────────────────────────────────────
    plt.show()

    displacement = np.linalg.norm(com_xy[-1] - com_xy[0])
    print(f'CoM displacement: {displacement:.4f} m')