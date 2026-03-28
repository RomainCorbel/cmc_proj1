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