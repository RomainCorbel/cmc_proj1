from matplotlib.pylab import norm
import matplotlib.pyplot as plt
import numpy as np
from cmc_controllers.metrics import LINKS_MASSES
from matplotlib.colors import SymLogNorm
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
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('CoM trajectory (2D)')
    plt.tight_layout()
    plt.savefig(base_path + 'com_trajectory.png')
    plt.show()
    print(f'CoM displacement: {np.linalg.norm(com_xy[-1] - com_xy[0]):.4f} m')

def plot_gridsearch_heatmaps(twl_range, amp_range, get_metrics, base_path):
    n_twl = len(twl_range)
    n_amp = len(amp_range)

    # 1. Initialize grids as (Rows=Amplitude, Cols=TWL)
    grid_speed = np.zeros((n_amp, n_twl))
    grid_ipl   = np.zeros((n_amp, n_twl))
    grid_cot   = np.zeros((n_amp, n_twl))

    for i, twl in enumerate(twl_range):
        for j, amp in enumerate(amp_range):
            val1, val2, val3 = get_metrics(twl, amp)
            # 2. Map j (amplitude) to the first dimension (Y-axis)
            # and i (twl) to the second dimension (X-axis)
            grid_speed[j, i] = val1
            grid_ipl[j, i]   = val3
            grid_cot[j, i]   = val2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    grids  = [grid_speed, grid_ipl, grid_cot]
    titles = ['Forward speed (m/s)', 'IPL (rad)', 'CoT (J/m)']

    for ax, grid, title in zip(axes, grids, titles):
        # 1. Normalisation et affichage de la Heatmap
        norm = SymLogNorm(linthresh=0.01, vmin=grid.min(), vmax=grid.max())
        im = ax.imshow(grid, norm=norm, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax)
        
        # --- 2. BOUCLE POUR AFFICHER LES VALEURS ---
        for i in range(n_amp):    # Axe Y
            for j in range(n_twl): # Axe X
                val = grid[i, j]
                
                # Choix de la couleur du texte pour la lisibilité
                # Si la valeur est élevée (couleur claire dans viridis), on écrit en noir
                # Si elle est basse (couleur sombre), on écrit en blanc
                color_text = "white" if val < (grid.max() * 0.5) else "black"
                
                ax.text(j, i, f'{val:.2f}', 
                        ha="center", va="center", 
                        color=color_text, fontsize=8)
        
        # 3. Configuration des Ticks et Labels
        ax.set_yticks(range(n_amp))
        ax.set_yticklabels([f'{a:.2f}' for a in amp_range])
        
        ax.set_xticks(range(n_twl))
        ax.set_xticklabels([f'{t:.2f}' for t in twl_range], rotation=45)
        
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('TWL')
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(base_path + 'gridsearch_heatmaps.png', dpi=150)
    plt.show()

def plot_drive_pl_heatmaps(drive_range, pl_vals, get_metrics, base_path):
    """
    Plots heatmaps of forward speed and CoT over a grid of drive and phase lag values.
    get_metrics(drive, pl) must return (speed_forward, cot).
    """
    n_drive = len(drive_range)
    n_pl = len(pl_vals)

    grid_speed = np.full((n_drive, n_pl), np.nan)
    grid_cot   = np.full((n_drive, n_pl), np.nan)

    for i, drive in enumerate(drive_range):
        for j, pl in enumerate(pl_vals):
            speed, cot = get_metrics(drive, pl)
            grid_speed[i, j] = speed
            grid_cot[i, j]   = cot

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    grids  = [grid_speed, grid_cot]
    titles = ['Forward speed (m/s)', 'CoT (J/m)']

    for ax, grid, title in zip(axes, grids, titles):
        im = ax.imshow(grid, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n_pl))
        ax.set_xticklabels([f'{p:.2f}' for p in pl_vals], rotation=45)
        ax.set_yticks(range(n_drive))
        ax.set_yticklabels([f'{d:.2f}' for d in drive_range])
        ax.set_xlabel('Phase lag per joint (rad)')
        ax.set_ylabel('Drive')
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(base_path + 'drive_pl_heatmaps.png', dpi=150)
    plt.show()


def plot_diff_drive_heatmaps(drive_vals, get_metrics, base_path):
    """
    Plots a heatmap of trajectory curvature over a grid of drive_left × drive_right values.
    get_metrics(drive_left, drive_right) must return curvature (scalar).
    """
    n = len(drive_vals)

    grid_curvature = np.full((n, n), np.nan)

    for i, dl in enumerate(drive_vals):
        for j, dr in enumerate(drive_vals):
            grid_curvature[i, j] = get_metrics(dl, dr)

    labels = [f'{v:.2f}' for v in drive_vals]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(grid_curvature, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Drive right')
    ax.set_ylabel('Drive left')
    ax.set_title('Trajectory curvature (1/m)')

    plt.tight_layout()
    plt.savefig(base_path + 'diff_drive_heatmaps.png', dpi=150)
    plt.show()
# =======
def plot_results_EXO2_3(f_animal, a_animal, ipl_animal, ipl_animal_mean, f_robot, a_robot, ipl_robot, ipl_robot_mean, base_path):
    """
    Plots per-joint bar comparisons for Frequency, Amplitude, and Phase Lag.
    """
    n_joints = len(a_animal)
    joints_labels = [f"J{i}" for i in range(n_joints)]
    couples_labels = [f"J{i}-{i+1}" for i in range(n_joints - 1)]

    # --- Configuration des couleurs (Viridis) ---
    cmap = plt.get_cmap('viridis')
    color_animal = cmap(0.3)  # Greenish
    color_robot  = cmap(0.7)  # Purplish
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    width = 0.35
    x_joints = np.arange(n_joints)
    x_couples = np.arange(n_joints - 1)

    # --- 1. Frequency Comparison  ---
    x_freq = np.arange(1) 
    
    # On calcule la moyenne globale si ce n'est pas déjà fait
    mean_f_animal = np.mean(f_animal)
    mean_f_robot  = np.mean(f_robot)

    axes[0].bar(0 - width/2, mean_f_animal, width, label='Animal', 
                color=color_animal, edgecolor='white', linewidth=0.5)
    axes[0].bar(0 + width/2, mean_f_robot, width, label='Controller', 
                color=color_robot, edgecolor='white', linewidth=0.5)
    
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('Global Average Frequency')
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(['All Joints']) # Ou laisser vide
    axes[0].legend()
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    
    # --- 2. Amplitude Comparison (Per Joint) ---
    axes[1].bar(x_joints - width/2, a_animal, width, label='Animal', 
                color=color_animal, edgecolor='white', linewidth=0.5)
    axes[1].bar(x_joints + width/2, a_robot, width, label='Controller', 
                color=color_robot, edgecolor='white', linewidth=0.5)
    
    axes[1].set_ylabel('Amplitude (rad)')
    axes[1].set_title('Amplitude per Joint')
    axes[1].set_xticks(x_joints)
    axes[1].legend()
    axes[1].set_xticklabels(joints_labels)
    axes[1].grid(axis='y', linestyle=':', alpha=0.6)

    # --- 3. Phase Lag Comparison (Per Couple) ---
    axes[2].bar(x_couples - width/2, ipl_animal, width, label='Animal', 
                color=color_animal, edgecolor='white', linewidth=0.5)
    axes[2].bar(x_couples + width/2, ipl_robot, width, label='Controller', 
                color=color_robot, edgecolor='white', linewidth=0.5)
    axes[2].axhline(y=ipl_animal_mean, color=color_animal, linestyle='--', 
                    linewidth=1.5, alpha=0.8, label='Mean Animal')
    axes[2].axhline(y=ipl_robot_mean, color=color_robot, linestyle='--', 
                    linewidth=1.5, alpha=0.8, label='Mean Controller')
    axes[2].text(1.01, ipl_animal_mean, f'{ipl_animal_mean:.3f}', 
                    color=color_animal, va='center', fontweight='bold',
                    transform=axes[2].get_yaxis_transform())
    axes[2].text(1.01, ipl_robot_mean, f'{ipl_robot_mean:.3f}', 
                 color=color_robot, va='center', fontweight='bold',
                 transform=axes[2].get_yaxis_transform())
    axes[2].set_ylabel('Phase Lag (rad)')
    axes[2].set_title('Inter-joint Phase Lags')
    axes[2].set_xticks(x_couples)
    axes[2].legend()
    axes[2].set_xticklabels(couples_labels)
    axes[2].grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(base_path + 'animal_controller_comparison.png', dpi=150)
    plt.show()

def plot_kinematics_comparison(animal_times, animal_joints, robot_times, robot_joints, base_path):
    """
    Compare animal and robot joint kinematics over one normalized cycle.
    Both arrays: shape (T, 8), angles in radians.
    Returns per-joint RMSE array.
    """
    from scipy.interpolate import interp1d

    n_pts = len(animal_times)
    n_joints = 8

    # Resample robot onto animal's time grid (normalized to [0,1])
    t_norm = np.linspace(0, 1, n_pts)
    t_robot_norm = np.linspace(0, 1, len(robot_times))
    robot_rs = np.zeros((n_pts, n_joints))
    for j in range(n_joints):
        robot_rs[:, j] = interp1d(t_robot_norm, robot_joints[:, j], kind='linear')(t_norm)

    # Remove DC offset from both (compare shape, not absolute angle)
    animal_c = animal_joints - animal_joints.mean(axis=0)
    robot_c  = robot_rs     - robot_rs.mean(axis=0)

    # Phase-align via cross-correlation on joint 0
    xcorr = np.correlate(animal_c[:, 0], robot_c[:, 0], mode='full')
    shift = np.argmax(xcorr) - (n_pts - 1)
    robot_aligned = np.roll(robot_c, shift, axis=0)

    rmse = np.sqrt(np.mean((animal_c - robot_aligned) ** 2, axis=0))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Animal
    ax = axes[0, 0]
    for j in range(n_joints):
        ax.plot(t_norm, animal_c[:, j], label=f'J{j}')
    ax.set_title('Animal joint angles (mean-removed)')
    ax.set_xlabel('Normalized cycle')
    ax.set_ylabel('Angle (rad)')
    ax.legend(fontsize=7, ncol=2)

    # Robot
    ax = axes[0, 1]
    for j in range(n_joints):
        ax.plot(t_norm, robot_aligned[:, j], label=f'J{j}')
    ax.set_title('Robot joint angles (best imitation, phase-aligned)')
    ax.set_xlabel('Normalized cycle')
    ax.set_ylabel('Angle (rad)')
    ax.legend(fontsize=7, ncol=2)

    # Overlay first 4 joints
    ax = axes[1, 0]
    colors4 = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
    for j in range(4):
        ax.plot(t_norm, animal_c[:, j],   '--', color=colors4[j], label=f'Animal J{j}')
        ax.plot(t_norm, robot_aligned[:, j], '-', color=colors4[j], label=f'Robot J{j}')
    ax.set_title('Overlay joints 0–3  (-- animal, — robot)')
    ax.set_xlabel('Normalized cycle')
    ax.set_ylabel('Angle (rad)')
    ax.legend(fontsize=6, ncol=2)

    # RMSE bar
    ax = axes[1, 1]
    ax.bar(range(n_joints), rmse, color='steelblue')
    ax.axhline(np.mean(rmse), color='red', linestyle='--', label=f'Mean = {np.mean(rmse):.4f} rad')
    ax.set_xticks(range(n_joints))
    ax.set_xticklabels([f'J{j}' for j in range(n_joints)])
    ax.set_ylabel('RMSE (rad)')
    ax.set_title('Per-joint RMSE (animal vs robot)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(base_path + 'kinematics_comparison.png', dpi=150)
    plt.show()

    return rmse


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

    for i in range(n_joints):
        axes[0].plot(t_ctrl, phases[:, i], alpha=0.7, label=f'Joint {i}')
    axes[0].legend(loc='lower right', fontsize=7, ncol=2)
    axes[0].set_ylabel('Phase θ [rad]')
    axes[0].set_title('Oscillator Phases')

    for i in range(n_joints):
        axes[1].plot(t_ctrl, amplitudes[:, i], alpha=0.7, label=f'Joint {i}')
    axes[1].set_ylabel('Amplitude r')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend(loc='lower right', fontsize=7, ncol=2)
    axes[1].set_title('Oscillator Amplitudes')

    plt.tight_layout()
    plt.savefig(base_path + 'oscillator_states.png', dpi=150)

    # ─────────────────────────────────────────────
    # FIG 2: Muscle outputs
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for i in range(n_joints):
        axes[0].plot(t_ctrl, M_sum[:, i], alpha=0.7, label=f'Joint {i}')
    axes[0].set_ylabel('ML + MR')
    axes[0].legend(loc='lower right', fontsize=7, ncol=2)
    axes[0].set_title('Muscle Output Sum')

    for i in range(n_joints):
        axes[1].plot(t_ctrl, M_diff[:, i], alpha=0.7, label=f'Joint {i}')
    axes[1].set_ylabel('ML - MR')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend(loc='lower right', fontsize=7, ncol=2)
    axes[1].set_title('Muscle Output Difference')

    plt.tight_layout()
    plt.savefig(base_path + 'muscle_outputs.png', dpi=150)

    # ─────────────────────────────────────────────
    # FIG 3: Joint angles (simulation)
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))

    for j in range(8):
        # 1. Generate the color directly from the string 'YlOrRd'
        color = plt.get_cmap('viridis')(j / 7) 
        
        # 2. Add the 0.2 offset and the color
        ax.plot(t_sim, sensor_data_joints_positions[sim_mask, j] + (j * 0.2), 
                label=f'joint {j}', color=color, alpha=0.9)

    ax.set_ylabel('Joint angle [rad]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Joint Angles (Simulation)')
    ax.legend(loc='upper left', fontsize=7, ncol=2)

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

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('CoM Trajectory (2D)')

    plt.tight_layout()
    plt.savefig(base_path + 'com_trajectory.png', dpi=150)

    # ─────────────────────────────────────────────
    # Final
    # ─────────────────────────────────────────────
    plt.show()

    displacement = np.linalg.norm(com_xy[-1] - com_xy[0])
    print(f'CoM displacement: {displacement:.4f} m')
