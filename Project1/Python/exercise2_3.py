#!/usr/bin/env python3

import os
import h5py
import matplotlib.pyplot as plt

from cmc_controllers import CPG_controller
from cmc_controllers.plot_utils import plot_gridsearch_heatmaps
from exercise1_2 import get_metrics
from farms_core import pylog

from cmc_controllers.metrics import *
from simulate import run_multiple, runsim


BASE_PATH = 'logs/exercise2_3/'
PLOT_PATH = 'results'
ANIMAL_DATA_PATH = 'cmc_project_pack/models/a2sw5_cycle_smoothed.csv'

def get_animal_data(path):
    """Extract metrics from animal data and apply dynamical scaling."""
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    times = data[:, 0]
    joint_angles = np.deg2rad(data[:, 1:9]) 
    
    # 1. Extract raw animal metrics
    freqs, amplitudes = compute_mechanical_frequency_amplitude_fft(times, joint_angles)
    f_animal = np.mean(freqs)
    amp_animal = np.mean(amplitudes)

    # 2. Extract Phase Lag (IPL)
    inds_couples = [[i, i+1] for i in range(7)]
    _, ipls_animal = compute_neural_phase_lags(
        times=times,
        smooth_signals=joint_angles,
        freqs=freqs,
        inds_couples=inds_couples
    )

    # 3. Dynamical Scaling (Project Ref [4])
    # Frequency scales by sqrt(1/6.5)
    freq_robot_scaled = np.sqrt(1 / 6.5) * f_animal
    
    return freq_robot_scaled, amp_animal, ipls_animal

# def find_best_imitation(target_f, target_ipl, target_amp):
#     def plot_error_heatmap(error_matrix, drives, phase_lags):
#         plt.figure(figsize=(10, 8))
#         # Use seaborn for a clean look
#         import seaborn as sns
#         ax = sns.heatmap(
#             error_matrix, 
#             annot=True, 
#             fmt=".3f", 
#             xticklabels=np.round(phase_lags, 3), 
#             yticklabels=np.round(drives, 2),
#             cmap="YlGnBu_r" # Reverse color map so low error is blue/dark
#         )
        
#         plt.title("Imitation Error Heatmap")
#         plt.xlabel("Phase Lag (PL)")
#         plt.ylabel("Drive (d)")
#         plt.show()
#     n_steps = 5
#     error_matrix = np.zeros((n_steps, n_steps))
#     drives = np.linspace(2.0, 4.0, n_steps) 
#     phase_lags = np.linspace(target_ipl * 0.8, target_ipl * 1.2, n_steps)
    
#     best_error = float('inf')
#     best_params = {}
#     i = 0
#     j = 0
#     for i, d in enumerate(drives):
#         # --- TUNING AMPLITUDE (Step 3) ---
#         # Solve Equation 7 for G_amp: 
#         # G_amp = (Target_R - offset) / (drive - d_low)
#         # Note: We use target_amp as our Target_R
#         d_low = 1.0
#         c_R0 = 0.5
        
#         # Calculate the G_amp needed to hit the target amplitude at this drive
#         # We add a small max(0.01, ...) to avoid division by zero if d == d_low
#         calculated_G_amp = (target_amp - c_R0) / max(0.01, (d - d_low))
        
#         # Clip it to a reasonable range (e.g., 0.0 to 1.0) to avoid unstable gait
#         calculated_G_amp = np.clip(calculated_G_amp, 0.01, 1.0)

#         for j, pl in enumerate(phase_lags):
#             print(f"\n\n\n\n\n\n\n\n\n Testing Drive: {d:.2f}, PL: {pl:.2f}, G_amp: {calculated_G_amp:.3f} \n\n\n\n\n\n\n\n\n")
#             controller = {
#                 'loader': 'cmc_controllers.CPG_controller.CPGController',
#                 'config': {
#                     'drive_left': d,
#                     'drive_right': d,
#                     'd_low': d_low,
#                     'd_high': 5,
#                     'a_rate': np.ones(8) * 3,
#                     'offset_freq': np.ones(8) * 1,
#                     'offset_amp': np.ones(8) * c_R0,
#                     'G_freq': np.ones(8) * 0.5,
#                     'G_amp': np.ones(8) * calculated_G_amp, # Applied here
#                     'PL': np.ones(7) * pl,
#                     'coupling_weights_rostral': 5,
#                     'coupling_weights_caudal': 5,
#                     'coupling_weights_contra': 10,
#                     'init_phase': np.random.default_rng(seed=42).uniform(0, 2*np.pi, 16)
#                 }
#             }

#             # Run and evaluate
#             runsim(controller=controller, base_path=BASE_PATH)
            
#             with h5py.File(os.path.join(BASE_PATH, 'simulation.hdf5'), "r") as f:
#                 times = f['times'][:]
#                 joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:, :8, 0]
                
#                 # Use steady state (last 50%)
#                 cut = len(times) // 2
#                 f_sim_arr, a_sim_arr = compute_mechanical_frequency_amplitude_fft(times[cut:], joints[cut:])
#                 f_sim = np.mean(f_sim_arr)
#                 ipl_sim = np.mean(compute_neural_phase_lags(times[cut:], joints[cut:], f_sim_arr, [[i, i+1] for i in range(7)])[1])
#                 a_sim = np.mean(a_sim_arr)

#             # Compute error
#             error = (abs(f_sim - target_f) / target_f) + \
#                     (abs(ipl_sim - target_ipl) / target_ipl) + \
#                     (abs(a_sim - target_amp) / target_amp)
#             print(f"Drive: {d:.2f}, PL: {pl:.2f} => Error: {error:.4f} (Freq: {f_sim:.2f}, IPL: {ipl_sim:.2f}, Amp: {a_sim:.2f})")
#             error_matrix[i, j] = error
#             if error < best_error:
#                 best_error = error
#                 best_params = {
#                     'drive': d, 
#                     'pl': pl, 
#                     'G_amp': calculated_G_amp,
#                     'f_final': f_sim,
#                     'amp_final': a_sim
#                 }
#                 print(f"New Best! Error: {error:.4f} (Freq: {f_sim:.2f}, Amp: {a_sim:.2f})")
#     plot_error_heatmap(error_matrix, drives, phase_lags)
#     return best_params

# import numpy as np
# import os
# import h5py
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import h5py
# import pickle
# import re



# def find_best_imitation(target_f, target_ipl, target_amp):
#     """
#     Analyse les simulations existantes pour trouver le meilleur match avec l'animal.
#     """
#     def get_metrics(drive, pl):
#         # 1. Construction du nom du dossier (doit être identique à celui généré par run_multiple)
#         # Note : Vérifie bien si c'est 'drive_left' ou 'drive' dans le nom du dossier
#         folder_name = f"controller_drive_left{drive:.3f}_drive_right{drive:.3f}_PLarr7_{pl:.3f}"
#         folder_path = os.path.join(BASE_PATH, folder_name)
        
#         path_hdf5 = os.path.join(folder_path, 'simulation.hdf5')
#         path_pkl = os.path.join(folder_path, 'controller.pkl')

#         if not os.path.exists(path_hdf5) or not os.path.exists(path_pkl):
#             print(f"⚠️ Manquant : {folder_name}")
#             return 0.0, float('inf'), 0.0 # Vitesse nulle, CoT infini, IPL nul

#         # --- Lecture HDF5 (Mécanique) ---
#         with h5py.File(path_hdf5, "r") as f:
#             sim_times = f['times'][:]
#             # On cible l'animat 0
#             links = f['FARMSLISTanimats']['0']['sensors']['links']['array']
#             joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array']

#             # Slicing correct des données
#             links_pos = links[:, :, 7:10]   # Position XYZ
#             links_vel = links[:, :, 14:17]  # Vitesse XYZ
#             joints_vel = joints[:, :, 1]    # Vitesse angulaire
#             joints_tau = joints[:, :, 2]    # Torque (couple)

#         # Calcul des métriques de performance physique
#         speed_forward, _ = compute_mechanical_speed(
#             links_positions=links_pos,
#             links_velocities=links_vel,
#         )
#         _, cot = compute_mechanical_energy_and_cot(
#             times=sim_times,
#             links_positions=links_pos,
#             joints_torques=joints_tau,
#             joints_velocities=joints_vel,
#         )

#         # --- Lecture PKL (Neural/Contrôleur) ---
#         with open(path_pkl, "rb") as f:
#             controller_data = pickle.load(f)

#         # Extraction des signaux neuraux (différence gauche/droite pour avoir l'oscillation)
#         indices = controller_data["indices"]
#         # state[:, 2*n_osc:] correspond généralement aux sorties moteur 'm'
#         n_osc = controller_data["state"].shape[1] // 3
#         motor_outputs = controller_data["state"][:, 2*n_osc:] 
        
#         neural_signals = (
#             motor_outputs[:, indices['left_body_idx']] - 
#             motor_outputs[:, indices['right_body_idx']]
#         )

#         # Filtrage et FFT pour obtenir le Phase Lag (IPL)
#         neural_smoothed = filter_signals(times=sim_times, signals=neural_signals)
        
#         signal_freqs, _, _ = compute_frequency_amplitude_fft(
#             times=sim_times,
#             smooth_signals=neural_smoothed,
#         )
        
#         inds_couples = [[i, i + 1] for i in range(neural_smoothed.shape[1] - 1)]
        
#         _, ipls_mean = compute_neural_phase_lags(
#             times=sim_times,
#             smooth_signals=neural_smoothed,
#             freqs=signal_freqs,
#             inds_couples=inds_couples,
#         )

#         return float(speed_forward), float(cot), float(ipls_mean)
#     n_steps = 5
#     drives = np.linspace(2.0, 4.0, n_steps) 
#     phase_lags = np.linspace(target_ipl * 0.8, target_ipl * 1.2, n_steps)
    
#     # Initialisation des matrices pour les heatmaps
#     res = {k: np.zeros((n_steps, n_steps)) for k in ['error', 'f', 'amp']}
#     best_error = float('inf')
#     best_params = {}

#     # =========================================================================
#     # PARTIE SIMULATION (Commentée pour exécution post-traitement uniquement)
#     # =========================================================================
#     """
#     base_controller = {
#         'loader': 'cmc_controllers.CPG_controller.CPGController',
#         'config': {
#             'd_low': 1.0, 'd_high': 5.0, 'a_rate': np.ones(8)*3,
#             'offset_freq': np.ones(8)*1, 'offset_amp': np.ones(8)*0.5,
#             'G_freq': np.ones(8)*0.5, 'G_amp': np.ones(8)*0.25,
#             'coupling_weights_rostral': 5, 'coupling_weights_caudal': 5,
#             'coupling_weights_contra': 10,
#             'init_phase': np.random.default_rng(seed=42).uniform(0, 2*np.pi, 16)
#         }
#     }
#     parameter_grid = {'drive': drives, 'phaselag': phase_lags}
    
#     from simulate import run_multiple
#     run_multiple(
#         max_workers=4, 
#         controller=base_controller, 
#         base_path=BASE_PATH,
#         parameter_grid=parameter_grid,
#         common_kwargs={'fast': True, 'headless': True}
#     )
#     """
#     # =========================================================================
#     print(f"\nAnalyse des résultats dans {BASE_PATH}...")

#     # 1. Scan des dossiers réels sur le disque
#     if not os.path.exists(BASE_PATH):
#         print(f"Erreur : {BASE_PATH} introuvable.")
#         return None

#     subfolders = [f.path for f in os.scandir(BASE_PATH) if f.is_dir()]

#     for folder_path in subfolders:
#         folder_name = os.path.basename(folder_path)
#         path_hdf5 = os.path.join(folder_path, 'simulation.hdf5')
#         path_pkl = os.path.join(folder_path, 'controller.pkl')
    
#         # --- EXTRACTION DES PARAMÈTRES ---
#         # Regex flexible : cherche 'drive' ou 'drive_left' suivi du nombre
#         m_d = re.search(r"drive(?:_left)?(\d+\.\d+)", folder_name)
#         # Cherche 'phaselag' ou 'PLarr7_' suivi du nombre
#         m_pl = re.search(r"(?:phaselag|PLarr7_)(\d+\.\d+)", folder_name)
        
#         d_val = float(m_d.group(1))
#         pl_val = float(m_pl.group(1))

#         # --- ANALYSE DES MÉTRIQUES (ta logique intégrée) ---
#         with h5py.File(path_hdf5, "r") as f:
#             times = f['times'][:]
#             # Extraction des articulations (8 premières)
#             joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:, :8, 0]
            
#             # Régime permanent
#             cut = len(times) // 2
#             t_steady, j_steady = times[cut:], joints[cut:]
            
#             # Métriques mécaniques
#             f_arr, a_arr = compute_mechanical_frequency_amplitude_fft(t_steady, j_steady)
#             f_sim, a_sim = np.mean(f_arr), np.mean(a_arr)
            
#             # Phase Lag Neural (IPL)
#             _, ipl_arr = compute_neural_phase_lags(t_steady, j_steady, f_arr, [[k, k+1] for k in range(7)])
#             ipl_sim = np.mean(ipl_arr)

#             # --- CALCUL DE L'ERREUR ---
#             err_f = abs(f_sim - target_f) / target_f
#             err_a = abs(a_sim - target_amp) / target_amp
#             err_ipl = abs(ipl_sim - target_ipl) / target_ipl
#             error = err_f + err_a + err_ipl

#             # --- MAPPING SUR LA GRILLE ---
#             i = np.argmin(np.abs(drives - d_val))
#             j = np.argmin(np.abs(phase_lags - pl_val))

#             res['error'][i, j] = error
#             res['f'][i, j] = f_sim
#             res['amp'][i, j] = a_sim

#             if error < best_error:
#                 best_error = error
#                 best_params = {
#                     'drive': d_val, 'PL': pl_val, 'error': error, 
#                     'f': f_sim, 'amp': a_sim, 'folder': folder_name
#                 }
#                 print(f"✅ Nouveau record : D={d_val:.2f}, PL={pl_val:.3f} | Erreur={error:.4f}")

#     targets = {'f': target_f, 'amp': target_amp}
#     plot_imitation_results(drives, phase_lags, res, targets)
#     print(f"\nMeilleure simulation trouvée dans : {best_params['folder']}")
#     return best_params

def find_best_imitation(target_f, target_ipl, target_amp):
    n_steps = 5
    drives = np.linspace(2.0, 4.0, n_steps) 
    phase_lags = np.linspace(target_ipl * 0.8, target_ipl * 1.2, n_steps)
    
    # Initialisation des matrices pour les heatmaps
    res = {k: np.zeros((n_steps, n_steps)) for k in ['error', 'f', 'amp']}
    best_error = float('inf')
    best_params = {}

    # =========================================================================
    # PARTIE SIMULATION (Commentée pour exécution post-traitement uniquement)
    # =========================================================================
    """
    base_controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'd_low': 1.0, 'd_high': 5.0, 'a_rate': np.ones(8)*3,
            'offset_freq': np.ones(8)*1, 'offset_amp': np.ones(8)*0.5,
            'G_freq': np.ones(8)*0.5, 'G_amp': np.ones(8)*0.25,
            'coupling_weights_rostral': 5, 'coupling_weights_caudal': 5,
            'coupling_weights_contra': 10,
            'init_phase': np.random.default_rng(seed=42).uniform(0, 2*np.pi, 16)
        }
    }
    parameter_grid = {'drive': drives, 'phaselag': phase_lags}
    
    from simulate import run_multiple
    run_multiple(
        max_workers=4, 
        controller=base_controller, 
        base_path=BASE_PATH,
        parameter_grid=parameter_grid,
        common_kwargs={'fast': True, 'headless': True}
    )
    """
    # TODO Analysis
    print("Bonus, maintenant il faut analyse des résultats pour trouver la meilleure imitation de l'animal...")
    # =========================================================================




def exercise2_3(**kwargs):
    """
    Analyze animal data and compare with the current controller's performance.
    """
    # Get the target metrics from the animal
    f_animal, a_animal, ipl_animal = get_animal_data(ANIMAL_DATA_PATH)
    
    # Extract data from the EXISTING simulation (Exercise 2.1/2.2 logs)
    # We load the HDF5 file generated by the controller
    BASE_PATH_CPG = 'logs/exercise2_1/'
    sim_result = os.path.join(BASE_PATH_CPG, 'simulation.hdf5')
    
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    joint_positions = sensor_data_joints[:, :8, 0]


    # Robot Metrics
    # Frequency and Amplitude
    f_robot_array, a_robot_array = compute_mechanical_frequency_amplitude_fft(sim_times, joint_positions)
    f_robot = np.mean(f_robot_array)
    a_robot = np.mean(a_robot_array)
    # Intersegmental Phase Lag (IPL)
    inds_couples = [[i, i+1] for i in range(7)]
    _, ipl_robot = compute_neural_phase_lags(sim_times, joint_positions, f_robot_array, inds_couples)
    # Speed and Efficiency cannot be computed for the animal as the CSV lacks 
    # global COM coordinates and muscle torque/effort data.

    
    # Step 5:  Performance Comparison
    print("\n" + "="*55)
    print(f"{'METRIC COMPARISON':^55}")
    print("="*55)
    print(f"{'Metric':<25} | {'Animal (Scaled)':<12} | {'CPG Controller':<12}")
    print("-" * 55)
    print(f"{'Frequency (Hz)':<25} | {f_animal:<12.3f} | {f_robot:<12.3f}")
    print(f"{'Joint Amplitude (rad)':<25} | {a_animal:<12.3f} | {a_robot:<12.3f}")
    print(f"{'Phase Lag / IPL (rad)':<25} | {ipl_animal:<12.3f} | {ipl_robot:<12.3f}")
    print("="*55 + "\n")

    find_best_imitation(f_animal, ipl_animal, a_animal)


if __name__ == '__main__':
    exercise2_3(plot=True)

