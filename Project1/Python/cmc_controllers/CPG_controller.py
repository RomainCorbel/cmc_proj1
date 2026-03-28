import numpy as np
from typing import List
from scipy.integrate import ode

from farms_core.model.data import AnimatData
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.options import AnimatOptions

from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.control.network import AnimatNetwork

from farms_core import pylog

from cmc_controllers.polymander_controller import PolymanderController, NeuralNetwork


class CPGNetwork(NeuralNetwork):
    """Dummy Network"""

    def __init__(
        self,
        data: AnimatData,
        drive_left: float,
        drive_right: float,
        d_low: float,
        d_high: float,
        a_rate: np.ndarray,  # (n_joints,)
        offset_freq: np.ndarray,  # (n_joints,)
        offset_amp: np.ndarray,  # (n_joints,)
        G_freq: np.ndarray,  # (n_joints,)
        G_amp: np.ndarray,  # (n_joints,)
        PL: np.ndarray,  # (n_joints,)
        coupling_weights_rostral: float,
        coupling_weights_caudal: float,
        coupling_weights_contra: float,
        init_phase: np.ndarray,  # (2*n_joints,)
        n_body_joints: int,
        left_body_idx: slice,
        right_body_idx: slice,
        **kwargs,
    ):
        super().__init__(data, **kwargs)

        # indexes
        self.n_body_joints = n_body_joints
        self.left_body_idx = left_body_idx
        self.right_body_idx = right_body_idx

        # Controller state
        self.n_body_joints = n_body_joints
        self.n_oscillators = 2*n_body_joints  # double chain
        # [phases, amplitudes, motor_outputs_storage]
        self.state = np.zeros((self.n_iterations, 3*self.n_oscillators))

        # init phase
        self.state[0, :self.n_oscillators] = init_phase

        # Solver
        self.solver = ode(f=self.network_ode)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.state[0], t=0.0)

        # CPG controller hyperparameters
        self.d_low = d_low
        self.d_high = d_high
        self.a_rate = a_rate
        self.offset_freq = offset_freq
        self.offset_amp = offset_amp
        self.G_freq = G_freq
        self.G_amp = G_amp
        self.PL = PL
        self.coupling_weights_rostral = coupling_weights_rostral
        self.coupling_weights_caudal = coupling_weights_caudal
        self.coupling_weights_contra = coupling_weights_contra

        pylog.warning("TODO 3.1 stretch feedback")
        self.w_ipsi = kwargs.pop('w_ipsi', None)

        pylog.warning("TODO 3.3 Disruption masks")
        self.disruption_p_sensors = kwargs.pop('disruption_p_sensors', 0.0)
        self.disruption_p_couplings = kwargs.pop('disruption_p_couplings', 0.0)
        self.random_seed = kwargs.pop('random_seed', 42)
        np.random.seed(self.random_seed)

        # CPG controller parameters
        #TODO EXO 2.1: Set nominal frequencies and amplitudes according to needs
        drive = np.concatenate([
            np.full(n_body_joints, drive_left),   # oscillateurs gauche
            np.full(n_body_joints, drive_right),  # oscillateurs droite
        ])
        active = (drive >= d_low) & (drive <= d_high)

        self.nominal_frequencies = np.where(
            active,
            G_freq * (drive - d_low) + offset_freq,
            0
        )

        self.nominal_amplitudes = np.where(
            active,
            G_amp * (drive - d_low) + offset_amp,
            0
        )

        self.coupling_weights = np.zeros((self.n_oscillators, self.n_oscillators))
        for i in range(n_body_joints):
            l = i                    
            r = i + n_body_joints   

            if i > 0:
                self.coupling_weights[l, l - 1] = coupling_weights_rostral
                self.coupling_weights[r, r - 1] = coupling_weights_rostral

            if i < n_body_joints - 1:
                self.coupling_weights[l, l + 1] = coupling_weights_caudal
                self.coupling_weights[r, r + 1] = coupling_weights_caudal

            self.coupling_weights[l, r] = coupling_weights_contra
            self.coupling_weights[r, l] = coupling_weights_contra


        self.phase_bias = np.zeros((self.n_oscillators, self.n_oscillators))
        PB = PL / (n_body_joints - 1)
        for i in range(n_body_joints):
            l = i
            r = i + n_body_joints

            if i > 0:
                self.phase_bias[l, l - 1] = +PB
                self.phase_bias[r, r - 1] = +PB

            if i < n_body_joints - 1:
                self.phase_bias[l, l + 1] = -PB
                self.phase_bias[r, r + 1] = -PB

            self.phase_bias[l, r] = +np.pi
            self.phase_bias[r, l] = -np.pi

        self.drive_left = drive_left
        self.drive_right = drive_right

    def motor_output(self, phase, amplitude):
        pylog.warning("TODO 2.1 CPG motor output implementation")
        oscillator_output = np.zeros_like(phase)
        return np.array(oscillator_output[self.left_body_idx]), np.array(
            oscillator_output[self.right_body_idx])

    def network_ode(self, _time, state, stretch_value):
        """
        Compute derivatives for the ODE system.
        state: [phases, amplitudes, dphases_storage, damplitudes_storage, motor_outputs_storage]
        stretch_value: array of stretch feedback values (or zeros if w_ipsi is None)
        Returns: derivatives for [phases, amplitudes]
        """
        phases = state[:self.n_oscillators]
        amplitudes = state[self.n_oscillators:2*self.n_oscillators]

        dstates = np.zeros_like(state)

        pylog.warning("TODO 2.1 CPG ODE implementation")

        pylog.warning("TODO 3.1 Stretch feedback")

        if self.w_ipsi is not None:
            pass
        return dstates

    def step(
        self,
        iteration: int,
        time: float,
        timestep: float,
        checks: bool = False,
        strict: bool = False,
    ):
        """
        Control step
        Called after obtaining all the current sensor data, and right before
        calling the physics.
        """

        phases = self.state[iteration, :self.n_oscillators]
        amplitudes = self.state[iteration,
                                self.n_oscillators:2*self.n_oscillators]

        # Compute stretch feedback value
        stretch_value = np.array(
            self.data.sensors.joints.array[iteration-1, :self.n_body_joints, 0]) if iteration > 0 else np.zeros(self.n_body_joints)

        pylog.warning("TODO 3.1 Stretch feedback")

        pylog.warning("TODO 3.3 Disruption to sensors")

        pylog.warning("TODO 3.3 Set ODE parameters with stretch value")
        # self.solver.set_f_params(np.zeros(self.n_oscillators))

        # Integrate ODE using dopri5 solver
        self.solver.integrate(time + timestep)
        integrated_state = self.solver.y

        # motor output from CPG state
        motor_output_left, motor_output_right = self.motor_output(
            phases, amplitudes)

        # Only set body joints in project 1
        # self.data.state.array[iteration, :] = 0
        self.data.state.array[iteration,
                              self.left_body_idx] = motor_output_left
        self.data.state.array[iteration,
                              self.right_body_idx] = motor_output_right

        # Controller state update
        left_storage_idx = slice(
            self.left_body_idx.start + self.n_oscillators*2,
            self.left_body_idx.stop + self.n_oscillators*2,
            self.left_body_idx.step)
        right_storage_idx = slice(
            self.right_body_idx.start + self.n_oscillators*2,
            self.right_body_idx.stop + self.n_oscillators*2,
            self.right_body_idx.step)
        self.state[iteration, left_storage_idx] = motor_output_left
        self.state[iteration, right_storage_idx] = motor_output_right

        if iteration + 1 >= self.n_iterations:
            return

        # Update state with integrated values
        self.state[iteration+1,
                   :self.n_oscillators] = integrated_state[:self.n_oscillators]
        self.state[iteration +
                   1, self.n_oscillators:2 *
                   self.n_oscillators] = integrated_state[self.n_oscillators:2 *
                                                          self.n_oscillators]


class CPGController(PolymanderController):
    """CPGController"""

    def __init__(self,
                 animat_options: AmphibiousOptions,
                 animat_data: AmphibiousData,
                 config):

        control_joint_names = [
            joint.joint_name for joint in animat_options.control.motors]
        body_joint_names = [
            name for name in control_joint_names if "body" in name and 'passive' not in name]
        leg_joint_names = [
            name for name in control_joint_names if "leg" in name]

        self.n_body_joints = len(body_joint_names)
        self.n_leg_joints = len(leg_joint_names)

        self.left_body_idx = slice(0, 2*self.n_body_joints, 2)
        self.right_body_idx = slice(1, 2*self.n_body_joints+1, 2)
        self.left_leg_idx = slice(
            2 *
            self.n_body_joints,
            2 *
            self.n_body_joints +
            2 *
            self.n_leg_joints,
            2)
        self.right_leg_idx = slice(
            2 *
            self.n_body_joints +
            1,
            2 *
            self.n_body_joints +
            2 *
            self.n_leg_joints +
            1,
            2)

        animat_network = CPGNetwork(
            data=animat_data,
            drive_left=config['drive_left'],
            drive_right=config['drive_right'],
            d_low=config['d_low'],
            d_high=config['d_high'],
            a_rate=config['a_rate'],
            offset_freq=config['offset_freq'],
            offset_amp=config['offset_amp'],
            G_freq=config['G_freq'],
            G_amp=config['G_amp'],
            PL=config['PL'],
            coupling_weights_rostral=config['coupling_weights_rostral'],
            coupling_weights_caudal=config['coupling_weights_caudal'],
            coupling_weights_contra=config['coupling_weights_contra'],
            init_phase=config['init_phase'],
            w_ipsi=config.get(
                'w_ipsi',
                None),
            disruption_p_sensors=config.get(
                'disruption_p_sensors',
                0.0),
            disruption_p_couplings=config.get(
                'disruption_p_couplings',
                0.0),
            random_seed=config.get(
                'random_seed',
                42),
            n_body_joints=self.n_body_joints,
            left_body_idx=self.left_body_idx,
            right_body_idx=self.right_body_idx,
        )

        super().__init__(
            animat_options=animat_options,
            animat_data=animat_data,
            animat_network=animat_network,
        )

        self.config = config

    @classmethod
    def from_options(
        cls,
        config: dict,
        experiment_options: ExperimentOptions,
        animat_i: int,
        animat_data: AnimatData,
        animat_options: AnimatOptions,
    ):
        del animat_i
        return cls(
            animat_options=animat_options,
            animat_data=animat_data,
            config=config,
        )

