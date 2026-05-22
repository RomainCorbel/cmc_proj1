"""Robot parameters"""

import numpy as np
from farms_core import pylog

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
}
limb_data = {
    'dlow': 1.0,
    'dhigh': 3.0,
    'cv1': 0.4,
    'cv0': 0.0,
    'cr1': 0.131,
    'cr0': 0.131,
    'vsat': 0.0,
    'Rsat': 0.0,
    'amp_rate': 20.0,
}
coupling_weights = {
    'body_ips_down': 10.0,  # Towards the tail
    'body_ips_up': 10.0,    # Away from the tail
    'body_contra': 10.0,    # Between left/right body oscillators
    'limb_close_body': 30,  # Between the inner limb joint and the next segment body joints
    'limb_contra': 10.0,    # Between the two oscillators for one limb joint
    'limb_ips': 10,         # Along one limb
    'limb_close_lr': 10,    # Between the inner limb joints on the same axis
    'limb_close_fb': 10,    # Between the inner limb joints on the same side of the body
    'other': 0,
}
phase_biases = {
    'body_ips_down': 2*np.pi/8,  # Towards the tail
    'body_ips_up': 2*np.pi/8,    # Away from the tail
    'body_contra': np.pi,        # Between left/right body oscillators
    'limb_close_body': np.pi,    # Between the inner limb joint and the next segment body joints
    'limb_contra': -np.pi,        # Between the two oscillators for one limb joint ggg
    'limb_ips': 0,               # Along one limb
    'limb_close_lr': np.pi,      # Between the inner limb joints on the same axis ggg
    'limb_close_fb': -np.pi,      # Between the inner limb joints on the same side of the body  gggg
    'other': 0,
}


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super().__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = 2*self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators,
        ])
        self.phase_bias = np.zeros([
            self.n_oscillators,
            self.n_oscillators,
        ])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)

        self.osc_left_index = parameters.osc_left_index
        self.osc_right_index = parameters.osc_right_index
        self.osc_legs_index = parameters.osc_legs_index

        self.body_data = parameters.body_data
        self.limb_data = parameters.limb_data
        self.sim_parameters = parameters

        # self.feedback_gains_swim = np.zeros(self.n_oscillators)
        # self.feedback_gains_walk = np.zeros(self.n_oscillators)

        # # gains for final motor output
        # self.position_body_gain = parameters.position_body_gain
        # self.position_limb_gain = parameters.position_limb_gain

        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        # This is probably constant, but update it just in case
        if hasattr(parameters, 'body_data'):
            self.body_data = parameters.body_data
        if hasattr(parameters, 'limb_data'):
            self.limb_data = parameters.limb_data

        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # psi_ij
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def step(self, time, iteration, salamandra_data):
        """Step function called at each iteration

        Parameters
        ----------

        salamanra_data: salamandra_simulation/data.py::SalamandraData
            Contains the robot data, including network and sensors.

        gps (within the method): Numpy array of shape [9x3]
            Numpy array of size 9x3 representing the GPS positions of each link
            of the robot along the body. The first index [0-8] coressponds to
            the link number from head to tail, and the second index [0,1,2]
            coressponds to the XYZ axis in world coordinate.

        """
        # Example to get global coordinates of robot links
        gps = np.array(
            salamandra_data.sensors.links.urdf_positions()[iteration, :9],
        )
        # Example to update the drive
        # self.sim_parameters.drive = ...
        # self.set_frequencies(self.sim_parameters)  # f_i
        # self.set_nominal_amplitudes(self.sim_parameters)  # R_i
        # print("GPGS: {}".format(gps[4, 0]))
        # print("drive: {}".format(self.sim_parameters.drive))
        if hasattr(self.sim_parameters, 'drive_ramp_end'):
            t_max = self.sim_parameters.drive_ramp_duration
            d_start = self.sim_parameters.drive_ramp_start
            d_end = self.sim_parameters.drive_ramp_end
            self.sim_parameters.drive = d_start + (d_end - d_start) * min(time / t_max, 1.0)
            self.set_frequencies(self.sim_parameters)
            self.set_nominal_amplitudes(self.sim_parameters)

    def set_frequencies(self, parameters):
        """Set frequencies"""

        if not hasattr(parameters, 'drive'):
            return
        
        bod = self.body_data
        leg = self.limb_data
        d = parameters.drive

        if bod['dlow'] < d and d < bod['dhigh']:
            bod_f = 2*np.pi * (bod['cv1']*d + bod['cv0'])
        else:
            bod_f = bod['vsat']
        if leg['dlow'] < d and d < leg['dhigh']:
            leg_f = 2*np.pi * (leg['cv1']*d + leg['cv0'])
        else:
            leg_f = leg['vsat']
        
        for i in range(len(self.freqs)):
            if i in self.osc_left_index or i in self.osc_right_index:
                self.freqs[i] = bod_f
            elif i in self.osc_legs_index:
                self.freqs[i] = leg_f
            else:
                pylog.error(f'Unassigned frequency index {i}')

    def get_pair_case(self, i, j):
        # Two body joints
        if (i < min(self.osc_legs_index)) and (j < min(self.osc_legs_index)):
            if i == j+2:
                return 'body_ips_up'
            elif j == i+2:
                return 'body_ips_down'
            elif abs(i-j) == 1 and min(i, j) in self.osc_left_index:
                return 'body_contra'
            else:
                return 'other'
        # Two leg joints
        elif (i >= min(self.osc_legs_index)) and (j >= min(self.osc_legs_index)):
            if abs(i-j) == 1 and min(i,j)%2 == 0:
                return 'limb_contra'
            elif abs(i-j) == 2 and min(i,j)%4 in [0,1]:
                return 'limb_ips'
            elif abs(i-j) == 4 and min(i,j)%4 in [0,1] and min(i,j) not in [20, 21]:
                return 'limb_close_lr'
            elif abs(i-j) == 8 and min(i,j)%4 in [0,1]:
                return 'limb_close_fb'
            else:
                return 'other'
        # One leg, one body
        else:
            leg = max(i,j)
            body = min(i,j)

            match_sets = {
                16: [3,5,7,9],
                17: [2,4,6,8],
                20: [2,4,6,8],
                21: [3,5,7,9],
                24: [11,13,15],
                25: [10,12,14],
                28: [10,12,14],
                29: [11,13,15],
            }
            if leg in match_sets.keys():
                if body in match_sets[leg]:
                    return 'limb_close_body'
                else:
                    return 'other'
            else:
                return 'other'

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        # If the weights aren't being updated, don't try to update them
        if not hasattr(parameters, 'coupling_weights'):
            return
        
        for i in range(len(self.coupling_weights)):
            for j in range(len(self.coupling_weights[0])):
                case = self.get_pair_case(i, j)
                self.coupling_weights[i,j] = parameters.coupling_weights[case]

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        # If the biases aren't being updated, don't try to update them
        if not hasattr(parameters, 'phase_biases'):
            return
        
        for i in range(len(self.coupling_weights)):
            for j in range(len(self.coupling_weights[0])):
                case = self.get_pair_case(i, j)
                self.phase_bias[i,j] = parameters.phase_biases[case]

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        body_rate = self.body_data['amp_rate']
        leg_rate = self.limb_data['amp_rate']
        
        for i in range(len(self.rates)):
            if i in self.osc_left_index or i in self.osc_right_index:
                self.rates[i] = body_rate
            elif i in self.osc_legs_index:
                self.rates[i] = leg_rate
            else:
                pylog.error(f'Unassigned amplitude index {i}')

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        if not hasattr(parameters, 'drive'):
            return
        bod = self.body_data
        leg = self.limb_data
        d = parameters.drive
        
        if bod['dlow'] < d and d < bod['dhigh']:
            bod_R = bod['cr1']*d + bod['cr0']
        else:
            bod_R = bod['Rsat']
        if leg['dlow'] < d and d < leg['dhigh']:
            leg_R = leg['cr1']*d + leg['cr0']
        else:
            leg_R = leg['Rsat']

        for i in range(len(self.nominal_amplitudes)):
            if i in self.osc_left_index or i in self.osc_right_index:
                self.nominal_amplitudes[i] = bod_R
            elif i in self.osc_legs_index:
                self.nominal_amplitudes[i] = leg_R
            else:
                pylog.error(f'Unassigned amplitude index {i}')

