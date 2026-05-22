"""Simulation parameters"""
import numpy as np
from robot_parameters import body_data, limb_data, coupling_weights, phase_biases

class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 8
        self.n_legs_joints = 8
        self.duration = 30
        self.initial_phases = None
        # self.position_body_gain = 0.6  # default do not change
        # self.position_limb_gain = 1  # default do not change
        self.phase_lag_body = None
        self.amplitude_gradient = None
        self.body_data = body_data
        self.limb_data = limb_data
        self.coupling_weights = coupling_weights
        self.phase_biases = phase_biases
        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...

        self.osc_left_index = np.arange(0, 16, 2)
        self.osc_right_index = np.arange(1, 16, 2)
        self.osc_legs_index = np.arange(16, 32)

        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)

