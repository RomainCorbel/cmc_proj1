"""Simulation parameters"""


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
        # MLR drive d: controls gait via saturation functions in robot_parameters.
        #   d ∈ (1, 3): walking  (body + limb active, low frequencies)
        #   d ∈ (3, 5): swimming (body only active, higher frequencies)
        self.drive = 2.0  # default: mid-walking regime
        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...

        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)

