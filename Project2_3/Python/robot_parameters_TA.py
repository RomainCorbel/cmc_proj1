"""Robot parameters"""

import numpy as np
from farms_core import pylog

BODY = {
    'dlow':  1.0,   # lower drive threshold; body silent below this
    'dhigh': 5.0,   # upper threshold; body saturates above this
    'cv1':   0.2,   # frequency slope  [Hz / drive unit]
    'cv0':   0.3,   # frequency offset [Hz]: nu=0.5 Hz at d=1, nu=1.3 Hz at d=5
    'cr1':   0.065, # amplitude slope  [rad / drive unit]
    'cr0':   0.196*2, # amplitude offset [rad]: R=0.261 at d=1, R=0.521 at d=5
    'rate':  20.0,  # convergence rate a_i [1/s]: controls r_i -> R_i speed
}

LIMB = {
    'dlow':  1.0,
    'dhigh': 3.0,   # limbs stop oscillating here; body continues (Hypothesis 3)
    'cv1':   0.2,   # same slope as body but lower range -> lower walk frequencies
    'cv0':   0.0,   # nu=0.2 Hz at d=1, nu=0.6 Hz at d=3
    'cr1':   0.131*10,
    'cr0':   0.131*10, # R=0.262 at d=1, R=0.524 at d=3
    'rate':  20.0,
}

W_BODY_IPSI   = 10.0   # w_{i,i+1}=20, i∈{1..7}  (body chain)
W_BODY_CONTRA = 10.0   # within-pair anti-phase (not in Table I, needed for architecture)
W_LIMB_IPSI   = 10.0   # w_{i,i+2}=20, i∈{9,10}  (FL-HL, FR-HR ipsilateral)
W_LIMB_CONTRA = 30.0   # w_{i,i+1}=10, i∈{9,11}  (FL-FR, HL-HR contralateral)
W_LIMB2BODY   = 30.0   # w_{limb,body}=10 walking / 0 swimming  (Table I)
W_BODY2LIMB   = 0   # symmetric return coupling
W_LIMB_PAIR   = 30.0   # within limb-joint pair anti-phase
W_HIP_KNEE    = 30.0   # hip→knee within same leg

PHI_BODY_WALK  = 0.0              # standing wave; no travelling phase lag during walking
PHI_BODY_SWIM  = 2.0 * np.pi / 9.0  # k=1 travelling wave over 9 body segments during swimming
PHI_BODY_CONTRA = np.pi           # left-right anti-phase (θ_L - θ_R = π)
PHI_LIMB_PAIR  = np.pi            # flexor-extensor anti-phase within each joint pair
PHI_LIMB_IPSI  = np.pi            # FL-HL and FR-HR anti-phase (ipsilateral)
PHI_LIMB_CONTRA = np.pi           # FL-FR and HL-HR anti-phase (contralateral)
PHI_LIMB2BODY  = 0.0              # limb hip flexor in-phase with body at girdle
PHI_HIP_KNEE   = -np.pi / 2      # hip leads knee by 90° within the same leg

def _sat_freq(d, p):
    """Piece-wise linear frequency saturation (returns nu in Hz)."""
    d = float(d[0]) if hasattr(d, '__len__') else d
    if p['dlow'] < d < p['dhigh']:
        return p['cv1'] * d + p['cv0']
    return 0.0


def _sat_amp(d, p):
    """Piece-wise linear amplitude saturation (returns R in rad)."""
    d = float(d[0]) if hasattr(d, '__len__') else d
    if p['dlow'] < d < p['dhigh']:
        return p['cr1'] * d + p['cr0']
    return 0.0

class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    # Robot-level constants
    N_LEGS        = 4   # four limbs
    N_DOF_PER_LEG = 2   # hip + knee per leg

    def __init__(self, parameters):
        super().__init__()
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2 * self.n_body_joints   # 16
        self.n_oscillators_legs = 2 * self.n_legs_joints   # 16
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs  # 32

        # --- Network state arrays (filled by set_* methods) ------------------
        # freqs[i]: intrinsic angular frequency omega_i = 2*pi*nu_i [rad/s]
        self.freqs = np.zeros(self.n_oscillators)

        # coupling_weights[j, i]: weight of oscillator j on oscillator i
        self.coupling_weights = np.zeros([self.n_oscillators, self.n_oscillators])

        # phase_bias[j, i]
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])

        # rates[i]: convergence rate a_i for amplitude dynamics
        self.rates = np.zeros(self.n_oscillators)

        # nominal_amplitudes[i]: target amplitude R_i [rad]
        self.nominal_amplitudes = np.zeros(self.n_oscillators)

        # Keep reference to sim_parameters for drive-ramp updates in step()
        self.sim_parameters = parameters

        self.update(parameters)

    # =========================================================================
    # Oscillator index helpers
    # =========================================================================

    def _body_left(self):
        """Indices of left-side body oscillators (head to tail)."""
        return np.arange(0, self.n_oscillators_body, 2)

    def _body_right(self):
        """Indices of right-side body oscillators (head to tail)."""
        return np.arange(1, self.n_oscillators_body, 2)

    def _hip_osc(self, leg):
        """(flexor_idx, extensor_idx) of hip joint for leg 0-3."""
        base = self.n_oscillators_body + leg * 2 * self.N_DOF_PER_LEG
        return base, base + 1

    def _knee_osc(self, leg):
        """(flexor_idx, extensor_idx) of knee joint for leg 0-3."""
        base = self.n_oscillators_body + leg * 2 * self.N_DOF_PER_LEG + 2
        return base, base + 1

    # =========================================================================
    # update
    # =========================================================================
    def update(self, parameters):
        """Update network from parameters"""
        self.sim_parameters = parameters
        self.set_frequencies(parameters)
        self.set_coupling_weights(parameters)
        self.set_phase_bias(parameters)
        self.set_amplitudes_rate(parameters)
        self.set_nominal_amplitudes(parameters)

    # =========================================================================
    # step
    # =========================================================================
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
        if hasattr(self.sim_parameters, 'drive_ramp_end'):
            t_max = self.sim_parameters.drive_ramp_duration
            d0    = self.sim_parameters.drive_ramp_start
            d1    = self.sim_parameters.drive_ramp_end
            self.sim_parameters.drive = d0 + (d1 - d0) * min(time / t_max, 1.0)
            self.set_frequencies(self.sim_parameters)
            self.set_nominal_amplitudes(self.sim_parameters)

    # =========================================================================
    # set_frequencies
    # =========================================================================
    def set_frequencies(self, parameters):
        d = parameters.drive
        # 2*pi converts Hz -> rad/s (matches dtheta/dt in the network ODE)
        omega_body = 2.0 * np.pi * _sat_freq(d, BODY)
        omega_limb = 2.0 * np.pi * _sat_freq(d, LIMB)

        self.freqs[:self.n_oscillators_body] = omega_body
        self.freqs[self.n_oscillators_body:] = omega_limb

    # =========================================================================
    # set_coupling_weights
    # =========================================================================
    def set_coupling_weights(self, parameters):
        w = self.coupling_weights
        w[:] = 0.0
        n = self.n_body_joints  # 8
        left  = self._body_left()
        right = self._body_right()

        # --- 1. Body-body ipsilateral (nearest-neighbor)
        for i in range(n - 1):
            w[left[i],     left[i + 1]]  = W_BODY_IPSI
            w[left[i + 1], left[i]]      = W_BODY_IPSI
            w[right[i],    right[i + 1]] = W_BODY_IPSI
            w[right[i + 1], right[i]]    = W_BODY_IPSI

        # --- 2. Body-body contralateral (left <-> right, same segment)
        for i in range(n):
            w[left[i],  right[i]] = W_BODY_CONTRA
            w[right[i], left[i]]  = W_BODY_CONTRA

        # --- 3. Limb <-> body at girdle segments
        fl_h = [16, 17] 
        fr_h = [20, 21] 
        hl_h = [24, 25] 
        hr_h = [28, 29] 

        fore_body_left = [0, 2, 4, 6]
        fore_body_right = [1, 3, 5, 7]
        hind_body_left = [8, 10, 12, 14]
        hind_body_right = [9, 11, 13, 15]

        for limb_osc in list(fl_h):
            for body_osc in fore_body_left:
                w[limb_osc, body_osc] = W_LIMB2BODY
                w[body_osc, limb_osc] = W_BODY2LIMB

        for limb_osc in list(fr_h):
            for body_osc in fore_body_right:
                w[limb_osc, body_osc] = W_LIMB2BODY
                w[body_osc, limb_osc] = W_BODY2LIMB
 
        for limb_osc in list(hr_h):
            for body_osc in hind_body_right:
                w[limb_osc, body_osc] = W_LIMB2BODY
                w[body_osc, limb_osc] = W_BODY2LIMB

        for limb_osc in list(hl_h):
            for body_osc in hind_body_left:
                w[limb_osc, body_osc] = W_LIMB2BODY
                w[body_osc, limb_osc] = W_BODY2LIMB

        # --- 4. Within-joint pair (flexor-extensor anti-phase) ---------------
        for j in range(self.n_legs_joints):
            even = self.n_oscillators_body + 2 * j
            odd  = even + 1
            w[even, odd] = W_LIMB_PAIR
            w[odd, even] = W_LIMB_PAIR

        # --- 5. Inter-leg hip coupling (trot) ------------------------------------
        # Ipsilateral fore-hind (FL-HL, FR-HR): (w_{i,i+2}, i∈{9,10})
        # Contralateral (FL-FR, HL-HR):         (w_{i,i+1}, i∈{9,11})
        ipsilateral = [(fl_h, hl_h), (fr_h, hr_h)]
        contralateral = [(fl_h, fr_h), (hl_h, hr_h)]
        for pair_a, pair_b in ipsilateral:
            for oa, ob in zip(pair_a, pair_b):
                w[oa, ob] = W_LIMB_IPSI
                w[ob, oa] = W_LIMB_IPSI
        for pair_a, pair_b in contralateral:
            for oa, ob in zip(pair_a, pair_b):
                w[oa, ob] = W_LIMB_CONTRA
                w[ob, oa] = W_LIMB_CONTRA

        # --- 6. Hip -> knee within the same leg ------------------------------
        for leg in range(self.N_LEGS):
            hip_f, hip_e = self._hip_osc(leg)
            kne_f, kne_e = self._knee_osc(leg)
            print(hip_f, hip_e, kne_f, kne_e,self.N_LEGS)
            w[hip_f, kne_f] = W_HIP_KNEE  # hip flexor  -> knee flexor
            w[kne_f, hip_f] = W_HIP_KNEE
            w[hip_e, kne_e] = W_HIP_KNEE  # hip extensor -> knee extensor
            w[kne_e, hip_e] = W_HIP_KNEE
        np.set_printoptions(threshold=100000, linewidth=1000)
                # Open Ipython to interact with the code (uv pip install ipython)
        # This can be useful for exploring the contents of data.sensors for example
        # from IPython import embed; embed()
        print(w)

    # =========================================================================
    # set_phase_bias
    # =========================================================================
    def set_phase_bias(self, parameters):
        phi = self.phase_bias
        phi[:] = 0.0
        n = self.n_body_joints  # 8
        left  = self._body_left()
        right = self._body_right()

        # --- Body phase lag--------------------------------------
        if hasattr(parameters, 'phase_lag_body') and parameters.phase_lag_body is not None:
            lag = parameters.phase_lag_body
        else:
            d_target = parameters.drive
            if hasattr(parameters, 'drive_ramp_end'):
                d_target = parameters.drive_ramp_end
            d_target = float(d_target[0]) if hasattr(d_target, '__len__') else float(d_target)
            if d_target <= 3.0:
                lag = PHI_BODY_WALK
            else:
                lag = PHI_BODY_SWIM

        # --- 1. Body traveling wave (head -> tail) ------------------------------
        for i in range(n - 1):
            phi[left[i],      left[i + 1]]  = -lag
            phi[left[i + 1],  left[i]]      = +lag
            phi[right[i],     right[i + 1]] = -lag
            phi[right[i + 1], right[i]]     = +lag

        # --- 2. Body left-right anti-phase (theta_L - theta_R = pi) -------------
        for i in range(n):
            phi[left[i],  right[i]] = -PHI_BODY_CONTRA
            phi[right[i], left[i]]  = +PHI_BODY_CONTRA

        # --- 3. Limb flexor-extensor anti-phase (all 8 limb joint pairs) --------
        for j in range(self.n_legs_joints):
            even = self.n_oscillators_body + 2 * j
            odd  = even + 1
            phi[even, odd] = -PHI_LIMB_PAIR
            phi[odd, even] = +PHI_LIMB_PAIR

        # --- 4. Trot inter-leg phase biases (Table I) ---------------------------
        # φ_{i,i+2}=π  i∈{9,10}: FL-HL and FR-HR anti-phase (ipsilateral)
        # φ_{i,i+1}=π  i∈{9,11}: FL-FR and HL-HR anti-phase (contralateral)
        fl_h = [16, 17] 
        fr_h = [20, 21] 
        hl_h = [24, 25] 
        hr_h = [28, 29] 

        for oa, ob in zip(fl_h, fr_h):  # FL-FR anti-phase (les 2 de devant)
            phi[oa, ob] = -PHI_LIMB_CONTRA
            phi[ob, oa] = +PHI_LIMB_CONTRA
        for oa, ob in zip(fl_h, hl_h):  # FL-HL anti-phase (les deux a gauche)
            phi[oa, ob] = -PHI_LIMB_IPSI
            phi[ob, oa] = +PHI_LIMB_IPSI
        for oa, ob in zip(fr_h, hr_h):  # FR-HR anti-phase (les deux a droite)
            phi[oa, ob] = -PHI_LIMB_IPSI
            phi[ob, oa] = +PHI_LIMB_IPSI
        for oa, ob in zip(hl_h, hr_h):  # HL-HR anti-phase (les 2 de derriere)
            phi[oa, ob] = -PHI_LIMB_CONTRA
            phi[ob, oa] = +PHI_LIMB_CONTRA

        # --- 5. Limb-body at girdle: (in-phase) -------------
        fore_body_left = [0, 2, 4, 6]
        fore_body_right = [1, 3, 5, 7]
        hind_body_left = [8, 10, 12, 14]
        hind_body_right = [9, 11, 13, 15]

        for i,limb_osc in enumerate(fl_h):
            for body_osc in fore_body_left:
                phi[limb_osc, body_osc] = PHI_LIMB2BODY if i % 2 == 0 else PHI_LIMB_PAIR
                phi[body_osc, limb_osc] = -PHI_LIMB2BODY if i % 2 == 0 else -PHI_LIMB_PAIR

        for i,limb_osc in enumerate(fr_h):
            for body_osc in fore_body_right:
                phi[limb_osc, body_osc] = PHI_LIMB2BODY if i % 2 == 0 else PHI_LIMB_PAIR
                phi[body_osc, limb_osc] = -PHI_LIMB2BODY if i % 2 == 0 else -PHI_LIMB_PAIR

        for i,limb_osc in enumerate(hr_h):
            for body_osc in hind_body_right:
                phi[limb_osc, body_osc] = PHI_LIMB2BODY if i % 2 == 0 else PHI_LIMB_PAIR
                phi[body_osc, limb_osc] = -PHI_LIMB2BODY if i % 2 == 0 else -PHI_LIMB_PAIR

        for i,limb_osc in enumerate(hl_h):
            for body_osc in hind_body_left:
                phi[limb_osc, body_osc] = PHI_LIMB2BODY if i % 2 == 0 else PHI_LIMB_PAIR
                phi[body_osc, limb_osc] = -PHI_LIMB2BODY if i % 2 == 0 else -PHI_LIMB_PAIR
 
        # --- 6. Hip-knee within same leg: hip leads knee by PHI_HIP_KNEE
        for leg in range(self.N_LEGS):
            hip_f, hip_e = self._hip_osc(leg)
            kne_f, kne_e = self._knee_osc(leg)
            phi[hip_f, kne_f] = PHI_HIP_KNEE    # hip flexor  -> knee flexor
            phi[kne_f, hip_f] = -PHI_HIP_KNEE
            phi[hip_e, kne_e] = PHI_HIP_KNEE    # hip extensor -> knee extensor
            phi[kne_e, hip_e] = -PHI_HIP_KNEE
        np.set_printoptions(threshold=100000, linewidth=1000)
        print(phi)
        # Open Ipython to interact with the code (uv pip install ipython)
        # This can be useful for exploring the contents of data.sensors for example
        # from IPython import embed; embed()
    # =========================================================================
    # set_amplitudes_rate
    # =========================================================================
    def set_amplitudes_rate(self, parameters):
        """Set convergence rates a_i (same for body and limb, from Table S1).

        The amplitude ODE is  dr_i/dt = a_i * (R_i - r_i).
        a_i = 20 s^-1 means the amplitude tracks the target R_i with a
        time constant of ~50 ms, fast enough to respond to drive changes
        during gait transitions without disrupting the limit cycle.
        """
        self.rates[:self.n_oscillators_body] = BODY['rate']  # 20.0
        self.rates[self.n_oscillators_body:]  = LIMB['rate']  # 20.0

    # =========================================================================
    # set_nominal_amplitudes
    # =========================================================================
    def set_nominal_amplitudes(self, parameters):
        """Set nominal (target) amplitudes R_i [rad].

        R_i is computed from the amplitude saturation function (Table S1):
          R_body(d) = 0.065*d + 0.196   for d in (1, 5), else 0
          R_limb(d) = 0.131*d + 0.131   for d in (1, 3), else 0

        Optional: if parameters.amplitude_gradient is set, the body amplitude
        increases linearly from head to tail:
          R_i(seg) = R_body * (1 + gradient * pos),  pos in [0, 1]
        A gradient > 0 enhances the traveling wave by giving the tail larger
        oscillations, as observed in swimming salamanders (Fig. 4 of the paper).
        """
        if not hasattr(parameters, 'drive'):
            pylog.warning('No drive parameter; nominal amplitudes left at zero.')
            return

        d = parameters.drive
        R_body = _sat_amp(d, BODY)
        R_limb = _sat_amp(d, LIMB)

        gradient = 0.0
        if hasattr(parameters, 'amplitude_gradient') and parameters.amplitude_gradient is not None:
            gradient = parameters.amplitude_gradient  # positive -> larger tail amplitude

        left  = self._body_left()
        right = self._body_right()
        n = self.n_body_joints

        for i in range(n):
            # pos = 0 at head, 1 at tail
            pos = i / (n - 1) if n > 1 else 0.0
            R_i = R_body * (1.0 + gradient * pos)
            self.nominal_amplitudes[left[i]]  = R_i
            self.nominal_amplitudes[right[i]] = R_i

        self.nominal_amplitudes[self.n_oscillators_body:] = R_limb

