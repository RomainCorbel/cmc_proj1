"""Robot parameters"""

import numpy as np
from farms_core import pylog

# =============================================================================
# Saturation-function parameters (Ijspeert et al. 2007, Table S1)
#
# Frequency and amplitude of each oscillator are set by a piece-wise linear
# saturation function driven by the MLR drive signal d (Fig. 2 in the paper):
#
#   nu(d) = cv1*d + cv0   if  dlow < d < dhigh,  else 0   [Hz]
#   R(d)  = cr1*d + cr0   if  dlow < d < dhigh,  else 0   [rad]
#
# Body and limb oscillators have DIFFERENT upper thresholds:
#   - body dhigh = 5 : body CPG active for the full drive range
#   - limb dhigh = 3 : limbs stop oscillating above drive 3
#     This asymmetry automatically switches gait:
#       d in (1,3) -> walking (body + limbs both oscillate)
#       d in (3,5) -> swimming (only body oscillates)
# =============================================================================

# Body oscillator saturation (axial CPG, 8 spine joints)
BODY = {
    'dlow':  1.0,   # lower drive threshold; body silent below this
    'dhigh': 5.0,   # upper threshold; body saturates above this
    'cv1':   0.2,   # frequency slope  [Hz / drive unit]
    'cv0':   0.3,   # frequency offset [Hz]: nu=0.5 Hz at d=1, nu=1.3 Hz at d=5
    'cr1':   0.065, # amplitude slope  [rad / drive unit]
    'cr0':   0.196, # amplitude offset [rad]: R=0.261 at d=1, R=0.521 at d=5
    'rate':  20.0,  # convergence rate a_i [1/s]: controls r_i -> R_i speed
}

# Limb oscillator saturation (limb CPG, 8 leg joints = 4 legs x 2 DOF)
# dhigh=3 < body dhigh=5 so limbs stop at the walking/swimming transition
# walking frequency range: nu in [0.2, 0.6] Hz for d in [1, 3]
LIMB = {
    'dlow':  1.0,
    'dhigh': 3.0,   # limbs stop oscillating here; body continues (Hypothesis 3)
    'cv1':   0.2,   # same slope as body but lower range -> lower walk frequencies
    'cv0':   0.0,   # nu=0.2 Hz at d=1, nu=0.6 Hz at d=3
    'cr1':   0.131,
    'cr0':   0.131, # R=0.262 at d=1, R=0.524 at d=3
    'rate':  20.0,
}


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


# =============================================================================
# Coupling weights  (Ijspeert 2007, supplementary)
#
# W_BODY_IPSI   : nearest-neighbour coupling along one side of the body chain.
#                 Creates the traveling-wave tendency in the axial CPG.
# W_BODY_CONTRA : left-right coupling within the same body segment.
#                 Keeps the two sides in anti-phase.
# W_LIMB2BODY   : limb-hip -> body at the girdle segments (unidirectional).
#                 Stronger than W_BODY_IPSI so limbs can override the traveling
#                 wave and impose a standing wave during walking (Hypothesis 2).
# W_BODY2LIMB   : weak body -> limb return coupling at the girdle.
# W_LIMB_PAIR   : within a joint pair (flexor-extensor); enforces anti-phase so
#                 the motor output (Eq. 3) yields a clean sinusoidal joint angle.
# W_INTER_LEG   : hip-to-hip coupling between different legs (trot coordination).
# W_HIP_KNEE    : coupling from the hip to the knee of the same leg; propagates
#                 the oscillation to the additional DOF (not present in the
#                 original 6-joint robot - our adaptation for 8 leg joints).
# =============================================================================

# Table I values (paper uses 1 osc/joint; here each joint = anti-phase pair)
W_BODY_IPSI   = 20.0   # w_{i,i+1}=20, i∈{1..7}  (body chain)
W_BODY_CONTRA = 20.0   # within-pair anti-phase (not in Table I, needed for architecture)
W_LIMB_IPSI   = 20.0   # w_{i,i+2}=20, i∈{9,10}  (FL-HL, FR-HR ipsilateral)
W_LIMB_CONTRA = 10.0   # w_{i,i+1}=10, i∈{9,11}  (FL-FR, HL-HR contralateral)
W_LIMB2BODY   = 10.0   # w_{limb,body}=10 walking / 0 swimming  (Table I)
W_BODY2LIMB   = 10.0   # symmetric return coupling
W_LIMB_PAIR   = 20.0   # within limb-joint pair anti-phase
W_HIP_KNEE    = 10.0   # hip→knee within same leg


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    # Robot-level constants (anatomical)
    N_LEGS        = 4   # four limbs
    N_DOF_PER_LEG = 2   # hip + knee per leg (adaptation vs original 1 DOF)

    def __init__(self, parameters):
        super().__init__()

        # --- Basic dimensions ------------------------------------------------
        # n_body_joints = 8 axial (spine) joints
        self.n_body_joints = parameters.n_body_joints
        # n_legs_joints = 8 leg joints (4 legs x 2 DOF = hip + knee each)
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases
        self.n_joints = self.n_body_joints + self.n_legs_joints

        # Each body joint -> 2 oscillators (left + right side of the spine)
        self.n_oscillators_body = 2 * self.n_body_joints   # 16
        # Each leg joint -> 2 oscillators (flexor + extensor activation)
        self.n_oscillators_legs = 2 * self.n_legs_joints   # 16
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs  # 32

        # --- Network state arrays (filled by set_* methods) ------------------
        # freqs[i]: intrinsic angular frequency omega_i = 2*pi*nu_i [rad/s]
        self.freqs = np.zeros(self.n_oscillators)
        # coupling_weights[j, i]: weight of oscillator j on oscillator i
        self.coupling_weights = np.zeros([self.n_oscillators, self.n_oscillators])
        # phase_bias[j, i]: satisfies theta_j - theta_i = -phase_bias[j,i] at eq.
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        # rates[i]: convergence rate a_i for amplitude dynamics
        self.rates = np.zeros(self.n_oscillators)
        # nominal_amplitudes[i]: target amplitude R_i [rad]
        self.nominal_amplitudes = np.zeros(self.n_oscillators)

        # --- Saturation-function tables (used by exercise_p1 plots) ----------
        self.body_data = BODY
        self.limb_data = LIMB

        # Keep reference to sim_parameters for drive-ramp updates in step()
        self.sim_parameters = parameters

        self.update(parameters)

    # =========================================================================
    # Oscillator index helpers
    # =========================================================================
    #
    # Oscillator layout (n_body_joints=8, n_legs_joints=8):
    #
    # Body (indices 0-15), interleaved left/right per segment:
    #   Left  side (flexor-like): 0, 2, 4, 6, 8, 10, 12, 14   (2*i)
    #   Right side (extensor-like): 1, 3, 5, 7, 9, 11, 13, 15 (2*i+1)
    #
    # Limb (indices 16-31), 4 legs x 2 DOF x 2 osc per joint:
    #   Leg 0 (FL fore-left) : hip  (16, 17),  knee (18, 19)
    #   Leg 1 (FR fore-right): hip  (20, 21),  knee (22, 23)
    #   Leg 2 (HL hind-left) : hip  (24, 25),  knee (26, 27)
    #   Leg 3 (HR hind-right): hip  (28, 29),  knee (30, 31)
    #
    # Within each joint pair the even oscillator drives flexion and the odd
    # oscillator drives extension; they are kept in anti-phase (pi) so that
    # motor_output (Eq. 3) x_i = r_i*(1+cos(theta_i)) produces a sinusoid.

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
        """Step function called at each simulation iteration.

        Supports a linear drive ramp when sim_parameters carries
        'drive_ramp_end', 'drive_ramp_start', and 'drive_ramp_duration'.
        The drive is updated from start to end over the ramp duration, then
        the frequency and amplitude arrays are refreshed accordingly.
        """
        # GPS positions available if needed for sensory feedback
        # gps = salamandra_data.sensors.links.urdf_positions()[iteration, :9]
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
        """Set intrinsic angular frequencies omega_i = 2*pi*nu_i [rad/s].

        nu_i is computed from the piece-wise linear saturation function
        (Ijspeert 2007, Eq. 1 / Table S1) evaluated at the current drive d:

          nu_body(d) = 0.2*d + 0.3  for d in (1, 5), else 0   [Hz]
          nu_limb(d) = 0.2*d + 0.0  for d in (1, 3), else 0   [Hz]

        Walking example (d=2): nu_body=0.7 Hz, nu_limb=0.4 Hz
        Swimming example (d=4): nu_body=1.1 Hz, nu_limb=0 (limbs stop)

        Adaptation: same formula as the original 6-joint robot; the higher
        joint count does not change the per-oscillator saturation function.
        """
        if not hasattr(parameters, 'drive'):
            pylog.warning('No drive parameter; frequencies left at zero.')
            return

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
        """Set coupling weight matrix w[source, target].

        Network ODE convention (see network.py):
          dtheta_i/dt = omega_i + sum_j  r_j * w[j,i] * sin(theta_j - theta_i + phi[j,i])
        So w[j,i] is the weight of the influence that oscillator j exerts on i.

        Coupling topology (Ijspeert 2007, Fig. 1A and supplementary):

        1. Body-body ipsilateral (nearest-neighbor along each side of the chain)
           Gives the body CPG its traveling-wave tendency.

        2. Body-body contralateral (left-right, same segment)
           Maintains anti-phase between the two sides of each spine segment.

        3. Limb-hip -> body at girdle segments (strong, W_LIMB2BODY=30)
           Forelimb hips couple to body segments 0-1 (oscillators 0-3).
           Hindlimb hips couple to body segments 4-5 (oscillators 8-11).
           Girdle positions chosen for an 8-joint spine: forelimb at ~1/8 and
           hindlimb at ~5/8 of the body length, consistent with salamander
           anatomy (Ijspeert 2007, Fig. 1A and Fig. 3).
           W_LIMB2BODY > W_BODY_IPSI -> limbs can override traveling wave and
           force a standing wave during walking (Hypothesis 2).

        4. Body -> limb at girdle segments (weak, W_BODY2LIMB=10)
           Provides sensory return coupling to the limb CPG.

        5. Within-joint pair (flexor-extensor)
           Keeps the two oscillators of each joint in anti-phase, which is
           required for a correct sinusoidal motor output via Eq. 3.

        6. Inter-leg hip coupling (trot coordination)
           Couples hip oscillators across all 4 legs to implement the trot
           gait (diagonal pairs FL-HR and FR-HL in phase; all other pairs
           anti-phase). Only hip oscillators are coupled inter-leg; the knee
           oscillators are coordinated via the within-leg hip->knee coupling.

        7. Hip -> knee within the same leg (W_HIP_KNEE=10)
           Propagates the oscillation from each hip to the same leg's knee.
           This coupling is our adaptation for the additional knee DOF which
           is not present in the original 6-joint robot.
        """
        w = self.coupling_weights
        w[:] = 0.0
        n = self.n_body_joints  # 8
        left  = self._body_left()
        right = self._body_right()

        # --- 1. Body-body ipsilateral (nearest-neighbor) ---------------------
        for i in range(n - 1):
            w[left[i],     left[i + 1]]  = W_BODY_IPSI
            w[left[i + 1], left[i]]      = W_BODY_IPSI
            w[right[i],    right[i + 1]] = W_BODY_IPSI
            w[right[i + 1], right[i]]    = W_BODY_IPSI

        # --- 2. Body-body contralateral (left <-> right, same segment) -------
        for i in range(n):
            w[left[i],  right[i]] = W_BODY_CONTRA
            w[right[i], left[i]]  = W_BODY_CONTRA

        # --- 3 & 4. Limb <-> body at girdle segments -------------------------
        # Forelimb girdle: body segments 0-1 -> oscillators 0,1,2,3
        # Hindlimb girdle: body segments 4-5 -> oscillators 8,9,10,11
        fore_body = [0, 1, 2, 3]
        hind_body = [8, 9, 10, 11]

        fl_h = self._hip_osc(0)  # (16, 17) FL hip
        fr_h = self._hip_osc(1)  # (20, 21) FR hip
        hl_h = self._hip_osc(2)  # (24, 25) HL hip
        hr_h = self._hip_osc(3)  # (28, 29) HR hip

        for limb_osc in list(fl_h) + list(fr_h):
            for body_osc in fore_body:
                w[limb_osc, body_osc] = W_LIMB2BODY  # strong limb -> body
                w[body_osc, limb_osc] = W_BODY2LIMB  # weak  body -> limb

        for limb_osc in list(hl_h) + list(hr_h):
            for body_osc in hind_body:
                w[limb_osc, body_osc] = W_LIMB2BODY
                w[body_osc, limb_osc] = W_BODY2LIMB

        # --- 5. Within-joint pair (flexor-extensor anti-phase) ---------------
        for j in range(self.n_legs_joints):
            even = self.n_oscillators_body + 2 * j
            odd  = even + 1
            w[even, odd] = W_LIMB_PAIR
            w[odd, even] = W_LIMB_PAIR

        # --- 6. Inter-leg hip coupling (trot) ------------------------------------
        # Ipsilateral fore-hind (FL-HL, FR-HR): w=20  (w_{i,i+2}, i∈{9,10})
        # Contralateral (FL-FR, HL-HR):         w=10  (w_{i,i+1}, i∈{9,11})
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

        # --- 7. Hip -> knee within the same leg ------------------------------
        for leg in range(self.N_LEGS):
            hip_f, hip_e = self._hip_osc(leg)
            kne_f, kne_e = self._knee_osc(leg)
            w[hip_f, kne_f] = W_HIP_KNEE  # hip flexor  -> knee flexor
            w[kne_f, hip_f] = W_HIP_KNEE
            w[hip_e, kne_e] = W_HIP_KNEE  # hip extensor -> knee extensor
            w[kne_e, hip_e] = W_HIP_KNEE

    # =========================================================================
    # set_phase_bias
    # =========================================================================
    def set_phase_bias(self, parameters):
        """Set phase-bias matrix phi[source, target].

        Sign convention (matches network ODE in network.py):
          dtheta_i/dt = ... + sum_j r_j * w[j,i] * sin(theta_j - theta_i + phi[j,i])
        At equilibrium: theta_j - theta_i = -phi[j, i]
        Therefore:  phi[source, target] = -(desired theta_source - theta_target)

        Phase relationships implemented:

        Body traveling wave (head leads tail):
          phase_lag = 2*pi/n_body_joints per segment (or parameters.phase_lag_body).
          phi[seg_i,   seg_{i+1}] = -lag  (source i leads, drives i+1 to lag)
          phi[seg_{i+1}, seg_i]   = +lag  (reverse: drives i to stay ahead)

        Body left-right anti-phase (theta_L - theta_R = pi):
          phi[L_i, R_i] = -pi,  phi[R_i, L_i] = +pi

        Limb flexor-extensor anti-phase within each joint pair:
          phi[even, odd] = -pi,  phi[odd, even] = +pi
          Same as body anti-phase, required for correct motor output (Eq. 3).

        Trot inter-leg (diagonal pairs FL-HR and FR-HL in phase):
          Contralateral (FL-FR, HL-HR) and ipsilateral fore-hind (FL-HL, FR-HR)
          are anti-phase (lag = pi).  Diagonal pairs (FL-HR, FR-HL) are
          in-phase (lag = 0).  This matches the walking pattern of the real
          salamander (Ijspeert 2007, Fig. 3).

        Limb-body at girdle (theta_limb - theta_body = pi):
          phi[limb, body] = -pi,  phi[body, limb] = +pi
          Forces the body at the girdle into the standing-wave pattern that
          is characteristic of salamander walking (Ijspeert 2007, Fig. 3).

        Hip-knee within the same leg (in-phase, lag = 0):
          The knee oscillates in phase with the hip (phi = 0).  This is an
          adaptation for the extra knee DOF; a phase offset could be tuned
          to improve gait naturalness.
        """
        phi = self.phase_bias
        phi[:] = 0.0
        n = self.n_body_joints  # 8
        left  = self._body_left()
        right = self._body_right()

        # --- Body phase lag from Table I --------------------------------------
        # Walking: k=0  → lag = 0           (standing wave)
        # Swimming: k≠0 → lag = 2πk/N, N=9  (traveling wave)
        # If phase_lag_body is explicitly set, use it directly.
        # Otherwise infer from the target drive (use drive_ramp_end if present).
        if hasattr(parameters, 'phase_lag_body') and parameters.phase_lag_body is not None:
            lag = parameters.phase_lag_body
        else:
            d_target = parameters.drive
            if hasattr(parameters, 'drive_ramp_end'):
                d_target = parameters.drive_ramp_end
            d_target = float(d_target[0]) if hasattr(d_target, '__len__') else float(d_target)
            if d_target <= 3.0:
                lag = 0.0                     # walking: standing wave (k=0)
            else:
                lag = 2.0 * np.pi / 9.0      # swimming: k=1, N=9

        # --- Body traveling wave (head -> tail) ------------------------------
        for i in range(n - 1):
            phi[left[i],      left[i + 1]]  = -lag
            phi[left[i + 1],  left[i]]      = +lag
            phi[right[i],     right[i + 1]] = -lag
            phi[right[i + 1], right[i]]     = +lag

        # --- Body left-right anti-phase (theta_L - theta_R = pi) -------------
        for i in range(n):
            phi[left[i],  right[i]] = -np.pi
            phi[right[i], left[i]]  = +np.pi

        # --- Limb flexor-extensor anti-phase (all 8 limb joint pairs) --------
        for j in range(self.n_legs_joints):
            even = self.n_oscillators_body + 2 * j
            odd  = even + 1
            phi[even, odd] = -np.pi
            phi[odd, even] = +np.pi

        # --- Trot inter-leg phase biases (Table I) ---------------------------
        # φ_{i,i+2}=π  i∈{9,10}: FL-HL and FR-HR anti-phase (ipsilateral)
        # φ_{i,i+1}=π  i∈{9,11}: FL-FR and HL-HR anti-phase (contralateral)
        # Diagonal pairs (FL-HR, FR-HL) in-phase emerges from transitivity;
        # no direct coupling weight → no explicit phase bias needed.
        def _set_pair(pair_a, pair_b, desired_lag):
            for oa, ob in zip(pair_a, pair_b):
                phi[oa, ob] = -desired_lag
                phi[ob, oa] = +desired_lag

        fl_h = self._hip_osc(0)  # (16, 17)
        fr_h = self._hip_osc(1)  # (20, 21)
        hl_h = self._hip_osc(2)  # (24, 25)
        hr_h = self._hip_osc(3)  # (28, 29)

        _set_pair(fl_h, fr_h, np.pi)   # FL-FR anti-phase  (φ_{9,10}=π)
        _set_pair(fl_h, hl_h, np.pi)   # FL-HL anti-phase  (φ_{9,11}=π)
        _set_pair(fr_h, hr_h, np.pi)   # FR-HR anti-phase  (φ_{10,12}=π)
        _set_pair(hl_h, hr_h, np.pi)   # HL-HR anti-phase  (φ_{11,12}=π)

        # --- Limb-body at girdle: Table I φ_{i,j}=0  (in-phase) -------------
        fore_body = [0, 1, 2, 3]
        hind_body = [8, 9, 10, 11]

        for limb_osc in list(fl_h) + list(fr_h):
            for body_osc in fore_body:
                phi[limb_osc, body_osc] = 0.0
                phi[body_osc, limb_osc] = 0.0

        for limb_osc in list(hl_h) + list(hr_h):
            for body_osc in hind_body:
                phi[limb_osc, body_osc] = 0.0
                phi[body_osc, limb_osc] = 0.0

        # Hip-knee within same leg: in-phase (phi=0, already default)

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
