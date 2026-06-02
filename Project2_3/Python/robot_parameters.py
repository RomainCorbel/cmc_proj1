"""Robot parameters"""

import numpy as np
from farms_core import pylog

BODY = {
    'dlow':  1.0,
    'dhigh': 5.0,
    'cv1':   0.2,
    'cv0':   0.3,
    'cr1':   0.5,#0.065*4,#*1.5,
    'cr0':   0,#0.196,#*2,
    'rate':  20.0,
}

LIMB = {
    'dlow':  1.0,
    'dhigh': 3.0,
    'cv1':   0.2,
    'cv0':   0.0,
    'cr1':   0.025,#0.131*2.5,
    'cr0':   0.575,#0.131*2,#*6,
    'rate':  20.0,
}

W_BODY_IPSI   = 10.0   # body: adjacent oscillators, same side, ex: 2,4
W_BODY_CONTRA = 10.0   # body: same position, left↔right oscillators ex: 0,1
W_LIMB_IPSI   = 10.0   # legs: front↔hind oscillators, same side (FL↔HL, FR↔HR)
W_LIMB_CONTRA = 30.0   # legs: same girdle, left↔right oscillators (FL↔FR, HL↔HR)
W_LIMB2BODY   = 30.0   # limb hip oscillators → body girdle oscillators
W_BODY2LIMB   = 0      # body girdle → limb hip oscillators (0 = unidirectional)
W_LIMB_PAIR   = 30.0   # within joint: flexor↔extensor oscillators of same hip or knee
W_HIP_KNEE    = 30.0   # within leg: hip↔knee oscillators

PHI_BODY_WALK   = 0.0               # body: phase lag between adjacent oscillators (walking)
PHI_BODY_SWIM   = 2.0 * np.pi / 8.0 # body: phase lag between adjacent oscillators (swimming)
PHI_BODY_CONTRA = np.pi             # body left↔right oscillators: anti-phase
PHI_LIMB_PAIR   = np.pi             # flexor↔extensor oscillators of same joint: anti-phase
PHI_LIMB_IPSI   = np.pi             # ipsilateral front↔hind oscillators: anti-phase (trot)
PHI_LIMB_CONTRA = np.pi             # contralateral same-girdle oscillators: anti-phase
PHI_LIMB2BODY   = 0.0               # limb→body oscillators: flexor in-phase with body
PHI_HIP_KNEE    = -np.pi / 2        # hip oscillator leads knee by 90°


def _sat_freq(d, p):
    d = float(d[0]) if hasattr(d, '__len__') else d
    if p['dlow'] < d < p['dhigh']:
        return p['cv1'] * d + p['cv0']
    return 0.0


def _sat_amp(d, p):
    d = float(d[0]) if hasattr(d, '__len__') else d
    if p['dlow'] < d < p['dhigh']:
        return p['cr1'] * d + p['cr0']
    return 0.0


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    N_LEGS        = 4
    N_DOF_PER_LEG = 2

    def __init__(self, parameters):
        super().__init__()
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2 * self.n_body_joints
        self.n_oscillators_legs = 2 * self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs

        self.freqs              = np.zeros(self.n_oscillators)
        self.coupling_weights   = np.zeros([self.n_oscillators, self.n_oscillators])
        self.phase_bias         = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates              = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)

        self.sim_parameters = parameters
        self.update_drive = getattr(parameters, 'update_drive', False)
        self._transitioned = False
        self.update(parameters)

    def _body_left(self):
        return np.arange(0, self.n_oscillators_body, 2)

    def _body_right(self):
        return np.arange(1, self.n_oscillators_body, 2)

    def _hip_osc(self, leg):
        base = self.n_oscillators_body + leg * 2 * self.N_DOF_PER_LEG
        return base, base + 1

    def _knee_osc(self, leg):
        base = self.n_oscillators_body + leg * 2 * self.N_DOF_PER_LEG + 2
        return base, base + 1

    def _index_groups(self):
        """Shared index groups used by both set_coupling_weights and set_phase_bias."""
        left, right = self._body_left(), self._body_right()
        half = self.n_body_joints // 2
        # fl/fr/hl/hr: [flexor_osc, extensor_osc] indices for each hip (front-left, front-right, hind-left, hind-right)
        fl, fr, hl, hr = [list(self._hip_osc(i)) for i in range(4)]
        # limb_body: pairs (hip_oscs, body_oscs) connecting each hip to its nearest body oscillators at the girdle
        limb_body = [
            (fl, list(left[:half])), (fr, list(right[:half])),
            (hl, list(left[half:])), (hr, list(right[half:])),
        ]
        # inter_leg: (pair_a, pair_b, weight, phase_bias) for all 4 hip-to-hip couplings
        inter_leg = [
            (fl, hl, W_LIMB_IPSI,   PHI_LIMB_IPSI),   # FL↔HL ipsilateral
            (fr, hr, W_LIMB_IPSI,   PHI_LIMB_IPSI),   # FR↔HR ipsilateral
            (fl, fr, W_LIMB_CONTRA, PHI_LIMB_CONTRA),  # FL↔FR contralateral
            (hl, hr, W_LIMB_CONTRA, PHI_LIMB_CONTRA),  # HL↔HR contralateral
        ]
        # names_il = ['FL↔HL', 'FR↔HR', 'FL↔FR', 'HL↔HR']
        # print("inter_leg (pair_a, pair_b, weight, phase_bias):")
        # for name, (a, b, w, phi) in zip(names_il, inter_leg):
        #     print(f"  {name}: {a} ↔ {b}  w={w}  phi={phi:.3f}")
        return left, right, limb_body, inter_leg

    def update(self, parameters):
        self.sim_parameters = parameters
        self.set_frequencies(parameters)
        self.set_coupling_weights(parameters)
        self.set_phase_bias(parameters)
        self.set_amplitudes_rate(parameters)
        self.set_nominal_amplitudes(parameters)

    def step(self, time, iteration, salamandra_data):
        if hasattr(self.sim_parameters, 'drive_ramp_end'):
            t_max = self.sim_parameters.drive_ramp_duration
            d0    = self.sim_parameters.drive_ramp_start
            d1    = self.sim_parameters.drive_ramp_end
            self.sim_parameters.drive = d0 + (d1 - d0) * min(time / t_max, 1.0)
            self.set_frequencies(self.sim_parameters)
            self.set_nominal_amplitudes(self.sim_parameters)

        if self.update_drive and not self._transitioned:
            index = 0 if iteration == 0 else (iteration - 1)
            contacts_all = np.linalg.norm(np.array(
                salamandra_data.sensors.contacts.totals()[index]
            ), axis=1)
            contacts_feet = contacts_all[10:18:2]

            if self.update_drive == 'swim2walk':
                if np.any(contacts_feet > 0):
                    self._transition_time = time
                    self._ramp_d0 = self.sim_parameters.drive
                    self._ramp_d1 = 2.0
                    self._ramp_duration = 3.0
                    self._transitioned = True

            elif self.update_drive == 'walk2swim':
                contacts_front_feet = contacts_all[10:13:2]  # FL and FR feet only
                if iteration > 200 and np.all(contacts_front_feet == 0):
                    self._no_contact_count = self.get('_no_contact_count', 0) + 1
                else:
                    self._no_contact_count = 0
                if self.get('_no_contact_count', 0) > 200:
                    self._transition_time = time
                    self._ramp_d0 = self.sim_parameters.drive
                    self._ramp_d1 = 4.0
                    self._ramp_duration = 3.0
                    self._transitioned = True

        # update phase bias once when transition is first detected
        if self._transitioned and not self.get('_phase_bias_updated', False):
            old_drive = self.sim_parameters.drive
            self.sim_parameters.drive = self._ramp_d1
            self.set_phase_bias(self.sim_parameters)
            self.sim_parameters.drive = old_drive
            self._phase_bias_updated = True

        # linear drive ramp after transition (same logic as exercise_p2 ramp)
        if self._transitioned and '_ramp_d0' in self:
            elapsed = time - self._transition_time
            alpha = min(elapsed / self._ramp_duration, 1.0)
            d = self._ramp_d0 + (self._ramp_d1 - self._ramp_d0) * alpha
            self.sim_parameters.drive = d
            self.set_frequencies(self.sim_parameters)
            self.set_nominal_amplitudes(self.sim_parameters)
            if alpha >= 1.0:
                self.pop('_ramp_d0')  # stop updating once ramp is complete

    def set_frequencies(self, parameters):
        d = parameters.drive
        self.freqs[:self.n_oscillators_body] = 2.0 * np.pi * _sat_freq(d, BODY)
        self.freqs[self.n_oscillators_body:] = 2.0 * np.pi * _sat_freq(d, LIMB)

    def set_coupling_weights(self, parameters):
        w = self.coupling_weights
        w[:] = 0.0
        n = self.n_body_joints
        left, right, limb_body, inter_leg = self._index_groups()

        # 1. Body ipsilateral nearest-neighbor
        pairs = []
        for i in range(n - 1):
            for side in (left, right):
                w[side[i], side[i+1]] = w[side[i+1], side[i]] = W_BODY_IPSI
                pairs.append(f"[{side[i]},{side[i+1]}]")
        # print(f"W1 body ipsi (w={W_BODY_IPSI}): {' '.join(pairs)}")

        # 2. Body contralateral same-segment
        pairs = []
        for i in range(n):
            w[left[i], right[i]] = w[right[i], left[i]] = W_BODY_CONTRA
            pairs.append(f"[{left[i]},{right[i]}]")
        # print(f"W2 body contra (w={W_BODY_CONTRA}): {' '.join(pairs)}")

        # 3. Limb-body at girdle
        names_lb = ['FL', 'FR', 'HL', 'HR']
        limb2body_w = 0.0 if getattr(parameters, 'disable_limb_spine_coupling', False) else W_LIMB2BODY
        # print(f"W3 limb→body (w={limb2body_w}), body→limb (w={W_BODY2LIMB}):")
        for _, (limb_oscs, body_oscs) in zip(names_lb, limb_body):
            for lo in limb_oscs:
                for bo in body_oscs:
                    w[lo, bo] = limb2body_w
                    w[bo, lo] = W_BODY2LIMB
            # print(f"  {name}: hip{limb_oscs} ↔ body{body_oscs}")

        # 4. Within-joint pair (flexor-extensor)
        pairs = []
        for j in range(self.n_legs_joints):
            even = self.n_oscillators_body + 2 * j
            w[even, even+1] = w[even+1, even] = W_LIMB_PAIR
            pairs.append(f"[{even},{even+1}]")
        # print(f"W4 limb pair flexor↔extensor (w={W_LIMB_PAIR}): {' '.join(pairs)}")

        # 5. Inter-leg hip coupling
        names_il = ['FL↔HL', 'FR↔HR', 'FL↔FR', 'HL↔HR']
        # print("W5 inter-leg hips:")
        for name, (pair_a, pair_b, weight, _) in zip(names_il, inter_leg):
            for oa, ob in zip(pair_a, pair_b):
                w[oa, ob] = w[ob, oa] = weight
            # print(f"  {name} (w={weight}): {list(zip(pair_a, pair_b))}")

        # 6. Hip-knee within same leg
        pairs = []
        for leg in range(self.N_LEGS):
            hip_f, hip_e = self._hip_osc(leg)
            kne_f, kne_e = self._knee_osc(leg)
            w[hip_f, kne_f] = w[kne_f, hip_f] = W_HIP_KNEE
            w[hip_e, kne_e] = w[kne_e, hip_e] = W_HIP_KNEE
            pairs.append(f"[{hip_f},{kne_f}] [{hip_e},{kne_e}]")
        # print(f"W6 hip↔knee (w={W_HIP_KNEE}): {' '.join(pairs)}")

    def set_phase_bias(self, parameters):
        phi = self.phase_bias
        phi[:] = 0.0
        n = self.n_body_joints
        left, right, limb_body, inter_leg = self._index_groups()

        if hasattr(parameters, 'phase_lag_body') and parameters.phase_lag_body is not None:
            lag = parameters.phase_lag_body
        else:
            d = parameters.drive
            if hasattr(parameters, 'drive_ramp_end'):
                d = parameters.drive_ramp_end
            d = float(d[0]) if hasattr(d, '__len__') else float(d)
            lag = PHI_BODY_WALK if d <= 3.0 else PHI_BODY_SWIM

        # 1. Body traveling wave
        pairs = []
        for i in range(n - 1):
            for side in (left, right):
                phi[side[i], side[i+1]] = -lag
                phi[side[i+1], side[i]] = +lag
                pairs.append(f"[{side[i]},{side[i+1]}]")
        print(f"P1 body ipsi traveling wave (lag={lag:.3f}): {' '.join(pairs)}")

        # 2. Body contralateral anti-phase
        pairs = []
        for i in range(n):
            phi[left[i], right[i]] = -PHI_BODY_CONTRA
            phi[right[i], left[i]] = +PHI_BODY_CONTRA
            pairs.append(f"[{left[i]},{right[i]}]")
        print(f"P2 body contra (phi={PHI_BODY_CONTRA:.3f}): {' '.join(pairs)}")

        # 3. Within-joint pair (flexor-extensor)
        pairs = []
        for j in range(self.n_legs_joints):
            even = self.n_oscillators_body + 2 * j
            phi[even, even+1] = -PHI_LIMB_PAIR
            phi[even+1, even] = +PHI_LIMB_PAIR
            pairs.append(f"[{even},{even+1}]")
        print(f"P3 limb pair flexor↔extensor (phi={PHI_LIMB_PAIR:.3f}): {' '.join(pairs)}")

        # 4. Inter-leg phase biases
        names_il = ['FL↔HL', 'FR↔HR', 'FL↔FR', 'HL↔HR']
        print("P4 inter-leg hips:")
        for name, (pair_a, pair_b, _, bias) in zip(names_il, inter_leg):
            for oa, ob in zip(pair_a, pair_b):
                phi[oa, ob] = -bias
                phi[ob, oa] = +bias
            print(f"  {name} (phi={bias:.3f}): {list(zip(pair_a, pair_b))}")

        # 5. Limb-body at girdle (flexor in-phase, extensor anti-phase)
        names_lb = ['FL', 'FR', 'HL', 'HR']
        flexor_bias = np.pi if getattr(parameters, 'limb_spine_antiphase', False) else PHI_LIMB2BODY
        print("P5 limb→body girdle:")
        for name, (limb_oscs, body_oscs) in zip(names_lb, limb_body):
            for i, lo in enumerate(limb_oscs):
                bias = flexor_bias if i % 2 == 0 else PHI_LIMB_PAIR
                for bo in body_oscs:
                    phi[lo, bo] = bias
                    phi[bo, lo] = -bias
            print(f"  {name}: hip{limb_oscs} ↔ body{body_oscs}  (flexor phi={flexor_bias:.3f}, extensor phi={PHI_LIMB_PAIR:.3f})")

        # 6. Hip-knee within same leg
        pairs = []
        for leg in range(self.N_LEGS):
            hip_f, hip_e = self._hip_osc(leg)
            kne_f, kne_e = self._knee_osc(leg)
            phi[hip_f, kne_f] = PHI_HIP_KNEE;  phi[kne_f, hip_f] = -PHI_HIP_KNEE
            phi[hip_e, kne_e] = PHI_HIP_KNEE;  phi[kne_e, hip_e] = -PHI_HIP_KNEE
            pairs.append(f"[{hip_f},{kne_f}] [{hip_e},{kne_e}]")
        print(phi)

    def set_amplitudes_rate(self, parameters):
        self.rates[:self.n_oscillators_body] = BODY['rate']
        self.rates[self.n_oscillators_body:]  = LIMB['rate']

    def set_nominal_amplitudes(self, parameters):
        if not hasattr(parameters, 'drive'):
            pylog.warning('No drive parameter; nominal amplitudes left at zero.')
            return

        d = parameters.drive
        R_body = _sat_amp(d, BODY)
        R_limb = _sat_amp(d, LIMB)

        gradient = 0.0
        if hasattr(parameters, 'amplitude_gradient') and parameters.amplitude_gradient is not None:
            gradient = parameters.amplitude_gradient

        left, right = self._body_left(), self._body_right()
        n = self.n_body_joints
        for i in range(n):
            pos = i / (n - 1) if n > 1 else 0.0
            R_i = R_body * (1.0 + gradient * pos)
            self.nominal_amplitudes[left[i]]  = R_i
            self.nominal_amplitudes[right[i]] = R_i

        self.nominal_amplitudes[self.n_oscillators_body:] = R_limb
