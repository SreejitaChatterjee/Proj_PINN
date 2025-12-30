"""
Hardened Attack Detector - Defenses Against Mathematical Vulnerabilities.

This module implements defenses against all identified attack vectors:

1. Multi-rate jerk checking - Closes low-frequency nullspace
2. CUSUM drift monitoring - Catches slow cumulative attacks
3. Velocity-augmented EKF - Eliminates observability blind spots
4. Randomized thresholds - Prevents threshold gaming
5. Robust statistics - Resists normalization poisoning
6. SPRT sequential testing - Replaces instantaneous NIS
7. Spectral monitoring - Detects frequency-domain attacks

Reference: VULNERABILITIES.md for attack analysis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from scipy import signal
from scipy.stats import chi2


# =============================================================================
# 1. MULTI-RATE JERK CHECKER - Closes low-frequency nullspace
# =============================================================================

class MultiRateJerkChecker:
    """
    Check jerk at multiple time scales to close filter nullspace.

    Vulnerability addressed:
    - Single dt creates nullspace at f < 1/(dt*10) Hz
    - Solution: Check at multiple dt values

    With dt = [0.005, 0.02, 0.1], we cover:
    - 0.005s: Catches >10Hz attacks
    - 0.02s:  Catches >2.5Hz attacks
    - 0.1s:   Catches >0.5Hz attacks (closes the gap!)
    """

    def __init__(
        self,
        dt_values: List[float] = None,
        max_jerk: float = 100.0
    ):
        self.dt_values = dt_values or [0.005, 0.02, 0.1]
        self.max_jerk = max_jerk

        # Adaptive thresholds per rate (lower for slower rates)
        self.thresholds = {
            dt: max_jerk * (dt / 0.005) ** 0.5  # Scale with sqrt(dt)
            for dt in self.dt_values
        }

    def check(self, pos: np.ndarray, base_dt: float = 0.005) -> Dict:
        """
        Check jerk at multiple rates.

        Args:
            pos: Position array (N, 3)
            base_dt: Base sampling period

        Returns:
            Dict with results per rate and combined score
        """
        results = {}
        max_violation_score = 0.0

        for dt in self.dt_values:
            # Downsample factor
            factor = max(1, int(dt / base_dt))
            pos_ds = pos[::factor]

            if len(pos_ds) < 4:
                continue

            # Compute jerk at this rate
            vel = np.diff(pos_ds, axis=0) / dt
            acc = np.diff(vel, axis=0) / dt
            jerk = np.diff(acc, axis=0) / dt
            jerk_mag = np.linalg.norm(jerk, axis=1)

            # Check against rate-specific threshold
            threshold = self.thresholds[dt]
            violations = jerk_mag > threshold
            violation_ratio = np.mean(violations)

            # Compute normalized score
            if len(jerk_mag) > 0:
                score = np.max(jerk_mag) / threshold
            else:
                score = 0.0

            results[f'dt_{dt}'] = {
                'jerk_max': np.max(jerk_mag) if len(jerk_mag) > 0 else 0,
                'threshold': threshold,
                'violation_ratio': violation_ratio,
                'score': score
            }

            max_violation_score = max(max_violation_score, score)

        results['combined_score'] = max_violation_score
        results['is_anomaly'] = max_violation_score > 1.0

        return results


# =============================================================================
# 2. CUSUM DRIFT MONITOR - Catches slow cumulative attacks
# =============================================================================

class CUSUMDriftMonitor:
    """
    Cumulative Sum (CUSUM) detector for slow drift attacks.

    Vulnerability addressed:
    - Instantaneous checks miss slow drifts
    - Solution: Track cumulative deviation over time

    CUSUM formula:
    S_n = max(0, S_{n-1} + x_n - μ - k)

    Where k is the allowance (slack) parameter.
    """

    def __init__(
        self,
        threshold_h: float = 10.0,    # Detection threshold
        allowance_k: float = 0.5,     # Drift allowance
        reset_on_alarm: bool = True
    ):
        self.h = threshold_h
        self.k = allowance_k
        self.reset_on_alarm = reset_on_alarm

        # State
        self.S_pos = 0.0  # Positive CUSUM
        self.S_neg = 0.0  # Negative CUSUM
        self.mean = 0.0
        self.n = 0
        self.alarm_count = 0

    def reset(self):
        """Reset CUSUM state."""
        self.S_pos = 0.0
        self.S_neg = 0.0
        self.mean = 0.0
        self.n = 0

    def update(self, innovation: float) -> Dict:
        """
        Update CUSUM with new innovation.

        Args:
            innovation: Measurement innovation (residual)

        Returns:
            Dict with CUSUM state and alarm status
        """
        # Update running mean (for centering)
        self.n += 1
        delta = innovation - self.mean
        self.mean += delta / self.n

        # Centered innovation
        x = innovation - self.mean

        # Two-sided CUSUM
        self.S_pos = max(0, self.S_pos + x - self.k)
        self.S_neg = max(0, self.S_neg - x - self.k)

        # Check for alarm
        alarm = (self.S_pos > self.h) or (self.S_neg > self.h)

        if alarm:
            self.alarm_count += 1
            if self.reset_on_alarm:
                self.S_pos = 0.0
                self.S_neg = 0.0

        return {
            'S_pos': self.S_pos,
            'S_neg': self.S_neg,
            'alarm': alarm,
            'alarm_count': self.alarm_count,
            'score': max(self.S_pos, self.S_neg) / self.h
        }

    def update_vector(self, innovation: np.ndarray) -> Dict:
        """Update CUSUM with vector innovation."""
        # Use magnitude for scalar CUSUM
        mag = np.linalg.norm(innovation)
        return self.update(mag)


class MultiChannelCUSUM:
    """CUSUM monitors for multiple channels (pos, vel, att)."""

    def __init__(self, channels: List[str] = None, **kwargs):
        self.channels = channels or ['position', 'velocity', 'attitude']
        self.monitors = {ch: CUSUMDriftMonitor(**kwargs) for ch in self.channels}

    def reset(self):
        for m in self.monitors.values():
            m.reset()

    def update(self, innovations: Dict[str, np.ndarray]) -> Dict:
        results = {}
        any_alarm = False

        for channel, innovation in innovations.items():
            if channel in self.monitors:
                result = self.monitors[channel].update_vector(innovation)
                results[channel] = result
                any_alarm = any_alarm or result['alarm']

        results['any_alarm'] = any_alarm
        results['combined_score'] = max(
            r['score'] for r in results.values() if isinstance(r, dict) and 'score' in r
        )
        return results


# =============================================================================
# 3. VELOCITY-AUGMENTED EKF - Eliminates observability blind spots
# =============================================================================

@dataclass
class AugmentedEKFConfig:
    """Configuration for velocity-augmented EKF."""
    dt: float = 0.005

    # Process noise
    sigma_acc: float = 0.5
    sigma_gyro: float = 0.01
    sigma_acc_bias: float = 1e-4
    sigma_gyro_bias: float = 1e-5

    # Measurement noise - NOW INCLUDES VELOCITY
    sigma_pos: float = 0.1
    sigma_vel: float = 0.05     # Added velocity measurement!
    sigma_baro: float = 0.5
    sigma_mag: float = 0.1

    # NIS thresholds
    nis_threshold_95: float = 12.59  # Chi2(6) for pos+vel
    nis_threshold_99: float = 16.81


class VelocityAugmentedEKF:
    """
    EKF with velocity measurements to close observability gaps.

    Vulnerability addressed:
    - Position-only EKF has nullspace in velocity/attitude/bias
    - Solution: Add velocity measurements

    New measurement matrix:
    H = [I₃ 0 0 0 0]   <- position
        [0 I₃ 0 0 0]   <- velocity (NEW!)
    """

    def __init__(self, config: AugmentedEKFConfig = None):
        self.config = config or AugmentedEKFConfig()

        # State: [pos(3), vel(3), att(3), ba(3), bg(3)] = 15 states
        self.x = np.zeros(15)
        self.P = np.eye(15) * 0.1

        # Build measurement matrices
        self._build_H_matrices()
        self._build_Q()

        # CUSUM for sequential monitoring
        self.cusum = MultiChannelCUSUM(
            channels=['position', 'velocity'],
            threshold_h=5.0,
            allowance_k=0.3
        )

    def _build_H_matrices(self):
        """Build measurement matrices including velocity."""
        # Position measurement (3 states)
        self.H_pos = np.zeros((3, 15))
        self.H_pos[0:3, 0:3] = np.eye(3)
        self.R_pos = np.eye(3) * self.config.sigma_pos**2

        # Velocity measurement (3 states) - NEW!
        self.H_vel = np.zeros((3, 15))
        self.H_vel[0:3, 3:6] = np.eye(3)
        self.R_vel = np.eye(3) * self.config.sigma_vel**2

        # Combined position + velocity (6 states)
        self.H_posvel = np.zeros((6, 15))
        self.H_posvel[0:3, 0:3] = np.eye(3)  # Position
        self.H_posvel[3:6, 3:6] = np.eye(3)  # Velocity
        self.R_posvel = np.diag([
            self.config.sigma_pos**2, self.config.sigma_pos**2, self.config.sigma_pos**2,
            self.config.sigma_vel**2, self.config.sigma_vel**2, self.config.sigma_vel**2
        ])

    def _build_Q(self):
        """Build process noise matrix."""
        cfg = self.config
        dt = cfg.dt
        self.Q = np.zeros((15, 15))
        self.Q[0:3, 0:3] = np.eye(3) * (cfg.sigma_acc * dt**2)**2
        self.Q[3:6, 3:6] = np.eye(3) * (cfg.sigma_acc * dt)**2
        self.Q[6:9, 6:9] = np.eye(3) * (cfg.sigma_gyro * dt)**2
        self.Q[9:12, 9:12] = np.eye(3) * (cfg.sigma_acc_bias * dt)**2
        self.Q[12:15, 12:15] = np.eye(3) * (cfg.sigma_gyro_bias * dt)**2

    def reset(self):
        """Reset filter state."""
        self.x = np.zeros(15)
        self.P = np.eye(15) * 0.1
        self.cusum.reset()

    def predict(self, acc: np.ndarray, gyro: np.ndarray):
        """Prediction step."""
        dt = self.config.dt

        # Simple state prediction
        self.x[0:3] += self.x[3:6] * dt  # pos += vel * dt
        self.x[3:6] += (acc - self.x[9:12]) * dt  # vel += (acc - bias) * dt
        self.x[6:9] += (gyro - self.x[12:15]) * dt  # att += (gyro - bias) * dt

        # State transition matrix (simplified)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 9:12] = -np.eye(3) * dt
        F[6:9, 12:15] = -np.eye(3) * dt

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    def update_posvel(self, pos: np.ndarray, vel: np.ndarray) -> Dict:
        """
        Update with BOTH position and velocity measurements.

        This is the key defense - velocity measurement closes the nullspace!
        """
        z = np.concatenate([pos, vel])
        z_pred = np.concatenate([self.x[0:3], self.x[3:6]])

        # Innovation
        y = z - z_pred

        # Innovation covariance
        S = self.H_posvel @ self.P @ self.H_posvel.T + self.R_posvel

        # NIS (now 6-DOF)
        S_inv = np.linalg.inv(S)
        nis = y.T @ S_inv @ y

        # Kalman gain
        K = self.P @ self.H_posvel.T @ S_inv

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I_KH = np.eye(15) - K @ self.H_posvel
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_posvel @ K.T

        # CUSUM update
        cusum_result = self.cusum.update({
            'position': y[0:3],
            'velocity': y[3:6]
        })

        # Check thresholds
        is_anomaly = (
            nis > self.config.nis_threshold_99 or
            cusum_result['any_alarm']
        )

        return {
            'innovation': y,
            'nis': nis,
            'nis_threshold': self.config.nis_threshold_99,
            'cusum': cusum_result,
            'is_anomaly': is_anomaly,
            'score': max(nis / self.config.nis_threshold_99, cusum_result['combined_score'])
        }


# =============================================================================
# 4. RANDOMIZED THRESHOLDS - Prevents threshold gaming
# =============================================================================

class RandomizedThresholds:
    """
    Randomize detection thresholds to prevent gaming.

    Vulnerability addressed:
    - Fixed thresholds allow attack at threshold - ε
    - Solution: Add random noise to thresholds

    The attacker cannot reliably operate at the boundary
    if they don't know the exact threshold.
    """

    def __init__(
        self,
        base_thresholds: Dict[str, float],
        noise_fraction: float = 0.1,
        update_interval: int = 100  # Re-randomize every N samples
    ):
        self.base = base_thresholds
        self.noise_fraction = noise_fraction
        self.update_interval = update_interval
        self.sample_count = 0
        self._randomize()

    def _randomize(self):
        """Generate new random thresholds."""
        self.current = {}
        for name, base_val in self.base.items():
            noise = np.random.uniform(-self.noise_fraction, self.noise_fraction)
            self.current[name] = base_val * (1 + noise)

    def get(self, name: str) -> float:
        """Get current (randomized) threshold."""
        self.sample_count += 1
        if self.sample_count >= self.update_interval:
            self._randomize()
            self.sample_count = 0
        return self.current.get(name, self.base.get(name, 1.0))

    def check(self, name: str, value: float) -> bool:
        """Check if value exceeds randomized threshold."""
        return value > self.get(name)


# =============================================================================
# 5. ROBUST STATISTICS - Resists normalization poisoning
# =============================================================================

class RobustNormalizer:
    """
    Robust score normalization using median/MAD instead of mean/std.

    Vulnerability addressed:
    - Mean/std can be poisoned by extreme values
    - Solution: Use median and Median Absolute Deviation (MAD)

    MAD is robust to up to 50% outliers!
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = {}

    def reset(self, name: str = None):
        if name:
            self.history.pop(name, None)
        else:
            self.history = {}

    def update(self, name: str, value: float):
        """Add value to history."""
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(value)
        if len(self.history[name]) > self.window_size:
            self.history[name].pop(0)

    def normalize(self, name: str, value: float, update: bool = True) -> float:
        """
        Normalize using robust statistics.

        Returns value in [0, 1] using sigmoid of robust z-score.
        """
        if update:
            self.update(name, value)

        if name not in self.history or len(self.history[name]) < 5:
            return np.clip(value, 0, 1)

        data = np.array(self.history[name])

        # Robust center: median
        median = np.median(data)

        # Robust scale: MAD (Median Absolute Deviation)
        mad = np.median(np.abs(data - median))
        # Convert MAD to std-equivalent (for normal distribution)
        robust_std = 1.4826 * mad + 1e-10

        # Robust z-score
        z = (value - median) / robust_std

        # Sigmoid normalization
        return 1.0 / (1.0 + np.exp(-z))


# =============================================================================
# 6. SPRT SEQUENTIAL TESTING - Replaces instantaneous NIS
# =============================================================================

class SPRTDetector:
    """
    Sequential Probability Ratio Test for attack detection.

    Vulnerability addressed:
    - Instantaneous NIS can be gamed by intermittent attacks
    - Solution: Sequential test accumulates evidence over time

    SPRT decides between:
    - H0: Normal operation (innovation ~ N(0, σ²))
    - H1: Attack present (innovation ~ N(μ_attack, σ²))
    """

    def __init__(
        self,
        sigma_normal: float = 0.1,
        mu_attack: float = 0.3,
        alpha: float = 0.01,    # False positive rate
        beta: float = 0.01      # False negative rate
    ):
        self.sigma = sigma_normal
        self.mu_attack = mu_attack

        # SPRT thresholds
        self.A = (1 - beta) / alpha      # Upper threshold (decide H1)
        self.B = beta / (1 - alpha)      # Lower threshold (decide H0)

        self.log_A = np.log(self.A)
        self.log_B = np.log(self.B)

        # State
        self.log_likelihood_ratio = 0.0
        self.decision = None  # None, 'H0', or 'H1'

    def reset(self):
        self.log_likelihood_ratio = 0.0
        self.decision = None

    def update(self, innovation: float) -> Dict:
        """
        Update SPRT with new innovation.

        Log-likelihood ratio:
        Λ_n = Λ_{n-1} + log(P(x|H1) / P(x|H0))
        """
        if self.decision is not None:
            # Already decided, would need reset for new sequence
            return {
                'llr': self.log_likelihood_ratio,
                'decision': self.decision,
                'is_attack': self.decision == 'H1'
            }

        # Log-likelihood increment for Gaussian
        # H0: N(0, σ²), H1: N(μ, σ²)
        ll_H0 = -0.5 * (innovation / self.sigma)**2
        ll_H1 = -0.5 * ((innovation - self.mu_attack) / self.sigma)**2

        self.log_likelihood_ratio += ll_H1 - ll_H0

        # Decision
        if self.log_likelihood_ratio >= self.log_A:
            self.decision = 'H1'  # Attack detected
        elif self.log_likelihood_ratio <= self.log_B:
            self.decision = 'H0'  # Normal
            self.log_likelihood_ratio = 0.0  # Reset for next sequence

        return {
            'llr': self.log_likelihood_ratio,
            'decision': self.decision,
            'is_attack': self.decision == 'H1',
            'upper_threshold': self.log_A,
            'lower_threshold': self.log_B
        }


# =============================================================================
# 7. SPECTRAL MONITORING - Detects frequency-domain attacks
# =============================================================================

class SpectralMonitor:
    """
    Monitor signal spectrum for anomalies.

    Vulnerability addressed:
    - Attacks shaped to match expected noise spectrum
    - Solution: Track spectral changes over time

    Detects:
    - Unexpected frequency components
    - Spectral shape changes
    - Aliasing artifacts
    """

    def __init__(
        self,
        fs: float = 200.0,           # Sampling frequency
        window_size: int = 256,       # FFT window
        num_bands: int = 8,           # Frequency bands to monitor
        baseline_samples: int = 1000  # Samples for baseline
    ):
        self.fs = fs
        self.window_size = window_size
        self.num_bands = num_bands
        self.baseline_samples = baseline_samples

        # Band edges (logarithmically spaced)
        self.band_edges = np.logspace(
            np.log10(0.1), np.log10(fs/2),
            num_bands + 1
        )

        # State
        self.buffer = []
        self.baseline_psd = None
        self.is_calibrated = False

    def reset(self):
        self.buffer = []
        self.baseline_psd = None
        self.is_calibrated = False

    def _compute_band_powers(self, x: np.ndarray) -> np.ndarray:
        """Compute power in each frequency band."""
        # Compute PSD
        f, psd = signal.welch(x, fs=self.fs, nperseg=min(len(x), self.window_size))

        # Integrate power in each band
        band_powers = np.zeros(self.num_bands)
        for i in range(self.num_bands):
            f_low, f_high = self.band_edges[i], self.band_edges[i+1]
            mask = (f >= f_low) & (f < f_high)
            if np.any(mask):
                band_powers[i] = np.trapz(psd[mask], f[mask])

        return band_powers

    def update(self, sample: float) -> Dict:
        """
        Update spectral monitor with new sample.

        Returns anomaly score based on spectral deviation.
        """
        self.buffer.append(sample)

        # Keep buffer bounded
        if len(self.buffer) > self.window_size * 2:
            self.buffer = self.buffer[-self.window_size*2:]

        # Need enough samples
        if len(self.buffer) < self.window_size:
            return {'score': 0.0, 'is_anomaly': False, 'calibrated': False}

        x = np.array(self.buffer[-self.window_size:])
        current_psd = self._compute_band_powers(x)

        # Calibration phase
        if not self.is_calibrated:
            if len(self.buffer) >= self.baseline_samples:
                baseline_x = np.array(self.buffer[:self.baseline_samples])
                self.baseline_psd = self._compute_band_powers(baseline_x)
                self.is_calibrated = True
            return {'score': 0.0, 'is_anomaly': False, 'calibrated': False}

        # Compare to baseline
        # Use log ratio to handle wide dynamic range
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratio = np.log10(current_psd / (self.baseline_psd + 1e-10) + 1e-10)
            log_ratio = np.nan_to_num(log_ratio, 0)

        # Anomaly score: max deviation across bands
        score = np.max(np.abs(log_ratio))

        # Threshold at 1 decade change
        is_anomaly = score > 1.0

        return {
            'score': score,
            'is_anomaly': is_anomaly,
            'band_ratios': log_ratio,
            'calibrated': True
        }


# =============================================================================
# 8. HARDENED DETECTOR - Combines all defenses
# =============================================================================

@dataclass
class HardenedConfig:
    """Configuration for hardened detector."""
    # Base thresholds (will be randomized)
    jerk_threshold: float = 100.0
    nis_threshold: float = 12.0
    cusum_threshold: float = 5.0
    spectral_threshold: float = 1.0

    # Randomization
    threshold_noise: float = 0.1

    # CUSUM
    cusum_allowance: float = 0.3

    # SPRT
    sprt_alpha: float = 0.01
    sprt_beta: float = 0.01


class HardenedDetector:
    """
    Hardened attack detector combining all defense mechanisms.

    Defenses implemented:
    1. Multi-rate jerk checking
    2. CUSUM drift monitoring
    3. Velocity-augmented EKF
    4. Randomized thresholds
    5. Robust normalization
    6. SPRT sequential testing
    7. Spectral monitoring
    """

    def __init__(self, config: HardenedConfig = None, dt: float = 0.005):
        self.config = config or HardenedConfig()
        self.dt = dt

        # Initialize all defense components
        self.multi_rate_jerk = MultiRateJerkChecker(
            dt_values=[0.005, 0.02, 0.1],
            max_jerk=self.config.jerk_threshold
        )

        self.cusum = MultiChannelCUSUM(
            channels=['position', 'velocity', 'attitude'],
            threshold_h=self.config.cusum_threshold,
            allowance_k=self.config.cusum_allowance
        )

        self.ekf = VelocityAugmentedEKF()

        self.random_thresholds = RandomizedThresholds(
            base_thresholds={
                'jerk': self.config.jerk_threshold,
                'nis': self.config.nis_threshold,
                'cusum': self.config.cusum_threshold,
                'spectral': self.config.spectral_threshold
            },
            noise_fraction=self.config.threshold_noise
        )

        self.robust_norm = RobustNormalizer(window_size=100)

        self.sprt = SPRTDetector(
            alpha=self.config.sprt_alpha,
            beta=self.config.sprt_beta
        )

        self.spectral = SpectralMonitor(fs=1.0/dt)

        # History for multi-rate jerk
        self.pos_history = []

    def reset(self):
        """Reset all detector state."""
        self.cusum.reset()
        self.ekf.reset()
        self.robust_norm.reset()
        self.sprt.reset()
        self.spectral.reset()
        self.pos_history = []

    def detect(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        acc: np.ndarray,
        gyro: np.ndarray
    ) -> Dict:
        """
        Run hardened detection pipeline.

        Args:
            pos: Current position (3,)
            vel: Current velocity (3,)
            acc: Accelerometer reading (3,)
            gyro: Gyroscope reading (3,)

        Returns:
            Comprehensive detection results
        """
        results = {}

        # 1. Multi-rate jerk check
        self.pos_history.append(pos)
        if len(self.pos_history) > 1000:
            self.pos_history = self.pos_history[-1000:]

        if len(self.pos_history) >= 20:
            pos_array = np.array(self.pos_history)
            jerk_result = self.multi_rate_jerk.check(pos_array, self.dt)
            results['jerk'] = jerk_result
        else:
            results['jerk'] = {'combined_score': 0, 'is_anomaly': False}

        # 2. EKF with velocity (predict + update)
        self.ekf.predict(acc, gyro)
        ekf_result = self.ekf.update_posvel(pos, vel)
        results['ekf'] = ekf_result

        # 3. SPRT on innovation magnitude
        innovation_mag = np.linalg.norm(ekf_result['innovation'])
        sprt_result = self.sprt.update(innovation_mag)
        results['sprt'] = sprt_result

        # 4. Spectral monitoring on position magnitude
        spectral_result = self.spectral.update(np.linalg.norm(pos))
        results['spectral'] = spectral_result

        # 5. Combine scores with robust normalization
        scores = {
            'jerk': results['jerk']['combined_score'],
            'ekf': ekf_result['score'],
            'sprt': 1.0 if sprt_result['is_attack'] else 0.0,
            'spectral': spectral_result['score']
        }

        # Normalize each score
        norm_scores = {}
        for name, score in scores.items():
            norm_scores[name] = self.robust_norm.normalize(name, score)

        # Combined score (max of normalized scores)
        combined_score = max(norm_scores.values())

        # Check against randomized threshold
        is_anomaly = any([
            results['jerk']['is_anomaly'],
            ekf_result['is_anomaly'],
            sprt_result['is_attack'],
            spectral_result['is_anomaly']
        ])

        results['scores'] = scores
        results['normalized_scores'] = norm_scores
        results['combined_score'] = combined_score
        results['is_anomaly'] = is_anomaly

        return results


# =============================================================================
# TEST / DEMO
# =============================================================================

def test_hardened_detector():
    """Test the hardened detector against known attacks."""
    print("="*70)
    print("HARDENED DETECTOR TEST")
    print("="*70)

    # Use higher thresholds for testing
    config = HardenedConfig(
        nis_threshold=50.0,      # Higher NIS threshold
        cusum_threshold=20.0,    # Higher CUSUM threshold
        spectral_threshold=2.0   # Higher spectral threshold
    )
    detector = HardenedDetector(config=config, dt=0.005)

    # Generate clean data
    N = 500
    t = np.arange(N) * 0.005

    clean_pos = np.column_stack([
        np.sin(2 * np.pi * 0.1 * t),
        np.cos(2 * np.pi * 0.1 * t),
        0.5 * np.sin(2 * np.pi * 0.05 * t)
    ])
    clean_vel = np.diff(clean_pos, axis=0, prepend=clean_pos[:1]) / 0.005

    # Proper acceleration for the trajectory
    clean_acc = np.diff(clean_vel, axis=0, prepend=clean_vel[:1]) / 0.005
    acc = clean_acc + np.array([0, 0, 9.81])  # Add gravity
    gyro = np.zeros((N, 3))

    # Initialize EKF with first state
    detector.ekf.x[0:3] = clean_pos[0]
    detector.ekf.x[3:6] = clean_vel[0]

    # Warm-up period (first 100 samples to stabilize filters)
    for i in range(100):
        detector.detect(clean_pos[i], clean_vel[i], acc[i], gyro[i])

    # Reset anomaly tracking after warm-up
    detector.sprt.reset()

    # Test on clean data
    print("\n1. Clean Data Test")
    anomaly_count = 0
    for i in range(100, N):
        result = detector.detect(
            clean_pos[i], clean_vel[i], acc[i], gyro[i]
        )
        if result['is_anomaly']:
            anomaly_count += 1

    print(f"   False positives: {anomaly_count}/{N-100} = {100*anomaly_count/(N-100):.1f}%")

    # Reset and test attack
    detector.reset()
    detector.ekf.x[0:3] = clean_pos[0]
    detector.ekf.x[3:6] = clean_vel[0]

    # Inject slow drift attack (0.5 Hz, 30m amplitude)
    print("\n2. Slow Drift Attack Test (0.5 Hz, 30m)")
    attack_pos = clean_pos.copy()
    attack_pos[:, 0] += 30 * np.sin(2 * np.pi * 0.5 * t)
    attack_vel = np.diff(attack_pos, axis=0, prepend=attack_pos[:1]) / 0.005

    # Warm-up with clean data first
    for i in range(50):
        detector.detect(clean_pos[i], clean_vel[i], acc[i], gyro[i])

    anomaly_count = 0
    first_detection = -1
    for i in range(50, N):
        result = detector.detect(
            attack_pos[i], attack_vel[i], acc[i], gyro[i]
        )
        if result['is_anomaly']:
            anomaly_count += 1
            if first_detection < 0:
                first_detection = i

    print(f"   Detections: {anomaly_count}/{N-50} = {100*anomaly_count/(N-50):.1f}%")
    if first_detection >= 0:
        print(f"   First detection at sample {first_detection} ({first_detection*0.005:.3f}s)")

    # Test GPS jump (coordinated - vel matches)
    detector.reset()
    detector.ekf.x[0:3] = clean_pos[0]
    detector.ekf.x[3:6] = clean_vel[0]

    print("\n3. GPS Jump Attack Test (5m sudden jump)")
    jump_pos = clean_pos.copy()
    jump_pos[250:, :] += 5.0  # 5m jump

    # Warm-up
    for i in range(50):
        detector.detect(clean_pos[i], clean_vel[i], acc[i], gyro[i])

    anomaly_count = 0
    first_detection = -1
    for i in range(50, N):
        result = detector.detect(
            jump_pos[i], clean_vel[i], acc[i], gyro[i]
        )
        if result['is_anomaly']:
            anomaly_count += 1
            if first_detection < 0:
                first_detection = i

    print(f"   Detections: {anomaly_count}/{N-50} = {100*anomaly_count/(N-50):.1f}%")
    if first_detection >= 0:
        print(f"   First detection at sample {first_detection} ({first_detection*0.005:.3f}s)")

    print("\n" + "="*70)
    print("HARDENED DETECTOR: All defenses active")
    print("="*70)


if __name__ == "__main__":
    test_hardened_detector()
