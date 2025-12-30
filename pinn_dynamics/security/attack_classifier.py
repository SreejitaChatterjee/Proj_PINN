"""
Multi-Class Attack Classification.

Combines ML-based classification with rule-based refinement for
CPU-friendly attack type identification.

Attack Categories:
1. GPS attacks (drift, jump, oscillation, meaconing, jamming, freeze)
2. IMU attacks (bias, drift, noise, saturation)
3. Sensor spoofing (baro, mag)
4. Actuator faults (stuck, degraded, hijack)
5. Coordinated attacks (GPS+IMU, stealthy)
6. Temporal attacks (replay, delay, dropout)

CPU Impact: Small MLP + rule engine; O(1) per sample.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class AttackCategory(Enum):
    """High-level attack categories."""
    NORMAL = 0
    GPS = 1
    IMU = 2
    SENSOR_SPOOF = 3
    ACTUATOR = 4
    COORDINATED = 5
    TEMPORAL = 6
    UNKNOWN = 7


class AttackType(Enum):
    """Detailed attack types."""
    NORMAL = 0
    # GPS
    GPS_DRIFT = 1
    GPS_JUMP = 2
    GPS_OSCILLATION = 3
    GPS_MEACONING = 4
    GPS_JAMMING = 5
    GPS_FREEZE = 6
    # IMU
    IMU_BIAS = 10
    IMU_DRIFT = 11
    IMU_NOISE = 12
    IMU_SATURATION = 13
    # Sensor
    BARO_SPOOF = 20
    MAG_SPOOF = 21
    # Actuator
    ACTUATOR_STUCK = 30
    ACTUATOR_DEGRADED = 31
    CONTROL_HIJACK = 32
    THRUST_MANIPULATION = 33
    # Coordinated
    COORDINATED_GPS_IMU = 40
    STEALTHY_COORDINATED = 41
    # Temporal
    REPLAY = 50
    TIME_DELAY = 51
    SENSOR_DROPOUT = 52
    # Unknown
    UNKNOWN = 99


@dataclass
class AttackSignature:
    """Feature signature for attack classification."""
    # Position-based
    pos_jump_magnitude: float = 0.0      # Max position change in window
    pos_drift_rate: float = 0.0          # Linear drift rate
    pos_oscillation_freq: float = 0.0    # Dominant oscillation frequency
    pos_freeze_duration: int = 0         # Consecutive unchanged samples

    # Velocity-based
    vel_pos_mismatch: float = 0.0        # Velocity vs position derivative

    # IMU-based
    accel_magnitude: float = 0.0         # Accelerometer magnitude
    gyro_magnitude: float = 0.0          # Gyroscope magnitude
    imu_saturation_ratio: float = 0.0    # Fraction at saturation limits

    # Sensor consistency
    baro_pos_diff: float = 0.0           # Baro vs GPS altitude
    mag_heading_diff: float = 0.0        # Mag vs yaw heading

    # Temporal
    autocorr_lag1: float = 0.0           # Autocorrelation at lag 1
    repeat_segment_score: float = 0.0    # Similarity to past segments

    # Control
    thrust_deviation: float = 0.0        # Deviation from expected thrust
    control_response_delay: float = 0.0  # Delay between control and response


class RuleBasedClassifier:
    """
    Rule-based attack type classifier.

    Uses physics-based rules to classify attack types from feature signatures.
    """

    def __init__(self):
        # Thresholds (tunable)
        self.jump_threshold = 0.5        # m - position jump
        self.drift_threshold = 0.01      # m/s - drift rate
        self.oscillation_threshold = 0.3  # Hz - min oscillation freq
        self.freeze_threshold = 10       # samples - freeze duration
        self.saturation_threshold = 0.1  # ratio - saturation
        self.baro_diff_threshold = 1.0   # m - baro/pos difference
        self.mag_diff_threshold = 0.2    # rad - heading difference
        self.replay_threshold = 0.9      # correlation - replay detection
        self.thrust_threshold = 3.0      # N - thrust deviation

    def classify(self, sig: AttackSignature) -> Tuple[AttackCategory, AttackType, float]:
        """
        Classify attack from signature.

        Args:
            sig: Attack signature features

        Returns:
            category: High-level category
            attack_type: Detailed type
            confidence: Classification confidence [0, 1]
        """
        scores = {}

        # GPS attacks
        if sig.pos_jump_magnitude > self.jump_threshold:
            scores[AttackType.GPS_JUMP] = min(sig.pos_jump_magnitude / self.jump_threshold, 3.0)

        if sig.pos_drift_rate > self.drift_threshold:
            scores[AttackType.GPS_DRIFT] = min(sig.pos_drift_rate / self.drift_threshold, 3.0)

        if sig.pos_oscillation_freq > self.oscillation_threshold:
            scores[AttackType.GPS_OSCILLATION] = 1.0 + sig.pos_oscillation_freq

        if sig.pos_freeze_duration > self.freeze_threshold:
            scores[AttackType.GPS_FREEZE] = min(sig.pos_freeze_duration / self.freeze_threshold, 3.0)

        # IMU attacks
        if sig.imu_saturation_ratio > self.saturation_threshold:
            scores[AttackType.IMU_SATURATION] = sig.imu_saturation_ratio / self.saturation_threshold

        if sig.vel_pos_mismatch > 0.5 and sig.pos_jump_magnitude < self.jump_threshold:
            scores[AttackType.IMU_BIAS] = sig.vel_pos_mismatch

        # Sensor spoofing
        if sig.baro_pos_diff > self.baro_diff_threshold:
            scores[AttackType.BARO_SPOOF] = sig.baro_pos_diff / self.baro_diff_threshold

        if sig.mag_heading_diff > self.mag_diff_threshold:
            scores[AttackType.MAG_SPOOF] = sig.mag_heading_diff / self.mag_diff_threshold

        # Control attacks
        if sig.thrust_deviation > self.thrust_threshold:
            scores[AttackType.CONTROL_HIJACK] = sig.thrust_deviation / self.thrust_threshold

        # Temporal attacks
        if sig.repeat_segment_score > self.replay_threshold:
            scores[AttackType.REPLAY] = sig.repeat_segment_score

        # Coordinated (multiple signals)
        gps_score = max(
            scores.get(AttackType.GPS_JUMP, 0),
            scores.get(AttackType.GPS_DRIFT, 0),
            scores.get(AttackType.GPS_OSCILLATION, 0)
        )
        imu_score = max(
            scores.get(AttackType.IMU_BIAS, 0),
            scores.get(AttackType.IMU_SATURATION, 0)
        )
        if gps_score > 0.5 and imu_score > 0.5:
            scores[AttackType.COORDINATED_GPS_IMU] = (gps_score + imu_score) / 2

        # Select best match
        if not scores:
            return AttackCategory.NORMAL, AttackType.NORMAL, 1.0

        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] / 3.0, 1.0)  # Normalize to [0, 1]

        # Map to category
        category = self._type_to_category(best_type)

        return category, best_type, confidence

    def _type_to_category(self, attack_type: AttackType) -> AttackCategory:
        """Map attack type to category."""
        if attack_type.value == 0:
            return AttackCategory.NORMAL
        elif 1 <= attack_type.value < 10:
            return AttackCategory.GPS
        elif 10 <= attack_type.value < 20:
            return AttackCategory.IMU
        elif 20 <= attack_type.value < 30:
            return AttackCategory.SENSOR_SPOOF
        elif 30 <= attack_type.value < 40:
            return AttackCategory.ACTUATOR
        elif 40 <= attack_type.value < 50:
            return AttackCategory.COORDINATED
        elif 50 <= attack_type.value < 60:
            return AttackCategory.TEMPORAL
        else:
            return AttackCategory.UNKNOWN


class FeatureExtractor:
    """Extract attack signature features from sensor data."""

    def __init__(self, window_size: int = 100, dt: float = 0.005):
        self.window_size = window_size
        self.dt = dt
        self.history = []

    def reset(self):
        """Reset feature extractor state."""
        self.history = []

    def extract(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        att: np.ndarray,
        rate: np.ndarray,
        control: Optional[np.ndarray] = None,
        baro_z: Optional[float] = None,
        mag_heading: Optional[float] = None
    ) -> AttackSignature:
        """
        Extract attack signature features from current window.

        Args:
            pos: Position history (window_size, 3)
            vel: Velocity history (window_size, 3)
            att: Attitude history (window_size, 3)
            rate: Angular rate history (window_size, 3)
            control: Control history (window_size, 4) optional
            baro_z: Current baro altitude
            mag_heading: Current mag heading

        Returns:
            Attack signature features
        """
        sig = AttackSignature()
        W = len(pos)

        # Position jump
        pos_diff = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        sig.pos_jump_magnitude = np.max(pos_diff)

        # Position drift rate (linear trend)
        t = np.arange(W) * self.dt
        for i in range(3):
            slope = np.polyfit(t, pos[:, i], 1)[0]
            sig.pos_drift_rate = max(sig.pos_drift_rate, abs(slope))

        # Oscillation frequency (via zero crossings)
        pos_centered = pos - np.mean(pos, axis=0)
        zero_crossings = np.sum(np.diff(np.sign(pos_centered[:, 0])) != 0)
        sig.pos_oscillation_freq = zero_crossings / (2 * W * self.dt)

        # Freeze detection (unchanged position)
        pos_changes = np.sum(np.abs(np.diff(pos, axis=0)), axis=1)
        freeze_mask = pos_changes < 1e-6
        if np.any(freeze_mask):
            # Find longest consecutive freeze
            changes = np.diff(np.concatenate([[0], freeze_mask.astype(int), [0]]))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            if len(starts) > 0 and len(ends) > 0:
                sig.pos_freeze_duration = np.max(ends - starts)

        # Velocity-position mismatch
        pos_deriv = np.diff(pos, axis=0) / self.dt
        vel_trimmed = vel[1:]
        sig.vel_pos_mismatch = np.mean(np.linalg.norm(pos_deriv - vel_trimmed, axis=1))

        # IMU magnitudes
        sig.gyro_magnitude = np.mean(np.linalg.norm(rate, axis=1))

        # Saturation detection (angular rates > 4 rad/s)
        saturation_mask = np.linalg.norm(rate, axis=1) > 4.0
        sig.imu_saturation_ratio = np.mean(saturation_mask)

        # Sensor consistency
        if baro_z is not None:
            sig.baro_pos_diff = abs(baro_z - pos[-1, 2])

        if mag_heading is not None:
            heading_diff = mag_heading - att[-1, 2]
            sig.mag_heading_diff = abs(np.mod(heading_diff + np.pi, 2*np.pi) - np.pi)

        # Autocorrelation (replay detection)
        if W > 10:
            centered = pos[:, 0] - np.mean(pos[:, 0])
            autocorr = np.correlate(centered, centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if autocorr[0] > 1e-10:
                sig.autocorr_lag1 = autocorr[1] / autocorr[0]

        # Replay segment detection
        if len(self.history) > 0:
            current_seg = pos.flatten()
            best_match = 0
            for hist_seg in self.history[-10:]:  # Check last 10 segments
                if len(hist_seg) == len(current_seg):
                    corr = np.corrcoef(current_seg, hist_seg)[0, 1]
                    if not np.isnan(corr):
                        best_match = max(best_match, abs(corr))
            sig.repeat_segment_score = best_match

        # Control deviation
        if control is not None and len(control) > 0:
            expected_thrust = 10.0  # Nominal hover thrust
            sig.thrust_deviation = abs(np.mean(control[:, 0]) - expected_thrust)

        # Store for replay detection
        self.history.append(pos.flatten())
        if len(self.history) > 20:
            self.history.pop(0)

        return sig


class HybridClassifier:
    """
    Hybrid ML + Rule-based attack classifier.

    Combines:
    1. Rule-based classification for interpretability
    2. Feature-based scoring for confidence
    3. Temporal smoothing for stability
    """

    def __init__(self, window_size: int = 100, dt: float = 0.005):
        self.feature_extractor = FeatureExtractor(window_size, dt)
        self.rule_classifier = RuleBasedClassifier()
        self.window_size = window_size
        self.dt = dt

        # Temporal smoothing
        self.category_history = []
        self.smoothing_window = 5

    def reset(self):
        """Reset classifier state."""
        self.feature_extractor.reset()
        self.category_history = []

    def classify(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        att: np.ndarray,
        rate: np.ndarray,
        control: Optional[np.ndarray] = None,
        baro_z: Optional[float] = None,
        mag_heading: Optional[float] = None
    ) -> Dict:
        """
        Classify attack type from sensor window.

        Args:
            pos: Position history (window_size, 3)
            vel: Velocity history (window_size, 3)
            att: Attitude history (window_size, 3)
            rate: Angular rate history (window_size, 3)
            control: Control history (window_size, 4) optional
            baro_z: Current baro altitude
            mag_heading: Current mag heading

        Returns:
            Dict with category, type, confidence, signature
        """
        # Extract features
        signature = self.feature_extractor.extract(
            pos, vel, att, rate, control, baro_z, mag_heading
        )

        # Rule-based classification
        category, attack_type, confidence = self.rule_classifier.classify(signature)

        # Temporal smoothing
        self.category_history.append(category.value)
        if len(self.category_history) > self.smoothing_window:
            self.category_history.pop(0)

        # Majority voting for stability
        if len(self.category_history) >= 3:
            counts = {}
            for c in self.category_history:
                counts[c] = counts.get(c, 0) + 1
            smoothed_category = max(counts, key=counts.get)
            category = AttackCategory(smoothed_category)

        return {
            'category': category,
            'type': attack_type,
            'confidence': confidence,
            'signature': signature,
            'category_name': category.name,
            'type_name': attack_type.name
        }


def run_attack_classifier(
    df,
    window_size: int = 100,
    emulated_sensors: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Run attack classifier on dataframe.

    Args:
        df: DataFrame with state columns
        window_size: Classification window size
        emulated_sensors: Optional emulated sensor data

    Returns:
        categories: Category predictions (N,)
        types: Type predictions (N,)
        results: Full classification results
    """
    classifier = HybridClassifier(window_size)

    N = len(df)
    categories = np.zeros(N, dtype=int)
    types = np.zeros(N, dtype=int)
    results = []

    for i in range(window_size, N):
        # Get window
        pos = df[['x', 'y', 'z']].values[i-window_size:i]
        vel = df[['vx', 'vy', 'vz']].values[i-window_size:i]
        att = df[['phi', 'theta', 'psi']].values[i-window_size:i]
        rate = df[['p', 'q', 'r']].values[i-window_size:i]

        control = None
        if 'thrust' in df.columns:
            control = df[['thrust', 'torque_x', 'torque_y', 'torque_z']].values[i-window_size:i]

        baro_z = emulated_sensors['baro_z'][i] if emulated_sensors else None
        mag_heading = emulated_sensors['mag_heading'][i] if emulated_sensors else None

        # Classify
        result = classifier.classify(pos, vel, att, rate, control, baro_z, mag_heading)

        categories[i] = result['category'].value
        types[i] = result['type'].value
        results.append(result)

    return categories, types, results
