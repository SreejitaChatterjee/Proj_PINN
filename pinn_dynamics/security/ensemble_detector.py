"""
Ensemble Anomaly Detector for UAV Attack Detection.

Combines multiple detection strategies to catch attacks that single methods miss:

1. PINN-based: Physics violation detection (high-magnitude attacks)
2. Sequence-PINN: Temporal pattern detection (gradual/stealth attacks)
3. Cross-sensor: GPS vs IMU consistency (spoofing attacks)
4. Statistical: Noise fingerprint analysis (synthetic/replay attacks)
5. Similarity: Sequence repetition detection (replay attacks)

Each detector votes, and ensemble combines votes with learned weights.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque
import warnings


@dataclass
class DetectionResult:
    """Result from ensemble detection."""
    is_anomaly: bool
    confidence: float  # 0-1, higher = more confident
    scores: Dict[str, float]  # Per-detector scores
    votes: Dict[str, bool]  # Per-detector votes
    triggered_detectors: List[str]  # Which detectors triggered
    attack_type_hint: Optional[str] = None  # Suggested attack type


class CrossSensorDetector:
    """
    Detect attacks by checking physics consistency of state transitions.

    Instead of dead-reckoning (which accumulates drift), we check:
    1. Velocity-position consistency: Does position change match velocity?
    2. Acceleration-velocity consistency: Does velocity change match expected dynamics?
    3. Sudden jump detection: Large instantaneous state changes
    """

    def __init__(
        self,
        position_velocity_threshold: float = 10.0,  # Very permissive - 10m inconsistency
        velocity_change_threshold: float = 20.0,  # Very permissive - 20 m/s change
        sudden_jump_threshold: float = 10.0,  # Very permissive - 10m jump
        dt: float = 0.005,  # 200 Hz
    ):
        self.position_velocity_threshold = position_velocity_threshold
        self.velocity_change_threshold = velocity_change_threshold
        self.sudden_jump_threshold = sudden_jump_threshold
        self.dt = dt

        # Previous state for consistency checks
        self.prev_position = None
        self.prev_velocity = None
        self.initialized = False

        # Running statistics for adaptive thresholds
        self.position_errors = []
        self.max_history = 200

    def reset(self):
        """Reset detector state."""
        self.prev_position = None
        self.prev_velocity = None
        self.initialized = False
        self.position_errors = []

    def update(
        self,
        gps_position: np.ndarray,  # [x, y, z]
        imu_acceleration: np.ndarray,  # [ax, ay, az] body frame (optional, can be None)
        attitude: np.ndarray,  # [phi, theta, psi]
        velocity: Optional[np.ndarray] = None,  # [vx, vy, vz] if available
    ) -> Tuple[float, bool]:
        """
        Check physics consistency of current state.

        Returns:
            (inconsistency_score, is_anomaly)
        """
        # Initialize on first call
        if not self.initialized:
            self.prev_position = gps_position.copy()
            self.prev_velocity = velocity.copy() if velocity is not None else np.zeros(3)
            self.initialized = True
            return 0.0, False

        scores = []
        anomalies = []

        # Check 1: Position change vs velocity consistency
        if velocity is not None:
            expected_position_change = self.prev_velocity * self.dt
            actual_position_change = gps_position - self.prev_position
            position_error = np.linalg.norm(actual_position_change - expected_position_change)

            # Adaptive threshold based on history
            self.position_errors.append(position_error)
            if len(self.position_errors) > self.max_history:
                self.position_errors.pop(0)

            if len(self.position_errors) >= 50:
                adaptive_threshold = np.percentile(self.position_errors, 99) * 2
            else:
                adaptive_threshold = self.position_velocity_threshold

            pos_score = position_error / max(adaptive_threshold, 0.01)
            scores.append(pos_score)
            anomalies.append(position_error > adaptive_threshold)

        # Check 2: Sudden position jump
        position_change = np.linalg.norm(gps_position - self.prev_position)
        max_expected_change = np.linalg.norm(self.prev_velocity) * self.dt * 3 + 0.1
        jump_threshold = max(self.sudden_jump_threshold, max_expected_change)

        if position_change > jump_threshold:
            scores.append(position_change / jump_threshold)
            anomalies.append(True)

        # Check 3: Velocity continuity (if available)
        if velocity is not None and self.prev_velocity is not None:
            velocity_change = np.linalg.norm(velocity - self.prev_velocity)
            max_expected_v_change = 50.0 * self.dt  # Max ~50 m/s^2 acceleration
            if velocity_change > max(self.velocity_change_threshold * self.dt, max_expected_v_change):
                v_score = velocity_change / self.velocity_change_threshold
                scores.append(v_score)
                anomalies.append(True)

        # Update state
        self.prev_position = gps_position.copy()
        if velocity is not None:
            self.prev_velocity = velocity.copy()

        # Combine scores
        if not scores:
            return 0.0, False

        total_score = max(scores)
        is_anomaly = any(anomalies)

        return float(total_score), is_anomaly


class StatisticalFingerprintDetector:
    """
    Detect attacks by analyzing statistical properties of state CHANGES.

    Key insight: We analyze rate-of-change (derivatives) not absolute values,
    because absolute position/velocity naturally changes during flight.

    Checks:
    1. Derivative smoothness - Real sensors have characteristic noise on derivatives
    2. Sudden discontinuities - Jumps in derivatives indicate tampering
    3. Unusual periodicity - Synthetic attacks often have periodic patterns
    """

    def __init__(
        self,
        window_size: int = 100,  # Larger window for more stable statistics
        derivative_jump_threshold: float = 20.0,  # Very high - 20 std devs
        smoothness_threshold: float = 15.0,  # Very high roughness threshold
    ):
        self.window_size = window_size
        self.derivative_jump_threshold = derivative_jump_threshold
        self.smoothness_threshold = smoothness_threshold

        # Buffers for windowed analysis
        self.signal_buffer = deque(maxlen=window_size)
        self.derivative_buffer = deque(maxlen=window_size - 1)

        # Calibrated derivative statistics
        self.derivative_mean = None
        self.derivative_std = None
        self.calibrated = False

    def calibrate(self, normal_signals: np.ndarray):
        """
        Calibrate on normal (attack-free) data derivatives.

        Args:
            normal_signals: [N, signal_dim] normal sensor readings
        """
        # Compute derivatives (differences)
        derivatives = np.diff(normal_signals, axis=0)

        self.derivative_mean = np.mean(derivatives, axis=0)
        self.derivative_std = np.std(derivatives, axis=0) + 1e-8

        # Second derivatives for smoothness calibration
        second_derivatives = np.diff(derivatives, axis=0)
        self.second_deriv_std = np.std(second_derivatives, axis=0) + 1e-8

        self.calibrated = True

    def update(self, signal: np.ndarray) -> Tuple[float, bool]:
        """
        Analyze signal changes for statistical anomalies.

        Args:
            signal: [signal_dim] current sensor reading

        Returns:
            (anomaly_score, is_anomaly)
        """
        self.signal_buffer.append(signal.copy())

        if len(self.signal_buffer) < 2:
            return 0.0, False

        # Compute derivative
        derivative = signal - np.array(self.signal_buffer)[-2]
        self.derivative_buffer.append(derivative)

        if not self.calibrated or len(self.derivative_buffer) < self.window_size - 1:
            return 0.0, False

        # Convert buffer to array
        derivatives = np.array(self.derivative_buffer)

        scores = []
        anomalies = []

        # Check 1: Derivative jump detection (sudden discontinuity)
        current_deriv = derivatives[-1]
        recent_deriv_mean = np.mean(derivatives[:-1], axis=0)
        recent_deriv_std = np.std(derivatives[:-1], axis=0) + 1e-8

        deriv_zscore = np.abs(current_deriv - recent_deriv_mean) / recent_deriv_std
        max_zscore = np.max(deriv_zscore)

        if max_zscore > self.derivative_jump_threshold:
            scores.append(max_zscore / self.derivative_jump_threshold)
            anomalies.append(True)

        # Check 2: Smoothness (second derivative magnitude)
        if len(derivatives) >= 2:
            second_deriv = derivatives[-1] - derivatives[-2]
            smoothness_score = np.linalg.norm(second_deriv / self.second_deriv_std)

            if smoothness_score > self.smoothness_threshold:
                scores.append(smoothness_score / self.smoothness_threshold)
                anomalies.append(True)

        # Check 3: Variance change in derivatives (attack may alter noise patterns)
        window_deriv_std = np.std(derivatives, axis=0)
        std_ratio = window_deriv_std / self.derivative_std
        # Only flag extreme changes (>3x or <0.3x normal)
        max_std_change = np.max(np.abs(np.log(std_ratio + 1e-8)))
        if max_std_change > 1.1:  # >3x change
            scores.append(max_std_change)
            anomalies.append(True)

        # Combine
        if not scores:
            return 0.0, False

        total_score = max(scores)
        is_anomaly = any(anomalies)

        return float(total_score), is_anomaly


class SequenceSimilarityDetector:
    """
    Detect replay attacks by finding suspiciously similar sequences.

    Key insight: Replay attacks copy EARLIER data to CURRENT time.
    We need dense history coverage to catch replays from any point.

    Also detects:
    1. Exact replay (high cosine similarity)
    2. Time-shifted replay (compare with time offset)
    3. Freeze attacks (very low variance in sequence)
    """

    def __init__(
        self,
        sequence_length: int = 30,
        similarity_threshold: float = 0.98,  # Cosine similarity
        history_size: int = 100,  # Reduced from 500 for performance
        freeze_threshold: float = 0.001,  # Max variance for freeze detection
        check_limit: int = 20,  # Max history entries to check per sample
    ):
        self.sequence_length = sequence_length
        self.similarity_threshold = similarity_threshold
        self.history_size = history_size
        self.freeze_threshold = freeze_threshold
        self.check_limit = check_limit

        # Current sequence buffer
        self.current_buffer = deque(maxlen=sequence_length)

        # History of past sequences with dense coverage
        self.history: List[np.ndarray] = []
        self.history_timestamps: List[int] = []
        self.sample_count = 0
        self.save_interval = max(20, sequence_length)  # Less frequent saves

        # For freeze detection
        self.recent_variances = deque(maxlen=20)

    def reset(self):
        """Reset detector state."""
        self.current_buffer.clear()
        self.history.clear()
        self.history_timestamps.clear()
        self.sample_count = 0
        self.recent_variances.clear()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two sequences."""
        a_flat = a.flatten()
        b_flat = b.flatten()

        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

    def _normalized_mse(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute normalized MSE for better replay detection."""
        a_norm = (a - np.mean(a)) / (np.std(a) + 1e-8)
        b_norm = (b - np.mean(b)) / (np.std(b) + 1e-8)
        return float(np.mean((a_norm - b_norm) ** 2))

    def update(self, state: np.ndarray) -> Tuple[float, bool]:
        """
        Check if current state sequence matches any past sequence.

        Args:
            state: [state_dim] current state

        Returns:
            (max_similarity, is_replay)
        """
        self.current_buffer.append(state.copy())
        self.sample_count += 1

        if len(self.current_buffer) < self.sequence_length:
            return 0.0, False

        current_seq = np.array(self.current_buffer)

        # Check for freeze attack (very low variance)
        seq_variance = np.var(current_seq)
        self.recent_variances.append(seq_variance)

        if len(self.recent_variances) >= 10:
            avg_variance = np.mean(self.recent_variances)
            if avg_variance < self.freeze_threshold:
                return 1.0, True  # Freeze detected

        # Check against history for replay
        max_similarity = 0.0
        is_replay = False

        # Only check if we have meaningful history (not too recent)
        min_time_gap = self.sequence_length * 3  # Require gap between current and history

        # Limit checks for performance - sample from history
        history_to_check = self.history
        timestamps_to_check = self.history_timestamps
        if len(self.history) > self.check_limit:
            # Sample evenly from history
            indices = np.linspace(0, len(self.history) - 1, self.check_limit, dtype=int)
            history_to_check = [self.history[i] for i in indices]
            timestamps_to_check = [self.history_timestamps[i] for i in indices]

        for i, past_seq in enumerate(history_to_check):
            past_time = timestamps_to_check[i]
            time_gap = self.sample_count - past_time

            # Skip if too recent (normal temporal correlation)
            if time_gap < min_time_gap:
                continue

            # Cosine similarity
            similarity = self._cosine_similarity(current_seq, past_seq)

            if similarity > max_similarity:
                max_similarity = similarity

            if similarity > self.similarity_threshold:
                is_replay = True
                break

            # Also check normalized MSE for replay detection
            nmse = self._normalized_mse(current_seq, past_seq)
            if nmse < 0.1:  # Very similar after normalization
                max_similarity = max(max_similarity, 1.0 - nmse)
                if nmse < 0.02:
                    is_replay = True
                    break

        # Save current sequence to history (more frequently for better coverage)
        if self.sample_count % self.save_interval == 0:
            self.history.append(current_seq.copy())
            self.history_timestamps.append(self.sample_count)

            # Limit history size (keep oldest and newest, drop middle randomly)
            while len(self.history) > self.history_size:
                # Remove from middle to keep both old and recent coverage
                mid_idx = len(self.history) // 2
                self.history.pop(mid_idx)
                self.history_timestamps.pop(mid_idx)

        return float(max_similarity), is_replay


class EnsembleDetector:
    """
    Ensemble detector combining multiple detection strategies.

    Strategies:
    1. Cross-sensor consistency (GPS vs IMU)
    2. Statistical fingerprinting (noise patterns)
    3. Sequence similarity (replay detection)
    4. Optional: PINN-based detectors (loaded separately)

    Voting:
    - Each detector provides a score and binary vote
    - Ensemble combines votes with configurable weights
    - Final decision based on weighted vote or max score
    """

    def __init__(
        self,
        cross_sensor_weight: float = 1.0,  # Equal weights for voting
        statistical_weight: float = 1.0,
        similarity_weight: float = 1.0,
        pinn_weight: float = 1.0,
        sequence_pinn_weight: float = 1.0,
        voting_threshold: float = 0.5,  # Require majority voting
        use_max_score: bool = True,  # Enable max score mode for high-confidence detections
        max_score_threshold: float = 3.0,  # Raised - only very high scores trigger alone
        min_detectors_agree: int = 2,  # Require at least 2 detectors for robustness
    ):
        # Weights
        self.weights = {
            'cross_sensor': cross_sensor_weight,
            'statistical': statistical_weight,
            'similarity': similarity_weight,
            'pinn': pinn_weight,
            'sequence_pinn': sequence_pinn_weight,
        }

        self.voting_threshold = voting_threshold
        self.use_max_score = use_max_score
        self.max_score_threshold = max_score_threshold
        self.min_detectors_agree = min_detectors_agree

        # Initialize sub-detectors with redesigned architecture
        self.cross_sensor = CrossSensorDetector()  # Uses velocity consistency, not DR
        self.statistical = StatisticalFingerprintDetector()  # Uses derivatives, not absolutes
        self.similarity = SequenceSimilarityDetector()  # Includes freeze detection

        # PINN detectors (optional, set via set_pinn_detector)
        self.pinn_detector = None
        self.sequence_pinn_detector = None

        # Detection history for analysis
        self.detection_history: List[DetectionResult] = []

    def set_pinn_detector(self, detector, detector_type: str = 'pinn'):
        """
        Set a PINN-based detector.

        Args:
            detector: Object with detect(state, control, next_state) method
            detector_type: 'pinn' or 'sequence_pinn'
        """
        if detector_type == 'pinn':
            self.pinn_detector = detector
        elif detector_type == 'sequence_pinn':
            self.sequence_pinn_detector = detector
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

    def calibrate(self, normal_data: np.ndarray):
        """
        Calibrate detectors on normal (attack-free) data.

        Args:
            normal_data: [N, features] normal sensor data
                Expected columns: [x, y, z, phi, theta, psi, p, q, r, vx, vy, vz, ...]
        """
        # Calibrate statistical detector on state derivatives
        self.statistical.calibrate(normal_data[:, :12])  # State variables

        # Initialize cross-sensor detector with warmup samples
        self.cross_sensor.reset()
        for i in range(min(200, len(normal_data))):
            gps_pos = normal_data[i, :3]
            attitude = normal_data[i, 3:6]
            velocity = normal_data[i, 9:12] if normal_data.shape[1] >= 12 else None
            self.cross_sensor.update(gps_pos, None, attitude, velocity)

        # Build similarity history for replay detection (sparse for performance)
        self.similarity.reset()
        # Use every 50th sample for reasonable coverage without slowdown
        for i in range(0, min(len(normal_data), 5000), 50):
            self.similarity.update(normal_data[i, :12])

    def reset(self):
        """Reset all detector states."""
        self.cross_sensor.reset()
        self.similarity.reset()
        self.statistical.signal_buffer.clear()
        self.statistical.derivative_buffer.clear()
        self.detection_history.clear()

    def detect(
        self,
        state: np.ndarray,  # [12] current state
        control: Optional[np.ndarray] = None,  # [4] control input
        imu_acceleration: Optional[np.ndarray] = None,  # [3] IMU reading
        next_state: Optional[np.ndarray] = None,  # [12] for PINN detector
    ) -> DetectionResult:
        """
        Run ensemble detection on current sample.

        Args:
            state: [x, y, z, phi, theta, psi, p, q, r, vx, vy, vz]
            control: [thrust, torque_x, torque_y, torque_z]
            imu_acceleration: [ax, ay, az] body frame
            next_state: Next state (for PINN prediction error)

        Returns:
            DetectionResult with ensemble decision
        """
        scores = {}
        votes = {}
        triggered = []

        # Extract state components
        gps_pos = state[:3]
        attitude = state[3:6]
        velocity = state[9:12] if len(state) >= 12 else None

        # 1. Cross-sensor consistency (now uses velocity, not dead-reckoning)
        score, vote = self.cross_sensor.update(gps_pos, imu_acceleration, attitude, velocity)
        scores['cross_sensor'] = score
        votes['cross_sensor'] = vote
        if vote:
            triggered.append('cross_sensor')

        # 2. Statistical fingerprinting
        stat_score, stat_vote = self.statistical.update(state)
        scores['statistical'] = stat_score
        votes['statistical'] = stat_vote
        if stat_vote:
            triggered.append('statistical')

        # 3. Sequence similarity (replay detection)
        sim_score, sim_vote = self.similarity.update(state)
        scores['similarity'] = sim_score
        votes['similarity'] = sim_vote
        if sim_vote:
            triggered.append('similarity')

        # 4. PINN detector (if available)
        if self.pinn_detector is not None and next_state is not None:
            try:
                pinn_result = self.pinn_detector.detect(state, control, next_state)
                scores['pinn'] = pinn_result.score if hasattr(pinn_result, 'score') else 0.0
                votes['pinn'] = pinn_result.is_anomaly if hasattr(pinn_result, 'is_anomaly') else False
                if votes['pinn']:
                    triggered.append('pinn')
            except Exception:
                pass

        # 5. Sequence-PINN detector (if available)
        if self.sequence_pinn_detector is not None:
            try:
                seq_result = self.sequence_pinn_detector.detect(state)
                scores['sequence_pinn'] = seq_result.score if hasattr(seq_result, 'score') else 0.0
                votes['sequence_pinn'] = seq_result.is_anomaly if hasattr(seq_result, 'is_anomaly') else False
                if votes['sequence_pinn']:
                    triggered.append('sequence_pinn')
            except Exception:
                pass

        # Compute weighted vote
        total_weight = 0.0
        weighted_votes = 0.0
        weighted_score = 0.0

        for detector_name, vote in votes.items():
            weight = self.weights.get(detector_name, 1.0)
            total_weight += weight
            if vote:
                weighted_votes += weight
            weighted_score += weight * scores.get(detector_name, 0.0)

        if total_weight > 0:
            vote_fraction = weighted_votes / total_weight
            avg_score = weighted_score / total_weight
        else:
            vote_fraction = 0.0
            avg_score = 0.0

        # Count how many detectors triggered
        num_triggered = len(triggered)

        # Final decision: require BOTH weighted vote threshold AND minimum detector agreement
        if self.use_max_score:
            max_score = max(scores.values()) if scores else 0.0
            is_anomaly = (
                (max_score > self.max_score_threshold or vote_fraction >= self.voting_threshold)
                and num_triggered >= self.min_detectors_agree
            )
        else:
            # Must have both: sufficient weighted votes AND enough detectors agree
            is_anomaly = (
                vote_fraction >= self.voting_threshold
                and num_triggered >= self.min_detectors_agree
            )

        # Confidence based on agreement
        confidence = vote_fraction if is_anomaly else (1.0 - vote_fraction)

        # Hint at attack type based on which detectors triggered
        attack_hint = self._infer_attack_type(triggered, scores)

        result = DetectionResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            scores=scores,
            votes=votes,
            triggered_detectors=triggered,
            attack_type_hint=attack_hint,
        )

        self.detection_history.append(result)

        return result

    def _infer_attack_type(
        self,
        triggered: List[str],
        scores: Dict[str, float],
    ) -> Optional[str]:
        """Infer likely attack type from detection pattern."""
        if not triggered:
            return None

        if 'similarity' in triggered:
            return 'replay_attack'

        if 'cross_sensor' in triggered and 'statistical' not in triggered:
            return 'gps_spoofing'

        if 'statistical' in triggered and 'cross_sensor' not in triggered:
            return 'sensor_noise_attack'

        if 'pinn' in triggered and 'sequence_pinn' in triggered:
            return 'physics_violation_attack'

        if 'sequence_pinn' in triggered and 'pinn' not in triggered:
            return 'temporal_attack'

        return 'unknown_attack'

    def get_detection_summary(self) -> Dict:
        """Get summary of detection history."""
        if not self.detection_history:
            return {'total': 0, 'anomalies': 0, 'by_detector': {}}

        total = len(self.detection_history)
        anomalies = sum(1 for r in self.detection_history if r.is_anomaly)

        by_detector = {}
        for detector_name in self.weights.keys():
            triggers = sum(
                1 for r in self.detection_history
                if detector_name in r.triggered_detectors
            )
            by_detector[detector_name] = triggers

        return {
            'total': total,
            'anomalies': anomalies,
            'anomaly_rate': anomalies / total if total > 0 else 0,
            'by_detector': by_detector,
        }


class BaseDetector:
    """Base class for individual detectors in the ensemble."""

    def __init__(self, name: str, threshold: float = 0.5):
        self.name = name
        self.threshold = threshold
        self.calibrated = False

    def detect(self, data: Dict[str, np.ndarray]) -> "DetectionResult":
        """Override in subclass."""
        raise NotImplementedError

    def calibrate(self, normal_data: List[Dict[str, np.ndarray]]):
        """Override in subclass."""
        pass


def prepare_detector_data(
    states: np.ndarray,
    controls: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Prepare data dict for detector input."""
    return {
        "states": states,
        "controls": controls,
    }


def create_ensemble_detector(
    normal_data: Optional[np.ndarray] = None,
    pinn_detector=None,
    sequence_pinn_detector=None,
) -> EnsembleDetector:
    """
    Factory function to create and optionally calibrate ensemble detector.

    Args:
        normal_data: [N, features] normal data for calibration
        pinn_detector: Optional PINN-based detector
        sequence_pinn_detector: Optional Sequence-PINN detector

    Returns:
        Configured EnsembleDetector
    """
    detector = EnsembleDetector()

    if pinn_detector is not None:
        detector.set_pinn_detector(pinn_detector, 'pinn')

    if sequence_pinn_detector is not None:
        detector.set_pinn_detector(sequence_pinn_detector, 'sequence_pinn')

    if normal_data is not None:
        detector.calibrate(normal_data)

    return detector
