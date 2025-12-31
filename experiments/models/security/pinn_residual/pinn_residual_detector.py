"""
PINN Residual-Based Anomaly Detector.

Uses trained PINN model's prediction residuals for attack detection.
Key insight: Single-step residuals are noisy, but they accumulate over time.

This detector:
1. Uses the trained PINN model via the Predictor interface (no modifications)
2. Accumulates prediction residuals over multiple steps
3. Adds minimal temporal pattern detection for replay/freeze attacks

Does NOT modify: QuadrotorPINN, Predictor, AnomalyDetector, or any existing code.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque

# Import existing infrastructure - read-only usage
from ..inference.predictor import Predictor


@dataclass
class DetectionResult:
    """Container for detection results."""
    is_anomaly: bool
    confidence: float  # 0-1, how confident the detection is
    residual_score: float  # Normalized accumulated residual
    temporal_score: float  # Temporal pattern score (for replay/freeze)
    triggered_by: str  # Which detector triggered: "residual", "temporal", "both", "none"
    details: Dict[str, float] = field(default_factory=dict)


class MultiStepResidualDetector:
    """
    Multi-step PINN residual detector.

    Accumulates prediction residuals over a window and detects when
    the accumulated error exceeds a calibrated threshold.

    Key insight: Single-step prediction errors are noisy and can be large
    even for normal data. But attack-induced errors accumulate consistently
    while normal errors fluctuate around zero.

    Args:
        predictor: Trained Predictor instance (uses existing API)
        window_size: Number of steps to accumulate residuals over
        threshold_percentile: Percentile of clean data residuals for threshold
    """

    def __init__(
        self,
        predictor: Predictor,
        window_size: int = 20,
        threshold_percentile: float = 99.0,
    ):
        self.predictor = predictor
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile

        # Calibration statistics (set during calibrate())
        self.threshold = float('inf')  # Will be set by calibration
        self.residual_mean = 0.0
        self.residual_std = 1.0

        # Online state
        self.residual_window: deque = deque(maxlen=window_size)
        self.is_calibrated = False

    def calibrate(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        """
        Calibrate on clean validation data.

        Computes single-step residuals and sets threshold based on
        the distribution of accumulated residuals over the window.

        Args:
            states: [N, state_dim] clean states
            controls: [N, control_dim] controls
            next_states: [N, state_dim] clean next states
        """
        print(f"Calibrating MultiStepResidualDetector on {len(states)} samples...")

        # Compute single-step residuals
        single_step_residuals = []
        for i in range(len(states)):
            predicted = self.predictor.predict(states[i], controls[i])
            residual = np.linalg.norm(next_states[i] - predicted)
            single_step_residuals.append(residual)

        single_step_residuals = np.array(single_step_residuals)

        # Compute statistics
        self.residual_mean = np.mean(single_step_residuals)
        self.residual_std = np.std(single_step_residuals) + 1e-6

        # Compute accumulated residuals over windows
        accumulated_residuals = []
        for i in range(self.window_size, len(single_step_residuals)):
            window = single_step_residuals[i - self.window_size:i]
            # Use mean of window (more stable than sum)
            accumulated = np.mean(window)
            accumulated_residuals.append(accumulated)

        accumulated_residuals = np.array(accumulated_residuals)

        # Set threshold at specified percentile
        self.threshold = np.percentile(accumulated_residuals, self.threshold_percentile)

        print(f"  Single-step residual: mean={self.residual_mean:.4f}, std={self.residual_std:.4f}")
        print(f"  Threshold ({self.threshold_percentile}th percentile): {self.threshold:.4f}")

        self.is_calibrated = True

    def reset(self) -> None:
        """Reset online state for new sequence."""
        self.residual_window.clear()

    def detect(
        self,
        state: np.ndarray,
        control: np.ndarray,
        measured_next_state: np.ndarray,
    ) -> Tuple[bool, float, float]:
        """
        Detect anomaly in single timestep.

        Args:
            state: [state_dim] current state
            control: [control_dim] control input
            measured_next_state: [state_dim] measured next state

        Returns:
            (is_anomaly, accumulated_score, normalized_score)
        """
        if not self.is_calibrated:
            raise RuntimeError("Detector not calibrated. Call calibrate() first.")

        # Compute prediction and residual using existing Predictor
        predicted = self.predictor.predict(state, control)
        residual = np.linalg.norm(measured_next_state - predicted)

        # Add to window
        self.residual_window.append(residual)

        # Need full window for detection
        if len(self.residual_window) < self.window_size:
            return False, 0.0, 0.0

        # Compute accumulated score (mean over window)
        accumulated = np.mean(list(self.residual_window))

        # Normalize
        normalized = (accumulated - self.residual_mean) / self.residual_std

        # Threshold check
        is_anomaly = accumulated > self.threshold

        return is_anomaly, accumulated, normalized


class TemporalPatternDetector:
    """
    Minimal temporal pattern detector for replay and freeze attacks.

    These attacks are difficult for PINN to detect because:
    - Freeze: The sensor values are constant (valid state, just not changing)
    - Replay: The sensor values are physically plausible (they happened before)

    This detector looks for:
    1. Freeze: Sequence has near-zero variance
    2. Replay: Exact match with historical sequences

    Args:
        sequence_length: Length of sequence to analyze
        freeze_threshold: Max std for freeze detection (very small)
        replay_similarity: Min similarity for replay detection (very high, like 0.999)
        history_size: Number of historical sequences to store
    """

    def __init__(
        self,
        sequence_length: int = 30,
        freeze_threshold: float = 1e-4,
        replay_similarity: float = 0.999,
        history_size: int = 50,
    ):
        self.sequence_length = sequence_length
        self.freeze_threshold = freeze_threshold
        self.replay_similarity = replay_similarity
        self.history_size = history_size

        # Online state
        self.current_sequence: deque = deque(maxlen=sequence_length)
        self.history: deque = deque(maxlen=history_size)
        self.sample_count = 0

    def reset(self) -> None:
        """Reset for new sequence."""
        self.current_sequence.clear()
        self.history.clear()
        self.sample_count = 0

    def update(self, state: np.ndarray) -> Tuple[bool, str, float]:
        """
        Update with new state and check for temporal attacks.

        Args:
            state: [state_dim] current state

        Returns:
            (is_anomaly, attack_type, score)
            attack_type: "freeze", "replay", or "none"
        """
        self.current_sequence.append(state.copy())
        self.sample_count += 1

        # Need full sequence
        if len(self.current_sequence) < self.sequence_length:
            return False, "none", 0.0

        current = np.array(list(self.current_sequence))

        # Check for freeze: very low variance across time
        temporal_std = np.std(current, axis=0).mean()
        if temporal_std < self.freeze_threshold:
            return True, "freeze", 1.0 - (temporal_std / self.freeze_threshold)

        # Check for replay: high similarity with history
        # Only check every sequence_length samples to avoid O(n^2)
        if self.sample_count % (self.sequence_length // 2) == 0:
            for hist_seq in self.history:
                similarity = self._compute_similarity(current, hist_seq)
                if similarity > self.replay_similarity:
                    return True, "replay", similarity

        # Store sequence periodically (not every sample)
        if self.sample_count % (self.sequence_length // 2) == 0:
            self.history.append(current.copy())

        return False, "none", 0.0

    def _compute_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute cosine similarity between two sequences."""
        flat1 = seq1.flatten()
        flat2 = seq2.flatten()

        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        return np.dot(flat1, flat2) / (norm1 * norm2)


class PINNResidualEnsemble:
    """
    PINN-based ensemble detector.

    Combines:
    1. Multi-step PINN residual detection (primary)
    2. Temporal pattern detection for replay/freeze (secondary)

    The key difference from the old EnsembleDetector:
    - Uses the trained PINN model's predictions (not hand-crafted heuristics)
    - Accumulates residuals over time (not single-step)
    - Only adds temporal detection for attacks PINN can't catch

    Args:
        predictor: Trained Predictor instance
        window_size: Residual accumulation window
        threshold_percentile: For residual threshold calibration
    """

    def __init__(
        self,
        predictor: Predictor,
        window_size: int = 20,
        threshold_percentile: float = 99.0,
    ):
        self.predictor = predictor

        # Primary: PINN residual detector
        self.residual_detector = MultiStepResidualDetector(
            predictor=predictor,
            window_size=window_size,
            threshold_percentile=threshold_percentile,
        )

        # Secondary: Temporal pattern detector (for replay/freeze only)
        self.temporal_detector = TemporalPatternDetector(
            sequence_length=30,
            freeze_threshold=1e-4,
            replay_similarity=0.999,
        )

        # Tracking
        self.detection_counts = {
            "residual": 0,
            "temporal": 0,
            "both": 0,
            "total": 0,
        }

    def calibrate(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        """
        Calibrate on clean data.

        Args:
            states: [N, state_dim]
            controls: [N, control_dim]
            next_states: [N, state_dim]
        """
        print("=" * 60)
        print("PINN Residual Ensemble Calibration")
        print("=" * 60)

        self.residual_detector.calibrate(states, controls, next_states)

        print("\nTemporal detector uses fixed thresholds (no calibration needed)")
        print(f"  Freeze threshold: {self.temporal_detector.freeze_threshold}")
        print(f"  Replay similarity: {self.temporal_detector.replay_similarity}")
        print("=" * 60)

    def reset(self) -> None:
        """Reset for new sequence."""
        self.residual_detector.reset()
        self.temporal_detector.reset()

    def detect(
        self,
        state: np.ndarray,
        control: np.ndarray,
        measured_next_state: np.ndarray,
    ) -> DetectionResult:
        """
        Detect anomaly using PINN residuals and temporal patterns.

        Args:
            state: [state_dim] current state
            control: [control_dim] control input
            measured_next_state: [state_dim] measured next state

        Returns:
            DetectionResult with scores and triggered detector
        """
        # Primary: PINN residual detection
        residual_anomaly, accumulated, normalized = self.residual_detector.detect(
            state, control, measured_next_state
        )

        # Secondary: Temporal pattern detection
        temporal_anomaly, attack_type, temporal_score = self.temporal_detector.update(
            measured_next_state
        )

        # Combine: Either detector can flag an anomaly
        is_anomaly = residual_anomaly or temporal_anomaly

        # Determine which detector triggered
        if residual_anomaly and temporal_anomaly:
            triggered_by = "both"
            self.detection_counts["both"] += 1
        elif residual_anomaly:
            triggered_by = "residual"
            self.detection_counts["residual"] += 1
        elif temporal_anomaly:
            triggered_by = "temporal"
            self.detection_counts["temporal"] += 1
        else:
            triggered_by = "none"

        if is_anomaly:
            self.detection_counts["total"] += 1

        # Confidence: max of the two scores
        confidence = max(
            min(normalized / 3.0, 1.0) if normalized > 0 else 0.0,  # Normalize to 0-1
            temporal_score
        )

        return DetectionResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            residual_score=normalized,
            temporal_score=temporal_score,
            triggered_by=triggered_by,
            details={
                "accumulated_residual": accumulated,
                "temporal_attack_type": attack_type if temporal_anomaly else "none",
            }
        )

    def get_detection_summary(self) -> Dict[str, any]:
        """Get summary of detections by detector type."""
        return {
            "total_detections": self.detection_counts["total"],
            "by_detector": {
                "residual_only": self.detection_counts["residual"],
                "temporal_only": self.detection_counts["temporal"],
                "both": self.detection_counts["both"],
            }
        }
