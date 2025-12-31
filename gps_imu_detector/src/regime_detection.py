"""
Regime Detection Module (Phase 1.1)

Defines flight regime taxonomy for the GPS-IMU anomaly detector.
Rule-based classification - no learning, no probabilities.

Regimes:
- HOVER: Low velocity, low angular rates
- FORWARD: Medium velocity, aligned heading
- AGGRESSIVE: High velocity or high angular rates
- GUSTY: High acceleration variance (turbulence indicator)

Output: Regime ID only (enum value).
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np


class FlightRegime(Enum):
    """Flight regime taxonomy."""
    HOVER = auto()      # Stationary or near-stationary
    FORWARD = auto()    # Normal forward flight
    AGGRESSIVE = auto() # High-dynamic maneuvers
    GUSTY = auto()      # Turbulent conditions
    UNKNOWN = auto()    # Insufficient data


@dataclass
class RegimeThresholds:
    """Thresholds for regime classification."""
    # Velocity thresholds (m/s)
    hover_velocity: float = 0.5
    aggressive_velocity: float = 8.0

    # Angular rate thresholds (rad/s)
    hover_angular_rate: float = 0.1
    aggressive_angular_rate: float = 1.0

    # Acceleration variance threshold for gusty detection
    gusty_acc_variance: float = 2.0

    # Minimum samples for variance estimation
    variance_window: int = 20


class RegimeClassifier:
    """
    Rule-based flight regime classifier.

    No learning, no probabilities - just deterministic rules.
    """

    def __init__(self, thresholds: Optional[RegimeThresholds] = None):
        self.thresholds = thresholds or RegimeThresholds()

        # Rolling buffer for acceleration variance
        self._acc_buffer: list = []
        self._buffer_size = self.thresholds.variance_window

    def classify(
        self,
        velocity: np.ndarray,
        angular_rate: np.ndarray,
        acceleration: Optional[np.ndarray] = None,
    ) -> FlightRegime:
        """
        Classify current flight regime.

        Args:
            velocity: [3] velocity vector (m/s)
            angular_rate: [3] angular rate vector (rad/s)
            acceleration: [3] acceleration vector (m/s^2), optional

        Returns:
            FlightRegime enum value
        """
        velocity = np.asarray(velocity)
        angular_rate = np.asarray(angular_rate)

        vel_norm = np.linalg.norm(velocity)
        ang_norm = np.linalg.norm(angular_rate)

        # Update acceleration buffer if provided
        if acceleration is not None:
            acceleration = np.asarray(acceleration)
            self._acc_buffer.append(acceleration.copy())
            if len(self._acc_buffer) > self._buffer_size:
                self._acc_buffer.pop(0)

        # Check for gusty conditions first (high priority)
        if len(self._acc_buffer) >= self._buffer_size:
            acc_array = np.array(self._acc_buffer)
            acc_variance = np.var(acc_array, axis=0).sum()
            if acc_variance > self.thresholds.gusty_acc_variance:
                return FlightRegime.GUSTY

        # Check for aggressive maneuvers
        if (vel_norm > self.thresholds.aggressive_velocity or
            ang_norm > self.thresholds.aggressive_angular_rate):
            return FlightRegime.AGGRESSIVE

        # Check for hover
        if (vel_norm < self.thresholds.hover_velocity and
            ang_norm < self.thresholds.hover_angular_rate):
            return FlightRegime.HOVER

        # Default: forward flight
        return FlightRegime.FORWARD

    def reset(self):
        """Reset internal state."""
        self._acc_buffer.clear()

    def get_regime_id(
        self,
        velocity: np.ndarray,
        angular_rate: np.ndarray,
        acceleration: Optional[np.ndarray] = None,
    ) -> int:
        """
        Get numeric regime ID.

        Returns:
            Integer regime ID (0-4)
        """
        regime = self.classify(velocity, angular_rate, acceleration)
        return regime.value


@dataclass
class RegimeStatistics:
    """Statistics for a specific regime."""
    regime: FlightRegime
    sample_count: int
    mean_velocity: float
    mean_angular_rate: float
    mean_acceleration_variance: float


class RegimeProfiler:
    """
    Collects statistics per regime for offline analysis.
    """

    def __init__(self, thresholds: Optional[RegimeThresholds] = None):
        self.classifier = RegimeClassifier(thresholds)
        self._stats: dict = {regime: [] for regime in FlightRegime}

    def add_sample(
        self,
        velocity: np.ndarray,
        angular_rate: np.ndarray,
        acceleration: Optional[np.ndarray] = None,
    ):
        """Add a sample and classify it."""
        regime = self.classifier.classify(velocity, angular_rate, acceleration)

        self._stats[regime].append({
            'velocity': np.linalg.norm(velocity),
            'angular_rate': np.linalg.norm(angular_rate),
            'acceleration': np.linalg.norm(acceleration) if acceleration is not None else 0.0,
        })

    def get_statistics(self) -> dict:
        """Get statistics per regime."""
        results = {}
        for regime, samples in self._stats.items():
            if not samples:
                continue
            results[regime.name] = RegimeStatistics(
                regime=regime,
                sample_count=len(samples),
                mean_velocity=np.mean([s['velocity'] for s in samples]),
                mean_angular_rate=np.mean([s['angular_rate'] for s in samples]),
                mean_acceleration_variance=np.var([s['acceleration'] for s in samples]),
            )
        return results

    def get_regime_distribution(self) -> dict:
        """Get distribution of samples across regimes."""
        total = sum(len(samples) for samples in self._stats.values())
        if total == 0:
            return {}
        return {
            regime.name: len(samples) / total
            for regime, samples in self._stats.items()
            if samples
        }


def classify_trajectory(
    trajectory: np.ndarray,
    thresholds: Optional[RegimeThresholds] = None,
) -> np.ndarray:
    """
    Classify regimes for an entire trajectory.

    Args:
        trajectory: [T, 12] state trajectory (pos, vel, orient, ang_vel)
        thresholds: Optional regime thresholds

    Returns:
        [T] array of regime IDs
    """
    classifier = RegimeClassifier(thresholds)
    T = trajectory.shape[0]
    regimes = np.zeros(T, dtype=np.int32)

    for t in range(T):
        state = trajectory[t]
        velocity = state[3:6]
        angular_rate = state[9:12]

        # Estimate acceleration from velocity differences
        if t > 0:
            acceleration = (trajectory[t, 3:6] - trajectory[t-1, 3:6]) / 0.005
        else:
            acceleration = np.zeros(3)

        regimes[t] = classifier.get_regime_id(velocity, angular_rate, acceleration)

    return regimes


# Regime-specific parameters (for conformal calibration)
REGIME_PARAMETERS = {
    FlightRegime.HOVER: {
        'residual_scale': 1.0,
        'envelope_multiplier': 1.0,
        'probe_allowed': True,
        'abstention_threshold': 0.1,
    },
    FlightRegime.FORWARD: {
        'residual_scale': 1.2,
        'envelope_multiplier': 1.1,
        'probe_allowed': True,
        'abstention_threshold': 0.15,
    },
    FlightRegime.AGGRESSIVE: {
        'residual_scale': 1.5,
        'envelope_multiplier': 1.3,
        'probe_allowed': False,  # No probing during aggressive maneuvers
        'abstention_threshold': 0.25,
    },
    FlightRegime.GUSTY: {
        'residual_scale': 2.0,
        'envelope_multiplier': 1.5,
        'probe_allowed': False,  # No probing in turbulence
        'abstention_threshold': 0.3,
    },
    FlightRegime.UNKNOWN: {
        'residual_scale': 1.5,
        'envelope_multiplier': 1.2,
        'probe_allowed': False,
        'abstention_threshold': 0.2,
    },
}


def get_regime_parameters(regime: FlightRegime) -> dict:
    """Get parameters for a specific regime."""
    return REGIME_PARAMETERS.get(regime, REGIME_PARAMETERS[FlightRegime.UNKNOWN])
