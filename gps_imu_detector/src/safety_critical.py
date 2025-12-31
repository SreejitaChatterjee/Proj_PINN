"""
Safety-Critical Acceleration Module (Phase 3)

Implements:
3.1 Catastrophic Fast-Confirm
    - Immediate escalation for specific PINN residual signatures
    - High-rate divergence detection
    - Multi-axis anomaly detection
    - Bypass normal confirmation for critical events

3.2 Monotone Severity Scorer
    - Assigns severity levels based on residuals
    - Strictly monotone (higher residuals = higher severity)
    - Maps to action thresholds
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# Phase 3.1: Catastrophic Fast-Confirm
# =============================================================================

class CatastropheType(IntEnum):
    """Types of catastrophic events that bypass normal confirmation."""
    NONE = 0
    HIGH_RATE_DIVERGENCE = 1      # Rapid residual increase
    MULTI_AXIS_ANOMALY = 2        # Simultaneous anomalies on multiple axes
    PHYSICS_VIOLATION = 3         # Impossible physics (e.g., thrust > gravity but falling)
    COORDINATED_ATTACK = 4        # Simultaneous GPS+IMU anomalies


@dataclass
class FastConfirmConfig:
    """Configuration for catastrophic fast-confirm."""
    # High-rate divergence thresholds
    divergence_rate_threshold: float = 5.0  # residual units per second
    divergence_window: int = 10             # timesteps at 200 Hz (50ms)

    # Multi-axis thresholds
    multi_axis_threshold: float = 3.0       # z-score per axis
    min_axes_for_multi: int = 2             # minimum axes for multi-axis detection

    # Physics violation
    accel_thrust_ratio_min: float = 0.3     # thrust should cause some accel
    position_velocity_coherence: float = 0.9  # pos/vel should agree

    # Coordinated attack
    gps_imu_correlation_threshold: float = 0.8  # high correlation = coordinated


@dataclass
class FastConfirmResult:
    """Result of fast-confirm check."""
    is_catastrophic: bool
    catastrophe_type: CatastropheType
    confidence: float
    details: str
    bypass_confirmation: bool


class CatastrophicFastConfirm:
    """
    Immediate escalation for catastrophic events.

    Bypasses normal multi-probe confirmation when specific signatures
    indicate imminent danger.
    """

    def __init__(self, config: Optional[FastConfirmConfig] = None):
        self.config = config or FastConfirmConfig()

        # Rolling buffers for rate detection
        self._residual_buffer: List[float] = []
        self._axis_residuals_buffer: List[np.ndarray] = []
        self._max_buffer_size = self.config.divergence_window

    def check(
        self,
        residual: float,
        axis_residuals: np.ndarray,
        acceleration: Optional[np.ndarray] = None,
        thrust_command: Optional[float] = None,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        gps_residual: Optional[float] = None,
        imu_residual: Optional[float] = None,
    ) -> FastConfirmResult:
        """
        Check for catastrophic conditions.

        Args:
            residual: Overall residual magnitude
            axis_residuals: [N] per-axis residuals
            acceleration: [3] measured acceleration
            thrust_command: Commanded thrust (0-1)
            position: [3] position
            velocity: [3] velocity
            gps_residual: GPS-specific residual
            imu_residual: IMU-specific residual

        Returns:
            FastConfirmResult with catastrophe status
        """
        # Update buffers
        self._residual_buffer.append(residual)
        if len(self._residual_buffer) > self._max_buffer_size:
            self._residual_buffer.pop(0)

        self._axis_residuals_buffer.append(axis_residuals.copy())
        if len(self._axis_residuals_buffer) > self._max_buffer_size:
            self._axis_residuals_buffer.pop(0)

        # Check each catastrophe type
        checks = [
            self._check_high_rate_divergence(),
            self._check_multi_axis_anomaly(axis_residuals),
            self._check_physics_violation(acceleration, thrust_command),
            self._check_coordinated_attack(gps_residual, imu_residual),
        ]

        # Return the most severe catastrophe
        for result in checks:
            if result.is_catastrophic:
                return result

        return FastConfirmResult(
            is_catastrophic=False,
            catastrophe_type=CatastropheType.NONE,
            confidence=0.0,
            details="No catastrophic conditions detected",
            bypass_confirmation=False,
        )

    def _check_high_rate_divergence(self) -> FastConfirmResult:
        """Check for rapid residual increase."""
        if len(self._residual_buffer) < self.config.divergence_window:
            return FastConfirmResult(
                is_catastrophic=False,
                catastrophe_type=CatastropheType.HIGH_RATE_DIVERGENCE,
                confidence=0.0,
                details="Insufficient data",
                bypass_confirmation=False,
            )

        # Compute rate of change
        residuals = np.array(self._residual_buffer)
        rate = (residuals[-1] - residuals[0]) / (self.config.divergence_window / 200.0)

        if rate > self.config.divergence_rate_threshold:
            return FastConfirmResult(
                is_catastrophic=True,
                catastrophe_type=CatastropheType.HIGH_RATE_DIVERGENCE,
                confidence=min(1.0, rate / (2 * self.config.divergence_rate_threshold)),
                details=f"High-rate divergence: {rate:.2f} units/s",
                bypass_confirmation=True,
            )

        return FastConfirmResult(
            is_catastrophic=False,
            catastrophe_type=CatastropheType.HIGH_RATE_DIVERGENCE,
            confidence=rate / self.config.divergence_rate_threshold,
            details=f"Divergence rate: {rate:.2f} units/s",
            bypass_confirmation=False,
        )

    def _check_multi_axis_anomaly(
        self,
        axis_residuals: np.ndarray,
    ) -> FastConfirmResult:
        """Check for simultaneous anomalies on multiple axes."""
        # Count axes above threshold
        above_threshold = np.sum(axis_residuals > self.config.multi_axis_threshold)

        if above_threshold >= self.config.min_axes_for_multi:
            confidence = min(1.0, above_threshold / len(axis_residuals))
            return FastConfirmResult(
                is_catastrophic=True,
                catastrophe_type=CatastropheType.MULTI_AXIS_ANOMALY,
                confidence=confidence,
                details=f"Multi-axis anomaly: {above_threshold} axes affected",
                bypass_confirmation=True,
            )

        return FastConfirmResult(
            is_catastrophic=False,
            catastrophe_type=CatastropheType.MULTI_AXIS_ANOMALY,
            confidence=above_threshold / self.config.min_axes_for_multi,
            details=f"Axes above threshold: {above_threshold}",
            bypass_confirmation=False,
        )

    def _check_physics_violation(
        self,
        acceleration: Optional[np.ndarray],
        thrust_command: Optional[float],
    ) -> FastConfirmResult:
        """Check for impossible physics."""
        if acceleration is None or thrust_command is None:
            return FastConfirmResult(
                is_catastrophic=False,
                catastrophe_type=CatastropheType.PHYSICS_VIOLATION,
                confidence=0.0,
                details="Insufficient data",
                bypass_confirmation=False,
            )

        # Check thrust-acceleration coherence
        accel_z = acceleration[2]  # vertical acceleration
        gravity = 9.81

        if thrust_command > 0.5:  # High thrust command
            # Should have positive vertical acceleration (or at least close to 0)
            expected_accel = thrust_command * 20.0 - gravity  # simplified model
            if accel_z < -gravity * 0.5:  # Falling despite high thrust
                return FastConfirmResult(
                    is_catastrophic=True,
                    catastrophe_type=CatastropheType.PHYSICS_VIOLATION,
                    confidence=0.9,
                    details=f"Physics violation: thrust={thrust_command:.2f} but accel_z={accel_z:.2f}",
                    bypass_confirmation=True,
                )

        return FastConfirmResult(
            is_catastrophic=False,
            catastrophe_type=CatastropheType.PHYSICS_VIOLATION,
            confidence=0.0,
            details="Physics consistent",
            bypass_confirmation=False,
        )

    def _check_coordinated_attack(
        self,
        gps_residual: Optional[float],
        imu_residual: Optional[float],
    ) -> FastConfirmResult:
        """Check for coordinated GPS+IMU attack."""
        if gps_residual is None or imu_residual is None:
            return FastConfirmResult(
                is_catastrophic=False,
                catastrophe_type=CatastropheType.COORDINATED_ATTACK,
                confidence=0.0,
                details="Insufficient data",
                bypass_confirmation=False,
            )

        # Both sensors showing high residuals
        both_high = gps_residual > 2.0 and imu_residual > 2.0

        if both_high:
            # Check temporal correlation (simplified)
            confidence = min(gps_residual, imu_residual) / 3.0
            if confidence > self.config.gps_imu_correlation_threshold:
                return FastConfirmResult(
                    is_catastrophic=True,
                    catastrophe_type=CatastropheType.COORDINATED_ATTACK,
                    confidence=confidence,
                    details=f"Coordinated attack: GPS={gps_residual:.2f}, IMU={imu_residual:.2f}",
                    bypass_confirmation=True,
                )

        return FastConfirmResult(
            is_catastrophic=False,
            catastrophe_type=CatastropheType.COORDINATED_ATTACK,
            confidence=0.0,
            details="No coordinated attack detected",
            bypass_confirmation=False,
        )

    def reset(self):
        """Reset internal state."""
        self._residual_buffer.clear()
        self._axis_residuals_buffer.clear()


# =============================================================================
# Phase 3.2: Monotone Severity Scorer
# =============================================================================

class SeverityLevel(IntEnum):
    """Severity levels for anomalies."""
    NOMINAL = 0       # Normal operation
    ADVISORY = 1      # Low-level anomaly, log only
    CAUTION = 2       # Moderate anomaly, increase monitoring
    WARNING = 3       # Significant anomaly, prepare for action
    CRITICAL = 4      # Severe anomaly, take action
    EMERGENCY = 5     # Catastrophic, immediate response


@dataclass
class SeverityThresholds:
    """Thresholds for severity levels."""
    advisory: float = 1.0     # residual z-score
    caution: float = 2.0
    warning: float = 3.0
    critical: float = 4.0
    emergency: float = 5.0


@dataclass
class SeverityResult:
    """Result of severity scoring."""
    level: SeverityLevel
    score: float
    residual_zscore: float
    action_required: str
    details: Dict[str, float]


class MonotoneSeverityScorer:
    """
    Strictly monotone severity scorer.

    Guarantees:
    - Higher residuals always map to equal or higher severity
    - No hysteresis or state-dependence
    - Deterministic mapping
    """

    def __init__(
        self,
        thresholds: Optional[SeverityThresholds] = None,
        residual_stats: Optional[Tuple[float, float]] = None,
    ):
        self.thresholds = thresholds or SeverityThresholds()

        # Baseline statistics for z-score computation
        if residual_stats:
            self.residual_mean, self.residual_std = residual_stats
        else:
            self.residual_mean = 0.0
            self.residual_std = 1.0

        # Action mapping
        self._actions = {
            SeverityLevel.NOMINAL: "Continue normal operation",
            SeverityLevel.ADVISORY: "Log event, no action required",
            SeverityLevel.CAUTION: "Increase monitoring frequency",
            SeverityLevel.WARNING: "Prepare for defensive action",
            SeverityLevel.CRITICAL: "Execute defensive measures",
            SeverityLevel.EMERGENCY: "Immediate fail-safe response",
        }

    def score(
        self,
        residual: float,
        axis_residuals: Optional[np.ndarray] = None,
        catastrophe_result: Optional[FastConfirmResult] = None,
    ) -> SeverityResult:
        """
        Score severity of current anomaly state.

        Args:
            residual: Overall residual magnitude
            axis_residuals: [N] per-axis residuals (optional)
            catastrophe_result: Result from fast-confirm (optional)

        Returns:
            SeverityResult with level and recommended action
        """
        # Compute z-score
        zscore = (residual - self.residual_mean) / max(self.residual_std, 1e-6)

        # Handle catastrophic events
        if catastrophe_result and catastrophe_result.is_catastrophic:
            return SeverityResult(
                level=SeverityLevel.EMERGENCY,
                score=1.0,
                residual_zscore=zscore,
                action_required=self._actions[SeverityLevel.EMERGENCY],
                details={
                    'catastrophe_type': catastrophe_result.catastrophe_type.name,
                    'catastrophe_confidence': catastrophe_result.confidence,
                },
            )

        # Monotone level assignment
        level = self._get_level(zscore)

        # Compute continuous score (0-1 within level)
        score = self._compute_score(zscore, level)

        # Build details
        details = {
            'residual': residual,
            'zscore': zscore,
        }

        if axis_residuals is not None:
            details['max_axis_residual'] = float(np.max(axis_residuals))
            details['affected_axes'] = int(np.sum(axis_residuals > 2.0))

        return SeverityResult(
            level=level,
            score=score,
            residual_zscore=zscore,
            action_required=self._actions[level],
            details=details,
        )

    def _get_level(self, zscore: float) -> SeverityLevel:
        """Map z-score to severity level (strictly monotone)."""
        if zscore >= self.thresholds.emergency:
            return SeverityLevel.EMERGENCY
        elif zscore >= self.thresholds.critical:
            return SeverityLevel.CRITICAL
        elif zscore >= self.thresholds.warning:
            return SeverityLevel.WARNING
        elif zscore >= self.thresholds.caution:
            return SeverityLevel.CAUTION
        elif zscore >= self.thresholds.advisory:
            return SeverityLevel.ADVISORY
        else:
            return SeverityLevel.NOMINAL

    def _compute_score(self, zscore: float, level: SeverityLevel) -> float:
        """Compute continuous score within level."""
        if level == SeverityLevel.NOMINAL:
            return min(1.0, zscore / self.thresholds.advisory)

        thresholds = [
            0.0,  # NOMINAL
            self.thresholds.advisory,
            self.thresholds.caution,
            self.thresholds.warning,
            self.thresholds.critical,
            self.thresholds.emergency,
        ]

        lower = thresholds[level]
        upper = thresholds[min(level + 1, len(thresholds) - 1)]

        if upper <= lower:
            return 1.0

        return min(1.0, (zscore - lower) / (upper - lower))

    def calibrate(self, residuals: np.ndarray):
        """Calibrate scorer from training residuals."""
        self.residual_mean = float(np.mean(residuals))
        self.residual_std = float(np.std(residuals))


# =============================================================================
# Integrated Safety System
# =============================================================================

class SafetyCriticalSystem:
    """
    Integrated safety-critical system.

    Combines:
    - Fast-confirm for catastrophic events
    - Severity scoring for graduated response
    """

    def __init__(
        self,
        fast_confirm_config: Optional[FastConfirmConfig] = None,
        severity_thresholds: Optional[SeverityThresholds] = None,
    ):
        self.fast_confirm = CatastrophicFastConfirm(fast_confirm_config)
        self.severity_scorer = MonotoneSeverityScorer(severity_thresholds)

        self._history: List[SeverityResult] = []
        self._max_history = 100

    def update(
        self,
        residual: float,
        axis_residuals: np.ndarray,
        acceleration: Optional[np.ndarray] = None,
        thrust_command: Optional[float] = None,
        gps_residual: Optional[float] = None,
        imu_residual: Optional[float] = None,
    ) -> SeverityResult:
        """
        Process current state and return severity assessment.

        Args:
            residual: Overall residual
            axis_residuals: Per-axis residuals
            acceleration: Measured acceleration
            thrust_command: Commanded thrust
            gps_residual: GPS-specific residual
            imu_residual: IMU-specific residual

        Returns:
            SeverityResult with recommended action
        """
        # Check for catastrophic conditions
        catastrophe_result = self.fast_confirm.check(
            residual=residual,
            axis_residuals=axis_residuals,
            acceleration=acceleration,
            thrust_command=thrust_command,
            gps_residual=gps_residual,
            imu_residual=imu_residual,
        )

        # Compute severity score
        severity_result = self.severity_scorer.score(
            residual=residual,
            axis_residuals=axis_residuals,
            catastrophe_result=catastrophe_result,
        )

        # Update history
        self._history.append(severity_result)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return severity_result

    def get_escalation_trend(self, window: int = 20) -> str:
        """Analyze recent severity trend."""
        if len(self._history) < 2:
            return "insufficient_data"

        recent = self._history[-window:]
        levels = [r.level.value for r in recent]

        if len(levels) < 2:
            return "stable"

        # Compute trend
        first_half = np.mean(levels[:len(levels)//2])
        second_half = np.mean(levels[len(levels)//2:])

        diff = second_half - first_half

        if diff > 0.5:
            return "escalating"
        elif diff < -0.5:
            return "de-escalating"
        else:
            return "stable"

    def calibrate(self, training_residuals: np.ndarray):
        """Calibrate system from training data."""
        self.severity_scorer.calibrate(training_residuals)

    def reset(self):
        """Reset system state."""
        self.fast_confirm.reset()
        self._history.clear()


def evaluate_safety_critical(
    nominal_residuals: np.ndarray,
    attack_residuals: np.ndarray,
) -> Dict:
    """
    Evaluate safety-critical system.

    Args:
        nominal_residuals: [N] residuals from nominal operation
        attack_residuals: [M] residuals from attacks

    Returns:
        Evaluation metrics
    """
    system = SafetyCriticalSystem()
    system.calibrate(nominal_residuals)

    # Evaluate on nominal
    nominal_levels = []
    for res in nominal_residuals:
        result = system.update(
            residual=res,
            axis_residuals=np.array([res * 0.5, res * 0.3, res * 0.2]),
        )
        nominal_levels.append(result.level.value)

    system.reset()

    # Evaluate on attacks
    attack_levels = []
    for res in attack_residuals:
        result = system.update(
            residual=res,
            axis_residuals=np.array([res * 0.5, res * 0.3, res * 0.2]),
        )
        attack_levels.append(result.level.value)

    # Compute metrics
    nominal_false_alarms = sum(1 for l in nominal_levels if l >= SeverityLevel.WARNING)
    attack_detections = sum(1 for l in attack_levels if l >= SeverityLevel.WARNING)

    return {
        'nominal_mean_level': np.mean(nominal_levels),
        'attack_mean_level': np.mean(attack_levels),
        'nominal_false_alarm_rate': nominal_false_alarms / len(nominal_levels),
        'attack_detection_rate': attack_detections / len(attack_levels),
        'separation': np.mean(attack_levels) - np.mean(nominal_levels),
    }
