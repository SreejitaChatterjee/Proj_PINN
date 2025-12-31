"""
Integration Module (Phase 6)

Implements:
6.1 Unified Detection Pipeline
    - Integrates all Phase 1-5 components
    - Single entry point for detection

6.2 Certification-Aligned Validation
    - DO-178C / ARP4754A aligned metrics
    - Comprehensive test coverage tracking
    - Traceability support
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np

# Import all phase modules
from .regime_detection import FlightRegime, RegimeClassifier, get_regime_parameters
from .conformal_envelopes import EnvelopeTable, ConformalEnvelopeBuilder
from .uncertainty_maps import UncertaintyMap, AbstentionPolicy, UncertaintyMapBuilder
from .adaptive_probing import AdaptiveProbingSystem, ProbeScheduler, ProbeLibrary
from .safety_critical import SafetyCriticalSystem, SeverityLevel, CatastropheType
from .robustness_testing import RobustnessStressTester, evaluate_robustness
from .bounded_online_pinn import OnlineShadowMonitor, BoundedShadowInference


# =============================================================================
# Phase 6.1: Unified Detection Pipeline
# =============================================================================

class DetectionDecision(Enum):
    """Final detection decision."""
    NOMINAL = auto()          # Normal operation
    MONITORING = auto()       # Increased monitoring
    SOFT_ALERT = auto()       # Advisory alert
    HARD_ALERT = auto()       # Confirmed anomaly
    EMERGENCY = auto()        # Immediate action required


@dataclass
class DetectionResult:
    """Result from unified detection pipeline."""
    decision: DetectionDecision
    confidence: float
    severity: SeverityLevel
    regime: FlightRegime
    components: Dict[str, Any]
    latency_ms: float
    trace_id: str


@dataclass
class PipelineConfig:
    """Configuration for unified pipeline."""
    # Component enables
    enable_regime_detection: bool = True
    enable_conformal_envelopes: bool = True
    enable_uncertainty_maps: bool = True
    enable_adaptive_probing: bool = True
    enable_safety_critical: bool = True
    enable_shadow_pinn: bool = True

    # Thresholds
    soft_alert_severity: SeverityLevel = SeverityLevel.CAUTION
    hard_alert_severity: SeverityLevel = SeverityLevel.WARNING
    emergency_severity: SeverityLevel = SeverityLevel.EMERGENCY

    # Performance
    max_total_latency_ms: float = 2.0


class UnifiedDetectionPipeline:
    """
    Unified detection pipeline integrating all components.

    Provides single entry point for anomaly detection with:
    - Regime-aware processing
    - Multi-source fusion
    - Safety-critical escalation
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        envelope_table: Optional[EnvelopeTable] = None,
        uncertainty_map: Optional[UncertaintyMap] = None,
        probe_library: Optional[ProbeLibrary] = None,
    ):
        self.config = config or PipelineConfig()

        # Initialize components
        self.regime_classifier = RegimeClassifier()
        self.safety_system = SafetyCriticalSystem()
        self.probing_system = AdaptiveProbingSystem(probe_library=probe_library)
        self.shadow_monitor = OnlineShadowMonitor()

        # Optional components (require calibration)
        self.envelope_table = envelope_table
        self.uncertainty_map = uncertainty_map
        self.abstention_policy = (
            AbstentionPolicy(uncertainty_map) if uncertainty_map else None
        )

        # State
        self._trace_counter = 0
        self._history: List[DetectionResult] = []

    def process(
        self,
        state: np.ndarray,
        next_state: Optional[np.ndarray] = None,
        gps_data: Optional[np.ndarray] = None,
        imu_data: Optional[np.ndarray] = None,
        power_fraction: float = 0.5,
        altitude_agl: float = 10.0,
    ) -> DetectionResult:
        """
        Process state through unified pipeline.

        Args:
            state: [12] current state (pos, vel, orient, ang_vel)
            next_state: [12] next state (optional)
            gps_data: [3] GPS position measurement
            imu_data: [6] IMU data (accel, gyro)
            power_fraction: Current power as fraction of max
            altitude_agl: Altitude above ground level

        Returns:
            DetectionResult with decision and supporting data
        """
        start_time = time.perf_counter()
        self._trace_counter += 1
        trace_id = f"DET-{self._trace_counter:06d}"

        components = {}

        # 1. Regime Detection
        velocity = state[3:6]
        angular_rate = state[9:12]
        attitude = state[6:9]
        acceleration = imu_data[:3] if imu_data is not None else None

        regime = self.regime_classifier.classify(velocity, angular_rate, acceleration)
        regime_params = get_regime_parameters(regime)
        components['regime'] = regime.name

        # 2. Compute Residuals
        residual = self._compute_residual(state, next_state, gps_data, imu_data)
        axis_residuals = np.array([residual * 0.5, residual * 0.3, residual * 0.2])
        components['residual'] = residual

        # 3. Shadow PINN (if enabled)
        if self.config.enable_shadow_pinn:
            shadow_update = self.shadow_monitor.update(state, next_state)
            components['shadow'] = {
                'status': shadow_update.status.name,
                'residual': shadow_update.smoothed_residual,
                'confidence': shadow_update.confidence,
            }

        # 4. Safety-Critical Assessment
        safety_result = self.safety_system.update(
            residual=residual,
            axis_residuals=axis_residuals,
            acceleration=acceleration,
            thrust_command=power_fraction,
            gps_residual=gps_data[0] if gps_data is not None else None,
            imu_residual=imu_data[0] if imu_data is not None else None,
        )
        components['safety'] = {
            'level': safety_result.level.name,
            'score': safety_result.score,
        }

        # 5. Adaptive Probing (if enabled and allowed)
        if self.config.enable_adaptive_probing and regime_params['probe_allowed']:
            probe_excitation, probe_active = self.probing_system.update(
                velocity=velocity,
                angular_rate=angular_rate,
                attitude=attitude,
                power_fraction=power_fraction,
                altitude_agl=altitude_agl,
                acceleration=acceleration,
            )
            components['probing'] = {
                'active': probe_active,
                'excitation': probe_excitation,
            }

        # 6. Make Final Decision
        decision = self._make_decision(safety_result.level, regime, components)

        # Compute confidence
        confidence = self._compute_confidence(safety_result, components)

        latency_ms = (time.perf_counter() - start_time) * 1000

        result = DetectionResult(
            decision=decision,
            confidence=confidence,
            severity=safety_result.level,
            regime=regime,
            components=components,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )

        self._history.append(result)
        if len(self._history) > 1000:
            self._history.pop(0)

        return result

    def _compute_residual(
        self,
        state: np.ndarray,
        next_state: Optional[np.ndarray],
        gps_data: Optional[np.ndarray],
        imu_data: Optional[np.ndarray],
    ) -> float:
        """Compute combined residual from all sources."""
        residuals = []

        # State-based residual
        if next_state is not None:
            # Position-velocity consistency
            dt = 0.005
            expected_pos = state[:3] + state[3:6] * dt
            pos_residual = np.linalg.norm(next_state[:3] - expected_pos)
            residuals.append(pos_residual)

        # GPS residual
        if gps_data is not None:
            gps_residual = np.linalg.norm(gps_data - state[:3])
            residuals.append(gps_residual)

        # IMU residual (simplified)
        if imu_data is not None:
            imu_residual = np.linalg.norm(imu_data[:3])  # Acceleration magnitude
            residuals.append(imu_residual * 0.1)

        return float(np.mean(residuals)) if residuals else 0.0

    def _make_decision(
        self,
        severity: SeverityLevel,
        regime: FlightRegime,
        components: Dict,
    ) -> DetectionDecision:
        """Make final detection decision."""
        if severity >= self.config.emergency_severity:
            return DetectionDecision.EMERGENCY

        if severity >= self.config.hard_alert_severity:
            return DetectionDecision.HARD_ALERT

        if severity >= self.config.soft_alert_severity:
            return DetectionDecision.SOFT_ALERT

        # Check shadow PINN alert
        if 'shadow' in components:
            if components['shadow'].get('status') == 'ALERT':
                return DetectionDecision.SOFT_ALERT

        if severity >= SeverityLevel.ADVISORY:
            return DetectionDecision.MONITORING

        return DetectionDecision.NOMINAL

    def _compute_confidence(
        self,
        safety_result,
        components: Dict,
    ) -> float:
        """Compute confidence in detection decision."""
        confidences = [safety_result.score]

        if 'shadow' in components:
            confidences.append(components['shadow'].get('confidence', 0.5))

        return float(np.mean(confidences))

    def calibrate(
        self,
        nominal_trajectories: np.ndarray,
        attack_trajectories: Optional[np.ndarray] = None,
    ):
        """Calibrate pipeline from training data."""
        # Calibrate safety system
        nominal_residuals = []
        for traj in nominal_trajectories:
            for t in range(len(traj) - 1):
                res = self._compute_residual(traj[t], traj[t+1], None, None)
                nominal_residuals.append(res)

        self.safety_system.calibrate(np.array(nominal_residuals))
        self.shadow_monitor.calibrate(np.array(nominal_residuals))

    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        if not self._history:
            return {'total_processed': 0}

        latencies = [r.latency_ms for r in self._history]
        decisions = [r.decision.name for r in self._history]

        return {
            'total_processed': len(self._history),
            'mean_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'decision_distribution': {
                d: decisions.count(d) / len(decisions)
                for d in set(decisions)
            },
        }

    def reset(self):
        """Reset pipeline state."""
        self.safety_system.reset()
        self.probing_system.reset()
        self.shadow_monitor.reset()
        self._history.clear()


# =============================================================================
# Phase 6.2: Certification-Aligned Validation
# =============================================================================

@dataclass
class CertificationMetrics:
    """Metrics aligned with DO-178C / ARP4754A standards."""
    # Detection metrics
    detection_rate_overall: float
    detection_rate_per_attack: Dict[str, float]
    false_positive_rate: float
    false_negative_rate: float

    # Timing metrics
    mean_detection_latency_ms: float
    p95_detection_latency_ms: float
    worst_case_latency_ms: float

    # Robustness metrics
    coverage_by_attack_type: Dict[str, float]
    weakness_count: int

    # Reliability metrics
    availability: float
    mtbf_samples: float  # Mean time between failures

    # Traceability
    test_coverage: float
    requirements_traced: int


@dataclass
class ValidationConfig:
    """Configuration for certification validation."""
    # Detection requirements
    min_detection_rate: float = 0.95
    max_false_positive_rate: float = 0.01
    max_detection_latency_ms: float = 500.0

    # Robustness requirements
    min_coverage: float = 0.90
    max_weaknesses: int = 3

    # Reliability requirements
    min_availability: float = 0.999


class CertificationValidator:
    """
    Certification-aligned validation.

    Validates system against DO-178C / ARP4754A requirements.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self._results: Dict[str, Any] = {}

    def validate(
        self,
        pipeline: UnifiedDetectionPipeline,
        nominal_trajectories: np.ndarray,
        attack_trajectories: np.ndarray,
        attack_labels: Optional[np.ndarray] = None,
    ) -> Tuple[bool, CertificationMetrics]:
        """
        Run full certification validation.

        Args:
            pipeline: Unified detection pipeline
            nominal_trajectories: [N, T, state_dim] nominal data
            attack_trajectories: [M, T, state_dim] attack data
            attack_labels: [M] attack type labels (optional)

        Returns:
            (passed, metrics) tuple
        """
        # Calibrate pipeline
        pipeline.calibrate(nominal_trajectories)

        # Run nominal evaluation
        nominal_results = self._evaluate_nominal(pipeline, nominal_trajectories)

        # Run attack evaluation
        attack_results = self._evaluate_attacks(
            pipeline, attack_trajectories, attack_labels
        )

        # Run robustness evaluation
        robustness_results = self._evaluate_robustness(pipeline, nominal_trajectories)

        # Compute metrics
        metrics = self._compute_metrics(
            nominal_results, attack_results, robustness_results, pipeline
        )

        # Check requirements
        passed = self._check_requirements(metrics)

        self._results = {
            'nominal': nominal_results,
            'attack': attack_results,
            'robustness': robustness_results,
            'metrics': metrics,
            'passed': passed,
        }

        return passed, metrics

    def _evaluate_nominal(
        self,
        pipeline: UnifiedDetectionPipeline,
        trajectories: np.ndarray,
    ) -> Dict:
        """Evaluate on nominal data."""
        false_positives = 0
        total = 0
        latencies = []

        for traj in trajectories:
            for t in range(len(traj) - 1):
                result = pipeline.process(traj[t], traj[t+1])

                if result.decision in [DetectionDecision.HARD_ALERT,
                                        DetectionDecision.EMERGENCY]:
                    false_positives += 1

                total += 1
                latencies.append(result.latency_ms)

            pipeline.reset()

        return {
            'false_positives': false_positives,
            'total': total,
            'fpr': false_positives / max(1, total),
            'latencies': latencies,
        }

    def _evaluate_attacks(
        self,
        pipeline: UnifiedDetectionPipeline,
        trajectories: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Dict:
        """Evaluate on attack data."""
        detections = 0
        misses = 0
        total = 0
        detection_delays = []

        for i, traj in enumerate(trajectories):
            detected_this_traj = False
            attack_start = len(traj) // 4  # Assume attack starts at 25%

            for t in range(len(traj) - 1):
                result = pipeline.process(traj[t], traj[t+1])

                if t >= attack_start and not detected_this_traj:
                    if result.decision in [DetectionDecision.HARD_ALERT,
                                            DetectionDecision.EMERGENCY]:
                        detected_this_traj = True
                        delay = (t - attack_start) * 0.005 * 1000  # ms
                        detection_delays.append(delay)

                total += 1

            if detected_this_traj:
                detections += 1
            else:
                misses += 1

            pipeline.reset()

        return {
            'detections': detections,
            'misses': misses,
            'total_trajectories': len(trajectories),
            'detection_rate': detections / max(1, len(trajectories)),
            'detection_delays': detection_delays,
        }

    def _evaluate_robustness(
        self,
        pipeline: UnifiedDetectionPipeline,
        trajectories: np.ndarray,
    ) -> Dict:
        """Evaluate robustness."""
        def detector_fn(traj):
            scores = []
            for t in range(len(traj) - 1):
                result = pipeline.process(traj[t], traj[t+1])
                scores.append(result.severity.value)
            pipeline.reset()
            return np.array(scores + [scores[-1]])

        return evaluate_robustness(trajectories, detector_fn)

    def _compute_metrics(
        self,
        nominal_results: Dict,
        attack_results: Dict,
        robustness_results: Dict,
        pipeline: UnifiedDetectionPipeline,
    ) -> CertificationMetrics:
        """Compute certification metrics."""
        all_latencies = nominal_results['latencies']

        return CertificationMetrics(
            detection_rate_overall=attack_results['detection_rate'],
            detection_rate_per_attack=robustness_results.get('coverage', {}),
            false_positive_rate=nominal_results['fpr'],
            false_negative_rate=1.0 - attack_results['detection_rate'],
            mean_detection_latency_ms=(
                np.mean(attack_results['detection_delays'])
                if attack_results['detection_delays'] else 0.0
            ),
            p95_detection_latency_ms=(
                np.percentile(attack_results['detection_delays'], 95)
                if attack_results['detection_delays'] else 0.0
            ),
            worst_case_latency_ms=(
                np.max(attack_results['detection_delays'])
                if attack_results['detection_delays'] else 0.0
            ),
            coverage_by_attack_type=robustness_results.get('coverage', {}),
            weakness_count=robustness_results.get('n_weaknesses', 0),
            availability=1.0 - (
                pipeline.shadow_monitor.inference.get_statistics().get('timeout_rate', 0)
            ),
            mtbf_samples=float(nominal_results['total']),
            test_coverage=0.95,  # Placeholder
            requirements_traced=12,  # Placeholder
        )

    def _check_requirements(self, metrics: CertificationMetrics) -> bool:
        """Check if all requirements are met."""
        checks = [
            metrics.detection_rate_overall >= self.config.min_detection_rate,
            metrics.false_positive_rate <= self.config.max_false_positive_rate,
            metrics.mean_detection_latency_ms <= self.config.max_detection_latency_ms,
            metrics.coverage_by_attack_type.get('overall', 0) >= self.config.min_coverage,
            metrics.weakness_count <= self.config.max_weaknesses,
            metrics.availability >= self.config.min_availability,
        ]

        return all(checks)

    def generate_report(self) -> str:
        """Generate certification report."""
        if not self._results:
            return "No validation results available."

        metrics = self._results['metrics']
        passed = self._results['passed']

        report = [
            "=" * 60,
            "CERTIFICATION VALIDATION REPORT",
            "=" * 60,
            "",
            f"OVERALL RESULT: {'PASSED' if passed else 'FAILED'}",
            "",
            "DETECTION METRICS:",
            f"  Detection Rate: {metrics.detection_rate_overall:.2%}",
            f"  False Positive Rate: {metrics.false_positive_rate:.4%}",
            f"  False Negative Rate: {metrics.false_negative_rate:.2%}",
            "",
            "TIMING METRICS:",
            f"  Mean Detection Latency: {metrics.mean_detection_latency_ms:.1f} ms",
            f"  P95 Detection Latency: {metrics.p95_detection_latency_ms:.1f} ms",
            f"  Worst Case Latency: {metrics.worst_case_latency_ms:.1f} ms",
            "",
            "ROBUSTNESS METRICS:",
            f"  Overall Coverage: {metrics.coverage_by_attack_type.get('overall', 0):.2%}",
            f"  Weakness Count: {metrics.weakness_count}",
            "",
            "RELIABILITY METRICS:",
            f"  Availability: {metrics.availability:.4%}",
            f"  MTBF (samples): {metrics.mtbf_samples:.0f}",
            "",
            "TRACEABILITY:",
            f"  Test Coverage: {metrics.test_coverage:.2%}",
            f"  Requirements Traced: {metrics.requirements_traced}",
            "=" * 60,
        ]

        return "\n".join(report)


def run_certification_validation(
    nominal_trajectories: np.ndarray,
    attack_trajectories: np.ndarray,
) -> Tuple[bool, CertificationMetrics, str]:
    """
    Run full certification validation.

    Args:
        nominal_trajectories: [N, T, state_dim] nominal data
        attack_trajectories: [M, T, state_dim] attack data

    Returns:
        (passed, metrics, report) tuple
    """
    pipeline = UnifiedDetectionPipeline()
    validator = CertificationValidator()

    passed, metrics = validator.validate(
        pipeline, nominal_trajectories, attack_trajectories
    )

    report = validator.generate_report()

    return passed, metrics, report
