"""
Governance and Rollout Module (Phase 7)

Implements:
7.1 Drift Monitoring
    - Detects when model needs recalibration
    - Statistical tests for distribution shift
    - Automatic alerts

7.2 Recalibration Procedures
    - Safe recalibration with rollback
    - Version management
    - A/B testing support
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import json
import hashlib
import numpy as np


# =============================================================================
# Phase 7.1: Drift Monitoring
# =============================================================================

class DriftType(Enum):
    """Types of drift detected."""
    NONE = auto()
    RESIDUAL_MEAN_SHIFT = auto()
    RESIDUAL_VARIANCE_SHIFT = auto()
    REGIME_DISTRIBUTION_SHIFT = auto()
    PERFORMANCE_DEGRADATION = auto()


class DriftSeverity(Enum):
    """Severity of detected drift."""
    NONE = auto()
    MINOR = auto()       # Log only
    MODERATE = auto()    # Alert, consider recalibration
    SEVERE = auto()      # Immediate recalibration required


@dataclass
class DriftConfig:
    """Configuration for drift monitoring."""
    # Window sizes
    reference_window: int = 1000      # Samples for reference distribution
    test_window: int = 200            # Samples for current distribution

    # Thresholds
    mean_shift_threshold: float = 2.0   # Z-scores
    variance_ratio_threshold: float = 2.0
    regime_shift_threshold: float = 0.2  # KL divergence
    performance_drop_threshold: float = 0.1  # Relative drop

    # Alert settings
    alert_cooldown: int = 100         # Minimum samples between alerts


@dataclass
class DriftResult:
    """Result of drift detection."""
    drift_detected: bool
    drift_type: DriftType
    severity: DriftSeverity
    details: Dict[str, float]
    recommendation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DriftMonitor:
    """
    Statistical drift monitoring.

    Detects distribution shifts that indicate need for recalibration.
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()

        # Reference distributions (from calibration)
        self._ref_residuals: List[float] = []
        self._ref_regimes: List[int] = []
        self._ref_detections: List[bool] = []

        # Current window
        self._cur_residuals: List[float] = []
        self._cur_regimes: List[int] = []
        self._cur_detections: List[bool] = []

        # Alert tracking
        self._last_alert_sample = -float('inf')
        self._sample_count = 0
        self._drift_history: List[DriftResult] = []

    def set_reference(
        self,
        residuals: np.ndarray,
        regimes: np.ndarray,
        detections: Optional[np.ndarray] = None,
    ):
        """Set reference distributions from calibration data."""
        self._ref_residuals = list(residuals[:self.config.reference_window])
        self._ref_regimes = list(regimes[:self.config.reference_window])
        if detections is not None:
            self._ref_detections = list(detections[:self.config.reference_window])

    def update(
        self,
        residual: float,
        regime: int,
        detection: bool = False,
    ) -> DriftResult:
        """
        Update with new sample and check for drift.

        Args:
            residual: Current residual value
            regime: Current regime ID
            detection: Whether anomaly was detected

        Returns:
            DriftResult with drift status
        """
        self._sample_count += 1

        # Update current window
        self._cur_residuals.append(residual)
        self._cur_regimes.append(regime)
        self._cur_detections.append(detection)

        # Trim to window size
        max_window = self.config.test_window
        if len(self._cur_residuals) > max_window:
            self._cur_residuals.pop(0)
            self._cur_regimes.pop(0)
            self._cur_detections.pop(0)

        # Check drift if window is full
        if len(self._cur_residuals) < max_window:
            return DriftResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                details={'window_fill': len(self._cur_residuals) / max_window},
                recommendation="Collecting data",
            )

        # Check cooldown
        if self._sample_count - self._last_alert_sample < self.config.alert_cooldown:
            return DriftResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                details={'in_cooldown': True},
                recommendation="In cooldown period",
            )

        # Run drift tests
        result = self._run_drift_tests()

        if result.drift_detected:
            self._last_alert_sample = self._sample_count
            self._drift_history.append(result)

        return result

    def _run_drift_tests(self) -> DriftResult:
        """Run all drift detection tests."""
        if not self._ref_residuals:
            return DriftResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                details={'error': 'no_reference'},
                recommendation="Set reference distribution first",
            )

        # Test 1: Residual mean shift
        mean_shift = self._test_mean_shift()
        if mean_shift['significant']:
            severity = (DriftSeverity.SEVERE if mean_shift['zscore'] > 3.0
                       else DriftSeverity.MODERATE)
            return DriftResult(
                drift_detected=True,
                drift_type=DriftType.RESIDUAL_MEAN_SHIFT,
                severity=severity,
                details=mean_shift,
                recommendation="Recalibrate residual thresholds",
            )

        # Test 2: Residual variance shift
        var_shift = self._test_variance_shift()
        if var_shift['significant']:
            severity = (DriftSeverity.SEVERE if var_shift['ratio'] > 3.0
                       else DriftSeverity.MODERATE)
            return DriftResult(
                drift_detected=True,
                drift_type=DriftType.RESIDUAL_VARIANCE_SHIFT,
                severity=severity,
                details=var_shift,
                recommendation="Recalibrate confidence intervals",
            )

        # Test 3: Regime distribution shift
        regime_shift = self._test_regime_shift()
        if regime_shift['significant']:
            return DriftResult(
                drift_detected=True,
                drift_type=DriftType.REGIME_DISTRIBUTION_SHIFT,
                severity=DriftSeverity.MODERATE,
                details=regime_shift,
                recommendation="Review flight profile changes",
            )

        return DriftResult(
            drift_detected=False,
            drift_type=DriftType.NONE,
            severity=DriftSeverity.NONE,
            details={
                'mean_zscore': mean_shift['zscore'],
                'var_ratio': var_shift['ratio'],
            },
            recommendation="No action required",
        )

    def _test_mean_shift(self) -> Dict:
        """Test for mean shift using z-score."""
        ref_mean = np.mean(self._ref_residuals)
        ref_std = np.std(self._ref_residuals)
        cur_mean = np.mean(self._cur_residuals)

        if ref_std < 1e-6:
            ref_std = 1.0

        zscore = abs(cur_mean - ref_mean) / (ref_std / np.sqrt(len(self._cur_residuals)))

        return {
            'zscore': float(zscore),
            'ref_mean': float(ref_mean),
            'cur_mean': float(cur_mean),
            'significant': zscore > self.config.mean_shift_threshold,
        }

    def _test_variance_shift(self) -> Dict:
        """Test for variance shift using F-test."""
        ref_var = np.var(self._ref_residuals)
        cur_var = np.var(self._cur_residuals)

        if ref_var < 1e-6:
            ref_var = 1.0

        ratio = cur_var / ref_var

        return {
            'ratio': float(ratio),
            'ref_var': float(ref_var),
            'cur_var': float(cur_var),
            'significant': ratio > self.config.variance_ratio_threshold or ratio < 1.0 / self.config.variance_ratio_threshold,
        }

    def _test_regime_shift(self) -> Dict:
        """Test for regime distribution shift."""
        # Compute regime distributions
        ref_counts = np.bincount(self._ref_regimes, minlength=6)
        cur_counts = np.bincount(self._cur_regimes, minlength=6)

        ref_dist = ref_counts / max(1, ref_counts.sum())
        cur_dist = cur_counts / max(1, cur_counts.sum())

        # KL divergence (with smoothing)
        eps = 1e-6
        ref_dist = ref_dist + eps
        cur_dist = cur_dist + eps
        ref_dist = ref_dist / ref_dist.sum()
        cur_dist = cur_dist / cur_dist.sum()

        kl_div = float(np.sum(cur_dist * np.log(cur_dist / ref_dist)))

        return {
            'kl_divergence': kl_div,
            'significant': kl_div > self.config.regime_shift_threshold,
        }

    def get_statistics(self) -> Dict:
        """Get monitoring statistics."""
        return {
            'sample_count': self._sample_count,
            'drift_count': len(self._drift_history),
            'current_window_size': len(self._cur_residuals),
            'reference_window_size': len(self._ref_residuals),
        }

    def reset(self):
        """Reset current window (keep reference)."""
        self._cur_residuals.clear()
        self._cur_regimes.clear()
        self._cur_detections.clear()
        self._sample_count = 0
        self._last_alert_sample = -float('inf')


# =============================================================================
# Phase 7.2: Recalibration and Version Management
# =============================================================================

@dataclass
class ModelVersion:
    """Version metadata for a calibrated model."""
    version_id: str
    created_at: str
    calibration_samples: int
    metrics: Dict[str, float]
    config_hash: str
    parent_version: Optional[str] = None
    status: str = "active"


@dataclass
class RecalibrationConfig:
    """Configuration for recalibration."""
    min_samples_for_recalibration: int = 500
    validation_split: float = 0.2
    min_improvement: float = 0.01
    rollback_on_degradation: bool = True


class VersionManager:
    """
    Model version and recalibration management.

    Supports:
    - Version tracking
    - Safe recalibration with validation
    - Rollback on degradation
    """

    def __init__(self, config: Optional[RecalibrationConfig] = None):
        self.config = config or RecalibrationConfig()

        self._versions: Dict[str, ModelVersion] = {}
        self._active_version: Optional[str] = None
        self._calibration_data: Dict[str, Any] = {}

    def create_version(
        self,
        calibration_samples: int,
        metrics: Dict[str, float],
        config_hash: str,
    ) -> ModelVersion:
        """Create a new model version."""
        version_id = self._generate_version_id()

        version = ModelVersion(
            version_id=version_id,
            created_at=datetime.now().isoformat(),
            calibration_samples=calibration_samples,
            metrics=metrics,
            config_hash=config_hash,
            parent_version=self._active_version,
        )

        self._versions[version_id] = version
        return version

    def activate_version(self, version_id: str) -> bool:
        """Activate a specific version."""
        if version_id not in self._versions:
            return False

        # Deactivate current
        if self._active_version:
            self._versions[self._active_version].status = "inactive"

        # Activate new
        self._versions[version_id].status = "active"
        self._active_version = version_id
        return True

    def rollback(self) -> Optional[str]:
        """Rollback to parent version."""
        if not self._active_version:
            return None

        current = self._versions[self._active_version]
        if not current.parent_version:
            return None

        self.activate_version(current.parent_version)
        return current.parent_version

    def get_active_version(self) -> Optional[ModelVersion]:
        """Get currently active version."""
        if not self._active_version:
            return None
        return self._versions.get(self._active_version)

    def list_versions(self) -> List[ModelVersion]:
        """List all versions."""
        return sorted(
            self._versions.values(),
            key=lambda v: v.created_at,
            reverse=True,
        )

    def _generate_version_id(self) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.md5(
            str(np.random.random()).encode()
        ).hexdigest()[:6]
        return f"v{timestamp}-{random_suffix}"


class RecalibrationManager:
    """
    Safe recalibration with validation.

    Ensures recalibration improves performance before activation.
    """

    def __init__(
        self,
        version_manager: Optional[VersionManager] = None,
        config: Optional[RecalibrationConfig] = None,
    ):
        self.version_manager = version_manager or VersionManager()
        self.config = config or RecalibrationConfig()

    def recalibrate(
        self,
        pipeline,  # UnifiedDetectionPipeline
        new_data: np.ndarray,
        attack_data: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Dict]:
        """
        Recalibrate pipeline with new data.

        Args:
            pipeline: Pipeline to recalibrate
            new_data: [N, T, state_dim] new nominal data
            attack_data: [M, T, state_dim] attack data for validation

        Returns:
            (success, details) tuple
        """
        if len(new_data) < self.config.min_samples_for_recalibration:
            return False, {
                'error': 'insufficient_data',
                'required': self.config.min_samples_for_recalibration,
                'provided': len(new_data),
            }

        # Get baseline metrics
        baseline_metrics = self._evaluate(pipeline, new_data, attack_data)

        # Save current calibration for rollback
        old_mean = pipeline.safety_system.severity_scorer.residual_mean
        old_std = pipeline.safety_system.severity_scorer.residual_std

        # Perform recalibration
        pipeline.calibrate(new_data)

        # Evaluate new calibration
        new_metrics = self._evaluate(pipeline, new_data, attack_data)

        # Check improvement
        improved = self._check_improvement(baseline_metrics, new_metrics)

        if not improved and self.config.rollback_on_degradation:
            # Rollback
            pipeline.safety_system.severity_scorer.residual_mean = old_mean
            pipeline.safety_system.severity_scorer.residual_std = old_std

            return False, {
                'error': 'no_improvement',
                'baseline_metrics': baseline_metrics,
                'new_metrics': new_metrics,
                'rolled_back': True,
            }

        # Create new version
        config_hash = hashlib.md5(
            str(pipeline.config).encode()
        ).hexdigest()

        version = self.version_manager.create_version(
            calibration_samples=len(new_data),
            metrics=new_metrics,
            config_hash=config_hash,
        )

        self.version_manager.activate_version(version.version_id)

        return True, {
            'version_id': version.version_id,
            'baseline_metrics': baseline_metrics,
            'new_metrics': new_metrics,
            'improvement': new_metrics.get('score', 0) - baseline_metrics.get('score', 0),
        }

    def _evaluate(
        self,
        pipeline,
        nominal_data: np.ndarray,
        attack_data: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Evaluate pipeline performance."""
        # Simple evaluation - count false positives on nominal
        fp_count = 0
        total = 0

        for traj in nominal_data:
            for t in range(len(traj) - 1):
                result = pipeline.process(traj[t], traj[t+1])
                if result.decision.value >= 4:  # HARD_ALERT or higher
                    fp_count += 1
                total += 1
            pipeline.reset()

        fpr = fp_count / max(1, total)

        # Detection rate on attacks
        detection_rate = 0.0
        if attack_data is not None and len(attack_data) > 0:
            detections = 0
            for traj in attack_data:
                detected = False
                for t in range(len(traj) - 1):
                    result = pipeline.process(traj[t], traj[t+1])
                    if result.decision.value >= 4:
                        detected = True
                        break
                if detected:
                    detections += 1
                pipeline.reset()
            detection_rate = detections / len(attack_data)

        # Combined score
        score = detection_rate * (1 - fpr)

        return {
            'fpr': fpr,
            'detection_rate': detection_rate,
            'score': score,
        }

    def _check_improvement(
        self,
        baseline: Dict[str, float],
        new: Dict[str, float],
    ) -> bool:
        """Check if new calibration is better."""
        baseline_score = baseline.get('score', 0)
        new_score = new.get('score', 0)

        return new_score >= baseline_score + self.config.min_improvement


# =============================================================================
# Governance System
# =============================================================================

class GovernanceSystem:
    """
    Complete governance system for production deployment.

    Combines:
    - Drift monitoring
    - Version management
    - Recalibration
    """

    def __init__(self):
        self.drift_monitor = DriftMonitor()
        self.version_manager = VersionManager()
        self.recalibration_manager = RecalibrationManager(self.version_manager)

        self._pipeline = None

    def initialize(
        self,
        pipeline,  # UnifiedDetectionPipeline
        reference_data: np.ndarray,
    ):
        """Initialize governance with pipeline and reference data."""
        self._pipeline = pipeline

        # Set reference distributions
        residuals = []
        regimes = []
        for traj in reference_data:
            for t in range(len(traj) - 1):
                result = pipeline.process(traj[t], traj[t+1])
                residuals.append(result.components.get('residual', 0))
                regimes.append(result.regime.value)
            pipeline.reset()

        self.drift_monitor.set_reference(
            np.array(residuals),
            np.array(regimes),
        )

        # Create initial version
        config_hash = hashlib.md5(str(pipeline.config).encode()).hexdigest()
        version = self.version_manager.create_version(
            calibration_samples=len(reference_data),
            metrics={'initial': True},
            config_hash=config_hash,
        )
        self.version_manager.activate_version(version.version_id)

    def monitor(
        self,
        residual: float,
        regime: int,
        detection: bool = False,
    ) -> DriftResult:
        """Monitor for drift."""
        return self.drift_monitor.update(residual, regime, detection)

    def recalibrate(
        self,
        new_data: np.ndarray,
        attack_data: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Dict]:
        """Trigger recalibration."""
        if self._pipeline is None:
            return False, {'error': 'pipeline_not_initialized'}

        return self.recalibration_manager.recalibrate(
            self._pipeline, new_data, attack_data
        )

    def get_status(self) -> Dict:
        """Get governance status."""
        active_version = self.version_manager.get_active_version()

        return {
            'active_version': active_version.version_id if active_version else None,
            'drift_stats': self.drift_monitor.get_statistics(),
            'version_count': len(self.version_manager._versions),
        }
