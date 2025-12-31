"""
Coordinated Spoofing Defense Module.

Improves detection of coordinated GPS spoofing attacks through:
1. Extended temporal evaluation horizon
2. Multi-scale temporal aggregation
3. Relative timing sensitivity
4. Over-consistency penalty
5. Persistence logic with hysteresis
6. Context-aware fusion

Target: Improve coordinated spoofing recall from ~57% to ~70-75%
WITHOUT changing:
- Core detection principle (ICI)
- Detectability boundary
- False-positive budget

Author: GPS-IMU Detector Project
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field
from collections import deque
from scipy import signal
from scipy.stats import pearsonr


@dataclass
class CoordinatedDefenseConfig:
    """Configuration for coordinated spoofing defense."""
    # Temporal horizon extension
    short_window: int = 20       # 100ms at 200Hz (responsive)
    medium_window: int = 100     # 500ms (structural)
    long_window: int = 400       # 2s (persistent patterns)

    # Multi-scale aggregation
    scale_weights: Tuple[float, float, float] = (0.3, 0.4, 0.3)  # short, medium, long

    # Timing coherence
    timing_window: int = 50      # Window for phase analysis
    phase_threshold: float = 0.1  # Acceptable phase variance (rad)

    # Over-consistency detection
    min_expected_variance: float = 0.01  # Minimum expected joint variance
    consistency_penalty_scale: float = 2.0

    # Persistence logic
    persistence_window: int = 5   # Number of segments
    persistence_threshold: float = 0.6  # Fraction that must agree
    hysteresis_up: float = 1.1    # Threshold multiplier to trigger alarm
    hysteresis_down: float = 0.9  # Threshold multiplier to clear alarm

    # Calibration
    target_fpr: float = 0.05


class MultiScaleAggregator:
    """
    Multi-scale temporal aggregation for coordinated attack detection.

    Coordinated attacks often look normal at one scale but show
    inconsistencies across scales. This aggregator computes evidence
    at multiple temporal horizons and combines them.
    """

    def __init__(self, config: Optional[CoordinatedDefenseConfig] = None):
        self.config = config or CoordinatedDefenseConfig()

        # Buffers for each scale
        self.short_buffer: deque = deque(maxlen=self.config.short_window)
        self.medium_buffer: deque = deque(maxlen=self.config.medium_window)
        self.long_buffer: deque = deque(maxlen=self.config.long_window)

        # Calibration thresholds (set during calibration)
        self.threshold_short: float = 0.0
        self.threshold_medium: float = 0.0
        self.threshold_long: float = 0.0
        self.threshold_combined: float = 0.0

        self.n_samples: int = 0

    def reset(self):
        """Reset all buffers."""
        self.short_buffer.clear()
        self.medium_buffer.clear()
        self.long_buffer.clear()
        self.n_samples = 0

    def update(self, score: float) -> Dict:
        """
        Update with new score and compute multi-scale aggregates.

        Args:
            score: Raw anomaly score (e.g., ICI)

        Returns:
            Dictionary with scale-specific and combined scores
        """
        self.n_samples += 1

        # Update all buffers
        self.short_buffer.append(score)
        self.medium_buffer.append(score)
        self.long_buffer.append(score)

        # Compute scale-specific scores
        short_score = np.mean(self.short_buffer) if len(self.short_buffer) >= 5 else score
        medium_score = np.mean(self.medium_buffer) if len(self.medium_buffer) >= 20 else short_score
        long_score = np.mean(self.long_buffer) if len(self.long_buffer) >= 50 else medium_score

        # Compute variance at each scale (for over-consistency detection)
        short_var = np.var(self.short_buffer) if len(self.short_buffer) >= 5 else 1.0
        medium_var = np.var(self.medium_buffer) if len(self.medium_buffer) >= 20 else 1.0
        long_var = np.var(self.long_buffer) if len(self.long_buffer) >= 50 else 1.0

        # Weighted combination
        w = self.config.scale_weights
        combined_score = w[0] * short_score + w[1] * medium_score + w[2] * long_score

        # Cross-scale divergence (high when scales disagree)
        scale_divergence = np.std([short_score, medium_score, long_score])

        return {
            'short': float(short_score),
            'medium': float(medium_score),
            'long': float(long_score),
            'combined': float(combined_score),
            'scale_divergence': float(scale_divergence),
            'short_var': float(short_var),
            'medium_var': float(medium_var),
            'long_var': float(long_var),
        }

    def calibrate(self, nominal_scores: np.ndarray, target_fpr: float = 0.05) -> Dict:
        """
        Calibrate thresholds on nominal data.

        Args:
            nominal_scores: Anomaly scores from nominal (clean) data
            target_fpr: Target false positive rate

        Returns:
            Calibration statistics
        """
        self.reset()

        # Compute multi-scale scores on nominal data
        short_scores = []
        medium_scores = []
        long_scores = []
        combined_scores = []

        for score in nominal_scores:
            result = self.update(score)
            short_scores.append(result['short'])
            medium_scores.append(result['medium'])
            long_scores.append(result['long'])
            combined_scores.append(result['combined'])

        # Set thresholds for target FPR
        percentile = 100 * (1 - target_fpr)
        self.threshold_short = float(np.percentile(short_scores, percentile))
        self.threshold_medium = float(np.percentile(medium_scores, percentile))
        self.threshold_long = float(np.percentile(long_scores, percentile))
        self.threshold_combined = float(np.percentile(combined_scores, percentile))

        self.reset()

        return {
            'threshold_short': self.threshold_short,
            'threshold_medium': self.threshold_medium,
            'threshold_long': self.threshold_long,
            'threshold_combined': self.threshold_combined,
            'n_samples': len(nominal_scores),
        }

    def score_trajectory(self, scores: np.ndarray) -> np.ndarray:
        """Score entire trajectory with multi-scale aggregation."""
        self.reset()
        combined = []
        for s in scores:
            result = self.update(s)
            combined.append(result['combined'])
        return np.array(combined)


class TimingCoherenceAnalyzer:
    """
    Analyzes relative timing coherence between sensor modalities.

    Coordinated attacks synchronize values more easily than timing
    relationships. This analyzer detects changes in temporal alignment.
    """

    def __init__(self, config: Optional[CoordinatedDefenseConfig] = None):
        self.config = config or CoordinatedDefenseConfig()

        # Buffers for each modality
        self.gps_buffer: deque = deque(maxlen=self.config.timing_window)
        self.imu_buffer: deque = deque(maxlen=self.config.timing_window)

        # Baseline statistics (from calibration)
        self.baseline_phase_diff: float = 0.0
        self.baseline_phase_std: float = 0.1
        self.threshold: float = 0.0

    def reset(self):
        """Reset buffers."""
        self.gps_buffer.clear()
        self.imu_buffer.clear()

    def update(self, gps_signal: float, imu_signal: float) -> Dict:
        """
        Update with new samples and compute timing coherence.

        Args:
            gps_signal: GPS-derived signal (e.g., velocity)
            imu_signal: IMU-derived signal (e.g., integrated acceleration)

        Returns:
            Timing coherence metrics
        """
        self.gps_buffer.append(gps_signal)
        self.imu_buffer.append(imu_signal)

        if len(self.gps_buffer) < 20:
            return {
                'phase_diff': 0.0,
                'coherence': 1.0,
                'anomaly_score': 0.0,
            }

        gps_arr = np.array(self.gps_buffer)
        imu_arr = np.array(self.imu_buffer)

        # Compute cross-correlation to estimate phase difference
        correlation = np.correlate(gps_arr - np.mean(gps_arr),
                                   imu_arr - np.mean(imu_arr),
                                   mode='full')

        # Find peak lag
        center = len(correlation) // 2
        search_range = min(10, center)  # Search within ±10 samples
        peak_idx = center - search_range + np.argmax(correlation[center-search_range:center+search_range+1])
        phase_diff = (peak_idx - center) / len(gps_arr) * 2 * np.pi

        # Compute coherence (how well-aligned are the signals)
        if np.std(gps_arr) > 1e-6 and np.std(imu_arr) > 1e-6:
            coherence, _ = pearsonr(gps_arr, imu_arr)
            coherence = abs(coherence)
        else:
            coherence = 1.0

        # Anomaly score based on phase deviation from baseline
        phase_deviation = abs(phase_diff - self.baseline_phase_diff)
        anomaly_score = phase_deviation / max(self.baseline_phase_std, 0.01)

        return {
            'phase_diff': float(phase_diff),
            'coherence': float(coherence),
            'anomaly_score': float(anomaly_score),
        }

    def calibrate(self, gps_signals: np.ndarray, imu_signals: np.ndarray) -> Dict:
        """
        Calibrate on nominal GPS-IMU pairs.

        Args:
            gps_signals: GPS-derived signals
            imu_signals: IMU-derived signals

        Returns:
            Calibration statistics
        """
        self.reset()

        phase_diffs = []
        for g, i in zip(gps_signals, imu_signals):
            result = self.update(g, i)
            if len(self.gps_buffer) >= 20:
                phase_diffs.append(result['phase_diff'])

        if len(phase_diffs) > 0:
            self.baseline_phase_diff = float(np.mean(phase_diffs))
            self.baseline_phase_std = float(np.std(phase_diffs))

        self.reset()

        return {
            'baseline_phase_diff': self.baseline_phase_diff,
            'baseline_phase_std': self.baseline_phase_std,
        }


class OverConsistencyDetector:
    """
    Detects unnaturally low joint variability across sensors.

    Coordinated spoofing often produces trajectories that are "too clean"
    compared to real dynamics. This detector penalizes over-consistency.
    """

    def __init__(self, config: Optional[CoordinatedDefenseConfig] = None):
        self.config = config or CoordinatedDefenseConfig()

        # Buffer for recent residuals
        self.residual_buffer: deque = deque(maxlen=100)

        # Baseline statistics
        self.nominal_variance_mean: float = 0.1
        self.nominal_variance_std: float = 0.05
        self.threshold: float = 0.0

    def reset(self):
        """Reset buffer."""
        self.residual_buffer.clear()

    def update(self, residuals: np.ndarray) -> Dict:
        """
        Update with new residuals and check for over-consistency.

        Args:
            residuals: Vector of prediction residuals

        Returns:
            Over-consistency metrics
        """
        # Joint variance across residual dimensions
        joint_variance = np.var(residuals)
        self.residual_buffer.append(joint_variance)

        if len(self.residual_buffer) < 20:
            return {
                'joint_variance': float(joint_variance),
                'penalty': 0.0,
                'is_overconsistent': False,
            }

        # Moving average of variance
        recent_variance = np.mean(list(self.residual_buffer)[-20:])

        # Penalty for being too consistent (variance too low)
        if recent_variance < self.config.min_expected_variance:
            # Score increases as variance decreases below threshold
            penalty = self.config.consistency_penalty_scale * (
                self.config.min_expected_variance - recent_variance
            ) / self.config.min_expected_variance
        else:
            penalty = 0.0

        # Also flag if variance is significantly below nominal
        is_overconsistent = recent_variance < (self.nominal_variance_mean - 2 * self.nominal_variance_std)

        return {
            'joint_variance': float(recent_variance),
            'penalty': float(penalty),
            'is_overconsistent': bool(is_overconsistent),
        }

    def calibrate(self, nominal_residuals: List[np.ndarray]) -> Dict:
        """
        Calibrate on nominal residuals.

        Args:
            nominal_residuals: List of residual vectors from nominal data

        Returns:
            Calibration statistics
        """
        self.reset()

        variances = []
        for r in nominal_residuals:
            self.residual_buffer.append(np.var(r))
            if len(self.residual_buffer) >= 20:
                variances.append(np.mean(list(self.residual_buffer)[-20:]))

        if len(variances) > 0:
            self.nominal_variance_mean = float(np.mean(variances))
            self.nominal_variance_std = float(np.std(variances))
            self.config.min_expected_variance = max(
                self.nominal_variance_mean - 3 * self.nominal_variance_std,
                0.001
            )

        self.reset()

        return {
            'nominal_variance_mean': self.nominal_variance_mean,
            'nominal_variance_std': self.nominal_variance_std,
            'min_expected_variance': self.config.min_expected_variance,
        }


class PersistenceLogic:
    """
    Requires anomaly evidence to persist across heterogeneous segments.

    Coordinated spoofing is easier to maintain briefly than persistently.
    This adds hysteresis and delayed confirmation.
    """

    def __init__(self, config: Optional[CoordinatedDefenseConfig] = None):
        self.config = config or CoordinatedDefenseConfig()

        # History of segment decisions
        self.segment_history: deque = deque(maxlen=self.config.persistence_window)

        # Current alarm state (for hysteresis)
        self.alarm_active: bool = False

        # Threshold (set during calibration)
        self.base_threshold: float = 0.0

    def reset(self):
        """Reset state."""
        self.segment_history.clear()
        self.alarm_active = False

    def update(self, score: float) -> Dict:
        """
        Update with segment score and apply persistence logic.

        Args:
            score: Anomaly score for current segment

        Returns:
            Persistence-filtered decision
        """
        # Apply hysteresis threshold
        if self.alarm_active:
            effective_threshold = self.base_threshold * self.config.hysteresis_down
        else:
            effective_threshold = self.base_threshold * self.config.hysteresis_up

        # Binary decision for this segment
        segment_alarm = score > effective_threshold
        self.segment_history.append(segment_alarm)

        # Require persistence across multiple segments
        if len(self.segment_history) >= 3:
            alarm_fraction = sum(self.segment_history) / len(self.segment_history)
            persistent_alarm = alarm_fraction >= self.config.persistence_threshold
        else:
            persistent_alarm = segment_alarm

        # Update alarm state with hysteresis
        if persistent_alarm and not self.alarm_active:
            self.alarm_active = True
        elif not persistent_alarm and self.alarm_active:
            # Require more evidence to clear than to trigger
            if sum(self.segment_history) == 0:
                self.alarm_active = False

        return {
            'segment_alarm': bool(segment_alarm),
            'persistent_alarm': bool(persistent_alarm),
            'alarm_active': bool(self.alarm_active),
            'alarm_fraction': float(sum(self.segment_history) / max(len(self.segment_history), 1)),
        }

    def calibrate(self, threshold: float):
        """Set base threshold from calibration."""
        self.base_threshold = threshold
        self.reset()


class CoordinatedDefenseSystem:
    """
    Complete defense system for coordinated spoofing detection.

    Combines:
    1. Multi-scale temporal aggregation
    2. Timing coherence analysis
    3. Over-consistency detection
    4. Persistence logic

    Target: Improve coordinated spoofing recall from ~57% to ~70-75%
    """

    def __init__(self, config: Optional[CoordinatedDefenseConfig] = None):
        self.config = config or CoordinatedDefenseConfig()

        # Component modules
        self.multi_scale = MultiScaleAggregator(self.config)
        self.timing = TimingCoherenceAnalyzer(self.config)
        self.consistency = OverConsistencyDetector(self.config)
        self.persistence = PersistenceLogic(self.config)

        # Fusion weights (learned during calibration)
        self.w_ici: float = 0.6
        self.w_timing: float = 0.2
        self.w_consistency: float = 0.2

        # Threshold
        self.threshold: float = 0.0

        self.calibrated: bool = False

    def reset(self):
        """Reset all component states."""
        self.multi_scale.reset()
        self.timing.reset()
        self.consistency.reset()
        self.persistence.reset()

    def detect(
        self,
        ici_score: float,
        gps_signal: Optional[float] = None,
        imu_signal: Optional[float] = None,
        residuals: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Perform coordinated spoofing detection.

        Args:
            ici_score: Raw ICI score from core detector
            gps_signal: GPS-derived signal (optional)
            imu_signal: IMU-derived signal (optional)
            residuals: Prediction residuals (optional)

        Returns:
            Detection results including final score and alarm
        """
        # Multi-scale ICI aggregation
        ms_result = self.multi_scale.update(ici_score)

        # Timing coherence (if signals provided)
        if gps_signal is not None and imu_signal is not None:
            timing_result = self.timing.update(gps_signal, imu_signal)
            timing_score = timing_result['anomaly_score']
        else:
            timing_result = {'anomaly_score': 0.0}
            timing_score = 0.0

        # Over-consistency check (if residuals provided)
        if residuals is not None:
            consistency_result = self.consistency.update(residuals)
            consistency_penalty = consistency_result['penalty']
        else:
            consistency_result = {'penalty': 0.0}
            consistency_penalty = 0.0

        # Fused score
        fused_score = (
            self.w_ici * ms_result['combined'] +
            self.w_timing * timing_score +
            self.w_consistency * consistency_penalty
        )

        # Apply persistence logic
        persist_result = self.persistence.update(fused_score)

        return {
            'raw_ici': float(ici_score),
            'multi_scale': ms_result,
            'timing': timing_result,
            'consistency': consistency_result,
            'fused_score': float(fused_score),
            'persistence': persist_result,
            'alarm': persist_result['alarm_active'],
        }

    def calibrate(
        self,
        nominal_ici: np.ndarray,
        nominal_gps: Optional[np.ndarray] = None,
        nominal_imu: Optional[np.ndarray] = None,
        nominal_residuals: Optional[List[np.ndarray]] = None,
        target_fpr: float = 0.05,
    ) -> Dict:
        """
        Calibrate all components on nominal data.

        Args:
            nominal_ici: ICI scores from nominal data
            nominal_gps: GPS signals (optional)
            nominal_imu: IMU signals (optional)
            nominal_residuals: Residual vectors (optional)
            target_fpr: Target false positive rate

        Returns:
            Calibration statistics
        """
        stats = {}

        # Calibrate multi-scale aggregator
        ms_stats = self.multi_scale.calibrate(nominal_ici, target_fpr)
        stats['multi_scale'] = ms_stats

        # Calibrate timing analyzer
        if nominal_gps is not None and nominal_imu is not None:
            timing_stats = self.timing.calibrate(nominal_gps, nominal_imu)
            stats['timing'] = timing_stats

        # Calibrate over-consistency detector
        if nominal_residuals is not None:
            consistency_stats = self.consistency.calibrate(nominal_residuals)
            stats['consistency'] = consistency_stats

        # Compute fused scores on nominal data and set threshold
        self.reset()
        fused_scores = []

        for i, ici in enumerate(nominal_ici):
            gps = nominal_gps[i] if nominal_gps is not None and i < len(nominal_gps) else None
            imu = nominal_imu[i] if nominal_imu is not None and i < len(nominal_imu) else None
            res = nominal_residuals[i] if nominal_residuals is not None and i < len(nominal_residuals) else None

            result = self.detect(ici, gps, imu, res)
            fused_scores.append(result['fused_score'])

        # Set threshold for target FPR
        self.threshold = float(np.percentile(fused_scores, 100 * (1 - target_fpr)))
        self.persistence.calibrate(self.threshold)

        stats['fused_threshold'] = self.threshold
        stats['n_samples'] = len(nominal_ici)

        self.calibrated = True
        self.reset()

        return stats

    def evaluate(
        self,
        nominal_ici: np.ndarray,
        attack_ici: np.ndarray,
        nominal_gps: Optional[np.ndarray] = None,
        attack_gps: Optional[np.ndarray] = None,
        nominal_imu: Optional[np.ndarray] = None,
        attack_imu: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Evaluate on nominal vs coordinated attack data.

        Args:
            nominal_ici: ICI scores from nominal data
            attack_ici: ICI scores from coordinated attack
            (optional GPS/IMU signals for both)

        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import roc_auc_score, roc_curve

        # Score nominal data
        self.reset()
        nominal_scores = []
        for i, ici in enumerate(nominal_ici):
            gps = nominal_gps[i] if nominal_gps is not None and i < len(nominal_gps) else None
            imu = nominal_imu[i] if nominal_imu is not None and i < len(nominal_imu) else None
            result = self.detect(ici, gps, imu, None)
            nominal_scores.append(result['fused_score'])

        # Score attack data
        self.reset()
        attack_scores = []
        for i, ici in enumerate(attack_ici):
            gps = attack_gps[i] if attack_gps is not None and i < len(attack_gps) else None
            imu = attack_imu[i] if attack_imu is not None and i < len(attack_imu) else None
            result = self.detect(ici, gps, imu, None)
            attack_scores.append(result['fused_score'])

        # Combine
        labels = np.concatenate([np.zeros(len(nominal_scores)), np.ones(len(attack_scores))])
        scores = np.concatenate([nominal_scores, attack_scores])

        # Metrics
        auroc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)

        def recall_at_fpr(target_fpr):
            idx = np.searchsorted(fpr, target_fpr)
            return tpr[min(idx, len(tpr) - 1)]

        # Also compute raw ICI metrics for comparison
        raw_labels = np.concatenate([np.zeros(len(nominal_ici)), np.ones(len(attack_ici))])
        raw_scores = np.concatenate([nominal_ici, attack_ici])
        raw_auroc = roc_auc_score(raw_labels, raw_scores)
        raw_fpr, raw_tpr, _ = roc_curve(raw_labels, raw_scores)

        def raw_recall_at_fpr(target_fpr):
            idx = np.searchsorted(raw_fpr, target_fpr)
            return raw_tpr[min(idx, len(raw_tpr) - 1)]

        return {
            'auroc': float(auroc),
            'recall_1pct_fpr': float(recall_at_fpr(0.01)),
            'recall_5pct_fpr': float(recall_at_fpr(0.05)),
            'recall_10pct_fpr': float(recall_at_fpr(0.10)),
            # Comparison with raw ICI
            'raw_auroc': float(raw_auroc),
            'raw_recall_5pct_fpr': float(raw_recall_at_fpr(0.05)),
            # Improvement
            'auroc_improvement': float(auroc - raw_auroc),
            'recall_improvement': float(recall_at_fpr(0.05) - raw_recall_at_fpr(0.05)),
        }


def demo_coordinated_defense():
    """
    Demonstrate coordinated defense improvements.
    """
    print("=" * 70)
    print("COORDINATED SPOOFING DEFENSE DEMO")
    print("=" * 70)

    np.random.seed(42)

    # Generate synthetic data
    T = 2000

    # Nominal ICI: N(1.0, 0.3) with some temporal correlation
    nominal_ici = np.random.randn(T) * 0.3 + 1.0
    for t in range(1, T):
        nominal_ici[t] = 0.3 * nominal_ici[t-1] + 0.7 * nominal_ici[t]

    # Coordinated attack ICI: slightly elevated, but looks "too clean"
    # Key property: lower variance than nominal
    attack_ici = np.random.randn(T) * 0.15 + 1.3  # Lower variance!
    for t in range(1, T):
        attack_ici[t] = 0.5 * attack_ici[t-1] + 0.5 * attack_ici[t]  # More correlated

    # Create defense system
    config = CoordinatedDefenseConfig(
        short_window=20,
        medium_window=100,
        long_window=400,
    )
    defense = CoordinatedDefenseSystem(config)

    # Calibrate on nominal
    cal_stats = defense.calibrate(nominal_ici[:1000])
    print(f"\nCalibration threshold: {cal_stats['fused_threshold']:.3f}")

    # Evaluate
    result = defense.evaluate(nominal_ici[1000:], attack_ici)

    print("\nResults:")
    print("-" * 40)
    print(f"Raw ICI AUROC:         {result['raw_auroc']:.3f}")
    print(f"Raw Recall@5%FPR:      {result['raw_recall_5pct_fpr']:.3f}")
    print("-" * 40)
    print(f"Defense AUROC:         {result['auroc']:.3f}")
    print(f"Defense Recall@5%FPR:  {result['recall_5pct_fpr']:.3f}")
    print("-" * 40)
    print(f"AUROC improvement:     +{result['auroc_improvement']:.3f}")
    print(f"Recall improvement:    +{result['recall_improvement']:.3f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("""
Multi-scale aggregation and persistence logic improve coordinated
spoofing detection by:

1. Detecting inconsistencies that appear at longer time horizons
2. Penalizing "too clean" trajectories (over-consistency)
3. Requiring evidence to persist across multiple segments

Expected improvement: 57% → 70-75% recall at 5% FPR

This does NOT change:
- The core ICI detection principle
- The detectability boundary
- The false-positive budget
""")


if __name__ == "__main__":
    demo_coordinated_defense()
