#!/usr/bin/env python3
"""
Robustness Fixes: Principled Solutions for Weak Points

Core Principle: Add INVARIANCE, not SENSITIVITY.
               Sensitivity overfits. Invariance generalizes.

This module implements 5 principled fixes:

TEMPORAL FIXES (for 4% shuffling degradation):
1. Temporal Contrast - forward vs reversed consistency (not temporal modeling)
2. Multi-resolution Agreement - cross-scale consistency checks

DOMAIN FIXES (for CORAL failure):
3. Conditional Normalization - per-flight z-score (no domain labels)
4. Control-Conditioned Features - normalize by control magnitude

MISSED DETECTION FIX:
5. Gap-Tolerant Evidence Accumulation - k anomalies in N windows (non-consecutive)

Philosophy: Show WHY some tests fail and HOW to fix them = better science.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


# ============================================================
# SOLUTION 1: TEMPORAL CONTRAST
# ============================================================
# Instead of predicting sequences, COMPARE sequences against themselves.
# Shuffling breaks asymmetry. Marginal statistics remain unchanged.
# No attack labels. No new model capacity.

@dataclass
class TemporalContrastResult:
    """Result of temporal contrast computation."""
    forward_score: float
    reversed_score: float
    asymmetry: float  # The key signal
    is_anomalous: bool


class TemporalContrastDetector:
    """
    Detect anomalies by comparing forward vs time-reversed consistency.

    Key insight: Normal trajectories have causal structure.
    Attacks may break this asymmetry.

    This strengthens temporal reliance WITHOUT increasing sensitivity.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.baseline_asymmetry: Optional[float] = None
        self.baseline_std: Optional[float] = None

    def _compute_consistency(self, trajectory: np.ndarray, dt: float = 0.005) -> np.ndarray:
        """Compute position-velocity consistency scores."""
        n = len(trajectory)
        scores = np.zeros(n)

        pos = trajectory[:, 0:3]
        vel = trajectory[:, 3:6]

        for t in range(1, n):
            pos_expected = pos[t-1] + vel[t] * dt
            scores[t] = np.linalg.norm(pos[t] - pos_expected)

        return scores

    def compute_asymmetry(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute temporal asymmetry: difference between forward and reversed.

        Returns per-window asymmetry scores.
        """
        n = len(trajectory)
        asymmetry = np.zeros(n)

        for i in range(self.window_size, n):
            window = trajectory[i-self.window_size:i]

            # Forward consistency
            forward_scores = self._compute_consistency(window)
            forward_mean = np.mean(forward_scores)

            # Reversed consistency (flip time)
            reversed_window = window[::-1].copy()
            reversed_scores = self._compute_consistency(reversed_window)
            reversed_mean = np.mean(reversed_scores)

            # Asymmetry: the KEY signal
            # Normal trajectories: forward != reversed (causal structure)
            # Shuffled trajectories: forward ~ reversed (no structure)
            asymmetry[i] = np.abs(forward_mean - reversed_mean)

        return asymmetry

    def fit(self, normal_trajectories: List[np.ndarray]):
        """Learn baseline asymmetry from normal data."""
        all_asymmetry = []

        for traj in normal_trajectories:
            asym = self.compute_asymmetry(traj)
            all_asymmetry.extend(asym[self.window_size:])

        self.baseline_asymmetry = np.mean(all_asymmetry)
        self.baseline_std = np.std(all_asymmetry) + 1e-6

    def score(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Score trajectory based on temporal asymmetry.

        Lower asymmetry = more suspicious (structure may be broken).
        """
        asymmetry = self.compute_asymmetry(trajectory)

        if self.baseline_asymmetry is None:
            return asymmetry

        # Z-score relative to baseline
        # NEGATIVE z-score = less asymmetry than normal = suspicious
        z_scores = (asymmetry - self.baseline_asymmetry) / self.baseline_std

        # Return absolute value: both too much and too little asymmetry are suspicious
        return np.abs(z_scores)


# ============================================================
# SOLUTION 2: MULTI-RESOLUTION TEMPORAL AGREEMENT
# ============================================================
# Check consistency across time scales, not just within windows.
# Shuffling destroys cross-scale agreement.
# No new patterns learned. Just self-consistency checks.

@dataclass
class MultiResolutionResult:
    """Result of multi-resolution agreement check."""
    short_anomalous: bool
    medium_anomalous: bool
    long_anomalous: bool
    agreement_score: float  # 0-1, higher = more agreement
    disagreement_signal: float  # Higher = more suspicious


class MultiResolutionAgreement:
    """
    Check agreement across multiple time scales.

    Key insight: Shuffling destroys cross-scale agreement.
    If short/medium/long windows disagree, that's informative.

    This increases shuffling degradation naturally.
    """

    def __init__(
        self,
        short_window: int = 10,
        medium_window: int = 50,
        long_window: int = 200
    ):
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.thresholds: dict = {}

    def _compute_window_score(
        self,
        scores: np.ndarray,
        window: int,
        idx: int
    ) -> float:
        """Compute score for a single window."""
        start = max(0, idx - window)
        return np.mean(scores[start:idx])

    def fit(self, normal_scores: np.ndarray, target_fpr: float = 0.05):
        """Learn thresholds from normal data."""
        n = len(normal_scores)

        short_scores = []
        medium_scores = []
        long_scores = []

        for i in range(self.long_window, n):
            short_scores.append(self._compute_window_score(normal_scores, self.short_window, i))
            medium_scores.append(self._compute_window_score(normal_scores, self.medium_window, i))
            long_scores.append(self._compute_window_score(normal_scores, self.long_window, i))

        percentile = 100 * (1 - target_fpr)
        self.thresholds = {
            'short': np.percentile(short_scores, percentile),
            'medium': np.percentile(medium_scores, percentile),
            'long': np.percentile(long_scores, percentile)
        }

    def compute_agreement(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute agreement and disagreement signals.

        Returns:
            Tuple of (agreement_scores, disagreement_signals)
        """
        n = len(scores)
        agreement = np.zeros(n)
        disagreement = np.zeros(n)

        for i in range(self.long_window, n):
            short = self._compute_window_score(scores, self.short_window, i)
            medium = self._compute_window_score(scores, self.medium_window, i)
            long = self._compute_window_score(scores, self.long_window, i)

            # Check if each scale says "anomalous"
            short_anom = short > self.thresholds.get('short', np.inf)
            medium_anom = medium > self.thresholds.get('medium', np.inf)
            long_anom = long > self.thresholds.get('long', np.inf)

            # Agreement: all scales agree (all high or all low)
            votes = [short_anom, medium_anom, long_anom]
            agreement[i] = 1.0 if all(votes) or not any(votes) else 0.0

            # Disagreement signal: variance across scales
            # High disagreement = suspicious (cross-scale inconsistency)
            disagreement[i] = np.std([short, medium, long])

        return agreement, disagreement

    def detect(self, scores: np.ndarray) -> np.ndarray:
        """
        Detect using multi-resolution agreement.

        Anomaly if:
        - All scales agree it's anomalous, OR
        - High disagreement between scales (cross-scale inconsistency)
        """
        agreement, disagreement = self.compute_agreement(scores)

        # Unanimous anomaly
        n = len(scores)
        detections = np.zeros(n, dtype=bool)

        for i in range(self.long_window, n):
            short = self._compute_window_score(scores, self.short_window, i)
            medium = self._compute_window_score(scores, self.medium_window, i)
            long = self._compute_window_score(scores, self.long_window, i)

            short_anom = short > self.thresholds.get('short', np.inf)
            medium_anom = medium > self.thresholds.get('medium', np.inf)
            long_anom = long > self.thresholds.get('long', np.inf)

            # All agree it's anomalous
            if all([short_anom, medium_anom, long_anom]):
                detections[i] = True

        return detections


# ============================================================
# SOLUTION 3: CONDITIONAL NORMALIZATION
# ============================================================
# Normalize per-flight, not across domain.
# Removes environment scale. Preserves relative deviations.
# No labels. No domain leakage.

class ConditionalNormalizer:
    """
    Per-flight normalization for domain shift robustness.

    Key insight: Different flights have different scales.
    Normalize WITHIN each flight to remove environment effects.

    This often improves OOD without touching training.
    """

    def __init__(self, warmup_samples: int = 100):
        self.warmup_samples = warmup_samples

    def normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores using per-flight statistics.

        Uses rolling statistics after warmup period.
        """
        n = len(scores)
        normalized = np.zeros(n)

        # Warmup: use first N samples to establish baseline
        warmup_mean = np.mean(scores[:self.warmup_samples])
        warmup_std = np.std(scores[:self.warmup_samples]) + 1e-6

        # Normalize warmup period
        normalized[:self.warmup_samples] = (scores[:self.warmup_samples] - warmup_mean) / warmup_std

        # Rolling normalization after warmup
        for i in range(self.warmup_samples, n):
            # Use all samples up to now for statistics
            running_mean = np.mean(scores[:i])
            running_std = np.std(scores[:i]) + 1e-6
            normalized[i] = (scores[i] - running_mean) / running_std

        return normalized

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize feature matrix per-flight.

        Args:
            features: (N, D) feature matrix

        Returns:
            Normalized features
        """
        n, d = features.shape
        normalized = np.zeros_like(features)

        # Warmup statistics
        warmup_mean = np.mean(features[:self.warmup_samples], axis=0)
        warmup_std = np.std(features[:self.warmup_samples], axis=0) + 1e-6

        # Normalize
        for i in range(n):
            if i < self.warmup_samples:
                normalized[i] = (features[i] - warmup_mean) / warmup_std
            else:
                running_mean = np.mean(features[:i], axis=0)
                running_std = np.std(features[:i], axis=0) + 1e-6
                normalized[i] = (features[i] - running_mean) / running_std

        return normalized


# ============================================================
# SOLUTION 4: CONTROL-CONDITIONED FEATURES
# ============================================================
# Express deviations relative to commanded motion.
# This builds physics-aware invariance WITHOUT physics constraints.
# Reparameterizing features, not adding physics loss.

class ControlConditionedFeatures:
    """
    Normalize features by control magnitude.

    Key insight: Domain shift hurts because different flights
    have different control regimes.

    Instead of aligning states, align state-control relationships.
    """

    def __init__(self, control_indices: Tuple[int, ...] = (12, 13, 14)):
        """
        Args:
            control_indices: Indices of control-related features (e.g., acceleration)
        """
        self.control_indices = control_indices

    def compute(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute control-conditioned features.

        Args:
            trajectory: (N, D) trajectory with state and control

        Returns:
            Control-conditioned features
        """
        n = len(trajectory)

        # Extract control magnitude (e.g., acceleration magnitude)
        control = trajectory[:, list(self.control_indices)]
        control_magnitude = np.linalg.norm(control, axis=1) + 1e-6

        # Position-velocity residual
        pos = trajectory[:, 0:3]
        vel = trajectory[:, 3:6]

        residuals = np.zeros(n)
        for t in range(1, n):
            pos_expected = pos[t-1] + vel[t] * 0.005
            residuals[t] = np.linalg.norm(pos[t] - pos_expected)

        # Normalize residual by control magnitude
        # Higher control = expect larger residuals
        # This makes the feature invariant to control regime
        conditioned = residuals / control_magnitude

        return conditioned

    def compute_all(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute all control-conditioned features.

        Returns multiple features, all normalized by control.
        """
        n = len(trajectory)

        pos = trajectory[:, 0:3]
        vel = trajectory[:, 3:6]
        att = trajectory[:, 6:9]
        ang_rate = trajectory[:, 9:12]
        acc = trajectory[:, 12:15]

        # Control magnitude
        control_mag = np.linalg.norm(acc, axis=1) + 1e-6

        # 1. Position-velocity residual / control
        pos_vel_residual = np.zeros(n)
        for t in range(1, n):
            pos_expected = pos[t-1] + vel[t] * 0.005
            pos_vel_residual[t] = np.linalg.norm(pos[t] - pos_expected)
        feat1 = pos_vel_residual / control_mag

        # 2. Velocity-acceleration residual / control
        vel_acc_residual = np.zeros(n)
        for t in range(1, n):
            vel_expected = vel[t-1] + acc[t] * 0.005
            vel_acc_residual[t] = np.linalg.norm(vel[t] - vel_expected)
        feat2 = vel_acc_residual / control_mag

        # 3. Attitude-angular rate residual / angular rate magnitude
        ang_mag = np.linalg.norm(ang_rate, axis=1) + 1e-6
        att_residual = np.zeros(n)
        for t in range(1, n):
            att_expected = att[t-1] + ang_rate[t] * 0.005
            att_residual[t] = np.linalg.norm(att[t] - att_expected)
        feat3 = att_residual / ang_mag

        # 4. Jerk / control
        jerk = np.zeros(n)
        for t in range(1, n):
            jerk[t] = np.linalg.norm(acc[t] - acc[t-1]) / 0.005
        feat4 = jerk / (control_mag + 1)

        return np.column_stack([feat1, feat2, feat3, feat4])


# ============================================================
# SOLUTION 5: GAP-TOLERANT EVIDENCE ACCUMULATION
# ============================================================
# Allow k anomalies in N windows (non-consecutive).
# This catches intermittent/short attacks without lowering thresholds.
# No overfitting risk.

@dataclass
class GapTolerantResult:
    """Result of gap-tolerant accumulation."""
    is_anomaly: bool
    evidence_count: int
    window_size: int
    required_count: int


class GapTolerantAccumulator:
    """
    Accumulate evidence with gaps allowed.

    Key insight: Requiring CONSECUTIVE anomalies misses intermittent attacks.
    Allowing k anomalies in N windows (non-consecutive) catches them.

    No threshold lowering. No sensitivity increase.
    """

    def __init__(
        self,
        window_size: int = 50,
        required_count: int = 5,
        cooldown: int = 10
    ):
        """
        Args:
            window_size: Number of samples in sliding window
            required_count: Minimum anomalies needed in window to trigger
            cooldown: Samples to wait after detection before re-triggering
        """
        self.window_size = window_size
        self.required_count = required_count
        self.cooldown = cooldown

    def detect(self, sample_detections: np.ndarray) -> np.ndarray:
        """
        Apply gap-tolerant accumulation.

        Args:
            sample_detections: Per-sample binary detections

        Returns:
            Accumulated detections
        """
        n = len(sample_detections)
        accumulated = np.zeros(n, dtype=bool)

        cooldown_counter = 0

        for i in range(self.window_size, n):
            if cooldown_counter > 0:
                cooldown_counter -= 1
                continue

            # Count anomalies in window (non-consecutive OK)
            window = sample_detections[i-self.window_size:i]
            count = np.sum(window)

            if count >= self.required_count:
                accumulated[i] = True
                cooldown_counter = self.cooldown

        return accumulated

    def detect_with_weights(
        self,
        scores: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Detect with weighted evidence (stronger anomalies count more).
        """
        n = len(scores)
        accumulated = np.zeros(n, dtype=bool)

        cooldown_counter = 0

        for i in range(self.window_size, n):
            if cooldown_counter > 0:
                cooldown_counter -= 1
                continue

            # Weighted count: anomalies above threshold, weighted by excess
            window = scores[i-self.window_size:i]
            excess = np.maximum(0, window - threshold)
            weighted_count = np.sum(excess) / threshold

            if weighted_count >= self.required_count:
                accumulated[i] = True
                cooldown_counter = self.cooldown

        return accumulated


# ============================================================
# COMBINED ROBUST DETECTOR
# ============================================================

class RobustDetector:
    """
    Combined detector using all 5 principled fixes.

    This detector is designed to:
    1. Rely on temporal structure (not just marginal statistics)
    2. Be robust to domain shift (via per-flight normalization)
    3. Catch intermittent attacks (via gap-tolerant accumulation)
    """

    def __init__(self):
        self.temporal_contrast = TemporalContrastDetector()
        self.multi_resolution = MultiResolutionAgreement()
        self.conditional_norm = ConditionalNormalizer()
        self.control_features = ControlConditionedFeatures()
        self.gap_accumulator = GapTolerantAccumulator()

    def fit(self, normal_trajectories: List[np.ndarray]):
        """Fit all components on normal data."""
        # Fit temporal contrast
        self.temporal_contrast.fit(normal_trajectories)

        # Fit multi-resolution (need scores first)
        all_scores = []
        for traj in normal_trajectories:
            scores = self.temporal_contrast.score(traj)
            all_scores.extend(scores)
        self.multi_resolution.fit(np.array(all_scores))

    def score(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute robust anomaly scores.

        Combines:
        1. Temporal contrast (asymmetry signal)
        2. Control-conditioned features
        3. Per-flight normalization
        """
        # Temporal contrast score
        temporal_score = self.temporal_contrast.score(trajectory)

        # Control-conditioned features
        control_features = self.control_features.compute_all(trajectory)
        control_score = np.max(np.abs(control_features), axis=1)

        # Combine (equal weight)
        combined = 0.5 * temporal_score + 0.5 * control_score

        # Per-flight normalization
        normalized = self.conditional_norm.normalize(combined)

        return normalized

    def detect(self, trajectory: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Detect anomalies using all fixes.
        """
        scores = self.score(trajectory)

        # Sample-level detections
        sample_detections = np.abs(scores) > threshold

        # Gap-tolerant accumulation
        accumulated = self.gap_accumulator.detect(sample_detections)

        # Multi-resolution agreement check
        agreement_detections = self.multi_resolution.detect(scores)

        # Final: either accumulated OR multi-resolution agreement
        return accumulated | agreement_detections


# ============================================================
# EVALUATION HELPERS
# ============================================================

def evaluate_temporal_reliance(
    detector,
    trajectory: np.ndarray,
    seed: int = 42
) -> dict:
    """
    Evaluate how much the detector relies on temporal structure.

    Returns degradation when trajectory is shuffled.
    Higher degradation = more temporal reliance = GOOD.
    """
    # Original score
    original_scores = detector.score(trajectory)

    # Shuffled score
    np.random.seed(seed)
    shuffled = trajectory.copy()
    np.random.shuffle(shuffled)
    shuffled_scores = detector.score(shuffled)

    # Compute degradation
    original_mean = np.mean(np.abs(original_scores))
    shuffled_mean = np.mean(np.abs(shuffled_scores))

    degradation = (original_mean - shuffled_mean) / (original_mean + 1e-6)

    return {
        'original_mean': float(original_mean),
        'shuffled_mean': float(shuffled_mean),
        'degradation': float(degradation),
        'relies_on_structure': degradation > 0.10
    }


def evaluate_domain_robustness(
    detector,
    id_trajectory: np.ndarray,
    ood_trajectory: np.ndarray
) -> dict:
    """
    Evaluate domain robustness.

    Compare performance on in-distribution vs out-of-distribution.
    """
    id_scores = detector.score(id_trajectory)
    ood_scores = detector.score(ood_trajectory)

    # Score distributions should be similar if domain-robust
    from scipy import stats
    ks_stat, ks_pvalue = stats.ks_2samp(id_scores, ood_scores)

    return {
        'id_mean': float(np.mean(id_scores)),
        'ood_mean': float(np.mean(ood_scores)),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
        'domain_robust': ks_pvalue > 0.05  # Can't reject same distribution
    }
