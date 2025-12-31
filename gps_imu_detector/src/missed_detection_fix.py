#!/usr/bin/env python3
"""
Missed Detection Fix: Reduce 6.63% -> <1% WITHOUT breaking FPR

Core Insight: We need PARALLEL detection paths, not lower thresholds.

Strategy:
1. DUAL-PATH DETECTION
   - Fast Path: Short window (K=5), very high threshold -> catches bursts
   - Slow Path: Long window (K=20), normal threshold -> catches sustained
   - Trigger if EITHER fires (OR logic)

2. MAGNITUDE-ADAPTIVE CONFIRMATION
   - Very high scores: require less confirmation (attack is obvious)
   - Marginal scores: require more confirmation (could be noise)

3. ENSEMBLE VOTING
   - Multiple feature views vote
   - Reduces variance without increasing base sensitivity

Key Principle: Add COVERAGE, not SENSITIVITY.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class DualPathResult:
    """Result from dual-path detection."""
    fast_path_triggered: bool
    slow_path_triggered: bool
    final_detection: bool
    confidence: float
    path_used: str  # 'fast', 'slow', 'both', 'none'


class DualPathDetector:
    """
    Dual-path detection for catching both short and sustained attacks.

    Fast Path: High threshold, short window -> catches obvious short bursts
    Slow Path: Normal threshold, long window -> catches subtle sustained attacks

    FPR is preserved because:
    - Fast path has HIGHER threshold (not lower)
    - Paths use OR logic, but both have strict thresholds
    """

    def __init__(
        self,
        # Fast path (for short-duration attacks)
        fast_window: int = 5,
        fast_threshold_percentile: float = 99.0,  # Very high threshold
        fast_required: int = 3,  # 3/5 must exceed

        # Slow path (for sustained attacks)
        slow_window: int = 20,
        slow_threshold_percentile: float = 95.0,
        slow_required: int = 10,  # 10/20 must exceed

        # Cooldown
        cooldown: int = 20
    ):
        self.fast_window = fast_window
        self.fast_threshold_percentile = fast_threshold_percentile
        self.fast_required = fast_required

        self.slow_window = slow_window
        self.slow_threshold_percentile = slow_threshold_percentile
        self.slow_required = slow_required

        self.cooldown = cooldown

        self.fast_threshold: Optional[float] = None
        self.slow_threshold: Optional[float] = None

    def calibrate(self, normal_scores: np.ndarray):
        """Calibrate thresholds from normal data."""
        self.fast_threshold = np.percentile(normal_scores, self.fast_threshold_percentile)
        self.slow_threshold = np.percentile(normal_scores, self.slow_threshold_percentile)

    def detect(self, scores: np.ndarray) -> Tuple[np.ndarray, List[DualPathResult]]:
        """
        Run dual-path detection.

        Returns:
            Tuple of (detection_array, detailed_results)
        """
        if self.fast_threshold is None:
            self.calibrate(scores[:100])

        n = len(scores)
        detections = np.zeros(n, dtype=bool)
        results = []

        cooldown_counter = 0

        for i in range(self.slow_window, n):
            if cooldown_counter > 0:
                cooldown_counter -= 1
                results.append(DualPathResult(
                    fast_path_triggered=False,
                    slow_path_triggered=False,
                    final_detection=False,
                    confidence=0.0,
                    path_used='cooldown'
                ))
                continue

            # Fast path check
            fast_window = scores[max(0, i - self.fast_window):i]
            fast_exceedances = np.sum(fast_window > self.fast_threshold)
            fast_triggered = fast_exceedances >= self.fast_required

            # Slow path check
            slow_window = scores[i - self.slow_window:i]
            slow_exceedances = np.sum(slow_window > self.slow_threshold)
            slow_triggered = slow_exceedances >= self.slow_required

            # OR logic: either path triggers detection
            final_detection = fast_triggered or slow_triggered

            if final_detection:
                detections[i] = True
                cooldown_counter = self.cooldown

            # Determine which path
            if fast_triggered and slow_triggered:
                path_used = 'both'
            elif fast_triggered:
                path_used = 'fast'
            elif slow_triggered:
                path_used = 'slow'
            else:
                path_used = 'none'

            # Confidence based on exceedance ratio
            confidence = max(
                fast_exceedances / self.fast_required if self.fast_required > 0 else 0,
                slow_exceedances / self.slow_required if self.slow_required > 0 else 0
            )

            results.append(DualPathResult(
                fast_path_triggered=fast_triggered,
                slow_path_triggered=slow_triggered,
                final_detection=final_detection,
                confidence=float(confidence),
                path_used=path_used
            ))

        return detections, results


class MagnitudeAdaptiveConfirmation:
    """
    Adaptive confirmation based on score magnitude.

    - Very high scores (>99th percentile): require only 2/5 confirmation
    - High scores (>97th percentile): require 3/5 confirmation
    - Normal high (>95th percentile): require 5/10 confirmation
    - Marginal (>90th percentile): require 10/20 confirmation

    This catches obvious attacks faster without lowering thresholds.
    """

    def __init__(self):
        self.thresholds = {}
        self.requirements = {
            'extreme': {'percentile': 99.5, 'window': 3, 'required': 2},
            'very_high': {'percentile': 99.0, 'window': 5, 'required': 3},
            'high': {'percentile': 97.0, 'window': 10, 'required': 5},
            'moderate': {'percentile': 95.0, 'window': 15, 'required': 8},
            'marginal': {'percentile': 90.0, 'window': 20, 'required': 10},
        }

    def calibrate(self, normal_scores: np.ndarray):
        """Calibrate thresholds from normal data."""
        for level, config in self.requirements.items():
            self.thresholds[level] = np.percentile(normal_scores, config['percentile'])

    def detect(self, scores: np.ndarray) -> np.ndarray:
        """
        Detect with magnitude-adaptive confirmation.
        """
        if not self.thresholds:
            self.calibrate(scores[:100])

        n = len(scores)
        detections = np.zeros(n, dtype=bool)
        cooldown = 0

        max_window = max(c['window'] for c in self.requirements.values())

        for i in range(max_window, n):
            if cooldown > 0:
                cooldown -= 1
                continue

            # Check each level from most to least strict
            for level in ['extreme', 'very_high', 'high', 'moderate', 'marginal']:
                config = self.requirements[level]
                threshold = self.thresholds[level]
                window = config['window']
                required = config['required']

                # Check window
                window_scores = scores[i - window:i]
                exceedances = np.sum(window_scores > threshold)

                if exceedances >= required:
                    detections[i] = True
                    cooldown = 20
                    break  # Stop at first triggered level

        return detections


class EnsembleVoter:
    """
    Ensemble voting across multiple feature views.

    Each feature view gets a vote. Detection if majority agrees.
    Reduces variance without increasing base sensitivity.
    """

    def __init__(self, n_features: int = 3, vote_threshold: float = 0.5):
        self.n_features = n_features
        self.vote_threshold = vote_threshold
        self.feature_thresholds = []

    def calibrate(self, normal_features: np.ndarray, percentile: float = 95.0):
        """Calibrate per-feature thresholds."""
        self.feature_thresholds = []
        for j in range(normal_features.shape[1]):
            thresh = np.percentile(normal_features[:, j], percentile)
            self.feature_thresholds.append(thresh)

    def vote(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ensemble votes.

        Returns:
            Tuple of (detection_array, vote_counts)
        """
        n = len(features)
        n_features = features.shape[1]

        if not self.feature_thresholds:
            self.calibrate(features[:100])

        votes = np.zeros((n, n_features), dtype=bool)

        for j in range(n_features):
            votes[:, j] = features[:, j] > self.feature_thresholds[j]

        vote_counts = np.sum(votes, axis=1)
        detections = vote_counts >= (n_features * self.vote_threshold)

        return detections, vote_counts


class MissedDetectionFixer:
    """
    Combined approach to fix missed detection without breaking FPR.

    Combines:
    1. Dual-path detection (fast + slow)
    2. Magnitude-adaptive confirmation
    3. Ensemble voting
    """

    def __init__(self):
        self.dual_path = DualPathDetector()
        self.adaptive = MagnitudeAdaptiveConfirmation()
        self.ensemble = EnsembleVoter()

    def calibrate(self, normal_scores: np.ndarray, normal_features: Optional[np.ndarray] = None):
        """Calibrate all components."""
        self.dual_path.calibrate(normal_scores)
        self.adaptive.calibrate(normal_scores)
        if normal_features is not None:
            self.ensemble.calibrate(normal_features)

    def detect(
        self,
        scores: np.ndarray,
        features: Optional[np.ndarray] = None,
        mode: str = 'dual_path'  # 'dual_path', 'adaptive', 'ensemble', 'combined'
    ) -> np.ndarray:
        """
        Detect with specified mode.

        Modes:
        - 'dual_path': Fast + slow path OR logic
        - 'adaptive': Magnitude-adaptive confirmation
        - 'ensemble': Feature voting
        - 'combined': All methods OR'd together
        """
        if mode == 'dual_path':
            detections, _ = self.dual_path.detect(scores)
            return detections

        elif mode == 'adaptive':
            return self.adaptive.detect(scores)

        elif mode == 'ensemble':
            if features is None:
                raise ValueError("Ensemble mode requires features")
            detections, _ = self.ensemble.vote(features)
            return detections

        elif mode == 'combined':
            # OR all methods together
            dual_det, _ = self.dual_path.detect(scores)
            adaptive_det = self.adaptive.detect(scores)

            combined = dual_det | adaptive_det

            if features is not None:
                ensemble_det, _ = self.ensemble.vote(features)
                combined = combined | ensemble_det

            return combined

        else:
            raise ValueError(f"Unknown mode: {mode}")


def evaluate_missed_detection_fix(
    normal_scores: np.ndarray,
    attack_scores: np.ndarray,
    attack_labels: np.ndarray,
    mode: str = 'dual_path'
) -> dict:
    """
    Evaluate the missed detection fix.

    Returns FPR and missed detection rate.
    """
    fixer = MissedDetectionFixer()
    fixer.calibrate(normal_scores)

    # Detect on normal (for FPR)
    normal_detections = fixer.detect(normal_scores, mode=mode)
    fpr = np.mean(normal_detections)

    # Detect on attack (for missed detection)
    attack_detections = fixer.detect(attack_scores, mode=mode)

    # Missed detection = attacks that weren't detected
    true_attacks = attack_labels == 1
    missed = np.sum(true_attacks & ~attack_detections)
    total_attacks = np.sum(true_attacks)
    missed_rate = missed / total_attacks if total_attacks > 0 else 0

    # Recall
    recall = 1 - missed_rate

    return {
        'fpr': float(fpr),
        'missed_rate': float(missed_rate),
        'recall': float(recall),
        'fpr_target_met': fpr < 0.01,
        'missed_target_met': missed_rate < 0.01,
        'mode': mode
    }
