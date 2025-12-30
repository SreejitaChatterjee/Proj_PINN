"""
Hybrid Score Fusion for GPS Spoofing Detection.

Combines EKF-NIS and ML (ICI) scores using principled weighted fusion:

    S(t) = w_e * S_ekf(t) + w_m * S_ml(t)

The weights are optimized via grid search to maximize worst-case recall
at a fixed false positive rate.

Key Insight:
- EKF-NIS: Good at detecting sudden jumps and dynamics violations
- ICI: Good at detecting consistency-preserving (stealthy) spoofing
- Hybrid: Covers both attack modalities

Author: GPS-IMU Detector Project
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass
class FusionConfig:
    """Configuration for hybrid fusion."""
    w_ekf: float = 0.3  # Weight for EKF-NIS score
    w_ml: float = 0.7   # Weight for ML (ICI) score
    n_consecutive: int = 5  # Required consecutive anomalies
    target_fpr: float = 0.05  # Target false positive rate

    def __post_init__(self):
        # Normalize weights
        total = self.w_ekf + self.w_ml
        self.w_ekf /= total
        self.w_ml /= total


@dataclass
class FusionResult:
    """Results from hybrid fusion."""
    hybrid_scores: np.ndarray
    ekf_scores: np.ndarray
    ml_scores: np.ndarray
    alarms: np.ndarray  # Boolean alarm array after N-consecutive rule
    threshold: float
    config: FusionConfig


class HybridFusion:
    """
    Hybrid GPS spoofing detector combining EKF-NIS and ML scores.

    Detection pipeline:
    1. Normalize EKF and ML scores to [0, 1] range
    2. Compute weighted fusion: S = w_e * S_ekf + w_m * S_ml
    3. Apply threshold calibrated for target FPR
    4. Apply N-consecutive rule for final alarm

    Attributes:
        config: FusionConfig with weights and parameters
        threshold: Calibrated threshold for hybrid score
        ekf_mean, ekf_std: EKF score normalization stats
        ml_mean, ml_std: ML score normalization stats
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize hybrid fusion.

        Args:
            config: FusionConfig (default: balanced weights)
        """
        self.config = config or FusionConfig()

        # Normalization statistics (set during calibrate)
        self.ekf_mean: float = 0.0
        self.ekf_std: float = 1.0
        self.ml_mean: float = 0.0
        self.ml_std: float = 1.0

        # Threshold (set during calibrate)
        self.threshold: float = 0.0

        # Calibration data for reproducibility
        self._calibration_data: Dict[str, Any] = {}

    def normalize_scores(
        self,
        ekf_scores: np.ndarray,
        ml_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize scores to z-scores using calibration statistics.

        Args:
            ekf_scores: Raw EKF-NIS scores
            ml_scores: Raw ML (ICI) scores

        Returns:
            (norm_ekf, norm_ml): Normalized scores
        """
        norm_ekf = (ekf_scores - self.ekf_mean) / max(self.ekf_std, 1e-6)
        norm_ml = (ml_scores - self.ml_mean) / max(self.ml_std, 1e-6)
        return norm_ekf, norm_ml

    def fuse(
        self,
        ekf_scores: np.ndarray,
        ml_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Compute hybrid fusion score.

        Args:
            ekf_scores: Raw EKF-NIS scores
            ml_scores: Raw ML (ICI) scores

        Returns:
            Hybrid scores (weighted sum of normalized scores)
        """
        norm_ekf, norm_ml = self.normalize_scores(ekf_scores, ml_scores)
        hybrid = self.config.w_ekf * norm_ekf + self.config.w_ml * norm_ml
        return hybrid

    def apply_n_consecutive(self, flags: np.ndarray, n: int) -> np.ndarray:
        """
        Apply N-consecutive rule to anomaly flags.

        An alarm is raised only if N consecutive samples are flagged.

        Args:
            flags: Boolean array of per-sample anomaly flags
            n: Required consecutive anomalies

        Returns:
            Boolean array of alarms after N-consecutive rule
        """
        if n <= 1:
            return flags.copy()

        alarms = np.zeros_like(flags)

        # Sliding window
        consecutive = 0
        for i, flag in enumerate(flags):
            if flag:
                consecutive += 1
                if consecutive >= n:
                    alarms[i] = True
            else:
                consecutive = 0

        return alarms

    def detect(
        self,
        ekf_scores: np.ndarray,
        ml_scores: np.ndarray,
    ) -> FusionResult:
        """
        Run full detection pipeline.

        Args:
            ekf_scores: Raw EKF-NIS scores
            ml_scores: Raw ML (ICI) scores

        Returns:
            FusionResult with hybrid scores and alarms
        """
        # Fuse scores
        hybrid = self.fuse(ekf_scores, ml_scores)

        # Apply threshold
        flags = hybrid > self.threshold

        # Apply N-consecutive rule
        alarms = self.apply_n_consecutive(flags, self.config.n_consecutive)

        norm_ekf, norm_ml = self.normalize_scores(ekf_scores, ml_scores)

        return FusionResult(
            hybrid_scores=hybrid,
            ekf_scores=norm_ekf,
            ml_scores=norm_ml,
            alarms=alarms,
            threshold=self.threshold,
            config=self.config,
        )

    def calibrate(
        self,
        ekf_scores_normal: np.ndarray,
        ml_scores_normal: np.ndarray,
        target_fpr: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calibrate fusion on normal (clean) data.

        Sets normalization statistics and threshold.

        Args:
            ekf_scores_normal: EKF scores on clean data
            ml_scores_normal: ML scores on clean data
            target_fpr: Target FPR (default: config.target_fpr)

        Returns:
            Calibration statistics
        """
        target_fpr = target_fpr or self.config.target_fpr

        # Compute normalization statistics
        self.ekf_mean = float(np.mean(ekf_scores_normal))
        self.ekf_std = float(np.std(ekf_scores_normal))
        self.ml_mean = float(np.mean(ml_scores_normal))
        self.ml_std = float(np.std(ml_scores_normal))

        # Compute hybrid scores on normal data
        hybrid_normal = self.fuse(ekf_scores_normal, ml_scores_normal)

        # Set threshold for target FPR
        self.threshold = float(np.percentile(hybrid_normal, 100 * (1 - target_fpr)))

        self._calibration_data = {
            'ekf_mean': self.ekf_mean,
            'ekf_std': self.ekf_std,
            'ml_mean': self.ml_mean,
            'ml_std': self.ml_std,
            'threshold': self.threshold,
            'target_fpr': target_fpr,
            'w_ekf': self.config.w_ekf,
            'w_ml': self.config.w_ml,
            'n_samples': len(ekf_scores_normal),
        }

        return self._calibration_data

    def save_calibration(self, path: str) -> None:
        """Save calibration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self._calibration_data, f, indent=2)

    def load_calibration(self, path: str) -> None:
        """Load calibration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.ekf_mean = data['ekf_mean']
        self.ekf_std = data['ekf_std']
        self.ml_mean = data['ml_mean']
        self.ml_std = data['ml_std']
        self.threshold = data['threshold']
        self.config.w_ekf = data['w_ekf']
        self.config.w_ml = data['w_ml']
        self._calibration_data = data


def grid_search_weights(
    ekf_scores_normal: np.ndarray,
    ml_scores_normal: np.ndarray,
    ekf_scores_attack: np.ndarray,
    ml_scores_attack: np.ndarray,
    attack_labels: np.ndarray,
    target_fpr: float = 0.05,
    weight_range: np.ndarray = None,
) -> Tuple[FusionConfig, Dict[str, Any]]:
    """
    Grid search for optimal fusion weights.

    Objective: Maximize worst-case recall at fixed FPR.

    Args:
        ekf_scores_normal: EKF scores on clean data
        ml_scores_normal: ML scores on clean data
        ekf_scores_attack: EKF scores on attack data
        ml_scores_attack: ML scores on attack data
        attack_labels: Per-attack labels for worst-case computation
        target_fpr: Target false positive rate
        weight_range: Range of EKF weights to try (default: 0.1 to 0.9)

    Returns:
        (best_config, search_results): Best config and full search results
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    if weight_range is None:
        weight_range = np.arange(0.1, 0.95, 0.1)

    results = []
    best_worst_recall = -1
    best_config = None

    for w_ekf in weight_range:
        w_ml = 1 - w_ekf

        # Create fusion with these weights
        config = FusionConfig(w_ekf=w_ekf, w_ml=w_ml, target_fpr=target_fpr)
        fusion = HybridFusion(config)

        # Calibrate on normal data
        fusion.calibrate(ekf_scores_normal, ml_scores_normal, target_fpr)

        # Compute hybrid scores on attack data
        hybrid_attack = fusion.fuse(ekf_scores_attack, ml_scores_attack)
        hybrid_normal = fusion.fuse(ekf_scores_normal, ml_scores_normal)

        # Compute AUROC
        y_true = np.concatenate([np.zeros(len(hybrid_normal)), np.ones(len(hybrid_attack))])
        y_scores = np.concatenate([hybrid_normal, hybrid_attack])

        try:
            auroc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auroc = 0.5

        # Compute recall at target FPR
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        idx = np.searchsorted(fpr, target_fpr)
        recall_at_fpr = tpr[min(idx, len(tpr) - 1)]

        # Compute per-attack recall (for worst-case)
        unique_attacks = np.unique(attack_labels)
        attack_recalls = []

        for attack in unique_attacks:
            mask = attack_labels == attack
            if not np.any(mask):
                continue

            attack_subset = hybrid_attack[mask]
            threshold = np.percentile(hybrid_normal, 100 * (1 - target_fpr))
            recall = np.mean(attack_subset > threshold)
            attack_recalls.append(recall)

        worst_recall = min(attack_recalls) if attack_recalls else 0.0

        results.append({
            'w_ekf': w_ekf,
            'w_ml': w_ml,
            'auroc': auroc,
            'recall_at_fpr': recall_at_fpr,
            'worst_recall': worst_recall,
            'mean_recall': np.mean(attack_recalls) if attack_recalls else 0.0,
        })

        if worst_recall > best_worst_recall:
            best_worst_recall = worst_recall
            best_config = config

    return best_config, {'grid_search': results, 'best_worst_recall': best_worst_recall}


def evaluate_hybrid(
    fusion: HybridFusion,
    ekf_normal: np.ndarray,
    ml_normal: np.ndarray,
    ekf_attack: np.ndarray,
    ml_attack: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate hybrid detector.

    Args:
        fusion: Calibrated HybridFusion
        ekf_normal: EKF scores on normal data
        ml_normal: ML scores on normal data
        ekf_attack: EKF scores on attack data
        ml_attack: ML scores on attack data

    Returns:
        Evaluation metrics
    """
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

    # Compute hybrid scores
    hybrid_normal = fusion.fuse(ekf_normal, ml_normal)
    hybrid_attack = fusion.fuse(ekf_attack, ml_attack)

    # Combined arrays
    y_true = np.concatenate([np.zeros(len(hybrid_normal)), np.ones(len(hybrid_attack))])
    y_scores = np.concatenate([hybrid_normal, hybrid_attack])

    # Metrics
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5

    fpr, tpr, _ = roc_curve(y_true, y_scores)

    def recall_at_fpr(target_fpr):
        idx = np.searchsorted(fpr, target_fpr)
        return tpr[min(idx, len(tpr) - 1)]

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_f1 = np.max(f1_scores)

    return {
        'auroc': float(auroc),
        'recall_1pct_fpr': float(recall_at_fpr(0.01)),
        'recall_5pct_fpr': float(recall_at_fpr(0.05)),
        'best_f1': float(best_f1),
        'n_normal': len(hybrid_normal),
        'n_attack': len(hybrid_attack),
    }
