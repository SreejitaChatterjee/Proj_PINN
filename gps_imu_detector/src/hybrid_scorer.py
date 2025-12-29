"""
Hybrid Scoring Function

Combines multiple anomaly signals:
1. Physics residuals (analytic + PINN)
2. EKF NIS (integrity proxy)
3. ML detector score
4. Temporal consistency score

Calibrates weights on clean validation data.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import json


@dataclass
class HybridScore:
    """Container for hybrid anomaly score."""
    total_score: float
    physics_score: float
    ekf_score: float
    ml_score: float
    temporal_score: float
    is_anomaly: bool


class HybridScorer:
    """
    Hybrid anomaly scorer combining multiple detection signals.

    Score = w1 * physics + w2 * ekf_nis + w3 * ml + w4 * temporal

    Weights calibrated on clean validation data using grid search.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        normalize: bool = True
    ):
        """
        Initialize hybrid scorer.

        Args:
            weights: Dict with 'physics', 'ekf', 'ml', 'temporal' weights
            threshold: Anomaly detection threshold
            normalize: Normalize component scores before combining
        """
        self.weights = weights or {
            'physics': 0.25,
            'ekf': 0.25,
            'ml': 0.25,
            'temporal': 0.25
        }

        self.threshold = threshold
        self.normalize = normalize

        # Scalers for each component (fit on normal data)
        self.scalers = {
            'physics': StandardScaler(),
            'ekf': StandardScaler(),
            'ml': StandardScaler(),
            'temporal': StandardScaler()
        }

        self._fitted = False

    def fit(
        self,
        physics_scores: np.ndarray,
        ekf_scores: np.ndarray,
        ml_scores: np.ndarray,
        temporal_scores: np.ndarray
    ):
        """
        Fit scalers on normal (clean) data.

        IMPORTANT: Call only with normal data, never with attacks.

        Args:
            physics_scores: [N] physics residual scores
            ekf_scores: [N] EKF NIS scores
            ml_scores: [N] ML detector scores
            temporal_scores: [N] temporal consistency scores
        """
        self.scalers['physics'].fit(physics_scores.reshape(-1, 1))
        self.scalers['ekf'].fit(ekf_scores.reshape(-1, 1))
        self.scalers['ml'].fit(ml_scores.reshape(-1, 1))
        self.scalers['temporal'].fit(temporal_scores.reshape(-1, 1))

        self._fitted = True

    def score(
        self,
        physics_score: float,
        ekf_score: float,
        ml_score: float,
        temporal_score: float
    ) -> HybridScore:
        """
        Compute hybrid anomaly score.

        Args:
            physics_score: Physics residual score
            ekf_score: EKF NIS score
            ml_score: ML detector score (probability)
            temporal_score: Temporal consistency score

        Returns:
            HybridScore with total and component scores
        """
        # Normalize if fitted
        if self.normalize and self._fitted:
            physics_norm = self._normalize('physics', physics_score)
            ekf_norm = self._normalize('ekf', ekf_score)
            ml_norm = ml_score  # Already 0-1 from sigmoid
            temporal_norm = self._normalize('temporal', temporal_score)
        else:
            physics_norm = physics_score
            ekf_norm = ekf_score
            ml_norm = ml_score
            temporal_norm = temporal_score

        # Clip to reasonable range
        physics_norm = np.clip(physics_norm, 0, 1)
        ekf_norm = np.clip(ekf_norm, 0, 1)
        temporal_norm = np.clip(temporal_norm, 0, 1)

        # Weighted combination
        total = (
            self.weights['physics'] * physics_norm +
            self.weights['ekf'] * ekf_norm +
            self.weights['ml'] * ml_norm +
            self.weights['temporal'] * temporal_norm
        )

        is_anomaly = total > self.threshold

        return HybridScore(
            total_score=total,
            physics_score=physics_norm,
            ekf_score=ekf_norm,
            ml_score=ml_norm,
            temporal_score=temporal_norm,
            is_anomaly=is_anomaly
        )

    def _normalize(self, component: str, value: float) -> float:
        """Normalize score using fitted scaler."""
        scaler = self.scalers[component]
        scaled = scaler.transform([[value]])[0, 0]
        # Convert to 0-1 range using sigmoid
        return 1 / (1 + np.exp(-scaled))

    def score_batch(
        self,
        physics_scores: np.ndarray,
        ekf_scores: np.ndarray,
        ml_scores: np.ndarray,
        temporal_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score batch of data.

        Args:
            physics_scores: [N] physics scores
            ekf_scores: [N] EKF scores
            ml_scores: [N] ML scores
            temporal_scores: [N] temporal scores

        Returns:
            total_scores: [N] hybrid scores
            anomaly_flags: [N] boolean flags
        """
        n = len(physics_scores)
        total_scores = np.zeros(n)
        anomaly_flags = np.zeros(n, dtype=bool)

        for i in range(n):
            result = self.score(
                physics_scores[i],
                ekf_scores[i],
                ml_scores[i],
                temporal_scores[i]
            )
            total_scores[i] = result.total_score
            anomaly_flags[i] = result.is_anomaly

        return total_scores, anomaly_flags

    def calibrate_weights(
        self,
        physics_scores: np.ndarray,
        ekf_scores: np.ndarray,
        ml_scores: np.ndarray,
        temporal_scores: np.ndarray,
        labels: np.ndarray,
        target_fpr: float = 0.05,
        grid_resolution: int = 5
    ) -> Dict[str, float]:
        """
        Calibrate weights using grid search on validation data.

        Finds weights that maximize recall at target FPR.

        Args:
            physics_scores, ekf_scores, ml_scores, temporal_scores: Component scores
            labels: [N] ground truth (1=attack, 0=normal)
            target_fpr: Target false positive rate
            grid_resolution: Number of grid points per weight

        Returns:
            Optimal weights
        """
        # First fit scalers on normal data
        normal_mask = labels == 0
        if np.sum(normal_mask) > 0:
            self.fit(
                physics_scores[normal_mask],
                ekf_scores[normal_mask],
                ml_scores[normal_mask],
                temporal_scores[normal_mask]
            )

        best_recall = 0
        best_weights = self.weights.copy()

        # Grid search
        weight_values = np.linspace(0.1, 0.5, grid_resolution)

        for w_physics in weight_values:
            for w_ekf in weight_values:
                for w_ml in weight_values:
                    w_temporal = 1.0 - w_physics - w_ekf - w_ml
                    if w_temporal < 0.05:
                        continue

                    self.weights = {
                        'physics': w_physics,
                        'ekf': w_ekf,
                        'ml': w_ml,
                        'temporal': w_temporal
                    }

                    # Score all data
                    total_scores, _ = self.score_batch(
                        physics_scores, ekf_scores, ml_scores, temporal_scores
                    )

                    # Find threshold for target FPR
                    normal_scores = total_scores[normal_mask]
                    threshold = np.percentile(normal_scores, (1 - target_fpr) * 100)

                    # Compute recall at this threshold
                    attack_mask = labels == 1
                    if np.sum(attack_mask) > 0:
                        recall = np.mean(total_scores[attack_mask] > threshold)

                        if recall > best_recall:
                            best_recall = recall
                            best_weights = self.weights.copy()
                            best_weights['threshold'] = threshold

        self.weights = {k: v for k, v in best_weights.items() if k != 'threshold'}
        self.threshold = best_weights.get('threshold', 0.5)

        print(f"Calibrated weights: {self.weights}")
        print(f"Threshold: {self.threshold:.3f}")
        print(f"Best recall@{target_fpr*100:.0f}%FPR: {best_recall:.1%}")

        return self.weights

    def save(self, path: str):
        """Save scorer configuration."""
        config = {
            'weights': self.weights,
            'threshold': self.threshold,
            'normalize': self.normalize,
            'scaler_means': {k: float(v.mean_[0]) for k, v in self.scalers.items()},
            'scaler_stds': {k: float(v.scale_[0]) for k, v in self.scalers.items()},
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, path: str):
        """Load scorer configuration."""
        with open(path) as f:
            config = json.load(f)

        self.weights = config['weights']
        self.threshold = config['threshold']
        self.normalize = config['normalize']

        for k in self.scalers:
            self.scalers[k].mean_ = np.array([config['scaler_means'][k]])
            self.scalers[k].scale_ = np.array([config['scaler_stds'][k]])
            self.scalers[k].var_ = self.scalers[k].scale_ ** 2

        self._fitted = True


class TemporalConsistencyScorer:
    """
    Compute temporal consistency score.

    Detects sudden jumps or inconsistent patterns in features.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.feature_buffer = []

    def reset(self):
        """Reset buffer."""
        self.feature_buffer = []

    def update(self, features: np.ndarray) -> float:
        """
        Update with new features and compute consistency score.

        Args:
            features: [D] feature vector

        Returns:
            Consistency score (higher = more anomalous)
        """
        self.feature_buffer.append(features)
        if len(self.feature_buffer) > self.window_size:
            self.feature_buffer.pop(0)

        if len(self.feature_buffer) < 2:
            return 0.0

        # Compute temporal derivative
        buffer_array = np.array(self.feature_buffer)
        diffs = np.diff(buffer_array, axis=0)

        # Anomaly score: large changes relative to recent history
        recent_std = np.std(diffs[:-1], axis=0) + 1e-8
        current_diff = np.abs(diffs[-1])
        z_scores = current_diff / recent_std

        # Max z-score as anomaly indicator
        return float(np.max(z_scores))

    def score_sequence(self, features: np.ndarray) -> np.ndarray:
        """Score entire sequence."""
        self.reset()
        scores = []

        for feat in features:
            scores.append(self.update(feat))

        return np.array(scores)


if __name__ == "__main__":
    # Test hybrid scorer
    n = 1000

    # Simulate component scores
    np.random.seed(42)

    # Normal data
    physics_normal = np.abs(np.random.randn(n // 2)) * 0.1
    ekf_normal = np.abs(np.random.randn(n // 2)) * 0.5
    ml_normal = np.random.rand(n // 2) * 0.3
    temporal_normal = np.abs(np.random.randn(n // 2)) * 0.2

    # Attack data (higher scores)
    physics_attack = np.abs(np.random.randn(n // 2)) * 0.5 + 0.3
    ekf_attack = np.abs(np.random.randn(n // 2)) * 1.0 + 1.0
    ml_attack = np.random.rand(n // 2) * 0.5 + 0.4
    temporal_attack = np.abs(np.random.randn(n // 2)) * 0.5 + 0.3

    # Combine
    physics = np.concatenate([physics_normal, physics_attack])
    ekf = np.concatenate([ekf_normal, ekf_attack])
    ml = np.concatenate([ml_normal, ml_attack])
    temporal = np.concatenate([temporal_normal, temporal_attack])
    labels = np.array([0] * (n // 2) + [1] * (n // 2))

    # Calibrate
    scorer = HybridScorer()
    scorer.calibrate_weights(physics, ekf, ml, temporal, labels, target_fpr=0.05)

    # Test scoring
    total_scores, anomaly_flags = scorer.score_batch(physics, ekf, ml, temporal)

    # Evaluate
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(labels, anomaly_flags))
