"""
Hybrid Scoring Fusion Module.

Combines multiple detection signals into a unified anomaly score:
1. PINN residual scores (physics violations)
2. EKF integrity scores (NIS-based)
3. ML classifier logits
4. Physics consistency scores

CPU Impact: O(N) with small constant factor; no heavy models.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class DetectionLevel(Enum):
    """Detection confidence levels."""
    NORMAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FusionWeights:
    """Weights for score fusion."""
    w_pinn: float = 0.25        # PINN residual weight
    w_ekf: float = 0.25         # EKF integrity weight
    w_ml: float = 0.25          # ML classifier weight
    w_physics: float = 0.25     # Physics consistency weight

    def normalize(self):
        """Normalize weights to sum to 1."""
        total = self.w_pinn + self.w_ekf + self.w_ml + self.w_physics
        if total > 0:
            self.w_pinn /= total
            self.w_ekf /= total
            self.w_ml /= total
            self.w_physics /= total


@dataclass
class FusionConfig:
    """Configuration for hybrid fusion."""
    # Thresholds for detection levels
    threshold_low: float = 0.3
    threshold_medium: float = 0.5
    threshold_high: float = 0.7
    threshold_critical: float = 0.9

    # Temporal smoothing
    smoothing_window: int = 10
    smoothing_alpha: float = 0.3  # EMA decay

    # Calibration bounds
    min_weight: float = 0.05
    max_weight: float = 0.6

    # Score normalization
    score_clip_min: float = 0.0
    score_clip_max: float = 1.0


@dataclass
class DetectorScores:
    """Container for individual detector scores."""
    pinn_score: float = 0.0          # PINN residual anomaly score [0, 1]
    ekf_nis_score: float = 0.0       # EKF NIS integrity score [0, 1]
    ml_logit: float = 0.0            # ML classifier output [0, 1]
    physics_score: float = 0.0       # Physics consistency score [0, 1]

    # Optional detailed scores
    pinn_residual_raw: float = 0.0   # Raw PINN residual magnitude
    ekf_nis_raw: float = 0.0         # Raw NIS value
    attack_confidence: float = 0.0   # From attack classifier


class ScoreNormalizer:
    """
    Normalize detector scores to [0, 1] range.

    Uses online statistics for adaptive normalization.
    """

    def __init__(self, ema_alpha: float = 0.01):
        self.ema_alpha = ema_alpha
        self.stats = {}  # {detector_name: {'mean': float, 'std': float, 'n': int}}

    def reset(self):
        """Reset normalizer state."""
        self.stats = {}

    def update_stats(self, name: str, value: float):
        """Update running statistics for a detector."""
        if name not in self.stats:
            self.stats[name] = {'mean': value, 'std': 1.0, 'n': 1}
            return

        s = self.stats[name]
        s['n'] += 1
        delta = value - s['mean']
        s['mean'] += delta / s['n']

        # Online variance update (Welford's algorithm)
        if s['n'] > 1:
            delta2 = value - s['mean']
            s['std'] = np.sqrt(
                (s['std']**2 * (s['n'] - 2) + delta * delta2) / (s['n'] - 1)
            )

    def normalize(self, name: str, value: float, update: bool = True) -> float:
        """
        Normalize a score using z-score then sigmoid.

        Args:
            name: Detector name for tracking statistics
            value: Raw score value
            update: Whether to update running statistics

        Returns:
            Normalized score in [0, 1]
        """
        if update:
            self.update_stats(name, value)

        if name not in self.stats or self.stats[name]['std'] < 1e-6:
            return np.clip(value, 0, 1)

        s = self.stats[name]
        z_score = (value - s['mean']) / (s['std'] + 1e-6)

        # Sigmoid normalization
        normalized = 1.0 / (1.0 + np.exp(-z_score))
        return float(normalized)


class HybridFusion:
    """
    Hybrid scoring fusion for multi-modal attack detection.

    Combines:
    - PINN residual scores (physics model violations)
    - EKF integrity scores (measurement consistency)
    - ML classifier logits (learned patterns)
    - Physics consistency scores (kinematic checks)

    Features:
    - Adaptive weight learning via grid search
    - Temporal smoothing for stability
    - Multi-level detection thresholds
    """

    def __init__(
        self,
        weights: Optional[FusionWeights] = None,
        config: Optional[FusionConfig] = None
    ):
        self.weights = weights or FusionWeights()
        self.config = config or FusionConfig()
        self.normalizer = ScoreNormalizer()

        # Temporal smoothing state
        self.score_history: List[float] = []
        self.ema_score = 0.0

        # Calibration state
        self.calibrated = False
        self.best_weights = None
        self.calibration_results = {}

    def reset(self):
        """Reset fusion state."""
        self.normalizer.reset()
        self.score_history = []
        self.ema_score = 0.0

    def fuse(self, scores: DetectorScores, smooth: bool = True) -> Tuple[float, DetectionLevel]:
        """
        Fuse detector scores into unified anomaly score.

        Args:
            scores: Individual detector scores
            smooth: Whether to apply temporal smoothing

        Returns:
            fused_score: Combined anomaly score [0, 1]
            level: Detection level
        """
        # Normalize individual scores
        s_pinn = self.normalizer.normalize('pinn', scores.pinn_score)
        s_ekf = self.normalizer.normalize('ekf', scores.ekf_nis_score)
        s_ml = self.normalizer.normalize('ml', scores.ml_logit)
        s_physics = self.normalizer.normalize('physics', scores.physics_score)

        # Weighted combination
        w = self.weights
        raw_score = (
            w.w_pinn * s_pinn +
            w.w_ekf * s_ekf +
            w.w_ml * s_ml +
            w.w_physics * s_physics
        )

        # Clip to valid range
        raw_score = np.clip(
            raw_score,
            self.config.score_clip_min,
            self.config.score_clip_max
        )

        # Temporal smoothing
        if smooth:
            self.score_history.append(raw_score)
            if len(self.score_history) > self.config.smoothing_window:
                self.score_history.pop(0)

            # EMA smoothing
            self.ema_score = (
                self.config.smoothing_alpha * raw_score +
                (1 - self.config.smoothing_alpha) * self.ema_score
            )

            # Combine EMA with windowed stats
            window_max = max(self.score_history)
            fused_score = 0.7 * self.ema_score + 0.3 * window_max
        else:
            fused_score = raw_score

        # Determine detection level
        level = self._score_to_level(fused_score)

        return float(fused_score), level

    def _score_to_level(self, score: float) -> DetectionLevel:
        """Map score to detection level."""
        if score >= self.config.threshold_critical:
            return DetectionLevel.CRITICAL
        elif score >= self.config.threshold_high:
            return DetectionLevel.HIGH
        elif score >= self.config.threshold_medium:
            return DetectionLevel.MEDIUM
        elif score >= self.config.threshold_low:
            return DetectionLevel.LOW
        else:
            return DetectionLevel.NORMAL

    def calibrate(
        self,
        validation_data: List[Tuple[DetectorScores, int]],
        metric: str = 'f1',
        grid_resolution: int = 5
    ) -> FusionWeights:
        """
        Calibrate fusion weights using grid search.

        Args:
            validation_data: List of (scores, label) tuples
            metric: Optimization metric ('f1', 'precision', 'recall', 'auc')
            grid_resolution: Number of weight values to try per dimension

        Returns:
            Optimal fusion weights
        """
        cfg = self.config
        weight_values = np.linspace(cfg.min_weight, cfg.max_weight, grid_resolution)

        best_metric = -1.0
        best_weights = self.weights

        # Grid search over weight combinations
        for w_pinn in weight_values:
            for w_ekf in weight_values:
                for w_ml in weight_values:
                    w_physics = 1.0 - w_pinn - w_ekf - w_ml

                    # Skip invalid combinations
                    if w_physics < cfg.min_weight or w_physics > cfg.max_weight:
                        continue

                    # Create weight configuration
                    weights = FusionWeights(
                        w_pinn=w_pinn,
                        w_ekf=w_ekf,
                        w_ml=w_ml,
                        w_physics=w_physics
                    )

                    # Evaluate this configuration
                    metric_value = self._evaluate_weights(
                        weights, validation_data, metric
                    )

                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_weights = weights

        self.weights = best_weights
        self.calibrated = True
        self.calibration_results = {
            'best_metric': metric,
            'best_value': best_metric,
            'weights': {
                'pinn': best_weights.w_pinn,
                'ekf': best_weights.w_ekf,
                'ml': best_weights.w_ml,
                'physics': best_weights.w_physics
            }
        }

        return best_weights

    def _evaluate_weights(
        self,
        weights: FusionWeights,
        data: List[Tuple[DetectorScores, int]],
        metric: str
    ) -> float:
        """Evaluate weights on validation data."""
        # Temporarily set weights
        original_weights = self.weights
        self.weights = weights
        self.reset()

        predictions = []
        labels = []

        for scores, label in data:
            fused_score, _ = self.fuse(scores, smooth=False)
            predictions.append(fused_score)
            labels.append(label)

        self.weights = original_weights

        # Find optimal threshold
        predictions = np.array(predictions)
        labels = np.array(labels)

        best_metric_value = 0.0
        for threshold in np.linspace(0.1, 0.9, 17):
            preds_binary = (predictions >= threshold).astype(int)

            tp = np.sum((preds_binary == 1) & (labels == 1))
            fp = np.sum((preds_binary == 1) & (labels == 0))
            fn = np.sum((preds_binary == 0) & (labels == 1))
            tn = np.sum((preds_binary == 0) & (labels == 0))

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            if metric == 'f1':
                value = f1
            elif metric == 'precision':
                value = precision
            elif metric == 'recall':
                value = recall
            elif metric == 'auc':
                # Simplified AUC approximation
                tpr = recall
                fpr = fp / (fp + tn + 1e-10)
                value = (1 + tpr - fpr) / 2
            else:
                value = f1

            best_metric_value = max(best_metric_value, value)

        return best_metric_value

    def get_config(self) -> Dict:
        """Get current configuration as dict."""
        return {
            'weights': {
                'pinn': self.weights.w_pinn,
                'ekf': self.weights.w_ekf,
                'ml': self.weights.w_ml,
                'physics': self.weights.w_physics
            },
            'thresholds': {
                'low': self.config.threshold_low,
                'medium': self.config.threshold_medium,
                'high': self.config.threshold_high,
                'critical': self.config.threshold_critical
            },
            'smoothing': {
                'window': self.config.smoothing_window,
                'alpha': self.config.smoothing_alpha
            },
            'calibrated': self.calibrated
        }


class MultiModalDetector:
    """
    Complete multi-modal attack detector.

    Integrates all detection components:
    - PINN dynamics model
    - EKF with integrity monitoring
    - Attack classifier
    - Physics consistency checks
    - Hybrid fusion
    """

    def __init__(
        self,
        pinn_model=None,
        ekf=None,
        classifier=None,
        physics_checker=None,
        fusion_weights: Optional[FusionWeights] = None
    ):
        self.pinn_model = pinn_model
        self.ekf = ekf
        self.classifier = classifier
        self.physics_checker = physics_checker
        self.fusion = HybridFusion(fusion_weights)

        # Detection statistics
        self.stats = {
            'total_samples': 0,
            'detections': {level.name: 0 for level in DetectionLevel}
        }

    def reset(self):
        """Reset detector state."""
        self.fusion.reset()
        if self.ekf is not None:
            self.ekf.reset()
        if self.classifier is not None:
            self.classifier.reset()
        self.stats = {
            'total_samples': 0,
            'detections': {level.name: 0 for level in DetectionLevel}
        }

    def detect(
        self,
        state: np.ndarray,
        control: Optional[np.ndarray] = None,
        imu: Optional[np.ndarray] = None,
        gps_pos: Optional[np.ndarray] = None,
        baro_z: Optional[float] = None,
        mag: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run full detection pipeline on single sample.

        Args:
            state: Current state vector [x,y,z,vx,vy,vz,phi,theta,psi,p,q,r]
            control: Control input [thrust, tx, ty, tz]
            imu: IMU data [ax, ay, az, gx, gy, gz]
            gps_pos: GPS position [x, y, z]
            baro_z: Barometric altitude
            mag: Magnetometer [mx, my, mz]

        Returns:
            Detection result dict
        """
        scores = DetectorScores()

        # 1. PINN residual score
        if self.pinn_model is not None and control is not None:
            pinn_residual = self._compute_pinn_residual(state, control)
            scores.pinn_residual_raw = pinn_residual
            scores.pinn_score = self._residual_to_score(pinn_residual)

        # 2. EKF integrity score
        if self.ekf is not None and imu is not None:
            acc, gyro = imu[:3], imu[3:6]
            self.ekf.predict(acc, gyro)
            if gps_pos is not None:
                self.ekf.update_position(gps_pos)
            if baro_z is not None:
                self.ekf.update_baro(baro_z)
            if mag is not None:
                self.ekf.update_mag(mag)

            ekf_integrity = self.ekf.get_integrity_score()
            scores.ekf_nis_raw = self.ekf.get_nis()
            scores.ekf_nis_score = ekf_integrity  # EKF score: 0=normal, 1=anomaly

        # 3. ML classifier score
        if self.classifier is not None:
            # Assuming classifier takes state history
            scores.ml_logit = 0.0  # Placeholder
            scores.attack_confidence = 0.0

        # 4. Physics consistency score
        if self.physics_checker is not None:
            scores.physics_score = self._check_physics_consistency(state, imu)

        # Fuse scores
        fused_score, level = self.fusion.fuse(scores)

        # Update stats
        self.stats['total_samples'] += 1
        self.stats['detections'][level.name] += 1

        return {
            'fused_score': fused_score,
            'level': level,
            'level_name': level.name,
            'individual_scores': {
                'pinn': scores.pinn_score,
                'ekf': scores.ekf_nis_score,
                'ml': scores.ml_logit,
                'physics': scores.physics_score
            },
            'raw_values': {
                'pinn_residual': scores.pinn_residual_raw,
                'ekf_nis': scores.ekf_nis_raw
            }
        }

    def _compute_pinn_residual(self, state: np.ndarray, control: np.ndarray) -> float:
        """Compute PINN dynamics residual."""
        # Placeholder - actual implementation would use PINN model
        return 0.0

    def _residual_to_score(self, residual: float, scale: float = 1.0) -> float:
        """Convert residual magnitude to anomaly score."""
        return 1.0 - np.exp(-residual / scale)

    def _check_physics_consistency(self, state: np.ndarray, imu: Optional[np.ndarray]) -> float:
        """Check physics consistency between state and IMU."""
        if imu is None:
            return 0.0

        # Extract relevant quantities
        vx, vy, vz = state[3:6]
        p, q, r = state[9:12]
        ax, ay, az, gx, gy, gz = imu

        # Check angular rate consistency
        rate_diff = np.linalg.norm(np.array([p, q, r]) - np.array([gx, gy, gz]))

        # Normalize to [0, 1]
        return min(rate_diff / 2.0, 1.0)


def run_hybrid_detection(
    df,
    pinn_model=None,
    window_size: int = 100,
    dt: float = 0.005
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Run hybrid detection pipeline on dataframe.

    Args:
        df: DataFrame with state columns
        pinn_model: Optional trained PINN model
        window_size: Detection window size
        dt: Time step

    Returns:
        scores: Fused anomaly scores (N,)
        levels: Detection levels (N,)
        results: Full detection results
    """
    from .emulated_sensors import SensorEmulationPipeline
    from .integrity_ekf import IntegrityEKF
    from .attack_classifier import HybridClassifier

    # Initialize components
    sensor_pipeline = SensorEmulationPipeline(dt=dt)
    ekf = IntegrityEKF(dt=dt)
    classifier = HybridClassifier(window_size, dt)
    fusion = HybridFusion()

    # Emulate missing sensors
    emulated = sensor_pipeline.emulate(df)

    N = len(df)
    scores = np.zeros(N)
    levels = np.zeros(N, dtype=int)
    results = []

    for i in range(window_size, N):
        # Get current data
        state = df[['x', 'y', 'z', 'vx', 'vy', 'vz',
                    'phi', 'theta', 'psi', 'p', 'q', 'r']].values[i]

        # Collect detector scores
        detector_scores = DetectorScores()

        # PINN residual (placeholder)
        if pinn_model is not None:
            detector_scores.pinn_score = 0.0

        # EKF integrity
        acc = df[['ax', 'ay', 'az']].values[i] if 'ax' in df.columns else np.zeros(3)
        gyro = df[['p', 'q', 'r']].values[i]
        ekf.predict(acc, gyro)
        ekf.update_position(state[:3])
        ekf.update_baro(emulated['baro_z'][i])
        detector_scores.ekf_nis_score = ekf.get_integrity_score()  # 0=normal, 1=anomaly

        # Attack classification
        pos = df[['x', 'y', 'z']].values[i-window_size:i]
        vel = df[['vx', 'vy', 'vz']].values[i-window_size:i]
        att = df[['phi', 'theta', 'psi']].values[i-window_size:i]
        rate = df[['p', 'q', 'r']].values[i-window_size:i]

        class_result = classifier.classify(
            pos, vel, att, rate,
            baro_z=emulated['baro_z'][i],
            mag_heading=emulated['mag_heading'][i]
        )
        detector_scores.ml_logit = class_result['confidence']
        detector_scores.attack_confidence = class_result['confidence']

        # Physics consistency
        detector_scores.physics_score = emulated['baro_pos_diff'][i] / 2.0

        # Fuse
        fused_score, level = fusion.fuse(detector_scores)

        scores[i] = fused_score
        levels[i] = level.value
        results.append({
            'fused_score': fused_score,
            'level': level.name,
            'attack_category': class_result['category_name'],
            'attack_type': class_result['type_name'],
            'individual_scores': {
                'pinn': detector_scores.pinn_score,
                'ekf': detector_scores.ekf_nis_score,
                'ml': detector_scores.ml_logit,
                'physics': detector_scores.physics_score
            }
        })

    return scores, levels, results
