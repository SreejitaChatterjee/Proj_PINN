#!/usr/bin/env python3
"""
Physics-Invariant Detector: Solving the Robustness Problems

This module implements three key fixes:

1. PHYSICS-INVARIANT FEATURES (solves temporal structure problem)
   - Position-velocity consistency residual
   - Velocity-acceleration consistency residual
   - Jerk (rate of acceleration change)
   - These features are physics-based and domain-invariant

2. PER-FLIGHT CALIBRATION (solves domain shift problem)
   - Use first N samples (assumed normal) to calibrate thresholds
   - No retraining needed
   - Adapts to each flight's distribution

3. ENSEMBLE DETECTOR (solves OOD robustness problem)
   - Train multiple models with different seeds
   - Average predictions
   - Reduces variance and improves robustness

Key Insight: Physics doesn't change across distributions.
Position-velocity consistency is the SAME in any domain.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class PhysicsFeatures:
    """Physics-based features that are domain-invariant."""
    pos_vel_residual: np.ndarray      # Position-velocity consistency
    vel_acc_residual: np.ndarray      # Velocity-acceleration consistency
    jerk: np.ndarray                   # Rate of acceleration change
    angular_consistency: np.ndarray    # Attitude-angular rate consistency
    energy_rate: np.ndarray            # Rate of kinetic energy change


def compute_physics_features(trajectory: np.ndarray, dt: float = 0.005) -> PhysicsFeatures:
    """
    Compute physics-invariant features from trajectory data.

    These features are based on physical laws and should be
    consistent across different data distributions.

    Args:
        trajectory: (N, 15) array with [pos(3), vel(3), att(3), ang_rate(3), acc(3)]
        dt: Time step in seconds

    Returns:
        PhysicsFeatures with all physics-based residuals
    """
    n = len(trajectory)

    pos = trajectory[:, 0:3]
    vel = trajectory[:, 3:6]
    att = trajectory[:, 6:9]
    ang_rate = trajectory[:, 9:12]
    acc = trajectory[:, 12:15]

    # 1. Position-Velocity Consistency (fundamental physics)
    # d(pos)/dt should equal velocity
    pos_vel_residual = np.zeros(n)
    for t in range(1, n):
        pos_expected = pos[t-1] + vel[t] * dt
        pos_vel_residual[t] = np.linalg.norm(pos[t] - pos_expected)

    # 2. Velocity-Acceleration Consistency
    # d(vel)/dt should equal acceleration
    vel_acc_residual = np.zeros(n)
    for t in range(1, n):
        vel_expected = vel[t-1] + acc[t] * dt
        vel_acc_residual[t] = np.linalg.norm(vel[t] - vel_expected)

    # 3. Jerk (smoothness of acceleration)
    # Attacks often cause discontinuous acceleration
    jerk = np.zeros(n)
    for t in range(1, n):
        jerk[t] = np.linalg.norm(acc[t] - acc[t-1]) / dt

    # 4. Attitude-Angular Rate Consistency
    # d(att)/dt should equal angular rate
    angular_consistency = np.zeros(n)
    for t in range(1, n):
        att_expected = att[t-1] + ang_rate[t] * dt
        angular_consistency[t] = np.linalg.norm(att[t] - att_expected)

    # 5. Energy Rate (conservation principle)
    # Kinetic energy: 0.5 * m * v^2
    # Rate of change should match work done
    energy = 0.5 * np.sum(vel**2, axis=1)
    energy_rate = np.zeros(n)
    energy_rate[1:] = np.abs(np.diff(energy)) / dt

    return PhysicsFeatures(
        pos_vel_residual=pos_vel_residual,
        vel_acc_residual=vel_acc_residual,
        jerk=jerk,
        angular_consistency=angular_consistency,
        energy_rate=energy_rate
    )


def compute_temporal_features(trajectory: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Compute temporal derivative features that capture dynamics.

    These features explicitly encode temporal structure.
    """
    n = len(trajectory)

    # First and second derivatives
    d1 = np.zeros_like(trajectory)
    d2 = np.zeros_like(trajectory)

    d1[1:] = np.diff(trajectory, axis=0)
    d2[2:] = np.diff(d1, axis=0)[1:]

    # Rolling statistics
    rolling_mean = np.zeros_like(trajectory)
    rolling_std = np.zeros_like(trajectory)

    for i in range(window, n):
        rolling_mean[i] = np.mean(trajectory[i-window:i], axis=0)
        rolling_std[i] = np.std(trajectory[i-window:i], axis=0)

    # Deviation from rolling mean (z-score)
    z_score = np.zeros_like(trajectory)
    mask = rolling_std > 1e-6
    z_score[mask] = (trajectory[mask] - rolling_mean[mask]) / rolling_std[mask]

    # Autocorrelation at lag 1
    autocorr = np.zeros(n)
    for i in range(window, n):
        segment = trajectory[i-window:i, 0]  # Use first dimension
        if np.std(segment) > 1e-6:
            autocorr[i] = np.corrcoef(segment[:-1], segment[1:])[0, 1]

    return np.column_stack([
        d1, d2, rolling_mean, rolling_std, z_score,
        autocorr.reshape(-1, 1)
    ])


@dataclass
class CalibrationResult:
    """Result of per-flight calibration."""
    mean: np.ndarray
    std: np.ndarray
    threshold_90: float
    threshold_95: float
    threshold_99: float


class PerFlightCalibrator:
    """
    Per-flight calibration for domain shift robustness.

    Uses the first N samples (assumed normal) to calibrate
    detection thresholds for each flight.
    """

    def __init__(self, calibration_samples: int = 100):
        self.calibration_samples = calibration_samples
        self.calibration: Optional[CalibrationResult] = None

    def calibrate(self, scores: np.ndarray) -> CalibrationResult:
        """
        Calibrate using first N samples.

        Args:
            scores: Anomaly scores for the flight

        Returns:
            CalibrationResult with thresholds
        """
        cal_scores = scores[:self.calibration_samples]

        self.calibration = CalibrationResult(
            mean=np.mean(cal_scores),
            std=np.std(cal_scores) + 1e-6,
            threshold_90=np.percentile(cal_scores, 90),
            threshold_95=np.percentile(cal_scores, 95),
            threshold_99=np.percentile(cal_scores, 99)
        )

        return self.calibration

    def normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores using calibration.

        Returns z-scores relative to the calibration distribution.
        """
        if self.calibration is None:
            self.calibrate(scores)

        return (scores - self.calibration.mean) / self.calibration.std

    def detect(self, scores: np.ndarray, fpr_target: float = 0.01) -> np.ndarray:
        """
        Detect anomalies using calibrated thresholds.

        Args:
            scores: Anomaly scores
            fpr_target: Target false positive rate

        Returns:
            Binary detection array
        """
        if self.calibration is None:
            self.calibrate(scores)

        if fpr_target <= 0.01:
            threshold = self.calibration.threshold_99
        elif fpr_target <= 0.05:
            threshold = self.calibration.threshold_95
        else:
            threshold = self.calibration.threshold_90

        return scores > threshold


class PhysicsInvariantDetector:
    """
    Detector using physics-invariant features.

    This detector is inherently robust to domain shift because
    it uses features based on physical laws that don't change
    across distributions.
    """

    def __init__(self, dt: float = 0.005):
        self.dt = dt
        self.calibrator = PerFlightCalibrator()
        self.baseline_stats: Optional[Dict] = None

    def fit(self, normal_trajectories: List[np.ndarray]):
        """
        Learn baseline statistics from normal trajectories.
        """
        all_features = []

        for traj in normal_trajectories:
            features = compute_physics_features(traj, self.dt)
            combined = np.column_stack([
                features.pos_vel_residual,
                features.vel_acc_residual,
                features.jerk,
                features.angular_consistency,
                features.energy_rate
            ])
            all_features.append(combined)

        all_features = np.vstack(all_features)

        self.baseline_stats = {
            'mean': np.mean(all_features, axis=0),
            'std': np.std(all_features, axis=0) + 1e-6,
            'percentile_95': np.percentile(all_features, 95, axis=0),
            'percentile_99': np.percentile(all_features, 99, axis=0)
        }

    def score(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Score a trajectory using physics-based features.

        Returns per-sample anomaly scores.
        """
        features = compute_physics_features(trajectory, self.dt)
        combined = np.column_stack([
            features.pos_vel_residual,
            features.vel_acc_residual,
            features.jerk,
            features.angular_consistency,
            features.energy_rate
        ])

        if self.baseline_stats is None:
            # No baseline, return raw scores
            return np.mean(combined, axis=1)

        # Z-score relative to baseline
        z_scores = (combined - self.baseline_stats['mean']) / self.baseline_stats['std']

        # Max z-score across features (any physics violation triggers)
        return np.max(np.abs(z_scores), axis=1)

    def detect(self, trajectory: np.ndarray, use_calibration: bool = True) -> np.ndarray:
        """
        Detect anomalies in trajectory.

        Args:
            trajectory: Input trajectory
            use_calibration: Whether to use per-flight calibration

        Returns:
            Binary detection array
        """
        scores = self.score(trajectory)

        if use_calibration:
            return self.calibrator.detect(scores)
        else:
            # Use baseline threshold
            return scores > 3.0  # 3-sigma threshold


class EnsembleDetector:
    """
    Ensemble of detectors for OOD robustness.

    Trains multiple models with different random seeds and
    averages their predictions.
    """

    def __init__(self, n_models: int = 5, seeds: Optional[List[int]] = None):
        self.n_models = n_models
        self.seeds = seeds or list(range(42, 42 + n_models))
        self.models: List = []
        self.scalers: List = []

    def fit(self, train_func, train_data: np.ndarray, train_attacks: Dict):
        """
        Train ensemble of models.

        Args:
            train_func: Function that trains a single model
            train_data: Training data
            train_attacks: Attack dictionary
        """
        for seed in self.seeds:
            np.random.seed(seed)
            model, scaler = train_func(train_data, train_attacks, seed=seed)
            self.models.append(model)
            self.scalers.append(scaler)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble predictions by averaging.
        """
        predictions = []

        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            predictions.append(pred)

        return np.mean(predictions, axis=0)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble anomaly scores.
        """
        scores = []

        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            score = model.score(X_scaled)
            scores.append(score)

        # Return mean and std for uncertainty quantification
        return np.mean(scores, axis=0), np.std(scores, axis=0)


class HybridPhysicsMLDetector:
    """
    Hybrid detector combining physics-invariant features with ML.

    Architecture:
    1. Compute physics-invariant features
    2. Compute ML features (CNN-GRU on raw data)
    3. Combine with learned weights
    4. Apply per-flight calibration
    """

    def __init__(self, physics_weight: float = 0.3, ml_weight: float = 0.7):
        self.physics_weight = physics_weight
        self.ml_weight = ml_weight
        self.physics_detector = PhysicsInvariantDetector()
        self.calibrator = PerFlightCalibrator()
        self.ml_model = None
        self.ml_scaler = None

    def set_ml_model(self, model, scaler):
        """Set the ML model and scaler."""
        self.ml_model = model
        self.ml_scaler = scaler

    def fit_physics(self, normal_trajectories: List[np.ndarray]):
        """Fit the physics detector."""
        self.physics_detector.fit(normal_trajectories)

    def score(self, trajectory: np.ndarray, ml_scores: np.ndarray) -> np.ndarray:
        """
        Compute hybrid score.

        Args:
            trajectory: Raw trajectory data
            ml_scores: Pre-computed ML scores

        Returns:
            Hybrid anomaly scores
        """
        # Physics scores (z-normalized)
        physics_scores = self.physics_detector.score(trajectory)
        physics_z = (physics_scores - np.mean(physics_scores[:100])) / (np.std(physics_scores[:100]) + 1e-6)

        # ML scores (z-normalized)
        ml_z = (ml_scores - np.mean(ml_scores[:100])) / (np.std(ml_scores[:100]) + 1e-6)

        # Combine
        hybrid = self.physics_weight * physics_z + self.ml_weight * ml_z

        return hybrid

    def detect_with_calibration(
        self,
        trajectory: np.ndarray,
        ml_scores: np.ndarray,
        calibration_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect with per-flight calibration.

        Returns:
            Tuple of (detections, calibrated_scores)
        """
        hybrid_scores = self.score(trajectory, ml_scores)

        # Calibrate on first N samples
        self.calibrator.calibration_samples = calibration_samples
        self.calibrator.calibrate(hybrid_scores)

        # Normalize and detect
        normalized = self.calibrator.normalize(hybrid_scores)
        detections = self.calibrator.detect(hybrid_scores)

        return detections, normalized


# ============================================================
# TEMPORAL STRUCTURE ENHANCEMENT
# ============================================================

class TemporalContrastiveLoss:
    """
    Contrastive loss that encourages temporal structure learning.

    The model should distinguish between:
    - Coherent sequences (normal temporal structure)
    - Shuffled sequences (broken temporal structure)

    This explicitly forces the model to learn temporal patterns.
    """

    @staticmethod
    def create_contrastive_pairs(X: np.ndarray, n_pairs: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create pairs of coherent and shuffled sequences.

        Returns:
            Tuple of (sequences, is_coherent_labels, pair_indices)
        """
        n, seq_len, features = X.shape

        pairs = []
        labels = []

        for _ in range(n_pairs):
            idx = np.random.randint(0, n)
            seq = X[idx].copy()

            # Original (coherent)
            pairs.append(seq)
            labels.append(1)

            # Shuffled (incoherent)
            shuffled = seq.copy()
            np.random.shuffle(shuffled)
            pairs.append(shuffled)
            labels.append(0)

        return np.array(pairs), np.array(labels), None

    @staticmethod
    def temporal_coherence_score(predictions: np.ndarray, window: int = 5) -> float:
        """
        Measure temporal coherence of predictions.

        Coherent predictions should be smooth over time.
        """
        if len(predictions) < window:
            return 0.0

        # Compute local variance
        local_vars = []
        for i in range(window, len(predictions)):
            local_vars.append(np.var(predictions[i-window:i]))

        return -np.mean(local_vars)  # Lower variance = more coherent


# ============================================================
# DOMAIN ADAPTATION WITHOUT LABELS
# ============================================================

class TestTimeAdaptation:
    """
    Test-time adaptation for domain shift robustness.

    Adapts the model to the test distribution without labels
    using entropy minimization.
    """

    def __init__(self, adaptation_steps: int = 10, lr: float = 0.001):
        self.adaptation_steps = adaptation_steps
        self.lr = lr

    def adapt_batch_norm(self, model, test_batch: np.ndarray):
        """
        Adapt batch normalization statistics to test distribution.

        This is the simplest and most effective form of TTA.
        """
        # Set model to eval mode but enable BN stats update
        model.eval()

        # Forward pass to update running stats
        with torch.no_grad():
            _ = model(test_batch)

        return model

    def entropy_minimization(self, model, test_batch: np.ndarray):
        """
        Adapt model by minimizing prediction entropy.

        Low entropy = confident predictions.
        High entropy = uncertain predictions.

        We adapt to reduce entropy (increase confidence).
        """
        import torch
        import torch.nn.functional as F

        model.train()  # Enable gradient computation

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for _ in range(self.adaptation_steps):
            outputs = model(test_batch)
            probs = torch.sigmoid(outputs)

            # Entropy: -p*log(p) - (1-p)*log(1-p)
            entropy = -probs * torch.log(probs + 1e-6) - (1 - probs) * torch.log(1 - probs + 1e-6)
            loss = entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        return model


# ============================================================
# ROBUST AGGREGATION
# ============================================================

class RobustAggregator:
    """
    Robust score aggregation that handles outliers and noise.

    Uses median and MAD instead of mean and std.
    """

    def __init__(self, window: int = 50):
        self.window = window

    def aggregate(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute robust running statistics.
        """
        n = len(scores)
        aggregated = np.zeros(n)

        for i in range(self.window, n):
            segment = scores[i-self.window:i]

            # Median (robust to outliers)
            median = np.median(segment)

            # MAD (robust scale estimate)
            mad = np.median(np.abs(segment - median))

            # Robust z-score
            if mad > 1e-6:
                aggregated[i] = (scores[i] - median) / (1.4826 * mad)  # 1.4826 for normal consistency
            else:
                aggregated[i] = scores[i] - median

        return aggregated

    def detect(self, scores: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Detect anomalies using robust aggregation.
        """
        aggregated = self.aggregate(scores)
        return np.abs(aggregated) > threshold
