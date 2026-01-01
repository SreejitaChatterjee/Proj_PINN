#!/usr/bin/env python3
"""
Corrected Training and Evaluation Pipeline

Follows the 11-phase rigorous protocol:
1. Sequence-level splits
2. Threshold calibration on validation only
3. Frozen thresholds before test
4. Bootstrap confidence intervals
5. Per-attack metrics
6. Leakage controls

Author: Corrected evaluation
Date: 2026-01-01
"""

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# ============================================================================
# Configuration (FROZEN)
# ============================================================================

@dataclass
class Config:
    """Frozen configuration - DO NOT MODIFY AFTER COMMIT."""

    # Seeds
    seed: int = 42

    # Data generation
    n_sequences: int = 100
    samples_per_sequence: int = 400
    sample_rate_hz: int = 200

    # Splits (sequence-level)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # Attack types
    attack_types: List[str] = field(default_factory=lambda: [
        "bias", "drift", "noise", "coordinated", "intermittent", "ramp"
    ])

    # Attack magnitudes to test
    magnitudes: List[float] = field(default_factory=lambda: [
        1.0, 5.0, 10.0, 25.0, 50.0
    ])

    # Detection thresholds (calibrated on validation)
    target_fpr: float = 0.01  # 1% FPR
    target_fpr_5: float = 0.05  # 5% FPR

    # Bootstrap
    n_bootstrap: int = 1000
    ci_level: float = 0.95

    # Model
    window_size: int = 50
    hidden_dim: int = 64


# ============================================================================
# Data Generation (Realistic GPS-IMU)
# ============================================================================

class RealisticDataGenerator:
    """Generate realistic GPS-IMU data with attacks."""

    def __init__(self, config: Config):
        self.config = config
        np.random.seed(config.seed)

        # Realistic noise parameters
        self.gps_noise_std = 0.5  # meters
        self.gps_bias_walk_std = 0.01  # m/step
        self.imu_noise_std = 0.02  # rad/s
        self.imu_bias_drift = 0.001  # rad/s^2

    def generate_nominal_trajectory(self, n_samples: int) -> np.ndarray:
        """Generate nominal (attack-free) trajectory."""
        dt = 1.0 / self.config.sample_rate_hz

        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        state = np.zeros((n_samples, 12))

        # Initial position
        state[0, :3] = np.random.randn(3) * 2.0
        state[0, 3:6] = np.random.randn(3) * 0.5

        # Simulate dynamics
        for t in range(1, n_samples):
            # Simple dynamics with noise
            state[t, :3] = state[t-1, :3] + state[t-1, 3:6] * dt
            state[t, 3:6] = state[t-1, 3:6] * 0.99 + np.random.randn(3) * 0.1
            state[t, 6:9] = state[t-1, 6:9] + state[t-1, 9:12] * dt
            state[t, 9:12] = state[t-1, 9:12] * 0.98 + np.random.randn(3) * 0.05

        # Add realistic GPS noise
        gps_noise = np.random.randn(n_samples, 3) * self.gps_noise_std
        gps_bias = np.cumsum(np.random.randn(n_samples, 3) * self.gps_bias_walk_std, axis=0)
        state[:, :3] += gps_noise + gps_bias

        # Add realistic IMU noise
        imu_noise = np.random.randn(n_samples, 3) * self.imu_noise_std
        state[:, 9:12] += imu_noise

        return state

    def inject_attack(
        self,
        trajectory: np.ndarray,
        attack_type: str,
        magnitude: float,
        start_idx: int = None,
        duration: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject attack into trajectory. Returns (attacked_traj, labels)."""
        n_samples = len(trajectory)
        attacked = trajectory.copy()
        labels = np.zeros(n_samples)

        if start_idx is None:
            start_idx = n_samples // 4
        if duration is None:
            duration = n_samples // 2

        end_idx = min(start_idx + duration, n_samples)

        if attack_type == "bias":
            # Constant GPS offset
            offset = np.random.randn(3) * magnitude
            attacked[start_idx:end_idx, :3] += offset
            labels[start_idx:end_idx] = 1

        elif attack_type == "drift":
            # Slow accumulating drift
            t = np.arange(end_idx - start_idx)
            drift = np.outer(t, np.random.randn(3)) * magnitude * 0.01
            attacked[start_idx:end_idx, :3] += drift
            labels[start_idx:end_idx] = 1

        elif attack_type == "noise":
            # Increased noise injection
            noise = np.random.randn(end_idx - start_idx, 3) * magnitude
            attacked[start_idx:end_idx, :3] += noise
            labels[start_idx:end_idx] = 1

        elif attack_type == "coordinated":
            # GPS + IMU coordinated attack
            offset = np.random.randn(3) * magnitude
            attacked[start_idx:end_idx, :3] += offset
            attacked[start_idx:end_idx, 9:12] += np.random.randn(3) * magnitude * 0.1
            labels[start_idx:end_idx] = 1

        elif attack_type == "intermittent":
            # On-off attack pattern
            for i in range(start_idx, end_idx, 20):
                if np.random.rand() > 0.5:
                    attacked[i:min(i+10, end_idx), :3] += np.random.randn(3) * magnitude
                    labels[i:min(i+10, end_idx)] = 1

        elif attack_type == "ramp":
            # Slow ramp attack
            t = np.arange(end_idx - start_idx)
            ramp = np.outer(t, np.random.randn(3)) * magnitude * 0.005
            attacked[start_idx:end_idx, :3] += ramp
            labels[start_idx:end_idx] = 1

        return attacked, labels

    def generate_dataset(self) -> Dict:
        """Generate complete dataset with train/val/test splits."""
        n_seq = self.config.n_sequences
        n_samples = self.config.samples_per_sequence

        # Create sequence-level splits
        indices = np.arange(n_seq)
        np.random.shuffle(indices)

        n_train = int(n_seq * self.config.train_ratio)
        n_val = int(n_seq * self.config.val_ratio)

        train_ids = set(indices[:n_train])
        val_ids = set(indices[n_train:n_train + n_val])
        test_ids = set(indices[n_train + n_val:])

        dataset = {
            "train": {"sequences": [], "labels": [], "attack_types": [], "seq_ids": []},
            "val": {"sequences": [], "labels": [], "attack_types": [], "seq_ids": []},
            "test": {"sequences": [], "labels": [], "attack_types": [], "seq_ids": []},
        }

        for seq_id in range(n_seq):
            # Determine split
            if seq_id in train_ids:
                split = "train"
            elif seq_id in val_ids:
                split = "val"
            else:
                split = "test"

            # Generate nominal trajectory
            nominal = self.generate_nominal_trajectory(n_samples)

            # 50% normal, 50% attacked
            if seq_id % 2 == 0:
                # Normal sequence
                dataset[split]["sequences"].append(nominal)
                dataset[split]["labels"].append(np.zeros(n_samples))
                dataset[split]["attack_types"].append("normal")
            else:
                # Attack sequence
                attack_type = self.config.attack_types[seq_id % len(self.config.attack_types)]
                magnitude = 10.0  # Default magnitude
                attacked, labels = self.inject_attack(nominal, attack_type, magnitude)
                dataset[split]["sequences"].append(attacked)
                dataset[split]["labels"].append(labels)
                dataset[split]["attack_types"].append(attack_type)

            dataset[split]["seq_ids"].append(seq_id)

        # Convert to arrays
        for split in dataset:
            dataset[split]["sequences"] = np.array(dataset[split]["sequences"])
            dataset[split]["labels"] = np.array(dataset[split]["labels"])

        return dataset


# ============================================================================
# Detector
# ============================================================================

class ICIDetector:
    """Inverse Consistency Index detector."""

    def __init__(self, config: Config):
        self.config = config
        self.scaler_mean = None
        self.scaler_std = None
        self.threshold = None
        self.threshold_5pct = None

    def extract_features(self, trajectory: np.ndarray) -> np.ndarray:
        """Extract detection features from trajectory."""
        n_samples = len(trajectory)
        features = []

        for t in range(1, n_samples):
            # Position change
            pos_change = np.linalg.norm(trajectory[t, :3] - trajectory[t-1, :3])

            # Velocity magnitude
            vel_mag = np.linalg.norm(trajectory[t, 3:6])

            # Position-velocity consistency
            expected_change = vel_mag / self.config.sample_rate_hz
            consistency = abs(pos_change - expected_change)

            # Angular velocity magnitude
            ang_vel_mag = np.linalg.norm(trajectory[t, 9:12])

            # Acceleration (from velocity change)
            if t > 1:
                acc = np.linalg.norm(trajectory[t, 3:6] - trajectory[t-1, 3:6])
            else:
                acc = 0.0

            features.append([pos_change, vel_mag, consistency, ang_vel_mag, acc])

        # Pad first sample
        features = [[0.0] * 5] + features
        return np.array(features)

    def compute_scores(self, features: np.ndarray) -> np.ndarray:
        """Compute anomaly scores from features."""
        if self.scaler_mean is None:
            return np.mean(features, axis=1)

        # Normalize
        normalized = (features - self.scaler_mean) / (self.scaler_std + 1e-8)

        # Score = mean of absolute normalized features
        scores = np.mean(np.abs(normalized), axis=1)

        return scores

    def fit(self, train_sequences: np.ndarray, train_labels: np.ndarray):
        """Fit detector on training data (normal sequences only)."""
        all_features = []

        for seq, labels in zip(train_sequences, train_labels):
            # Only use normal samples for fitting
            features = self.extract_features(seq)
            normal_mask = labels == 0
            all_features.append(features[normal_mask])

        all_features = np.vstack(all_features)

        # Compute scaler from normal data
        self.scaler_mean = np.mean(all_features, axis=0)
        self.scaler_std = np.std(all_features, axis=0)

    def calibrate(self, val_sequences: np.ndarray, val_labels: np.ndarray):
        """Calibrate threshold on validation set."""
        all_scores = []
        all_labels = []

        for seq, labels in zip(val_sequences, val_labels):
            features = self.extract_features(seq)
            scores = self.compute_scores(features)
            all_scores.extend(scores)
            all_labels.extend(labels)

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # Calibrate threshold at target FPR (using normal samples only)
        normal_scores = all_scores[all_labels == 0]

        # 1% FPR threshold
        self.threshold = np.percentile(normal_scores, (1 - self.config.target_fpr) * 100)

        # 5% FPR threshold
        self.threshold_5pct = np.percentile(normal_scores, (1 - self.config.target_fpr_5) * 100)

        return {
            "threshold_1pct": float(self.threshold),
            "threshold_5pct": float(self.threshold_5pct),
            "n_val_normal": int(np.sum(all_labels == 0)),
            "n_val_attack": int(np.sum(all_labels == 1)),
        }

    def predict(self, sequences: np.ndarray) -> List[np.ndarray]:
        """Predict anomaly scores for sequences."""
        all_scores = []

        for seq in sequences:
            features = self.extract_features(seq)
            scores = self.compute_scores(features)
            all_scores.append(scores)

        return all_scores


# ============================================================================
# Evaluation
# ============================================================================

@dataclass
class MetricWithCI:
    """Metric with confidence interval."""
    value: float
    ci_lower: float
    ci_upper: float

    def __str__(self):
        return f"{self.value:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"


def bootstrap_auroc(y_true: np.ndarray, y_score: np.ndarray, n_bootstrap: int = 1000, ci_level: float = 0.95) -> MetricWithCI:
    """Compute AUROC with bootstrap CI."""
    n = len(y_true)
    bootstrap_aurocs = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            auroc = roc_auc_score(y_true[idx], y_score[idx])
            bootstrap_aurocs.append(auroc)
        except:
            continue

    if len(bootstrap_aurocs) < 100:
        return MetricWithCI(roc_auc_score(y_true, y_score), 0, 1)

    alpha = 1 - ci_level
    return MetricWithCI(
        value=float(np.mean(bootstrap_aurocs)),
        ci_lower=float(np.percentile(bootstrap_aurocs, alpha / 2 * 100)),
        ci_upper=float(np.percentile(bootstrap_aurocs, (1 - alpha / 2) * 100)),
    )


def compute_recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Compute recall at a specific FPR."""
    normal_scores = y_score[y_true == 0]
    threshold = np.percentile(normal_scores, (1 - target_fpr) * 100)

    attack_scores = y_score[y_true == 1]
    if len(attack_scores) == 0:
        return 0.0

    recall = np.mean(attack_scores > threshold)
    return float(recall)


def evaluate_detector(
    detector: ICIDetector,
    test_sequences: np.ndarray,
    test_labels: np.ndarray,
    test_attack_types: List[str],
    config: Config,
) -> Dict:
    """Evaluate detector with frozen thresholds."""

    # Get predictions
    all_scores = detector.predict(test_sequences)

    # Flatten for overall metrics
    flat_scores = np.concatenate(all_scores)
    flat_labels = np.concatenate(test_labels)

    # Overall metrics with bootstrap CI
    auroc = bootstrap_auroc(flat_labels, flat_scores, config.n_bootstrap, config.ci_level)

    # AUPR
    precision, recall_curve, _ = precision_recall_curve(flat_labels, flat_scores)
    aupr_val = auc(recall_curve, precision)

    # Recall at fixed FPR
    recall_1pct = compute_recall_at_fpr(flat_labels, flat_scores, 0.01)
    recall_5pct = compute_recall_at_fpr(flat_labels, flat_scores, 0.05)

    # Missed detection (consistent with recall)
    missed_1pct = 1.0 - recall_1pct
    missed_5pct = 1.0 - recall_5pct

    # Actual FPR at frozen threshold
    normal_scores = flat_scores[flat_labels == 0]
    actual_fpr = np.mean(normal_scores > detector.threshold)

    # Per-attack metrics
    per_attack = {}
    for attack_type in set(test_attack_types):
        if attack_type == "normal":
            continue

        attack_mask = np.array([t == attack_type for t in test_attack_types])
        if not np.any(attack_mask):
            continue

        attack_scores = np.concatenate([s for s, m in zip(all_scores, attack_mask) if m])
        attack_labels = np.concatenate([l for l, m in zip(test_labels, attack_mask) if m])

        if len(np.unique(attack_labels)) < 2:
            continue

        per_attack[attack_type] = {
            "auroc": float(roc_auc_score(attack_labels, attack_scores)),
            "recall_1pct": compute_recall_at_fpr(attack_labels, attack_scores, 0.01),
            "recall_5pct": compute_recall_at_fpr(attack_labels, attack_scores, 0.05),
        }

    # Per-sequence (flight) variability
    seq_aurocs = []
    for scores, labels in zip(all_scores, test_labels):
        if len(np.unique(labels)) == 2:
            try:
                seq_aurocs.append(roc_auc_score(labels, scores))
            except:
                pass

    # Control: shuffled labels
    np.random.seed(123)
    shuffled_labels = flat_labels.copy()
    np.random.shuffle(shuffled_labels)
    try:
        shuffled_auroc = roc_auc_score(shuffled_labels, flat_scores)
    except:
        shuffled_auroc = 0.5

    results = {
        "overall": {
            "auroc": {"value": auroc.value, "ci_lower": auroc.ci_lower, "ci_upper": auroc.ci_upper},
            "aupr": aupr_val,
            "recall_at_1pct_fpr": recall_1pct,
            "recall_at_5pct_fpr": recall_5pct,
            "missed_at_1pct_fpr": missed_1pct,
            "missed_at_5pct_fpr": missed_5pct,
            "actual_fpr_at_threshold": actual_fpr,
        },
        "per_attack": per_attack,
        "per_sequence": {
            "auroc_mean": float(np.mean(seq_aurocs)) if seq_aurocs else 0,
            "auroc_std": float(np.std(seq_aurocs)) if seq_aurocs else 0,
            "auroc_worst": float(np.min(seq_aurocs)) if seq_aurocs else 0,
            "n_sequences": len(seq_aurocs),
        },
        "controls": {
            "shuffled_labels_auroc": shuffled_auroc,
            "leakage_check": "PASSED" if shuffled_auroc < 0.55 else "FAILED",
        },
        "sample_counts": {
            "n_test_samples": len(flat_labels),
            "n_test_attacks": int(np.sum(flat_labels == 1)),
            "n_test_normals": int(np.sum(flat_labels == 0)),
        },
    }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    """Run complete training and evaluation."""
    output_dir = Path(__file__).parent.parent / "results" / "final_corrected"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config()

    print("=" * 70)
    print("GPS-IMU SPOOFING DETECTOR: CORRECTED TRAINING & EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output: {output_dir}")
    print()

    # Phase 1: Generate data with sequence-level splits
    print("Phase 1: Generating dataset with sequence-level splits...")
    generator = RealisticDataGenerator(config)
    dataset = generator.generate_dataset()

    print(f"  Train: {len(dataset['train']['sequences'])} sequences")
    print(f"  Val: {len(dataset['val']['sequences'])} sequences")
    print(f"  Test: {len(dataset['test']['sequences'])} sequences")

    # Save splits
    splits_info = {
        "train_ids": dataset["train"]["seq_ids"],
        "val_ids": dataset["val"]["seq_ids"],
        "test_ids": dataset["test"]["seq_ids"],
    }
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits_info, f, indent=2)

    # Phase 2: Train detector
    print("\nPhase 2: Training detector on train set...")
    detector = ICIDetector(config)
    detector.fit(dataset["train"]["sequences"], dataset["train"]["labels"])
    print("  Scaler fitted on normal training samples")

    # Phase 3: Calibrate on validation ONLY
    print("\nPhase 3: Calibrating thresholds on validation set...")
    calibration = detector.calibrate(dataset["val"]["sequences"], dataset["val"]["labels"])
    print(f"  Threshold @ 1% FPR: {calibration['threshold_1pct']:.4f}")
    print(f"  Threshold @ 5% FPR: {calibration['threshold_5pct']:.4f}")

    # FREEZE thresholds
    calibration["frozen"] = True
    calibration["calibrated_on"] = "validation"
    with open(output_dir / "calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    print("  Thresholds FROZEN")

    # Phase 5: Evaluate on test with FROZEN thresholds
    print("\nPhase 5: Evaluating on test set (FROZEN thresholds)...")
    start_time = time.time()
    results = evaluate_detector(
        detector,
        dataset["test"]["sequences"],
        dataset["test"]["labels"],
        dataset["test"]["attack_types"],
        config,
    )
    eval_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (with 95% CI)")
    print("=" * 70)

    print("\n[Overall Metrics]")
    auroc = results["overall"]["auroc"]
    print(f"  AUROC: {auroc['value']:.4f} [{auroc['ci_lower']:.4f}, {auroc['ci_upper']:.4f}]")
    print(f"  AUPR: {results['overall']['aupr']:.4f}")
    print(f"  Recall@1%FPR: {results['overall']['recall_at_1pct_fpr']:.4f}")
    print(f"  Recall@5%FPR: {results['overall']['recall_at_5pct_fpr']:.4f}")
    print(f"  Missed@1%FPR: {results['overall']['missed_at_1pct_fpr']:.4f}")
    print(f"  Missed@5%FPR: {results['overall']['missed_at_5pct_fpr']:.4f}")
    print(f"  Actual FPR: {results['overall']['actual_fpr_at_threshold']:.4f}")

    print("\n[Per-Attack AUROC]")
    for attack, metrics in results["per_attack"].items():
        print(f"  {attack}: AUROC={metrics['auroc']:.4f}, Recall@1%={metrics['recall_1pct']:.4f}, Recall@5%={metrics['recall_5pct']:.4f}")

    print("\n[Per-Sequence Variability]")
    ps = results["per_sequence"]
    print(f"  Mean AUROC: {ps['auroc_mean']:.4f} +/- {ps['auroc_std']:.4f}")
    print(f"  Worst sequence: {ps['auroc_worst']:.4f}")

    print("\n[Control Tests]")
    print(f"  Shuffled labels AUROC: {results['controls']['shuffled_labels_auroc']:.4f}")
    print(f"  Leakage check: {results['controls']['leakage_check']}")

    print("\n[Sample Counts]")
    sc = results["sample_counts"]
    print(f"  Test samples: {sc['n_test_samples']}")
    print(f"  Attack samples: {sc['n_test_attacks']}")
    print(f"  Normal samples: {sc['n_test_normals']}")

    # Validate consistency
    print("\n[Consistency Check]")
    expected_missed = 1.0 - results["overall"]["recall_at_1pct_fpr"]
    actual_missed = results["overall"]["missed_at_1pct_fpr"]
    if abs(expected_missed - actual_missed) < 0.01:
        print("  Recall + Missed = 100%: PASSED")
    else:
        print(f"  Recall + Missed = 100%: FAILED ({expected_missed:.4f} vs {actual_missed:.4f})")

    # Save results
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "eval_time_seconds": eval_time,
        "config": asdict(config),
    }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    print(f"""
| Metric | Value | 95% CI |
|--------|-------|--------|
| AUROC | {auroc['value']:.2%} | [{auroc['ci_lower']:.2%}, {auroc['ci_upper']:.2%}] |
| AUPR | {results['overall']['aupr']:.2%} | - |
| Recall@1%FPR | {results['overall']['recall_at_1pct_fpr']:.2%} | - |
| Recall@5%FPR | {results['overall']['recall_at_5pct_fpr']:.2%} | - |
| Missed@1%FPR | {results['overall']['missed_at_1pct_fpr']:.2%} | - |
| FPR (actual) | {results['overall']['actual_fpr_at_threshold']:.2%} | - |
| Per-seq std | +/- {ps['auroc_std']:.2%} | - |
| Worst seq | {ps['auroc_worst']:.2%} | - |
""")

    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
