#!/usr/bin/env python3
"""
Comprehensive Evaluation with Multiple Attack Magnitudes

Tests across:
- 6 attack types
- 5 magnitudes (1m, 5m, 10m, 25m, 50m)
- Temporal aggregation
- CUSUM detection
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Data Generation
# ============================================================================

class AttackGenerator:
    """Generate attacks at various magnitudes."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.gps_noise_std = 0.5  # meters (realistic)

    def generate_trajectory(self, n_samples: int = 400) -> np.ndarray:
        """Generate nominal trajectory."""
        dt = 0.005  # 200Hz

        state = np.zeros((n_samples, 12))
        state[0, :3] = np.random.randn(3) * 2.0
        state[0, 3:6] = np.random.randn(3) * 0.5

        for t in range(1, n_samples):
            state[t, :3] = state[t-1, :3] + state[t-1, 3:6] * dt
            state[t, 3:6] = state[t-1, 3:6] * 0.99 + np.random.randn(3) * 0.1
            state[t, 6:9] = state[t-1, 6:9] + state[t-1, 9:12] * dt
            state[t, 9:12] = state[t-1, 9:12] * 0.98 + np.random.randn(3) * 0.05

        # Add GPS noise
        state[:, :3] += np.random.randn(n_samples, 3) * self.gps_noise_std

        return state

    def inject_attack(
        self, traj: np.ndarray, attack_type: str, magnitude_m: float
    ) -> tuple:
        """Inject attack with specified magnitude in meters."""
        n = len(traj)
        attacked = traj.copy()
        labels = np.zeros(n)

        start = n // 4
        end = 3 * n // 4

        if attack_type == "bias":
            offset = np.array([magnitude_m, 0, 0])
            attacked[start:end, :3] += offset
            labels[start:end] = 1

        elif attack_type == "drift":
            t = np.arange(end - start)
            drift = np.outer(t / (end - start), [magnitude_m, 0, 0])
            attacked[start:end, :3] += drift
            labels[start:end] = 1

        elif attack_type == "noise":
            noise = np.random.randn(end - start, 3) * magnitude_m
            attacked[start:end, :3] += noise
            labels[start:end] = 1

        elif attack_type == "coordinated":
            offset = np.array([magnitude_m, 0, 0])
            attacked[start:end, :3] += offset
            attacked[start:end, 9:12] += np.random.randn(3) * 0.05
            labels[start:end] = 1

        elif attack_type == "intermittent":
            for i in range(start, end, 40):
                if np.random.rand() > 0.5:
                    attacked[i:min(i+20, end), :3] += [magnitude_m, 0, 0]
                    labels[i:min(i+20, end)] = 1

        elif attack_type == "ramp":
            t = np.arange(end - start)
            ramp = np.outer(t / (end - start), [magnitude_m, 0, 0])
            attacked[start:end, :3] += ramp
            labels[start:end] = 1

        return attacked, labels


# ============================================================================
# Enhanced Detector with Temporal Aggregation
# ============================================================================

class EnhancedDetector:
    """Detector with CUSUM and temporal aggregation."""

    def __init__(self):
        self.mean_features = None
        self.std_features = None
        self.threshold = None

    def extract_features(self, traj: np.ndarray) -> np.ndarray:
        """Extract multi-scale features."""
        n = len(traj)
        features = []

        for t in range(n):
            f = []

            # Position-velocity consistency
            if t > 0:
                pos_change = np.linalg.norm(traj[t, :3] - traj[t-1, :3])
                vel_pred = np.linalg.norm(traj[t-1, 3:6]) * 0.005
                f.append(abs(pos_change - vel_pred))
            else:
                f.append(0)

            # Velocity magnitude
            f.append(np.linalg.norm(traj[t, 3:6]))

            # Acceleration (velocity change)
            if t > 0:
                f.append(np.linalg.norm(traj[t, 3:6] - traj[t-1, 3:6]))
            else:
                f.append(0)

            # Position jump
            if t > 0:
                f.append(np.linalg.norm(traj[t, :3] - traj[t-1, :3]))
            else:
                f.append(0)

            # Jerk (acceleration change)
            if t > 1:
                acc_prev = traj[t-1, 3:6] - traj[t-2, 3:6]
                acc_curr = traj[t, 3:6] - traj[t-1, 3:6]
                f.append(np.linalg.norm(acc_curr - acc_prev))
            else:
                f.append(0)

            features.append(f)

        return np.array(features)

    def fit(self, trajectories: List[np.ndarray], labels_list: List[np.ndarray]):
        """Fit on normal data only."""
        all_features = []

        for traj, labels in zip(trajectories, labels_list):
            features = self.extract_features(traj)
            normal_mask = labels == 0
            if np.any(normal_mask):
                all_features.append(features[normal_mask])

        if all_features:
            all_features = np.vstack(all_features)
            self.mean_features = np.mean(all_features, axis=0)
            self.std_features = np.std(all_features, axis=0) + 1e-8

    def score(self, traj: np.ndarray) -> np.ndarray:
        """Compute anomaly scores with temporal smoothing."""
        features = self.extract_features(traj)

        if self.mean_features is None:
            return np.zeros(len(traj))

        # Normalize
        normalized = (features - self.mean_features) / self.std_features

        # Raw score = mean of squared normalized features
        raw_scores = np.mean(normalized ** 2, axis=1)

        # Temporal smoothing (window=20)
        window = 20
        smoothed = np.convolve(raw_scores, np.ones(window)/window, mode='same')

        return smoothed


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_at_magnitude(
    detector: EnhancedDetector,
    generator: AttackGenerator,
    attack_type: str,
    magnitude_m: float,
    n_sequences: int = 50,
) -> Dict:
    """Evaluate detector at specific attack magnitude."""
    all_scores = []
    all_labels = []

    for _ in range(n_sequences // 2):
        # Normal sequence
        traj = generator.generate_trajectory()
        scores = detector.score(traj)
        all_scores.extend(scores)
        all_labels.extend(np.zeros(len(traj)))

        # Attack sequence
        traj = generator.generate_trajectory()
        attacked, labels = generator.inject_attack(traj, attack_type, magnitude_m)
        scores = detector.score(attacked)
        all_scores.extend(scores)
        all_labels.extend(labels)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Compute metrics
    if len(np.unique(all_labels)) < 2:
        return {"auroc": 0.5, "recall_5pct": 0.0}

    auroc = roc_auc_score(all_labels, all_scores)

    # Recall at 5% FPR
    normal_scores = all_scores[all_labels == 0]
    threshold = np.percentile(normal_scores, 95)
    attack_scores = all_scores[all_labels == 1]
    recall_5pct = np.mean(attack_scores > threshold) if len(attack_scores) > 0 else 0

    return {
        "auroc": float(auroc),
        "recall_5pct": float(recall_5pct),
    }


def main():
    output_dir = Path(__file__).parent.parent / "results" / "comprehensive"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE GPS-IMU DETECTOR EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Setup
    generator = AttackGenerator(seed=42)
    detector = EnhancedDetector()

    # Generate training data (normal only)
    print("Phase 1: Training on normal sequences...")
    train_trajs = [generator.generate_trajectory() for _ in range(100)]
    train_labels = [np.zeros(len(t)) for t in train_trajs]
    detector.fit(train_trajs, train_labels)
    print("  Detector fitted")

    # Attack types and magnitudes
    attack_types = ["bias", "drift", "noise", "coordinated", "intermittent", "ramp"]
    magnitudes = [1.0, 5.0, 10.0, 25.0, 50.0]

    results = {
        "by_attack": {},
        "by_magnitude": {},
        "full_matrix": {},
    }

    # Evaluate each combination
    print("\nPhase 2: Evaluating attack combinations...")
    print()

    header = "| Attack Type     |" + "".join([f" {m:4.0f}m |" for m in magnitudes]) + " Mean |"
    print(header)
    print("|" + "-" * 17 + "|" + "".join(["-" * 7 + "|" for _ in magnitudes]) + "------|")

    attack_means = {}

    for attack_type in attack_types:
        results["by_attack"][attack_type] = {}
        row = f"| {attack_type:15} |"

        aurocs = []
        for mag in magnitudes:
            key = f"{attack_type}_{mag}m"
            r = evaluate_at_magnitude(detector, generator, attack_type, mag, n_sequences=40)
            results["full_matrix"][key] = r
            results["by_attack"][attack_type][f"{mag}m"] = r
            row += f" {r['auroc']*100:5.1f} |"
            aurocs.append(r['auroc'])

        mean_auroc = np.mean(aurocs)
        attack_means[attack_type] = mean_auroc
        row += f" {mean_auroc*100:4.1f} |"
        print(row)

    # Compute by-magnitude means
    print("|" + "-" * 17 + "|" + "".join(["-" * 7 + "|" for _ in magnitudes]) + "------|")
    row = "| Mean            |"
    for mag in magnitudes:
        mag_aurocs = [results["by_attack"][at][f"{mag}m"]["auroc"] for at in attack_types]
        mean_mag = np.mean(mag_aurocs)
        results["by_magnitude"][f"{mag}m"] = mean_mag
        row += f" {mean_mag*100:5.1f} |"
    overall_mean = np.mean(list(attack_means.values()))
    row += f" {overall_mean*100:4.1f} |"
    print(row)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n[Per-Attack Mean AUROC]")
    for at, auroc in sorted(attack_means.items(), key=lambda x: -x[1]):
        print(f"  {at:15}: {auroc*100:5.1f}%")

    print(f"\n[Overall Mean AUROC]: {overall_mean*100:.1f}%")

    print("\n[Detectability Floor Analysis]")
    for at in attack_types:
        for mag in magnitudes:
            auroc = results["by_attack"][at][f"{mag}m"]["auroc"]
            if auroc >= 0.80:
                print(f"  {at}: detectable at {mag}m (AUROC={auroc*100:.1f}%)")
                break
        else:
            print(f"  {at}: NOT reliably detectable up to 50m")

    # Per-attack at 10m (your document's reference)
    print("\n[Results at 10m offset (comparable to your document)]")
    print("| Attack Type     | AUROC | Recall@5%FPR |")
    print("|-----------------|-------|--------------|")
    for at in attack_types:
        r = results["by_attack"][at]["10.0m"]
        print(f"| {at:15} | {r['auroc']*100:5.1f}% | {r['recall_5pct']*100:10.1f}% |")

    # Comparison with document
    print("\n[Comparison with Your Document]")
    print("| Attack Type     | Your Doc | Current | Diff |")
    print("|-----------------|----------|---------|------|")
    doc_values = {
        "noise": 0.764,
        "drift": 0.807,
        "bias": 0.606,
        "coordinated": 0.570,
        "intermittent": 0.307,
    }
    for at, doc_val in doc_values.items():
        if at in attack_means:
            curr = attack_means[at]
            diff = (curr - doc_val) * 100
            print(f"| {at:15} | {doc_val*100:6.1f}% | {curr*100:5.1f}% | {diff:+4.1f} |")

    # Save results
    results["summary"] = {
        "overall_mean_auroc": overall_mean,
        "per_attack_mean": attack_means,
        "per_magnitude_mean": results["by_magnitude"],
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "comprehensive_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    main()
