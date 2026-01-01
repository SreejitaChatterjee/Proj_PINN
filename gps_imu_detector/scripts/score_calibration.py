#!/usr/bin/env python3
"""
Score Calibration - Improve operational metrics via calibration

This script applies calibration techniques to improve recall at low FPR:
1. Platt Scaling (logistic regression on scores)
2. Isotonic Regression (non-parametric monotonic calibration)
3. Temperature Scaling (simple single-parameter calibration)

Goal: Convert raw anomaly scores to calibrated probabilities
that better align with actual attack likelihood.

Author: Claude Code
Date: 2026-01-01
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid

sys.path.insert(0, str(Path(__file__).parent))
from honest_evaluation import (
    HonestConfig,
    RealisticAttackGenerator,
    HonestDetector,
    compute_auroc_with_ci,
)


# ============================================================================
# Calibration Methods
# ============================================================================

class PlattScaling:
    """Platt scaling using logistic regression."""

    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """Fit Platt scaling on calibration set."""
        self.model.fit(scores.reshape(-1, 1), labels)

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]


class IsotonicCalibration:
    """Isotonic regression calibration."""

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """Fit isotonic regression on calibration set."""
        self.model.fit(scores, labels)

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        return self.model.predict(scores)


class TemperatureScaling:
    """Temperature scaling (single parameter)."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """Find optimal temperature via cross-entropy minimization."""
        def objective(T):
            if T <= 0:
                return float('inf')
            probs = expit(scores / T)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            return log_loss(labels, probs)

        result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Return temperature-scaled probabilities."""
        return expit(scores / self.temperature)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_calibration(
    val_labels: np.ndarray,
    val_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    calibrated_probs: np.ndarray,
    method_name: str,
) -> Dict:
    """Evaluate calibration quality."""

    # Calibration metrics
    brier = brier_score_loss(test_labels, calibrated_probs)

    # Reliability diagram bins
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_counts = []
    bin_accuracies = []
    bin_confidences = []

    for i in range(n_bins):
        mask = (calibrated_probs >= bin_edges[i]) & (calibrated_probs < bin_edges[i+1])
        if mask.sum() > 0:
            bin_counts.append(int(mask.sum()))
            bin_accuracies.append(float(test_labels[mask].mean()))
            bin_confidences.append(float(calibrated_probs[mask].mean()))
        else:
            bin_counts.append(0)
            bin_accuracies.append(0.0)
            bin_confidences.append(bin_centers[i])

    # Expected Calibration Error (ECE)
    total_samples = len(test_labels)
    ece = sum(
        (bc / total_samples) * abs(acc - conf)
        for bc, acc, conf in zip(bin_counts, bin_accuracies, bin_confidences)
        if bc > 0
    )

    # AUROC (should be same as uncalibrated)
    auroc = roc_auc_score(test_labels, calibrated_probs)

    # Recall at fixed FPR
    recalls = {}
    for target_fpr in [0.005, 0.01, 0.05]:
        # Find threshold for target FPR on calibrated probs
        normal_probs = calibrated_probs[test_labels == 0]
        attack_probs = calibrated_probs[test_labels == 1]
        thresh = np.percentile(normal_probs, 100 * (1 - target_fpr))
        tpr = np.mean(attack_probs > thresh)
        recalls[f"recall_at_{int(target_fpr*100)}pct_fpr"] = float(tpr)

    return {
        "method": method_name,
        "brier_score": float(brier),
        "ece": float(ece),
        "auroc": float(auroc),
        **recalls,
        "reliability_diagram": {
            "bin_centers": [float(b) for b in bin_centers],
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
        },
    }


# ============================================================================
# Main
# ============================================================================

def run_calibration_analysis():
    """Run calibration analysis on GPS-IMU detector."""

    np.random.seed(HonestConfig.RANDOM_SEED)

    output_dir = Path(__file__).parent.parent / "results" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCORE CALIBRATION ANALYSIS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Initialize
    generator = RealisticAttackGenerator(seed=HonestConfig.RANDOM_SEED)
    detector = HonestDetector()

    # Train detector
    print("\nTraining detector...")
    train_trajs = [generator.generate_nominal_trajectory() for _ in range(100)]
    detector.fit(train_trajs)

    results = {
        "timestamp": datetime.now().isoformat(),
        "by_attack": {},
    }

    # Test each attack type
    attack_types = ["noise_injection", "intermittent", "bias", "coordinated"]

    for attack_type in attack_types:
        print(f"\n{'='*70}")
        print(f"Attack: {attack_type}")
        print("="*70)

        # Generate calibration and test data
        cal_scores = []
        cal_labels = []
        test_scores = []
        test_labels = []

        for i in range(100):
            # Normal
            traj = generator.generate_nominal_trajectory()
            scores = detector.score(traj)
            if i < 50:
                cal_scores.extend(scores)
                cal_labels.extend(np.zeros(len(scores)))
            else:
                test_scores.extend(scores)
                test_labels.extend(np.zeros(len(scores)))

            # Attack (at 5x magnitude for meaningful calibration)
            traj = generator.generate_nominal_trajectory()
            attacked, labels = generator.inject_attack(traj, attack_type, 5.0)
            scores = detector.score(attacked)
            if i < 50:
                cal_scores.extend(scores)
                cal_labels.extend(labels)
            else:
                test_scores.extend(scores)
                test_labels.extend(labels)

        cal_scores = np.array(cal_scores)
        cal_labels = np.array(cal_labels)
        test_scores = np.array(test_scores)
        test_labels = np.array(test_labels)

        results["by_attack"][attack_type] = {"methods": {}}

        # Uncalibrated baseline
        print("\n[Uncalibrated]")
        uncal_probs = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-10)
        uncal_results = evaluate_calibration(
            cal_labels, cal_scores, test_labels, test_scores,
            uncal_probs, "uncalibrated"
        )
        results["by_attack"][attack_type]["methods"]["uncalibrated"] = uncal_results
        print(f"  Brier Score: {uncal_results['brier_score']:.4f}")
        print(f"  ECE: {uncal_results['ece']:.4f}")
        print(f"  Recall @1% FPR: {uncal_results['recall_at_1pct_fpr']*100:.1f}%")
        print(f"  Recall @5% FPR: {uncal_results['recall_at_5pct_fpr']*100:.1f}%")

        # Platt Scaling
        print("\n[Platt Scaling]")
        platt = PlattScaling()
        platt.fit(cal_scores, cal_labels)
        platt_probs = platt.calibrate(test_scores)
        platt_results = evaluate_calibration(
            cal_labels, cal_scores, test_labels, test_scores,
            platt_probs, "platt"
        )
        results["by_attack"][attack_type]["methods"]["platt"] = platt_results
        print(f"  Brier Score: {platt_results['brier_score']:.4f}")
        print(f"  ECE: {platt_results['ece']:.4f}")
        print(f"  Recall @1% FPR: {platt_results['recall_at_1pct_fpr']*100:.1f}%")
        print(f"  Recall @5% FPR: {platt_results['recall_at_5pct_fpr']*100:.1f}%")

        # Isotonic Regression
        print("\n[Isotonic Regression]")
        isotonic = IsotonicCalibration()
        isotonic.fit(cal_scores, cal_labels)
        isotonic_probs = isotonic.calibrate(test_scores)
        isotonic_results = evaluate_calibration(
            cal_labels, cal_scores, test_labels, test_scores,
            isotonic_probs, "isotonic"
        )
        results["by_attack"][attack_type]["methods"]["isotonic"] = isotonic_results
        print(f"  Brier Score: {isotonic_results['brier_score']:.4f}")
        print(f"  ECE: {isotonic_results['ece']:.4f}")
        print(f"  Recall @1% FPR: {isotonic_results['recall_at_1pct_fpr']*100:.1f}%")
        print(f"  Recall @5% FPR: {isotonic_results['recall_at_5pct_fpr']*100:.1f}%")

        # Temperature Scaling
        print("\n[Temperature Scaling]")
        temp = TemperatureScaling()
        temp.fit(cal_scores, cal_labels)
        temp_probs = temp.calibrate(test_scores)
        temp_results = evaluate_calibration(
            cal_labels, cal_scores, test_labels, test_scores,
            temp_probs, "temperature"
        )
        temp_results["temperature"] = float(temp.temperature)
        results["by_attack"][attack_type]["methods"]["temperature"] = temp_results
        print(f"  Temperature: {temp.temperature:.3f}")
        print(f"  Brier Score: {temp_results['brier_score']:.4f}")
        print(f"  ECE: {temp_results['ece']:.4f}")
        print(f"  Recall @1% FPR: {temp_results['recall_at_1pct_fpr']*100:.1f}%")
        print(f"  Recall @5% FPR: {temp_results['recall_at_5pct_fpr']*100:.1f}%")

        # Best method
        methods = results["by_attack"][attack_type]["methods"]
        best_method = min(methods.keys(), key=lambda m: methods[m]["brier_score"])
        results["by_attack"][attack_type]["best_method"] = best_method
        print(f"\n  Best method (by Brier): {best_method}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n| Attack Type     | Best Method | Recall@1% (uncal) | Recall@1% (best) | Improvement |")
    print("|" + "-"*17 + "|" + "-"*13 + "|" + "-"*19 + "|" + "-"*18 + "|" + "-"*13 + "|")

    for attack_type in attack_types:
        data = results["by_attack"][attack_type]
        best = data["best_method"]
        uncal = data["methods"]["uncalibrated"]["recall_at_1pct_fpr"]
        best_recall = data["methods"][best]["recall_at_1pct_fpr"]
        improvement = (best_recall - uncal) * 100

        print(f"| {attack_type:15} | {best:11} | {uncal*100:16.1f}% | {best_recall*100:15.1f}% | {improvement:+10.1f}% |")

    print("\nKey Finding: Calibration improves probability estimates but does NOT")
    print("change the fundamental AUROC or overcome information-theoretic limits.")
    print("For indistinguishable attacks, no calibration can help.")

    # Save
    with open(output_dir / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'calibration_results.json'}")

    return results


if __name__ == "__main__":
    run_calibration_analysis()
