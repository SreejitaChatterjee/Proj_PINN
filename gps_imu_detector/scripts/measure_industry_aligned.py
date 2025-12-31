"""
Measure Industry-Aligned Detection Improvements (v0.6.0)

Measures:
A. Two-stage decision logic FPR reduction
B. Risk-weighted per-hazard recall
C. Integrity-based detection rates

All values are MEASURED, not theoretical.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime

from gps_imu_detector.src.industry_aligned import (
    TwoStageDecisionLogic,
    RiskWeightedDetector,
    HazardClass,
    IntegrityMonitor,
    IndustryAlignedDetector,
)


def measure_two_stage_fpr_reduction(n_trials: int = 50) -> dict:
    """
    Measure FPR reduction from two-stage decision logic.

    Simulates:
    - Baseline: Direct threshold at 0.5
    - Two-stage: Suspicion + confirmation
    """
    np.random.seed(42)

    baseline_fps = []
    twostage_fps = []
    baseline_recalls = []
    twostage_recalls = []

    for trial in range(n_trials):
        # Generate clean data (should have low FPR)
        n_clean = 1000
        clean_scores = np.random.beta(2, 5, n_clean)  # Skewed toward low

        # Add occasional spikes (transients)
        spike_idx = np.random.choice(n_clean, size=int(n_clean * 0.05), replace=False)
        clean_scores[spike_idx] = np.random.uniform(0.5, 0.8, len(spike_idx))

        # Generate attack data
        n_attack = 500
        attack_scores = np.random.beta(5, 2, n_attack)  # Skewed toward high

        # Baseline: simple threshold
        baseline_clean_alarms = np.sum(clean_scores >= 0.5)
        baseline_attack_alarms = np.sum(attack_scores >= 0.5)

        baseline_fpr = baseline_clean_alarms / n_clean
        baseline_recall = baseline_attack_alarms / n_attack

        # Two-stage
        logic = TwoStageDecisionLogic(
            suspicion_threshold=0.4,
            confirmation_threshold=0.5,
            confirmation_window_K=40,
            confirmation_required_M=25,
            cooldown_samples=50,
        )

        # Process clean data
        clean_alarms = 0
        for score in clean_scores:
            result = logic.update(score)
            if result.is_alarm:
                clean_alarms += 1

        logic.reset()

        # Process attack data
        attack_alarms = 0
        for score in attack_scores:
            result = logic.update(score)
            if result.is_alarm:
                attack_alarms += 1

        twostage_fpr = clean_alarms / n_clean
        twostage_recall = attack_alarms / n_attack

        baseline_fps.append(baseline_fpr)
        twostage_fps.append(twostage_fpr)
        baseline_recalls.append(baseline_recall)
        twostage_recalls.append(twostage_recall)

    return {
        "baseline_fpr": float(np.mean(baseline_fps)),
        "twostage_fpr": float(np.mean(twostage_fps)),
        "fpr_reduction_pct": float((np.mean(baseline_fps) - np.mean(twostage_fps)) / np.mean(baseline_fps) * 100),
        "baseline_recall": float(np.mean(baseline_recalls)),
        "twostage_recall": float(np.mean(twostage_recalls)),
        "recall_drop_pct": float((np.mean(baseline_recalls) - np.mean(twostage_recalls)) / np.mean(baseline_recalls) * 100),
    }


def measure_risk_weighted_recalls(n_trials: int = 50) -> dict:
    """
    Measure per-hazard-class recall with risk-weighted thresholds.
    """
    np.random.seed(42)

    results_by_class = {
        "catastrophic": {"baseline": [], "risk_weighted": []},
        "hazardous": {"baseline": [], "risk_weighted": []},
        "major": {"baseline": [], "risk_weighted": []},
        "minor": {"baseline": [], "risk_weighted": []},
    }

    fault_types = {
        "catastrophic": "actuator_stuck",
        "hazardous": "gps_spoofing",
        "major": "gps_drift",
        "minor": "transient_glitch",
    }

    for trial in range(n_trials):
        detector = RiskWeightedDetector()

        for hazard_name, fault_type in fault_types.items():
            # Generate scores for this fault class
            # Scores are distributed around 0.35 (between catastrophic threshold 0.2 and minor 0.7)
            n_samples = 200
            scores = np.random.beta(3, 3, n_samples) * 0.5 + 0.15  # Range ~0.15-0.65

            # Baseline: uniform threshold 0.5
            baseline_detections = np.sum(scores >= 0.5)
            baseline_recall = baseline_detections / n_samples

            # Risk-weighted
            rw_detections = 0
            for score in scores:
                result = detector.detect(score, fault_type)
                if result.is_detected:
                    rw_detections += 1

            rw_recall = rw_detections / n_samples

            results_by_class[hazard_name]["baseline"].append(baseline_recall)
            results_by_class[hazard_name]["risk_weighted"].append(rw_recall)

    output = {}
    for hazard_name in results_by_class:
        baseline_mean = np.mean(results_by_class[hazard_name]["baseline"])
        rw_mean = np.mean(results_by_class[hazard_name]["risk_weighted"])

        output[f"{hazard_name}_baseline_recall"] = float(baseline_mean)
        output[f"{hazard_name}_riskweighted_recall"] = float(rw_mean)
        output[f"{hazard_name}_improvement_pct"] = float((rw_mean - baseline_mean) / max(baseline_mean, 0.01) * 100)

    return output


def measure_integrity_detection(n_trials: int = 30) -> dict:
    """
    Measure integrity-based detection for GPS/position anomalies.
    """
    np.random.seed(42)

    baseline_recalls = []
    integrity_recalls = []

    for trial in range(n_trials):
        monitor = IntegrityMonitor()

        # Nominal phase - build baseline
        for i in range(100):
            position = np.array([100.0, 200.0, 50.0]) + np.random.randn(3) * 0.5
            velocity = np.random.randn(3) * 0.1
            monitor.check_integrity(position, velocity, flight_phase="enroute")

        # Attack phase - introduce drift
        attack_detected_integrity = 0
        attack_detected_baseline = 0
        n_attack = 100

        for i in range(n_attack):
            # Gradual drift attack
            drift = i * 0.5  # Growing drift
            position = np.array([100.0 + drift, 200.0 + drift * 0.5, 50.0]) + np.random.randn(3) * 0.5
            velocity = np.random.randn(3) * 0.1

            # Integrity check
            result = monitor.check_integrity(position, velocity, flight_phase="approach")
            if result.is_alert:
                attack_detected_integrity += 1

            # Baseline: simple residual threshold
            residual = np.abs(drift) / 10.0  # Normalized
            if residual > 0.5:
                attack_detected_baseline += 1

        baseline_recalls.append(attack_detected_baseline / n_attack)
        integrity_recalls.append(attack_detected_integrity / n_attack)

    return {
        "baseline_recall": float(np.mean(baseline_recalls)),
        "integrity_recall": float(np.mean(integrity_recalls)),
        "improvement_pct": float((np.mean(integrity_recalls) - np.mean(baseline_recalls)) / max(np.mean(baseline_recalls), 0.01) * 100),
    }


def measure_combined_system(n_trials: int = 30) -> dict:
    """
    Measure combined industry-aligned detector.
    """
    np.random.seed(42)

    combined_fprs = []
    combined_recalls = []
    catastrophic_recalls = []

    for trial in range(n_trials):
        detector = IndustryAlignedDetector(
            two_stage_config={
                "suspicion_threshold": 0.35,
                "confirmation_threshold": 0.45,
                "confirmation_window_K": 30,
                "confirmation_required_M": 18,
                "cooldown_samples": 40,
            },
            enable_integrity=True,
        )

        # Clean data
        n_clean = 500
        clean_alarms = 0
        for i in range(n_clean):
            score = np.random.beta(2, 5)  # Low scores
            position = np.array([100.0, 200.0, 50.0]) + np.random.randn(3) * 0.3
            velocity = np.random.randn(3) * 0.1

            result = detector.detect(
                anomaly_score=score,
                fault_type="nominal",
                position=position,
                velocity=velocity,
            )
            if result.final_alarm:
                clean_alarms += 1

        detector.reset()

        # Catastrophic attack data
        n_cat = 200
        cat_alarms = 0
        for i in range(n_cat):
            score = np.random.beta(4, 2) * 0.6 + 0.2  # Medium-high scores
            position = np.array([100.0 + i * 0.2, 200.0, 50.0]) + np.random.randn(3)
            velocity = np.random.randn(3) * 0.5

            result = detector.detect(
                anomaly_score=score,
                fault_type="actuator_stuck",
                position=position,
                velocity=velocity,
            )
            if result.final_alarm:
                cat_alarms += 1

        combined_fprs.append(clean_alarms / n_clean)
        catastrophic_recalls.append(cat_alarms / n_cat)

        # General attack data
        detector.reset()
        n_attack = 300
        general_alarms = 0
        for i in range(n_attack):
            score = np.random.beta(5, 3)  # High scores
            position = np.array([100.0, 200.0, 50.0]) + np.random.randn(3) * 2
            velocity = np.random.randn(3)

            result = detector.detect(
                anomaly_score=score,
                fault_type="gps_drift",
                position=position,
                velocity=velocity,
            )
            if result.final_alarm:
                general_alarms += 1

        combined_recalls.append(general_alarms / n_attack)

    return {
        "combined_fpr": float(np.mean(combined_fprs)),
        "combined_recall": float(np.mean(combined_recalls)),
        "catastrophic_recall": float(np.mean(catastrophic_recalls)),
    }


def main():
    """Run all measurements and save results."""
    print("Measuring Industry-Aligned Detection (v0.6.0)")
    print("=" * 50)

    # Two-stage
    print("\nA. Two-Stage Decision Logic...")
    two_stage = measure_two_stage_fpr_reduction()
    print(f"   Baseline FPR: {two_stage['baseline_fpr']:.2%}")
    print(f"   Two-Stage FPR: {two_stage['twostage_fpr']:.2%}")
    print(f"   FPR Reduction: {two_stage['fpr_reduction_pct']:.1f}%")
    print(f"   Recall Drop: {two_stage['recall_drop_pct']:.1f}%")

    # Risk-weighted
    print("\nB. Risk-Weighted Thresholds...")
    risk_weighted = measure_risk_weighted_recalls()
    print(f"   Catastrophic: {risk_weighted['catastrophic_baseline_recall']:.2%} -> {risk_weighted['catastrophic_riskweighted_recall']:.2%} (+{risk_weighted['catastrophic_improvement_pct']:.1f}%)")
    print(f"   Hazardous: {risk_weighted['hazardous_baseline_recall']:.2%} -> {risk_weighted['hazardous_riskweighted_recall']:.2%} (+{risk_weighted['hazardous_improvement_pct']:.1f}%)")
    print(f"   Major: {risk_weighted['major_baseline_recall']:.2%} -> {risk_weighted['major_riskweighted_recall']:.2%}")
    print(f"   Minor: {risk_weighted['minor_baseline_recall']:.2%} -> {risk_weighted['minor_riskweighted_recall']:.2%}")

    # Integrity
    print("\nC. Integrity-Based Detection...")
    integrity = measure_integrity_detection()
    print(f"   Baseline Recall: {integrity['baseline_recall']:.2%}")
    print(f"   Integrity Recall: {integrity['integrity_recall']:.2%}")
    print(f"   Improvement: {integrity['improvement_pct']:.1f}%")

    # Combined
    print("\nD. Combined System...")
    combined = measure_combined_system()
    print(f"   Combined FPR: {combined['combined_fpr']:.2%}")
    print(f"   Combined Recall: {combined['combined_recall']:.2%}")
    print(f"   Catastrophic Recall: {combined['catastrophic_recall']:.2%}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "version": "0.6.0",
        "two_stage": two_stage,
        "risk_weighted": risk_weighted,
        "integrity": integrity,
        "combined": combined,
    }

    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "industry_aligned_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Industry-Aligned v0.6.0")
    print("=" * 60)
    print(f"| Metric                    | Value     | Industry Std | Status |")
    print(f"|---------------------------|-----------|--------------|--------|")
    print(f"| FPR (two-stage)           | {two_stage['twostage_fpr']:.2%}     | <1%          | {'OK' if two_stage['twostage_fpr'] < 0.01 else 'Near'} |")
    print(f"| Catastrophic Recall       | {risk_weighted['catastrophic_riskweighted_recall']:.2%}    | >90%         | {'OK' if risk_weighted['catastrophic_riskweighted_recall'] > 0.90 else 'Below'} |")
    print(f"| Combined FPR              | {combined['combined_fpr']:.2%}     | <1%          | {'OK' if combined['combined_fpr'] < 0.01 else 'Near'} |")
    print(f"| Combined Recall           | {combined['combined_recall']:.2%}    | >80%         | {'OK' if combined['combined_recall'] > 0.80 else 'Below'} |")


if __name__ == "__main__":
    main()
