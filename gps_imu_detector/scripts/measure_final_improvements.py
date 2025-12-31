"""
Measure PRECISE improvements from v0.5.1 final improvements.

All numbers are measured, not projected.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gps_imu_detector.src.final_improvements import (
    FaultPersistenceScorer,
    CostAwareThresholder,
    FaultClass,
    AsymmetricThresholds,
    TTDAnalyzer,
    FinalDetector,
    ControllerPredictor,
    CrossAxisCouplingChecker,
)


def generate_attack_scenario(n_clean=500, n_attack=500, attack_strength=0.6, seed=42):
    """Generate clean + attack scenario with realistic score distributions."""
    np.random.seed(seed)

    # Clean: mostly low scores with occasional spikes (transients)
    clean_base = np.random.beta(2, 8, n_clean) * 0.4  # 0-0.4 range
    # Add transient spikes (10% of clean samples)
    n_transients = int(n_clean * 0.1)
    transient_idx = np.random.choice(n_clean, n_transients, replace=False)
    clean_base[transient_idx] = np.random.uniform(0.5, 0.8, n_transients)
    clean_scores = clean_base

    # Attack: scores ramp up then stay elevated
    ramp_length = min(50, n_attack // 4)
    ramp = np.linspace(0.3, 0.7, ramp_length)
    sustained = np.random.beta(5, 3, n_attack - ramp_length) * 0.4 + 0.5  # 0.5-0.9 range
    attack_scores = np.concatenate([ramp, sustained]) * attack_strength

    scores = np.concatenate([clean_scores, attack_scores])
    labels = np.concatenate([np.zeros(n_clean), np.ones(n_attack)])

    return scores, labels


def measure_persistence_improvement():
    """Measure improvement 1: Persistence scoring - filters transient false alarms."""
    print("\n## 1. Fault Persistence Scoring (Transient Filtering)")
    print("-" * 50)

    n_trials = 50
    baseline_fprs = []
    baseline_recalls = []
    persistence_fprs = []
    persistence_recalls = []

    for trial in range(n_trials):
        scores, labels = generate_attack_scenario(seed=trial)
        threshold = 0.45

        # Process each segment independently (reset between clean and attack)
        clean_scores = scores[labels == 0]
        attack_scores = scores[labels == 1]

        # Baseline: raw threshold
        baseline_clean_det = clean_scores > threshold
        baseline_attack_det = attack_scores > threshold
        baseline_fprs.append(np.mean(baseline_clean_det))
        baseline_recalls.append(np.mean(baseline_attack_det))

        # Persistence on CLEAN data (should filter transients)
        scorer = FaultPersistenceScorer(k=3, n=8, base_threshold=threshold)
        pers_clean_det = []
        for s in clean_scores:
            result = scorer.update(s)
            pers_clean_det.append(result.is_persistent)
        persistence_fprs.append(np.mean(pers_clean_det))

        # Persistence on ATTACK data (should still detect sustained)
        scorer.reset()
        pers_attack_det = []
        for s in attack_scores:
            result = scorer.update(s)
            pers_attack_det.append(result.is_persistent)
        persistence_recalls.append(np.mean(pers_attack_det))

    # Compute means
    base_fpr = np.mean(baseline_fprs)
    base_recall = np.mean(baseline_recalls)
    pers_fpr = np.mean(persistence_fprs)
    pers_recall = np.mean(persistence_recalls)

    fpr_reduction = (base_fpr - pers_fpr) / base_fpr * 100 if base_fpr > 0 else 0
    recall_change = (pers_recall - base_recall) * 100

    print(f"Baseline FPR:           {base_fpr:.4f} ({base_fpr*100:.2f}%)")
    print(f"Persistence FPR:        {pers_fpr:.4f} ({pers_fpr*100:.2f}%)")
    print(f"FPR Reduction:          {fpr_reduction:+.1f}%")
    print(f"Baseline Recall:        {base_recall:.4f} ({base_recall*100:.2f}%)")
    print(f"Persistence Recall:     {pers_recall:.4f} ({pers_recall*100:.2f}%)")
    print(f"Recall Change:          {recall_change:+.2f}%")

    return {
        "baseline_fpr": float(base_fpr),
        "persistence_fpr": float(pers_fpr),
        "fpr_reduction_pct": float(fpr_reduction),
        "baseline_recall": float(base_recall),
        "persistence_recall": float(pers_recall),
        "recall_change_pct": float(recall_change),
    }


def measure_asymmetric_threshold_improvement():
    """Measure improvement 2: Cost-aware asymmetric thresholds."""
    print("\n## 2. Cost-Aware Asymmetric Thresholds")
    print("-" * 50)

    n_trials = 50
    thresholder = CostAwareThresholder()

    # Test on actuator vs sensor faults
    uniform_recalls = {"actuator": [], "sensor": []}
    asymmetric_recalls = {"actuator": [], "sensor": []}

    uniform_threshold = 0.5

    for trial in range(n_trials):
        # Actuator faults (harder to detect, more critical)
        np.random.seed(trial)
        actuator_scores = np.random.beta(3, 3, 100)  # Medium-difficulty

        # Uniform threshold
        uniform_det = actuator_scores > uniform_threshold
        uniform_recalls["actuator"].append(np.mean(uniform_det))

        # Asymmetric threshold (lower for actuator)
        asymmetric_det = [thresholder.detect(s, FaultClass.ACTUATOR).is_detected for s in actuator_scores]
        asymmetric_recalls["actuator"].append(np.mean(asymmetric_det))

        # Sensor faults (easier)
        sensor_scores = np.random.beta(4, 2, 100)

        uniform_det = sensor_scores > uniform_threshold
        uniform_recalls["sensor"].append(np.mean(uniform_det))

        asymmetric_det = [thresholder.detect(s, FaultClass.SENSOR).is_detected for s in sensor_scores]
        asymmetric_recalls["sensor"].append(np.mean(asymmetric_det))

    # Results
    actuator_uniform = np.mean(uniform_recalls["actuator"])
    actuator_asymmetric = np.mean(asymmetric_recalls["actuator"])
    sensor_uniform = np.mean(uniform_recalls["sensor"])
    sensor_asymmetric = np.mean(asymmetric_recalls["sensor"])

    actuator_improvement = (actuator_asymmetric - actuator_uniform) * 100

    print(f"Actuator (uniform t=0.5):     {actuator_uniform:.4f} ({actuator_uniform*100:.2f}%)")
    print(f"Actuator (asymmetric t=0.3):  {actuator_asymmetric:.4f} ({actuator_asymmetric*100:.2f}%)")
    print(f"Actuator Improvement:         {actuator_improvement:+.2f}%")
    print(f"Sensor (uniform t=0.5):       {sensor_uniform:.4f} ({sensor_uniform*100:.2f}%)")
    print(f"Sensor (asymmetric t=0.45):   {sensor_asymmetric:.4f} ({sensor_asymmetric*100:.2f}%)")

    return {
        "actuator_uniform_recall": float(actuator_uniform),
        "actuator_asymmetric_recall": float(actuator_asymmetric),
        "actuator_improvement_pct": float(actuator_improvement),
        "sensor_uniform_recall": float(sensor_uniform),
        "sensor_asymmetric_recall": float(sensor_asymmetric),
    }


def measure_ttd_metrics():
    """Measure improvement 3: TTD metrics."""
    print("\n## 3. Time-to-Detection Metrics")
    print("-" * 50)

    analyzer = TTDAnalyzer(dt=0.005)  # 200 Hz

    n_trials = 50
    ttd_medians = []
    ttd_95s = []

    for trial in range(n_trials):
        np.random.seed(trial)

        # Generate scenario with fault onset at sample 200
        n = 500
        fault_onset = 200

        # Scores ramp up after fault
        scores = np.concatenate([
            np.random.beta(2, 5, fault_onset),  # Clean
            np.random.beta(4, 2, n - fault_onset),  # Attack
        ])
        labels = np.concatenate([np.zeros(fault_onset), np.ones(n - fault_onset)])

        metrics = analyzer.compute_ttd(scores, labels, np.array([fault_onset]), threshold=0.5)

        if metrics.n_detected > 0:
            ttd_medians.append(metrics.median_ttd)
            ttd_95s.append(metrics.ttd_95)

    median_ttd = np.mean(ttd_medians)
    median_ttd_ms = median_ttd * 5  # 200 Hz
    ttd_95 = np.mean(ttd_95s)
    ttd_95_ms = ttd_95 * 5

    print(f"Median TTD:             {median_ttd:.1f} samples ({median_ttd_ms:.1f} ms)")
    print(f"TTD@95%:                {ttd_95:.1f} samples ({ttd_95_ms:.1f} ms)")
    print(f"Detection Rate:         {len(ttd_medians)/n_trials*100:.1f}%")

    return {
        "median_ttd_samples": float(median_ttd),
        "median_ttd_ms": float(median_ttd_ms),
        "ttd_95_samples": float(ttd_95),
        "ttd_95_ms": float(ttd_95_ms),
        "detection_rate": float(len(ttd_medians) / n_trials),
    }


def measure_controller_predictor_improvement():
    """Measure improvement 4: Controller-in-loop prediction."""
    print("\n## 4. Controller-in-Loop Prediction")
    print("-" * 50)

    n_trials = 30
    baseline_recalls = []
    controller_recalls = []

    for trial in range(n_trials):
        np.random.seed(trial)

        # Generate nominal training data
        n_train = 500
        states = np.random.randn(n_train, 6) * 0.5
        state_dots = np.random.randn(n_train, 6) * 0.1
        controls = states[:, :4] * 0.3 + 5 + np.random.randn(n_train, 4) * 0.2

        predictor = ControllerPredictor(state_dim=6, control_dim=4, threshold=0.5)
        predictor.fit(states, state_dots, controls)

        # Test on degraded actuator (control doesn't match expected)
        n_test = 100

        # Normal test
        normal_states = np.random.randn(n_test, 6) * 0.5
        normal_dots = np.random.randn(n_test, 6) * 0.1
        normal_controls = normal_states[:, :4] * 0.3 + 5 + np.random.randn(n_test, 4) * 0.2

        # Degraded actuator: control is offset
        degraded_controls = normal_controls + np.random.randn(n_test, 4) * 2  # Large offset

        # Baseline: just check if control magnitude is unusual
        baseline_threshold = np.percentile(np.linalg.norm(controls, axis=1), 95)
        baseline_det = np.linalg.norm(degraded_controls, axis=1) > baseline_threshold
        baseline_recalls.append(np.mean(baseline_det))

        # Controller predictor
        controller_det = []
        for i in range(n_test):
            result = predictor.predict(normal_states[i], normal_dots[i], degraded_controls[i])
            controller_det.append(result.is_anomalous)
        controller_recalls.append(np.mean(controller_det))

    baseline_recall = np.mean(baseline_recalls)
    controller_recall = np.mean(controller_recalls)
    improvement = (controller_recall - baseline_recall) * 100

    print(f"Baseline Recall:        {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
    print(f"Controller Recall:      {controller_recall:.4f} ({controller_recall*100:.2f}%)")
    print(f"Improvement:            {improvement:+.2f}%")

    return {
        "baseline_recall": float(baseline_recall),
        "controller_recall": float(controller_recall),
        "improvement_pct": float(improvement),
    }


def measure_cross_axis_improvement():
    """Measure improvement 5: Cross-axis coupling."""
    print("\n## 5. Cross-Axis Coupling Consistency")
    print("-" * 50)

    n_trials = 30
    baseline_recalls = []
    coupling_recalls = []

    for trial in range(n_trials):
        np.random.seed(trial)
        n = 200

        # Generate coupled nominal data
        t = np.linspace(0, 10, n)
        omega_nominal = np.column_stack([np.sin(t), np.cos(t), np.zeros(n)])
        vel_nominal = np.column_stack([
            np.zeros(n),
            np.sin(t) * 0.8,  # Coupled with omega_x
            np.zeros(n),
        ])
        acc_nominal = np.column_stack([
            np.zeros(n),
            np.zeros(n),
            np.cos(t) * 0.8,  # Coupled with omega_y
        ])

        checker = CrossAxisCouplingChecker()
        checker.calibrate(omega_nominal, vel_nominal, acc_nominal)

        # Attack: break coupling (omega doesn't correlate with velocity)
        omega_attack = omega_nominal.copy()
        vel_attack = np.random.randn(n, 3) * 0.5  # Uncorrelated!
        acc_attack = np.random.randn(n, 3) * 0.5

        # Baseline: just check velocity magnitude
        vel_mag = np.linalg.norm(vel_attack, axis=1)
        baseline_det = np.mean(vel_mag) > np.mean(np.linalg.norm(vel_nominal, axis=1)) * 0.5
        baseline_recalls.append(float(baseline_det))

        # Cross-axis coupling
        result = checker.check_coupling(omega_attack, vel_attack, acc_attack)
        coupling_recalls.append(float(result.is_anomalous))

    baseline_recall = np.mean(baseline_recalls)
    coupling_recall = np.mean(coupling_recalls)
    improvement = (coupling_recall - baseline_recall) * 100

    print(f"Baseline Recall:        {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
    print(f"Cross-Axis Recall:      {coupling_recall:.4f} ({coupling_recall*100:.2f}%)")
    print(f"Improvement:            {improvement:+.2f}%")

    return {
        "baseline_recall": float(baseline_recall),
        "coupling_recall": float(coupling_recall),
        "improvement_pct": float(improvement),
    }


def measure_combined_improvement():
    """Measure combined effect of all improvements."""
    print("\n## Combined Effect (Persistence + Asymmetric)")
    print("-" * 50)

    n_trials = 50
    baseline_recalls = []
    combined_recalls = []
    baseline_fprs = []
    combined_fprs = []

    for trial in range(n_trials):
        scores, labels = generate_attack_scenario(seed=trial, attack_strength=0.5)

        # Baseline: uniform threshold
        baseline_det = scores > 0.5
        baseline_recalls.append(np.mean(baseline_det[labels == 1]))
        baseline_fprs.append(np.mean(baseline_det[labels == 0]))

        # Combined: persistence + asymmetric
        detector = FinalDetector(
            persistence_k=3,
            persistence_n=10,
            base_threshold=0.4,
            thresholds=AsymmetricThresholds(actuator=0.3, default=0.4),
        )

        combined_det = []
        for s in scores:
            result = detector.detect(s, FaultClass.ACTUATOR)
            combined_det.append(result.is_final_detection)
        combined_det = np.array(combined_det)

        combined_recalls.append(np.mean(combined_det[labels == 1]))
        combined_fprs.append(np.mean(combined_det[labels == 0]))

    baseline_recall = np.mean(baseline_recalls)
    combined_recall = np.mean(combined_recalls)
    baseline_fpr = np.mean(baseline_fprs)
    combined_fpr = np.mean(combined_fprs)

    recall_improvement = (combined_recall - baseline_recall) * 100
    fpr_reduction = (baseline_fpr - combined_fpr) / baseline_fpr * 100 if baseline_fpr > 0 else 0

    print(f"Baseline Recall:        {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
    print(f"Combined Recall:        {combined_recall:.4f} ({combined_recall*100:.2f}%)")
    print(f"Recall Improvement:     {recall_improvement:+.2f}%")
    print(f"Baseline FPR:           {baseline_fpr:.4f} ({baseline_fpr*100:.2f}%)")
    print(f"Combined FPR:           {combined_fpr:.4f} ({combined_fpr*100:.2f}%)")
    print(f"FPR Reduction:          {fpr_reduction:.1f}%")

    return {
        "baseline_recall": float(baseline_recall),
        "combined_recall": float(combined_recall),
        "recall_improvement_pct": float(recall_improvement),
        "baseline_fpr": float(baseline_fpr),
        "combined_fpr": float(combined_fpr),
        "fpr_reduction_pct": float(fpr_reduction),
    }


def main():
    print("=" * 60)
    print("PRECISE MEASURED IMPROVEMENTS (v0.5.1)")
    print("=" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "version": "0.5.1",
        "note": "All values are MEASURED across 30-50 trials each",
    }

    results["persistence"] = measure_persistence_improvement()
    results["asymmetric_thresholds"] = measure_asymmetric_threshold_improvement()
    results["ttd"] = measure_ttd_metrics()
    results["controller_predictor"] = measure_controller_predictor_improvement()
    results["cross_axis"] = measure_cross_axis_improvement()
    results["combined"] = measure_combined_improvement()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF PRECISE MEASUREMENTS")
    print("=" * 60)
    print(f"{'Improvement':<30} {'Metric':<20} {'Value':>12}")
    print("-" * 62)
    print(f"{'Persistence Scoring':<30} {'FPR Reduction':<20} {results['persistence']['fpr_reduction_pct']:>+10.1f}%")
    print(f"{'Persistence Scoring':<30} {'Recall Change':<20} {results['persistence']['recall_change_pct']:>+10.2f}%")
    print(f"{'Asymmetric Thresholds':<30} {'Actuator +Recall':<20} {results['asymmetric_thresholds']['actuator_improvement_pct']:>+10.2f}%")
    print(f"{'TTD Metrics':<30} {'Median TTD':<20} {results['ttd']['median_ttd_ms']:>10.1f} ms")
    print(f"{'TTD Metrics':<30} {'TTD@95%':<20} {results['ttd']['ttd_95_ms']:>10.1f} ms")
    print(f"{'Controller Predictor':<30} {'Recall Improvement':<20} {results['controller_predictor']['improvement_pct']:>+10.2f}%")
    print(f"{'Cross-Axis Coupling':<30} {'Recall Improvement':<20} {results['cross_axis']['improvement_pct']:>+10.2f}%")
    print(f"{'COMBINED':<30} {'Recall Improvement':<20} {results['combined']['recall_improvement_pct']:>+10.2f}%")
    print(f"{'COMBINED':<30} {'FPR Reduction':<20} {results['combined']['fpr_reduction_pct']:>+10.1f}%")

    # Save
    output_dir = Path(__file__).parent.parent / "results/final_improvements"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "measured_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'measured_results.json'}")

    return results


if __name__ == "__main__":
    main()
