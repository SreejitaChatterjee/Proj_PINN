"""
Evaluate Ensemble Detector on all 30 synthetic attack types.

Combines:
1. Cross-sensor consistency (GPS vs IMU)
2. Statistical fingerprinting (noise patterns)
3. Sequence similarity (replay detection)

Usage:
    python scripts/security/evaluate_ensemble_detector.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinn_dynamics.security.ensemble_detector import EnsembleDetector

sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic_attacks import SyntheticAttackGenerator


def evaluate_attack(
    detector: EnsembleDetector,
    attack_data: pd.DataFrame,
    attack_name: str,
) -> dict:
    """Evaluate ensemble detector on a single attack type."""
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]

    states = attack_data[state_cols].values
    labels = attack_data["label"].values

    # Get acceleration if available (for cross-sensor)
    if all(c in attack_data.columns for c in ["ax", "ay", "az"]):
        accelerations = attack_data[["ax", "ay", "az"]].values
    else:
        accelerations = None

    # Reset detector for new attack
    detector.reset()

    # Warm up detector with first 500 samples (assume clean)
    # Increased from 100 to allow statistical detector to stabilize
    warmup_size = min(500, len(states) // 2)
    for i in range(warmup_size):
        imu_acc = accelerations[i] if accelerations is not None else None
        detector.detect(states[i], imu_acceleration=imu_acc)

    # Evaluate on remaining samples
    predictions = []
    for i in range(warmup_size, len(states)):
        imu_acc = accelerations[i] if accelerations is not None else None
        result = detector.detect(states[i], imu_acceleration=imu_acc)
        predictions.append(1 if result.is_anomaly else 0)

    predictions = np.array(predictions)
    true_labels = labels[warmup_size:]

    # Metrics
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Get detector breakdown
    summary = detector.get_detection_summary()

    return {
        "samples": len(true_labels),
        "attack_samples": int(np.sum(true_labels)),
        "normal_samples": int(np.sum(true_labels == 0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "detector_triggers": summary.get("by_detector", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble detector")
    parser.add_argument("--data", type=str, default="data/euroc", help="Path to EuRoC data")
    parser.add_argument("--output", type=str, default="research/security/ensemble_results", help="Output directory")
    parser.add_argument("--voting-threshold", type=float, default=0.3, help="Voting threshold")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ENSEMBLE DETECTOR EVALUATION")
    print("=" * 70)
    print(f"Voting threshold: {args.voting_threshold}")

    # Load EuRoC data
    print("\n[1/4] Loading EuRoC data...")
    csv_file = None
    for name in ["all_sequences.csv", "euroc_processed.csv"]:
        if (data_path / name).exists():
            csv_file = data_path / name
            break
    if csv_file is None:
        csv_files = list(data_path.glob("*.csv"))
        if csv_files:
            csv_file = csv_files[0]

    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df):,} samples")

    # Normalize columns
    for old, new in [("roll", "phi"), ("pitch", "theta"), ("yaw", "psi")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "thrust" not in df.columns:
        df["thrust"] = df["az"] + 9.81 if "az" in df.columns else 9.81
    for col in ["torque_x", "torque_y", "torque_z"]:
        if col not in df.columns:
            df[col] = 0.0

    # Create and calibrate detector with conservative settings
    print("\n[2/4] Creating and calibrating ensemble detector...")
    detector = EnsembleDetector(
        cross_sensor_weight=1.0,
        statistical_weight=1.0,
        similarity_weight=1.0,
        voting_threshold=0.5,  # Conservative - require majority
        min_detectors_agree=2,  # Require 2+ detectors for robustness
    )

    # Calibrate on clean data
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    calibration_data = df[state_cols].values[:10000]  # First 10K samples
    detector.calibrate(calibration_data)
    print("  Calibration complete")
    print(f"  Active detectors: cross_sensor, statistical, similarity")

    # Generate attacks
    print("\n[3/4] Generating and evaluating attacks...")
    generator = SyntheticAttackGenerator(df, seed=42, randomize=False)
    attacks = generator.generate_all_attacks(handle_nan=True)

    results = {}
    print()

    categories = {
        "GPS": ["gps_gradual_drift", "gps_sudden_jump", "gps_oscillating", "gps_meaconing",
                "gps_jamming", "gps_freeze", "gps_multipath"],
        "IMU": ["imu_constant_bias", "imu_gradual_drift", "imu_sinusoidal",
                "imu_noise_injection", "imu_scale_factor", "gyro_saturation", "accel_saturation"],
        "Mag/Baro": ["magnetometer_spoofing", "barometer_spoofing"],
        "Actuator": ["actuator_stuck", "actuator_degraded", "control_hijack", "thrust_manipulation"],
        "Coordinated": ["coordinated_gps_imu", "stealthy_coordinated"],
        "Temporal": ["replay_attack", "time_delay", "sensor_dropout"],
        "Stealth": ["adaptive_attack", "intermittent_attack", "slow_ramp", "resonance_attack", "false_data_injection"],
    }

    for attack_name, attack_data in attacks.items():
        metrics = evaluate_attack(detector, attack_data, attack_name)
        results[attack_name] = metrics

        if attack_name == "clean":
            print(f"  {attack_name:30s} | FPR: {metrics['fpr']*100:5.1f}% | (baseline)")
        else:
            print(f"  {attack_name:30s} | Recall: {metrics['recall']*100:5.1f}% | "
                  f"F1: {metrics['f1']*100:5.1f}% | FPR: {metrics['fpr']*100:5.1f}%")

    # Category results
    print("\n[4/4] Computing aggregate metrics...")
    print("\n" + "=" * 70)
    print("RESULTS BY CATEGORY")
    print("=" * 70)

    category_results = {}
    for cat_name, attack_list in categories.items():
        cat_recalls = []
        for attack_name in attack_list:
            if attack_name in results and results[attack_name]["attack_samples"] > 0:
                cat_recalls.append(results[attack_name]["recall"])

        if cat_recalls:
            avg_recall = np.mean(cat_recalls)
            category_results[cat_name] = avg_recall
            print(f"  {cat_name:15s}: {avg_recall*100:5.1f}% avg recall")

    # Overall metrics
    all_tp = sum(r["true_positives"] for k, r in results.items() if k != "clean")
    all_fp = sum(r["false_positives"] for k, r in results.items() if k != "clean")
    all_fn = sum(r["false_negatives"] for k, r in results.items() if k != "clean")

    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
        if (overall_precision + overall_recall) > 0 else 0

    clean_fpr = results["clean"]["fpr"] if "clean" in results else 0

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nOverall Metrics:")
    print(f"  Precision: {overall_precision*100:.1f}%")
    print(f"  Recall:    {overall_recall*100:.1f}%")
    print(f"  F1 Score:  {overall_f1*100:.1f}%")
    print(f"\nClean Data FPR: {clean_fpr*100:.1f}%")

    # Key improvements
    print("\n** KEY IMPROVEMENTS (vs single detectors) **")
    print("Temporal Attack Performance:")
    for attack_name in ["replay_attack", "time_delay", "sensor_dropout"]:
        if attack_name in results:
            r = results[attack_name]
            print(f"  {attack_name:20s}: {r['recall']*100:5.1f}% recall")

    # Save results
    summary = {
        "model_type": "EnsembleDetector",
        "voting_threshold": args.voting_threshold,
        "overall_precision": float(overall_precision),
        "overall_recall": float(overall_recall),
        "overall_f1": float(overall_f1),
        "clean_fpr": float(clean_fpr),
        "category_results": category_results,
        "per_attack_results": results,
    }

    results_path = output_path / "ensemble_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
