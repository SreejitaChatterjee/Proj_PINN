"""
Evaluate PINN Residual-Based Detector on synthetic attacks.

Uses the trained PINN model's prediction residuals for detection,
with multi-step accumulation and minimal temporal pattern detection.

Usage:
    python scripts/security/evaluate_pinn_residual_detector.py
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinn_dynamics import QuadrotorPINN, Predictor
from pinn_dynamics.security.pinn_residual_detector import PINNResidualEnsemble

sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic_attacks import SyntheticAttackGenerator


def evaluate_attack(
    detector: PINNResidualEnsemble,
    attack_data: pd.DataFrame,
    attack_name: str,
) -> dict:
    """Evaluate detector on a single attack type."""
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    states = attack_data[state_cols].values
    controls = attack_data[control_cols].values
    labels = attack_data["label"].values

    # Reset detector
    detector.reset()

    # Warm up on first 200 samples (assume clean)
    warmup_size = min(200, len(states) // 3)
    for i in range(warmup_size):
        if i < len(states) - 1:
            detector.detect(states[i], controls[i], states[i + 1])

    # Evaluate on remaining
    predictions = []
    triggered_by = {"residual": 0, "temporal": 0, "both": 0, "none": 0}

    for i in range(warmup_size, len(states) - 1):
        result = detector.detect(states[i], controls[i], states[i + 1])
        predictions.append(1 if result.is_anomaly else 0)
        triggered_by[result.triggered_by] += 1

    predictions = np.array(predictions)
    true_labels = labels[warmup_size:-1]  # -1 because we use i+1 for next_state

    # Ensure same length
    min_len = min(len(predictions), len(true_labels))
    predictions = predictions[:min_len]
    true_labels = true_labels[:min_len]

    # Metrics
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

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
        "triggered_by": triggered_by,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PINN Residual Detector")
    parser.add_argument("--data", type=str, default="data/euroc", help="EuRoC data path")
    parser.add_argument("--model", type=str, default="models/euroc_pinn.pth", help="Trained PINN model")
    parser.add_argument("--scalers", type=str, default="models/euroc_scalers.pkl", help="Scalers file")
    parser.add_argument("--output", type=str, default="research/security/pinn_residual_results", help="Output dir")
    parser.add_argument("--window-size", type=int, default=20, help="Residual accumulation window")
    parser.add_argument("--threshold-percentile", type=float, default=99.0, help="Threshold percentile")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    scalers_path = Path(args.scalers)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PINN RESIDUAL-BASED DETECTOR EVALUATION")
    print("=" * 70)

    # Load model
    print("\n[1/5] Loading trained PINN model...")

    # Prefer security model trained with QuadrotorPINN
    security_model = Path("models/security/pinn_synthetic_detector.pth")
    security_scalers = Path("models/security/scalers_synthetic.pkl")

    if security_model.exists():
        model_path = security_model
        scalers_path = security_scalers
        print(f"  Using security model: {model_path}")
    elif not model_path.exists():
        print(f"  ERROR: No model found")
        print("  Please train a PINN first with: python scripts/security/train_synthetic_detector.py")
        return

    model = QuadrotorPINN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"  Loaded model from {model_path}")

    # Load scalers
    scaler_X, scaler_y = None, None
    if scalers_path.exists():
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)
            if isinstance(scalers, dict):
                scaler_X = scalers.get("scaler_X")
                scaler_y = scalers.get("scaler_y")
            elif isinstance(scalers, (list, tuple)) and len(scalers) == 2:
                scaler_X, scaler_y = scalers
        print(f"  Loaded scalers from {scalers_path}")
    else:
        print("  WARNING: No scalers found, using unscaled predictions")

    # Create predictor
    predictor = Predictor(model, scaler_X, scaler_y)

    # Load EuRoC data
    print("\n[2/5] Loading EuRoC data...")
    csv_file = None
    for name in ["all_sequences.csv", "euroc_processed.csv"]:
        if (data_path / name).exists():
            csv_file = data_path / name
            break
    if csv_file is None:
        csv_files = list(data_path.glob("*.csv"))
        if csv_files:
            csv_file = csv_files[0]

    if csv_file is None:
        print(f"  ERROR: No CSV files found in {data_path}")
        return

    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df):,} samples from {csv_file.name}")

    # Normalize columns
    for old, new in [("roll", "phi"), ("pitch", "theta"), ("yaw", "psi")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "thrust" not in df.columns:
        df["thrust"] = df["az"] + 9.81 if "az" in df.columns else 9.81
    for col in ["torque_x", "torque_y", "torque_z"]:
        if col not in df.columns:
            df[col] = 0.0

    # Create detector
    print("\n[3/5] Creating and calibrating PINN Residual Detector...")
    detector = PINNResidualEnsemble(
        predictor=predictor,
        window_size=args.window_size,
        threshold_percentile=args.threshold_percentile,
    )

    # Calibrate on clean data
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    calib_size = min(20000, len(df) - 1)
    states = df[state_cols].values[:calib_size]
    controls = df[control_cols].values[:calib_size]
    next_states = df[state_cols].values[1:calib_size + 1]

    detector.calibrate(states, controls, next_states)

    # Generate attacks
    print("\n[4/5] Generating and evaluating attacks...")
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
    print("\n[5/5] Computing aggregate metrics...")
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
    all_tn = sum(r["true_negatives"] for k, r in results.items() if k != "clean")

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

    # Detector breakdown
    print("\n** DETECTOR CONTRIBUTION **")
    total_residual = sum(r["triggered_by"]["residual"] for r in results.values())
    total_temporal = sum(r["triggered_by"]["temporal"] for r in results.values())
    total_both = sum(r["triggered_by"]["both"] for r in results.values())
    print(f"  Residual-only detections: {total_residual:,}")
    print(f"  Temporal-only detections: {total_temporal:,}")
    print(f"  Both detectors agreed:    {total_both:,}")

    # Temporal attack performance
    print("\n** TEMPORAL ATTACK PERFORMANCE **")
    for attack_name in ["replay_attack", "time_delay", "sensor_dropout", "gps_freeze"]:
        if attack_name in results:
            r = results[attack_name]
            print(f"  {attack_name:20s}: {r['recall']*100:5.1f}% recall | "
                  f"Temporal: {r['triggered_by']['temporal']} | "
                  f"Residual: {r['triggered_by']['residual']}")

    # Save results
    summary = {
        "model_type": "PINNResidualEnsemble",
        "window_size": args.window_size,
        "threshold_percentile": args.threshold_percentile,
        "overall_precision": float(overall_precision),
        "overall_recall": float(overall_recall),
        "overall_f1": float(overall_f1),
        "clean_fpr": float(clean_fpr),
        "category_results": category_results,
        "per_attack_results": results,
    }

    results_path = output_path / "pinn_residual_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
