"""
Evaluate Sequence-PINN detector on all 30 synthetic attack types.

Compares performance with single-step PINN, especially on temporal attacks
that the single-step model struggles with.

Usage:
    python scripts/security/evaluate_sequence_detector.py
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
from pinn_dynamics import SequencePINN

sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic_attacks import SyntheticAttackGenerator


def load_model_and_scalers(model_dir: Path, device: str = "cpu"):
    """Load trained model and scalers."""
    config_path = model_dir / "sequence_detector_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    model = SequencePINN(
        sequence_length=config["sequence_length"],
        hidden_size=config["hidden_size"],
        num_lstm_layers=config.get("num_lstm_layers", 2),
        fc_hidden_size=config.get("fc_hidden_size", 256),
        dropout=config.get("dropout", 0.1),
    )

    model_path = model_dir / "sequence_pinn_detector.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    scaler_path = model_dir / "scalers_sequence.pkl"
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    return model, scalers, config


def create_sequences(
    states: np.ndarray,
    controls: np.ndarray,
    sequence_length: int,
    labels: np.ndarray = None,
):
    """Create sequences from time series data."""
    n_samples = len(states) - sequence_length

    sequences = []
    targets = []
    seq_labels = []

    for i in range(n_samples):
        seq_states = states[i : i + sequence_length]
        seq_controls = controls[i : i + sequence_length]
        sequence = np.concatenate([seq_states, seq_controls], axis=1)
        sequences.append(sequence)
        targets.append(states[i + sequence_length])

        if labels is not None:
            seq_labels.append(labels[i + sequence_length])

    return np.array(sequences), np.array(targets), np.array(seq_labels) if labels is not None else None


def evaluate_attack(
    model,
    attack_data: pd.DataFrame,
    scalers: dict,
    sequence_length: int,
    threshold: float,
    device: str = "cpu",
):
    """Evaluate detection on a single attack type."""
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    states = attack_data[state_cols].values
    controls = attack_data[control_cols].values
    labels = attack_data["label"].values

    # Scale data
    combined = np.concatenate([states, controls], axis=1)
    combined_scaled = scalers["scaler_X"].transform(combined)
    states_scaled = combined_scaled[:, :12]
    controls_scaled = combined_scaled[:, 12:]

    # Create sequences
    sequences, targets, seq_labels = create_sequences(
        states_scaled, controls_scaled, sequence_length, labels
    )

    if len(sequences) == 0:
        return None

    # Predict
    model.eval()
    batch_size = 256

    all_errors = []
    for i in range(0, len(sequences), batch_size):
        batch_seq = torch.FloatTensor(sequences[i : i + batch_size]).to(device)
        batch_target = torch.FloatTensor(targets[i : i + batch_size]).to(device)

        with torch.no_grad():
            predictions = model(batch_seq)
            errors = torch.norm(predictions - batch_target, dim=1).cpu().numpy()
            all_errors.extend(errors)

    all_errors = np.array(all_errors)
    predictions = (all_errors > threshold).astype(int)

    # Metrics
    tp = np.sum((predictions == 1) & (seq_labels == 1))
    fp = np.sum((predictions == 1) & (seq_labels == 0))
    tn = np.sum((predictions == 0) & (seq_labels == 0))
    fn = np.sum((predictions == 0) & (seq_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "samples": len(seq_labels),
        "attack_samples": int(np.sum(seq_labels)),
        "normal_samples": int(np.sum(seq_labels == 0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "mean_error": float(np.mean(all_errors)),
        "max_error": float(np.max(all_errors)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate sequence detector")
    parser.add_argument("--model-dir", type=str, default="models/security", help="Model directory")
    parser.add_argument("--data", type=str, default="data/euroc", help="Path to EuRoC data")
    parser.add_argument("--output", type=str, default="research/security/sequence_results", help="Output directory")
    parser.add_argument("--threshold-percentile", type=float, default=95.0, help="Threshold percentile")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("SEQUENCE-PINN DETECTOR EVALUATION")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    model, scalers, config = load_model_and_scalers(model_dir, device)
    sequence_length = config["sequence_length"]
    threshold = config["detection_threshold"]
    print(f"  Sequence length: {sequence_length}")
    print(f"  Detection threshold: {threshold:.4f}")

    # Load EuRoC data
    print("\n[2/4] Loading EuRoC data...")
    csv_files = list(data_path.glob("*.csv"))
    euroc_file = None
    for name in ["all_sequences.csv", "euroc_processed.csv"]:
        if (data_path / name).exists():
            euroc_file = data_path / name
            break
    if euroc_file is None and csv_files:
        euroc_file = csv_files[0]

    df = pd.read_csv(euroc_file)
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

    # Generate attacks
    print("\n[3/4] Generating and evaluating attacks...")
    generator = SyntheticAttackGenerator(df, seed=42, randomize=False)
    attacks = generator.generate_all_attacks(handle_nan=True)

    results = {}
    print()

    # Categories for organized output
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
        metrics = evaluate_attack(
            model, attack_data, scalers, sequence_length, threshold, device
        )

        if metrics is None:
            print(f"  {attack_name:30s} | SKIPPED (too short)")
            continue

        results[attack_name] = metrics

        if attack_name == "clean":
            print(f"  {attack_name:30s} | FPR: {metrics['fpr']*100:5.1f}% | (baseline)")
        else:
            print(f"  {attack_name:30s} | Recall: {metrics['recall']*100:5.1f}% | "
                  f"F1: {metrics['f1']*100:5.1f}% | FPR: {metrics['fpr']*100:5.1f}%")

    # Aggregate results
    print("\n[4/4] Computing aggregate metrics...")

    # Overall metrics
    all_tp = sum(r["true_positives"] for k, r in results.items() if k != "clean")
    all_fp = sum(r["false_positives"] for k, r in results.items() if k != "clean")
    all_tn = sum(r["true_negatives"] for k, r in results.items() if k != "clean")
    all_fn = sum(r["false_negatives"] for k, r in results.items() if k != "clean")

    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
        if (overall_precision + overall_recall) > 0 else 0

    clean_fpr = results["clean"]["fpr"] if "clean" in results else 0

    # Category-level results
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

    # Summary
    summary = {
        "model_type": "SequencePINN",
        "sequence_length": sequence_length,
        "overall_precision": float(overall_precision),
        "overall_recall": float(overall_recall),
        "overall_f1": float(overall_f1),
        "clean_fpr": float(clean_fpr),
        "total_attacks": len(results) - 1,
        "threshold": float(threshold),
        "category_results": category_results,
        "per_attack_results": results,
    }

    # Save results
    results_path = output_path / "sequence_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nOverall Metrics:")
    print(f"  Precision: {overall_precision*100:.1f}%")
    print(f"  Recall:    {overall_recall*100:.1f}%")
    print(f"  F1 Score:  {overall_f1*100:.1f}%")
    print(f"\nClean Data FPR: {clean_fpr*100:.1f}%")

    # Highlight temporal attacks (key improvement area)
    print("\nTemporal Attack Performance (key improvement area):")
    for attack_name in ["replay_attack", "time_delay", "sensor_dropout"]:
        if attack_name in results:
            r = results[attack_name]
            print(f"  {attack_name:20s}: {r['recall']*100:5.1f}% recall")

    # Best/worst
    attack_results = {k: v for k, v in results.items() if k != "clean" and v["attack_samples"] > 0}
    if attack_results:
        best = max(attack_results.items(), key=lambda x: x[1]["recall"])
        worst = min(attack_results.items(), key=lambda x: x[1]["recall"])
        print(f"\nBest detected:  {best[0]} ({best[1]['recall']*100:.1f}%)")
        print(f"Worst detected: {worst[0]} ({worst[1]['recall']*100:.1f}%)")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
