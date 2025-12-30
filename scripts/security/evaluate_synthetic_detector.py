"""
Evaluate PINN detector on synthetic attacks.

Tests the trained detector on all 30 attack types and reports:
- Per-attack-type detection rates
- Overall precision, recall, F1
- False positive rate on clean data

Usage:
    python scripts/security/evaluate_synthetic_detector.py
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

try:
    from pinn_dynamics import QuadrotorPINN
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pinn_dynamics import QuadrotorPINN

from generate_synthetic_attacks import SyntheticAttackGenerator


def load_model_and_scalers(model_dir: Path, device: str = "cpu"):
    """Load trained model and scalers."""
    # Load model
    model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)
    model_path = model_dir / "pinn_synthetic_detector.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load scalers
    scaler_path = model_dir / "scalers_synthetic.pkl"
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    # Load config
    config_path = model_dir / "synthetic_detector_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    return model, scalers, config


def prepare_data(df: pd.DataFrame, scalers: dict):
    """Prepare data for detection."""
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    states = df[state_cols].values
    controls = df[control_cols].values

    X = np.concatenate([states[:-1], controls[:-1]], axis=1)
    y = states[1:]
    labels = df["label"].values[1:]

    X_scaled = scalers["scaler_X"].transform(X)
    y_scaled = scalers["scaler_y"].transform(y)

    return X_scaled, y_scaled, labels


def detect_anomalies(model, X, y, scalers, threshold, device="cpu"):
    """
    Detect anomalies using prediction error.

    Returns:
        predictions: binary predictions (1=anomaly, 0=normal)
        errors: prediction errors
    """
    model.eval()

    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        y_pred_scaled = model(X_t).cpu().numpy()

    # Inverse transform
    y_orig = scalers["scaler_y"].inverse_transform(y)
    y_pred = scalers["scaler_y"].inverse_transform(y_pred_scaled)

    # Compute errors
    errors = np.linalg.norm(y_orig - y_pred, axis=1)

    # Threshold
    predictions = (errors > threshold).astype(int)

    return predictions, errors


def evaluate_attack(
    model, attack_data, scalers, threshold, device="cpu"
):
    """Evaluate detection on a single attack type."""
    X, y, labels = prepare_data(attack_data, scalers)
    predictions, errors = detect_anomalies(model, X, y, scalers, threshold, device)

    # Metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "samples": len(labels),
        "attack_samples": int(np.sum(labels)),
        "normal_samples": int(np.sum(labels == 0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic attack detector")
    parser.add_argument("--model-dir", type=str, default="models/security",
                        help="Model directory")
    parser.add_argument("--data", type=str, default="data/euroc",
                        help="Path to EuRoC data")
    parser.add_argument("--output", type=str, default="research/security/synthetic_results",
                        help="Output directory for results")
    parser.add_argument("--threshold-percentile", type=float, default=99.0,
                        help="Threshold percentile (default: 99)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("PINN SYNTHETIC ATTACK DETECTOR EVALUATION")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    model, scalers, config = load_model_and_scalers(model_dir, device)
    threshold = config["detection_threshold"]
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
    print(f"  Loaded {len(df):,} total samples")

    # =========================================================================
    # CRITICAL: Only use TEST sequences for evaluation (prevent data leakage)
    # =========================================================================
    test_sequences = config.get("sequence_split", {})
    if test_sequences and test_sequences.get("test_sequences"):
        test_seqs = test_sequences["test_sequences"]
        df = df[df["sequence"].isin(test_seqs)].reset_index(drop=True)  # Reset index for attack generator
        print(f"  Using held-out test sequences: {test_seqs}")
        print(f"  Test samples: {len(df):,}")
    else:
        print("  WARNING: No sequence split info found. Using all data (may cause overestimation).")

    # Normalize columns
    for old, new in [("roll", "phi"), ("pitch", "theta"), ("yaw", "psi")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "thrust" not in df.columns:
        df["thrust"] = df["az"] + 9.81 if "az" in df.columns else 9.81
    for col in ["torque_x", "torque_y", "torque_z"]:
        if col not in df.columns:
            df[col] = 0.0

    # Generate attacks ONLY from test sequences
    print("\n[3/4] Generating and evaluating attacks on held-out test data...")
    generator = SyntheticAttackGenerator(df, seed=42, randomize=False)
    attacks = generator.generate_all_attacks(handle_nan=True)

    results = {}
    print()

    for attack_name, attack_data in attacks.items():
        metrics = evaluate_attack(model, attack_data, scalers, threshold, device)
        results[attack_name] = metrics

        # Print summary
        if attack_name == "clean":
            print(f"  {attack_name:30s} | FPR: {metrics['fpr']*100:5.1f}% | (baseline)")
        else:
            print(f"  {attack_name:30s} | Recall: {metrics['recall']*100:5.1f}% | "
                  f"F1: {metrics['f1']*100:5.1f}% | FPR: {metrics['fpr']*100:5.1f}%")

    # Aggregate results
    print("\n[4/4] Computing aggregate metrics...")

    # Combine all attack results (excluding clean)
    all_tp = sum(r["true_positives"] for k, r in results.items() if k != "clean")
    all_fp = sum(r["false_positives"] for k, r in results.items() if k != "clean")
    all_tn = sum(r["true_negatives"] for k, r in results.items() if k != "clean")
    all_fn = sum(r["false_negatives"] for k, r in results.items() if k != "clean")

    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
        if (overall_precision + overall_recall) > 0 else 0
    overall_fpr = all_fp / (all_fp + all_tn) if (all_fp + all_tn) > 0 else 0

    # Clean data FPR
    clean_fpr = results["clean"]["fpr"]

    # Summary
    summary = {
        "overall_precision": float(overall_precision),
        "overall_recall": float(overall_recall),
        "overall_f1": float(overall_f1),
        "overall_fpr": float(overall_fpr),
        "clean_fpr": float(clean_fpr),
        "total_attacks": len(results) - 1,
        "threshold": float(threshold),
        "per_attack_results": results,
    }

    # Save results
    results_path = output_path / "synthetic_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nOverall Metrics (across all {summary['total_attacks']} attack types):")
    print(f"  Precision: {overall_precision*100:.1f}%")
    print(f"  Recall:    {overall_recall*100:.1f}%")
    print(f"  F1 Score:  {overall_f1*100:.1f}%")
    print(f"  FPR:       {overall_fpr*100:.1f}%")
    print(f"\nClean Data FPR: {clean_fpr*100:.1f}%")

    # Best/worst attacks
    attack_results = {k: v for k, v in results.items() if k != "clean" and v["attack_samples"] > 0}
    if attack_results:
        best_attack = max(attack_results.items(), key=lambda x: x[1]["recall"])
        worst_attack = min(attack_results.items(), key=lambda x: x[1]["recall"])

        print(f"\nBest detected:  {best_attack[0]} (Recall: {best_attack[1]['recall']*100:.1f}%)")
        print(f"Worst detected: {worst_attack[0]} (Recall: {worst_attack[1]['recall']*100:.1f}%)")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
