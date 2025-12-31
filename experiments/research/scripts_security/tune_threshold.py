"""
Tune anomaly detection threshold to optimize F1 score.

Analyzes anomaly score distributions and finds optimal threshold using
validation data from ALFA dataset.

Usage:
    python scripts/security/tune_threshold.py \
        --model models/security/pinn_w0_best.pth \
        --data data/attack_datasets/processed/alfa \
        --metric f1
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Import PINN framework (install with: pip install -e .)
try:
    from pinn_dynamics import Predictor, QuadrotorPINN
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pinn_dynamics import QuadrotorPINN, Predictor

from pinn_dynamics.security import AnomalyDetector


def load_trained_model(model_path: Path, scalers_path: Path):
    """Load trained PINN model and scalers."""
    model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    return model, scalers["scaler_X"], scalers["scaler_y"]


def prepare_test_data(df: pd.DataFrame):
    """Convert dataframe to test format."""
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    states = df[state_cols].values[:-1]
    controls = df[control_cols].values[:-1]
    next_states = df[state_cols].values[1:]
    labels = df["label"].values[:-1]

    return states, controls, next_states, labels


def compute_anomaly_scores(detector, data_dir: Path):
    """
    Compute anomaly scores for all flights.

    Returns:
        scores: List of anomaly scores
        labels: List of ground truth labels (0=normal, 1=fault)
        fault_types: List of fault type for each sample
    """
    all_scores = []
    all_labels = []
    all_fault_types = []

    csv_files = list(data_dir.glob("*.csv"))

    print(f"Computing anomaly scores on {len(csv_files)} flights...")

    for i, csv_file in enumerate(csv_files):
        if csv_file.name.startswith("summary"):
            continue

        df = pd.read_csv(csv_file)
        if len(df) == 0:
            continue

        fault_type = df["fault_type"].iloc[0]
        states, controls, next_states, labels = prepare_test_data(df)

        # Compute scores
        for j in range(len(states)):
            result = detector.detect(states[j], controls[j], next_states[j])
            all_scores.append(result.total_score)
            all_labels.append(labels[j])
            all_fault_types.append(fault_type)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(csv_files)} flights...")

    return np.array(all_scores), np.array(all_labels), np.array(all_fault_types)


def tune_threshold(scores, labels, metric="f1"):
    """
    Find optimal threshold by grid search.

    Args:
        scores: Anomaly scores
        labels: Ground truth labels
        metric: Metric to optimize ('f1', 'balanced_accuracy', 'youden')

    Returns:
        best_threshold, best_metric_value, all_results
    """
    print(f"\nTuning threshold to maximize {metric}...")

    # Try thresholds from min to max score
    thresholds = np.percentile(scores, np.linspace(0, 99.9, 200))

    results = []
    best_threshold = 0
    best_metric_value = 0

    for thresh in thresholds:
        predictions = (scores > thresh).astype(int)

        # Compute metrics
        TP = np.sum((predictions == 1) & (labels == 1))
        TN = np.sum((predictions == 0) & (labels == 0))
        FP = np.sum((predictions == 1) & (labels == 0))
        FN = np.sum((predictions == 0) & (labels == 1))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        tpr = recall
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        tnr = 1 - fpr

        balanced_acc = (tpr + tnr) / 2
        youden_j = tpr - fpr  # Youden's J statistic

        results.append(
            {
                "threshold": thresh,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tpr": tpr,
                "fpr": fpr,
                "balanced_accuracy": balanced_acc,
                "youden_j": youden_j,
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
            }
        )

        # Check if this is the best
        if metric == "f1":
            metric_value = f1
        elif metric == "balanced_accuracy":
            metric_value = balanced_acc
        elif metric == "youden":
            metric_value = youden_j
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = thresh

    return best_threshold, best_metric_value, results


def plot_threshold_analysis(results, best_threshold, output_dir: Path):
    """Plot threshold vs metrics to visualize tuning."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # F1 vs Threshold
    axes[0, 0].plot(df["threshold"], df["f1"], linewidth=2)
    axes[0, 0].axvline(
        best_threshold, color="r", linestyle="--", label=f"Optimal={best_threshold:.2f}"
    )
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].set_title("F1 Score vs Threshold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Precision-Recall vs Threshold
    axes[0, 1].plot(df["threshold"], df["precision"], label="Precision", linewidth=2)
    axes[0, 1].plot(df["threshold"], df["recall"], label="Recall", linewidth=2)
    axes[0, 1].axvline(best_threshold, color="r", linestyle="--", label="Optimal")
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Precision/Recall vs Threshold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # TPR-FPR vs Threshold
    axes[1, 0].plot(df["threshold"], df["tpr"], label="TPR (Recall)", linewidth=2)
    axes[1, 0].plot(df["threshold"], df["fpr"], label="FPR", linewidth=2)
    axes[1, 0].axvline(best_threshold, color="r", linestyle="--", label="Optimal")
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].set_title("TPR/FPR vs Threshold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Confusion matrix counts vs Threshold
    axes[1, 1].plot(df["threshold"], df["TP"], label="TP", linewidth=2)
    axes[1, 1].plot(df["threshold"], df["FP"], label="FP", linewidth=2)
    axes[1, 1].plot(df["threshold"], df["FN"], label="FN", linewidth=2)
    axes[1, 1].axvline(best_threshold, color="r", linestyle="--", label="Optimal")
    axes[1, 1].set_xlabel("Threshold")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Confusion Matrix Counts vs Threshold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale("log")

    plt.tight_layout()
    output_path = output_dir / "threshold_tuning.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved threshold analysis plot: {output_path}")
    plt.close()


def plot_score_distributions(scores, labels, fault_types, output_dir: Path):
    """Plot anomaly score distributions by fault type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Normal vs Fault
    normal_scores = scores[labels == 0]
    fault_scores = scores[labels == 1]

    axes[0].hist(
        normal_scores,
        bins=50,
        alpha=0.6,
        label=f"Normal (n={len(normal_scores)})",
        density=True,
    )
    axes[0].hist(
        fault_scores,
        bins=50,
        alpha=0.6,
        label=f"Fault (n={len(fault_scores)})",
        density=True,
    )
    axes[0].set_xlabel("Anomaly Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score Distribution: Normal vs Fault")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # By fault type
    unique_faults = np.unique(fault_types)
    for fault in unique_faults:
        if fault == "Normal":
            continue
        fault_mask = (fault_types == fault) & (labels == 1)
        if np.sum(fault_mask) > 0:
            axes[1].hist(scores[fault_mask], bins=30, alpha=0.5, label=fault, density=True)

    axes[1].set_xlabel("Anomaly Score")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Score Distribution by Fault Type")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "score_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved score distributions: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Tune anomaly detection threshold")
    parser.add_argument(
        "--model",
        type=str,
        default="models/security/pinn_w0_best.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/attack_datasets/processed/alfa",
        help="Path to ALFA dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="research/security/threshold_tuning",
        help="Output directory",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["f1", "balanced_accuracy", "youden"],
        default="f1",
        help="Metric to optimize",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("THRESHOLD TUNING FOR PINN ANOMALY DETECTOR")
    print("=" * 80)

    # Load model
    print("\n[1/6] Loading trained model...")
    scalers_path = model_path.parent / "scalers.pkl"
    model, scaler_X, scaler_y = load_trained_model(model_path, scalers_path)

    # Create predictor and detector
    predictor = Predictor(model, scaler_X, scaler_y, device="cpu")
    detector = AnomalyDetector(predictor, threshold=3.0)  # Initial threshold doesn't matter

    # Calibrate on normal flights
    print("\n[2/6] Calibrating detector on normal flights...")
    normal_flights = []
    for csv_file in data_dir.glob("*no_failure*.csv"):
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            normal_flights.append(df)

    if normal_flights:
        normal_df = pd.concat(normal_flights, ignore_index=True)
        state_cols = [
            "x",
            "y",
            "z",
            "phi",
            "theta",
            "psi",
            "p",
            "q",
            "r",
            "vx",
            "vy",
            "vz",
        ]
        control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]
        states = normal_df[state_cols].values[:-1]
        controls = normal_df[control_cols].values[:-1]
        next_states = normal_df[state_cols].values[1:]
        detector.calibrate(states, controls, next_states)

    # Compute anomaly scores on all data
    print("\n[3/6] Computing anomaly scores on all flights...")
    scores, labels, fault_types = compute_anomaly_scores(detector, data_dir)

    print(f"\n  Total samples: {len(scores)}")
    print(f"  Normal samples: {np.sum(labels == 0)}")
    print(f"  Fault samples: {np.sum(labels == 1)}")
    print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")
    print(f"  Score mean: {scores.mean():.2f} +/- {scores.std():.2f}")

    # Plot score distributions
    print("\n[4/6] Plotting score distributions...")
    plot_score_distributions(scores, labels, fault_types, output_dir)

    # Tune threshold
    print(f"\n[5/6] Tuning threshold (optimizing {args.metric})...")
    best_threshold, best_metric_value, results = tune_threshold(scores, labels, args.metric)

    print(f"\n  OPTIMAL THRESHOLD: {best_threshold:.4f}")
    print(f"  {args.metric.upper()}: {best_metric_value:.4f}")

    # Get performance at optimal threshold
    best_result = [r for r in results if r["threshold"] == best_threshold][0]
    print(f"\n  Performance at optimal threshold:")
    print(f"    Precision: {best_result['precision']:.4f}")
    print(f"    Recall:    {best_result['recall']:.4f}")
    print(f"    F1:        {best_result['f1']:.4f}")
    print(f"    TPR:       {best_result['tpr']:.4f}")
    print(f"    FPR:       {best_result['fpr']:.4f}")
    print(
        f"    TP={best_result['TP']}, TN={best_result['TN']}, FP={best_result['FP']}, FN={best_result['FN']}"
    )

    # Plot threshold analysis
    print("\n[6/6] Plotting threshold analysis...")
    plot_threshold_analysis(results, best_threshold, output_dir)

    # Save results
    output_file = output_dir / "optimal_threshold.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "optimal_threshold": float(best_threshold),
                "metric": args.metric,
                "metric_value": float(best_metric_value),
                "performance": {
                    "precision": best_result["precision"],
                    "recall": best_result["recall"],
                    "f1": best_result["f1"],
                    "tpr": best_result["tpr"],
                    "fpr": best_result["fpr"],
                    "balanced_accuracy": best_result["balanced_accuracy"],
                },
                "confusion_matrix": {
                    "TP": int(best_result["TP"]),
                    "TN": int(best_result["TN"]),
                    "FP": int(best_result["FP"]),
                    "FN": int(best_result["FN"]),
                },
                "score_statistics": {
                    "min": float(scores.min()),
                    "max": float(scores.max()),
                    "mean": float(scores.mean()),
                    "std": float(scores.std()),
                    "n_samples": int(len(scores)),
                    "n_normal": int(np.sum(labels == 0)),
                    "n_fault": int(np.sum(labels == 1)),
                },
            },
            f,
            indent=2,
        )

    print(f"\n  Saved optimal threshold config: {output_file}")

    print("\n" + "=" * 80)
    print("THRESHOLD TUNING COMPLETE!")
    print("=" * 80)
    print(f"\nUse --threshold {best_threshold:.4f} when running evaluate_detector.py")


if __name__ == "__main__":
    main()
