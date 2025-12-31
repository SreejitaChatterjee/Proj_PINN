"""
Multi-seed ALFA evaluation for reproducible fault detection metrics.

This script trains a simple neural network detector on ALFA data
with leave-one-flight-out cross-validation across multiple seeds.

Usage:
    python scripts/security/multiseed_alfa_eval.py --seeds 5
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


class SimpleAnomalyDetector(nn.Module):
    """Simple feedforward network for anomaly detection."""

    def __init__(self, input_dim=12, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_alfa_data(data_dir: Path):
    """Load all preprocessed ALFA flights."""
    flights = []
    csv_files = sorted(data_dir.glob("*.csv"))

    for csv_file in csv_files:
        if csv_file.name.startswith("summary"):
            continue

        df = pd.read_csv(csv_file)
        if len(df) < 10:  # Skip very short flights
            continue

        fault_type = df["fault_type"].iloc[0]
        label = 1 if fault_type != "Normal" else 0

        state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
        X = df[state_cols].values

        flights.append(
            {
                "name": csv_file.stem,
                "X": X,
                "label": label,
                "fault_type": fault_type,
                "n_samples": len(X),
            }
        )

    return flights


def train_detector(train_flights, val_flight, seed, device="cpu", epochs=50):
    """Train detector with LOFO cross-validation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Combine training data
    X_train = np.vstack([f["X"] for f in train_flights])
    y_train = np.concatenate([np.full(len(f["X"]), f["label"]) for f in train_flights])

    # Validation data
    X_val = val_flight["X"]
    y_val = np.full(len(X_val), val_flight["label"])

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val_scaled).to(device)

    # Model
    model = SimpleAnomalyDetector(input_dim=12, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy().flatten()

    return val_pred, y_val


def run_lofo_cv(flights, seed, device="cpu"):
    """Run leave-one-flight-out CV."""
    results = []

    for i, val_flight in enumerate(flights):
        train_flights = flights[:i] + flights[i + 1 :]

        # Skip if no normal flights in training
        n_normal = sum(1 for f in train_flights if f["label"] == 0)
        n_fault = sum(1 for f in train_flights if f["label"] == 1)

        if n_normal == 0 or n_fault == 0:
            continue

        pred, true = train_detector(train_flights, val_flight, seed, device)

        # Per-flight metrics (all samples have same label)
        flight_pred = pred.mean()  # Average prediction
        flight_true = val_flight["label"]

        results.append(
            {
                "flight": val_flight["name"],
                "fault_type": val_flight["fault_type"],
                "true_label": flight_true,
                "pred_score": float(flight_pred),
                "n_samples": val_flight["n_samples"],
            }
        )

    return results


def compute_metrics(results, threshold=0.5):
    """Compute detection metrics from LOFO results."""
    y_true = np.array([r["true_label"] for r in results])
    y_scores = np.array([r["pred_score"] for r in results])
    y_pred = (y_scores > threshold).astype(int)

    metrics = {
        "n_flights": len(results),
        "n_normal": int(sum(y_true == 0)),
        "n_fault": int(sum(y_true == 1)),
        "threshold": threshold,
    }

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_scores))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["true_negatives"] = int(tn)
        metrics["false_negatives"] = int(fn)
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    else:
        metrics["auroc"] = 0.5
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-seed ALFA evaluation")
    parser.add_argument(
        "--data", type=str, default="data/alfa/preprocessed", help="Path to preprocessed ALFA data"
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to test")
    parser.add_argument(
        "--output",
        type=str,
        default="research/security/alfa_multiseed_results.json",
        help="Output file",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_path = Path(args.output)

    print("=" * 60)
    print("MULTI-SEED ALFA EVALUATION")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {data_dir}...")
    flights = load_alfa_data(data_dir)
    print(f"Loaded {len(flights)} flights")

    n_normal = sum(1 for f in flights if f["label"] == 0)
    n_fault = sum(1 for f in flights if f["label"] == 1)
    print(f"  Normal: {n_normal}, Fault: {n_fault}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Multi-seed evaluation
    all_metrics = []
    seeds = list(range(42, 42 + args.seeds))  # [42, 43, 44, ...]

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        results = run_lofo_cv(flights, seed, device)
        metrics = compute_metrics(results)
        metrics["seed"] = seed
        all_metrics.append(metrics)

        print(f"  AUROC: {metrics['auroc']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  FPR: {metrics['fpr']:.3f}")

    # Aggregate stats
    aurocs = [m["auroc"] for m in all_metrics]
    f1s = [m["f1"] for m in all_metrics]
    fprs = [m["fpr"] for m in all_metrics]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "n_seeds": args.seeds,
        "seeds": seeds,
        "n_flights": len(flights),
        "n_normal": n_normal,
        "n_fault": n_fault,
        "aggregate": {
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "fpr_mean": float(np.mean(fprs)),
            "fpr_std": float(np.std(fprs)),
        },
        "per_seed": all_metrics,
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(
        f"AUROC: {summary['aggregate']['auroc_mean']:.3f} +/- {summary['aggregate']['auroc_std']:.3f}"
    )
    print(f"F1:    {summary['aggregate']['f1_mean']:.3f} +/- {summary['aggregate']['f1_std']:.3f}")
    print(
        f"FPR:   {summary['aggregate']['fpr_mean']:.3f} +/- {summary['aggregate']['fpr_std']:.3f}"
    )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
