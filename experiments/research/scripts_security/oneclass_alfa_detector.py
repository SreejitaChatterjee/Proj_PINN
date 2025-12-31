"""
One-class anomaly detector for ALFA fault detection.

This approach:
1. Trains ONLY on normal samples (label=0)
2. Learns to reconstruct normal behavior
3. Detects anomalies as high reconstruction error

Usage:
    python scripts/security/oneclass_alfa_detector.py --seeds 5
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler


class AnomalyAutoencoder(nn.Module):
    """Autoencoder for anomaly detection."""

    def __init__(self, input_dim=12, hidden_dim=32, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def get_anomaly_score(self, x):
        """Reconstruction error as anomaly score."""
        with torch.no_grad():
            recon = self.forward(x)
            mse = ((x - recon) ** 2).mean(dim=1)
        return mse


def load_temporal_data(data_dir: Path):
    """Load temporally-labeled ALFA data."""
    flights = []
    csv_files = sorted(data_dir.glob("*.csv"))

    for csv_file in csv_files:
        if csv_file.name.startswith("processing"):
            continue

        df = pd.read_csv(csv_file)
        if len(df) < 10:
            continue

        state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
        X = df[state_cols].values
        labels = df["label"].values
        fault_type = df["fault_type"].iloc[0]

        flights.append(
            {
                "name": csv_file.stem,
                "X": X,
                "labels": labels,
                "fault_type": fault_type,
                "is_fault_flight": fault_type != "Normal",
                "n_normal": (labels == 0).sum(),
                "n_fault": (labels == 1).sum(),
            }
        )

    return flights


def train_autoencoder(X_train, seed, device="cpu", epochs=100, lr=1e-3):
    """Train autoencoder on normal samples only."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    model = AnomalyAutoencoder(input_dim=12, hidden_dim=32, latent_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon = model(X_tensor)
        loss = ((X_tensor - recon) ** 2).mean()
        loss.backward()
        optimizer.step()

    return model, scaler


def evaluate_flight(model, scaler, flight, device="cpu"):
    """Evaluate a single flight with the trained model."""
    X_scaled = scaler.transform(flight["X"])
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    model.eval()
    scores = model.get_anomaly_score(X_tensor).cpu().numpy()

    return {
        "name": flight["name"],
        "fault_type": flight["fault_type"],
        "is_fault_flight": flight["is_fault_flight"],
        "labels": flight["labels"],
        "scores": scores,
        "n_samples": len(scores),
    }


def compute_metrics(all_results, threshold_percentile=95, min_consecutive=3):
    """Compute detection metrics across all flights.

    Args:
        all_results: List of flight results
        threshold_percentile: Percentile of normal scores for threshold
        min_consecutive: Minimum consecutive samples above threshold for detection
    """
    # Aggregate all samples
    all_labels = np.concatenate([r["labels"] for r in all_results])
    all_scores = np.concatenate([r["scores"] for r in all_results])

    # Set threshold based on normal samples
    normal_scores = all_scores[all_labels == 0]
    threshold = np.percentile(normal_scores, threshold_percentile)

    # Sample-level metrics
    predictions = (all_scores > threshold).astype(int)

    # AUROC
    if len(np.unique(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_scores)
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        aupr = auc(recall, precision)
    else:
        auroc = 0.5
        aupr = 0.5

    # Confusion matrix
    tp = ((predictions == 1) & (all_labels == 1)).sum()
    fp = ((predictions == 1) & (all_labels == 0)).sum()
    tn = ((predictions == 0) & (all_labels == 0)).sum()
    fn = ((predictions == 0) & (all_labels == 1)).sum()

    # Helper: find consecutive runs above threshold
    def has_consecutive_above(above_threshold, min_len):
        """Check if there are min_len consecutive True values."""
        count = 0
        for val in above_threshold:
            if val:
                count += 1
                if count >= min_len:
                    return True
            else:
                count = 0
        return False

    def first_consecutive_index(above_threshold, min_len):
        """Find index of first consecutive run of min_len."""
        count = 0
        for i, val in enumerate(above_threshold):
            if val:
                count += 1
                if count >= min_len:
                    return i - min_len + 1
            else:
                count = 0
        return -1

    # Flight-level metrics
    n_fault_flights_detected = 0
    n_normal_flights_alarmed = 0
    detection_delays = []

    for r in all_results:
        above_threshold = r["scores"] > threshold

        if r["is_fault_flight"]:
            # Did we detect the fault with consecutive samples?
            fault_indices = np.where(r["labels"] == 1)[0]
            if len(fault_indices) > 0:
                fault_start = fault_indices[0]
                fault_region = above_threshold[fault_start:]
                if has_consecutive_above(fault_region, min_consecutive):
                    n_fault_flights_detected += 1
                    # Detection delay
                    first_idx = first_consecutive_index(fault_region, min_consecutive)
                    if first_idx >= 0:
                        detection_delays.append(first_idx)
        else:
            # False alarm on normal flight? Need consecutive to count
            if has_consecutive_above(above_threshold, min_consecutive):
                n_normal_flights_alarmed += 1

    n_fault_flights = sum(1 for r in all_results if r["is_fault_flight"])
    n_normal_flights = sum(1 for r in all_results if not r["is_fault_flight"])

    return {
        "threshold_percentile": threshold_percentile,
        "min_consecutive": min_consecutive,
        "threshold_value": float(threshold),
        "sample_auroc": float(auroc),
        "sample_aupr": float(aupr),
        "sample_tp": int(tp),
        "sample_fp": int(fp),
        "sample_tn": int(tn),
        "sample_fn": int(fn),
        "sample_precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "sample_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "sample_fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "flight_detection_rate": (
            float(n_fault_flights_detected / n_fault_flights) if n_fault_flights > 0 else 0.0
        ),
        "flight_false_alarm_rate": (
            float(n_normal_flights_alarmed / n_normal_flights) if n_normal_flights > 0 else 0.0
        ),
        "mean_detection_delay": float(np.mean(detection_delays)) if detection_delays else 0.0,
        "n_fault_flights_detected": n_fault_flights_detected,
        "n_normal_flights_alarmed": n_normal_flights_alarmed,
        "n_fault_flights": n_fault_flights,
        "n_normal_flights": n_normal_flights,
    }


def run_lofo_cv(flights, seed, device="cpu"):
    """Run leave-one-flight-out cross-validation."""
    all_results = []

    for i, test_flight in enumerate(flights):
        # Train on all OTHER flights' normal samples
        train_flights = flights[:i] + flights[i + 1 :]

        # Collect normal samples (label=0) from training flights
        train_normal = []
        for f in train_flights:
            normal_mask = f["labels"] == 0
            if normal_mask.any():
                train_normal.append(f["X"][normal_mask])

        if len(train_normal) == 0:
            continue

        X_train = np.vstack(train_normal)

        # Train autoencoder
        model, scaler = train_autoencoder(X_train, seed, device)

        # Evaluate on test flight
        result = evaluate_flight(model, scaler, test_flight, device)
        all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="One-class ALFA detector")
    parser.add_argument(
        "--data", type=str, default="data/alfa/temporal", help="Path to temporal ALFA data"
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument(
        "--output", type=str, default="research/security/oneclass_alfa_results.json"
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_path = Path(args.output)

    print("=" * 70)
    print("ONE-CLASS ANOMALY DETECTOR (Autoencoder)")
    print("=" * 70)
    print("\nApproach: Train ONLY on normal samples, detect via reconstruction error\n")

    # Load data
    flights = load_temporal_data(data_dir)
    print(f"Loaded {len(flights)} flights")

    total_normal = sum(f["n_normal"] for f in flights)
    total_fault = sum(f["n_fault"] for f in flights)
    print(f"  Normal samples: {total_normal}")
    print(f"  Fault samples: {total_fault}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Multi-seed evaluation
    all_seed_metrics = []
    seeds = list(range(42, 42 + args.seeds))

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        results = run_lofo_cv(flights, seed, device)
        metrics = compute_metrics(results, threshold_percentile=95)

        all_seed_metrics.append(metrics)
        print(f"  Sample AUROC: {metrics['sample_auroc']:.3f}")
        print(f"  Sample FPR: {metrics['sample_fpr']:.3f}")
        print(f"  Flight Detection Rate: {metrics['flight_detection_rate']:.1%}")
        print(f"  Flight False Alarm Rate: {metrics['flight_false_alarm_rate']:.1%}")

    # Aggregate
    aurocs = [m["sample_auroc"] for m in all_seed_metrics]
    fprs = [m["sample_fpr"] for m in all_seed_metrics]
    detection_rates = [m["flight_detection_rate"] for m in all_seed_metrics]
    false_alarm_rates = [m["flight_false_alarm_rate"] for m in all_seed_metrics]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "method": "One-Class Autoencoder",
        "data_dir": str(data_dir),
        "n_seeds": args.seeds,
        "n_flights": len(flights),
        "n_normal_samples": int(total_normal),
        "n_fault_samples": int(total_fault),
        "aggregate": {
            "sample_auroc_mean": float(np.mean(aurocs)),
            "sample_auroc_std": float(np.std(aurocs)),
            "sample_fpr_mean": float(np.mean(fprs)),
            "sample_fpr_std": float(np.std(fprs)),
            "flight_detection_rate_mean": float(np.mean(detection_rates)),
            "flight_detection_rate_std": float(np.std(detection_rates)),
            "flight_false_alarm_rate_mean": float(np.mean(false_alarm_rates)),
            "flight_false_alarm_rate_std": float(np.std(false_alarm_rates)),
        },
        "per_seed": all_seed_metrics,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (One-Class Autoencoder)")
    print("=" * 70)
    print(
        f"Sample AUROC:           {summary['aggregate']['sample_auroc_mean']:.3f} +/- {summary['aggregate']['sample_auroc_std']:.3f}"
    )
    print(
        f"Sample FPR:             {summary['aggregate']['sample_fpr_mean']:.3f} +/- {summary['aggregate']['sample_fpr_std']:.3f}"
    )
    print(
        f"Flight Detection Rate:  {summary['aggregate']['flight_detection_rate_mean']:.1%} +/- {summary['aggregate']['flight_detection_rate_std']:.1%}"
    )
    print(
        f"Flight False Alarm Rate:{summary['aggregate']['flight_false_alarm_rate_mean']:.1%} +/- {summary['aggregate']['flight_false_alarm_rate_std']:.1%}"
    )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
