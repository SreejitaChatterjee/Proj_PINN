"""
Feature-based fault detector for ALFA dataset.

Uses statistical features computed over sliding windows,
which is more robust to the low (~1 Hz) sampling rate.

Features:
- State means, variances
- State derivatives (finite differences)
- Control-state correlations
- Angular rate magnitudes

Usage:
    python scripts/security/feature_alfa_detector.py --seeds 5
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def extract_features(states, controls, window_size=5):
    """Extract statistical features from sliding windows."""
    n_samples = len(states)
    features_list = []

    for i in range(n_samples):
        start = max(0, i - window_size + 1)
        window_states = states[start:i + 1]
        window_controls = controls[start:i + 1]

        # State statistics
        state_mean = window_states.mean(axis=0)
        state_std = window_states.std(axis=0) + 1e-6

        # State derivatives (if enough samples)
        if len(window_states) >= 2:
            state_diff = np.diff(window_states, axis=0).mean(axis=0)
            state_diff_std = np.diff(window_states, axis=0).std(axis=0) + 1e-6
        else:
            state_diff = np.zeros(12)
            state_diff_std = np.zeros(12)

        # Control statistics
        control_mean = window_controls.mean(axis=0)
        control_std = window_controls.std(axis=0) + 1e-6

        # Angular rate magnitudes
        angular_rates = window_states[:, 6:9]  # p, q, r
        angular_mag = np.sqrt((angular_rates ** 2).sum(axis=1)).mean()

        # Velocity magnitude
        velocities = window_states[:, 9:12]  # vx, vy, vz
        vel_mag = np.sqrt((velocities ** 2).sum(axis=1)).mean()

        # Position variance (instability indicator)
        positions = window_states[:, 0:3]  # x, y, z
        pos_var = positions.var(axis=0).sum()

        # Attitude variance
        attitudes = window_states[:, 3:6]  # phi, theta, psi
        att_var = attitudes.var(axis=0).sum()

        # Compile features
        feature_vec = np.concatenate([
            state_mean,           # 12
            state_std,            # 12
            state_diff,           # 12
            state_diff_std,       # 12
            control_mean,         # 4
            control_std,          # 4
            [angular_mag],        # 1
            [vel_mag],            # 1
            [pos_var],            # 1
            [att_var],            # 1
        ])
        features_list.append(feature_vec)

    return np.array(features_list)


class FeatureAutoencoder(nn.Module):
    """Autoencoder for feature-based anomaly detection."""

    def __init__(self, input_dim=60, hidden_dim=32, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def get_anomaly_score(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            mse = ((x - recon) ** 2).mean(dim=1)
        return mse


def load_and_featurize(data_dir: Path, window_size=5):
    """Load data and extract features."""
    flights = []
    csv_files = sorted(data_dir.glob("*.csv"))

    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    for csv_file in csv_files:
        if csv_file.name.startswith("processing"):
            continue

        df = pd.read_csv(csv_file)
        if len(df) < window_size + 5:
            continue

        states = df[state_cols].values
        controls = df[control_cols].values
        labels = df["label"].values
        fault_type = df["fault_type"].iloc[0]

        # Extract features
        features = extract_features(states, controls, window_size)

        flights.append({
            "name": csv_file.stem,
            "features": features,
            "labels": labels,
            "fault_type": fault_type,
            "is_fault_flight": fault_type != "Normal",
        })

    return flights


def train_feature_model(train_flights, seed, device="cpu", epochs=100):
    """Train autoencoder on normal features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Collect normal samples
    normal_features = []
    for f in train_flights:
        mask = f["labels"] == 0
        if mask.any():
            normal_features.append(f["features"][mask])

    X = np.vstack(normal_features)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    # Model
    input_dim = X.shape[1]
    model = FeatureAutoencoder(input_dim=input_dim, hidden_dim=32, latent_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon = model(X_tensor)
        loss = ((X_tensor - recon) ** 2).mean()
        loss.backward()
        optimizer.step()

    return model, scaler


def run_lofo_cv(flights, seed, device="cpu"):
    """LOFO CV with feature-based model."""
    all_results = []

    for i, test_flight in enumerate(flights):
        train_flights = flights[:i] + flights[i + 1:]

        # Need enough normal samples
        n_normal = sum((f["labels"] == 0).sum() for f in train_flights)
        if n_normal < 50:
            continue

        model, scaler = train_feature_model(train_flights, seed, device)

        # Evaluate
        X_test = scaler.transform(test_flight["features"])
        X_test_t = torch.FloatTensor(X_test).to(device)
        model.eval()
        scores = model.get_anomaly_score(X_test_t).cpu().numpy()

        all_results.append({
            "name": test_flight["name"],
            "labels": test_flight["labels"],
            "scores": scores,
            "is_fault_flight": test_flight["is_fault_flight"],
        })

    return all_results


def compute_metrics(results, threshold_pct=95, min_consec=3):
    """Compute detection metrics."""
    all_labels = np.concatenate([r["labels"] for r in results])
    all_scores = np.concatenate([r["scores"] for r in results])

    # Threshold
    normal_scores = all_scores[all_labels == 0]
    threshold = np.percentile(normal_scores, threshold_pct)

    # Sample AUROC
    auroc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.5

    # Flight-level
    def has_consec(arr, n):
        count = 0
        for v in arr:
            count = count + 1 if v else 0
            if count >= n:
                return True
        return False

    n_det = 0
    n_fa = 0
    for r in results:
        above = r["scores"] > threshold
        if r["is_fault_flight"]:
            fault_idx = np.where(r["labels"] == 1)[0]
            if len(fault_idx) > 0 and has_consec(above[fault_idx[0]:], min_consec):
                n_det += 1
        else:
            if has_consec(above, min_consec):
                n_fa += 1

    n_fault = sum(1 for r in results if r["is_fault_flight"])
    n_normal = sum(1 for r in results if not r["is_fault_flight"])

    return {
        "sample_auroc": float(auroc),
        "flight_detection_rate": float(n_det / n_fault) if n_fault > 0 else 0,
        "flight_false_alarm_rate": float(n_fa / n_normal) if n_normal > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/alfa/temporal")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", default="research/security/feature_alfa_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("FEATURE-BASED FAULT DETECTOR")
    print("=" * 70)
    print("\nApproach: Statistical features over sliding windows\n")

    flights = load_and_featurize(Path(args.data))
    print(f"Loaded {len(flights)} flights")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_metrics = []
    for seed in range(42, 42 + args.seeds):
        print(f"\n--- Seed {seed} ---")
        results = run_lofo_cv(flights, seed, device)
        metrics = compute_metrics(results)
        all_metrics.append(metrics)
        print(f"  AUROC: {metrics['sample_auroc']:.3f}")
        print(f"  Detection: {metrics['flight_detection_rate']:.1%}")
        print(f"  False Alarm: {metrics['flight_false_alarm_rate']:.1%}")

    aurocs = [m["sample_auroc"] for m in all_metrics]
    det = [m["flight_detection_rate"] for m in all_metrics]
    fa = [m["flight_false_alarm_rate"] for m in all_metrics]

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"AUROC:      {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}")
    print(f"Detection:  {np.mean(det):.1%} +/- {np.std(det):.1%}")
    print(f"False Alarm:{np.mean(fa):.1%} +/- {np.std(fa):.1%}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "method": "Feature-based Autoencoder",
        "aggregate": {
            "auroc_mean": float(np.mean(aurocs)),
            "detection_rate_mean": float(np.mean(det)),
            "false_alarm_rate_mean": float(np.mean(fa)),
        },
        "per_seed": all_metrics,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
