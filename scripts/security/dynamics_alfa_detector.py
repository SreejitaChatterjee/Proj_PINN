"""
Dynamics-based fault detector for ALFA dataset.

This approach:
1. Learns dynamics: next_state = f(state, control)
2. Trains ONLY on normal samples
3. Detects faults as high prediction residuals (physics violations)

Usage:
    python scripts/security/dynamics_alfa_detector.py --seeds 5
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


class DynamicsPredictor(nn.Module):
    """Neural network to predict next state from current state + control."""

    def __init__(self, state_dim=12, control_dim=4, hidden_dim=64):
        super().__init__()
        input_dim = state_dim + control_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim),  # Predict delta (change in state)
        )

    def forward(self, state, control):
        x = torch.cat([state, control], dim=1)
        delta = self.net(x)
        return state + delta  # Residual connection: predict next = current + delta


def load_temporal_data(data_dir: Path):
    """Load temporally-labeled ALFA data with state transitions."""
    flights = []
    csv_files = sorted(data_dir.glob("*.csv"))

    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    for csv_file in csv_files:
        if csv_file.name.startswith("processing"):
            continue

        df = pd.read_csv(csv_file)
        if len(df) < 10:
            continue

        states = df[state_cols].values
        controls = df[control_cols].values
        labels = df["label"].values
        fault_type = df["fault_type"].iloc[0]

        # Create transition pairs: (state_t, control_t) -> state_{t+1}
        flights.append({
            "name": csv_file.stem,
            "states": states[:-1],
            "controls": controls[:-1],
            "next_states": states[1:],
            "labels": labels[:-1],  # Label of the transition
            "fault_type": fault_type,
            "is_fault_flight": fault_type != "Normal",
            "n_normal": (labels[:-1] == 0).sum(),
            "n_fault": (labels[:-1] == 1).sum(),
        })

    return flights


def train_dynamics_model(train_data, seed, device="cpu", epochs=100, lr=1e-3):
    """Train dynamics predictor on normal transitions only."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Extract normal transitions
    states_list, controls_list, next_states_list = [], [], []
    for flight in train_data:
        normal_mask = flight["labels"] == 0
        if normal_mask.any():
            states_list.append(flight["states"][normal_mask])
            controls_list.append(flight["controls"][normal_mask])
            next_states_list.append(flight["next_states"][normal_mask])

    states = np.vstack(states_list)
    controls = np.vstack(controls_list)
    next_states = np.vstack(next_states_list)

    # Normalize
    state_scaler = StandardScaler()
    control_scaler = StandardScaler()

    states_scaled = state_scaler.fit_transform(states)
    controls_scaled = control_scaler.fit_transform(controls)
    next_states_scaled = state_scaler.transform(next_states)

    # Tensors
    states_t = torch.FloatTensor(states_scaled).to(device)
    controls_t = torch.FloatTensor(controls_scaled).to(device)
    next_states_t = torch.FloatTensor(next_states_scaled).to(device)

    # Model
    model = DynamicsPredictor(state_dim=12, control_dim=4, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(states_t, controls_t)
        loss = ((pred - next_states_t) ** 2).mean()
        loss.backward()
        optimizer.step()

    return model, state_scaler, control_scaler


def get_residuals(model, state_scaler, control_scaler, flight, device="cpu"):
    """Compute prediction residuals for a flight."""
    states_scaled = state_scaler.transform(flight["states"])
    controls_scaled = control_scaler.transform(flight["controls"])
    next_states_scaled = state_scaler.transform(flight["next_states"])

    states_t = torch.FloatTensor(states_scaled).to(device)
    controls_t = torch.FloatTensor(controls_scaled).to(device)
    next_states_t = torch.FloatTensor(next_states_scaled).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(states_t, controls_t)
        residuals = ((pred - next_states_t) ** 2).mean(dim=1)

    return residuals.cpu().numpy()


def run_lofo_cv(flights, seed, device="cpu"):
    """Run leave-one-flight-out CV with dynamics model."""
    all_results = []

    for i, test_flight in enumerate(flights):
        train_flights = flights[:i] + flights[i + 1:]

        # Check we have normal samples to train on
        total_normal = sum(f["n_normal"] for f in train_flights)
        if total_normal < 50:
            continue

        # Train dynamics model
        model, state_scaler, control_scaler = train_dynamics_model(
            train_flights, seed, device
        )

        # Get residuals for test flight
        residuals = get_residuals(model, state_scaler, control_scaler, test_flight, device)

        all_results.append({
            "name": test_flight["name"],
            "fault_type": test_flight["fault_type"],
            "is_fault_flight": test_flight["is_fault_flight"],
            "labels": test_flight["labels"],
            "residuals": residuals,
        })

    return all_results


def compute_metrics(all_results, threshold_percentile=95, min_consecutive=3):
    """Compute detection metrics."""
    all_labels = np.concatenate([r["labels"] for r in all_results])
    all_residuals = np.concatenate([r["residuals"] for r in all_results])

    # Threshold from normal residuals
    normal_residuals = all_residuals[all_labels == 0]
    threshold = np.percentile(normal_residuals, threshold_percentile)

    # Sample-level AUROC
    if len(np.unique(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_residuals)
    else:
        auroc = 0.5

    # Helper for consecutive detection
    def has_consecutive(arr, min_len):
        count = 0
        for val in arr:
            if val:
                count += 1
                if count >= min_len:
                    return True
            else:
                count = 0
        return False

    # Flight-level metrics
    n_detected = 0
    n_false_alarm = 0

    for r in all_results:
        above = r["residuals"] > threshold

        if r["is_fault_flight"]:
            fault_indices = np.where(r["labels"] == 1)[0]
            if len(fault_indices) > 0:
                fault_start = fault_indices[0]
                if has_consecutive(above[fault_start:], min_consecutive):
                    n_detected += 1
        else:
            if has_consecutive(above, min_consecutive):
                n_false_alarm += 1

    n_fault = sum(1 for r in all_results if r["is_fault_flight"])
    n_normal = sum(1 for r in all_results if not r["is_fault_flight"])

    # Sample-level FPR
    predictions = (all_residuals > threshold).astype(int)
    fp = ((predictions == 1) & (all_labels == 0)).sum()
    tn = ((predictions == 0) & (all_labels == 0)).sum()
    sample_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0

    return {
        "sample_auroc": float(auroc),
        "sample_fpr": sample_fpr,
        "flight_detection_rate": float(n_detected / n_fault) if n_fault > 0 else 0,
        "flight_false_alarm_rate": float(n_false_alarm / n_normal) if n_normal > 0 else 0,
        "threshold_percentile": threshold_percentile,
        "min_consecutive": min_consecutive,
        "n_detected": n_detected,
        "n_false_alarm": n_false_alarm,
        "n_fault_flights": n_fault,
        "n_normal_flights": n_normal,
    }


def main():
    parser = argparse.ArgumentParser(description="Dynamics-based ALFA detector")
    parser.add_argument("--data", type=str, default="data/alfa/temporal")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=str, default="research/security/dynamics_alfa_results.json")
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_path = Path(args.output)

    print("=" * 70)
    print("DYNAMICS-BASED FAULT DETECTOR")
    print("=" * 70)
    print("\nApproach: Learn dynamics on normal data, detect via prediction residuals\n")

    flights = load_temporal_data(data_dir)
    print(f"Loaded {len(flights)} flights")

    total_normal = sum(f["n_normal"] for f in flights)
    total_fault = sum(f["n_fault"] for f in flights)
    print(f"  Normal transitions: {total_normal}")
    print(f"  Fault transitions: {total_fault}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_seed_metrics = []
    seeds = list(range(42, 42 + args.seeds))

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        results = run_lofo_cv(flights, seed, device)
        metrics = compute_metrics(results, threshold_percentile=95, min_consecutive=3)
        all_seed_metrics.append(metrics)

        print(f"  Sample AUROC: {metrics['sample_auroc']:.3f}")
        print(f"  Flight Detection Rate: {metrics['flight_detection_rate']:.1%}")
        print(f"  Flight False Alarm Rate: {metrics['flight_false_alarm_rate']:.1%}")

    # Aggregate
    aurocs = [m["sample_auroc"] for m in all_seed_metrics]
    det_rates = [m["flight_detection_rate"] for m in all_seed_metrics]
    fa_rates = [m["flight_false_alarm_rate"] for m in all_seed_metrics]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "method": "Dynamics Predictor (MLP)",
        "n_seeds": args.seeds,
        "n_flights": len(flights),
        "aggregate": {
            "sample_auroc_mean": float(np.mean(aurocs)),
            "sample_auroc_std": float(np.std(aurocs)),
            "flight_detection_rate_mean": float(np.mean(det_rates)),
            "flight_detection_rate_std": float(np.std(det_rates)),
            "flight_false_alarm_rate_mean": float(np.mean(fa_rates)),
            "flight_false_alarm_rate_std": float(np.std(fa_rates)),
        },
        "per_seed": all_seed_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (Dynamics Predictor)")
    print("=" * 70)
    print(f"Sample AUROC:           {summary['aggregate']['sample_auroc_mean']:.3f} +/- {summary['aggregate']['sample_auroc_std']:.3f}")
    print(f"Flight Detection Rate:  {summary['aggregate']['flight_detection_rate_mean']:.1%} +/- {summary['aggregate']['flight_detection_rate_std']:.1%}")
    print(f"Flight False Alarm Rate:{summary['aggregate']['flight_false_alarm_rate_mean']:.1%} +/- {summary['aggregate']['flight_false_alarm_rate_std']:.1%}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
