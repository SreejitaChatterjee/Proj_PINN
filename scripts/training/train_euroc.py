#!/usr/bin/env python3
"""
Train PINN on EuRoC MAV Dataset

This script downloads and trains on real quadrotor data from ETH Zurich's
EuRoC MAV benchmark dataset.

Usage:
    python scripts/train_euroc.py                    # Download + train on MH_01_easy
    python scripts/train_euroc.py --sequence V1_01_easy
    python scripts/train_euroc.py --skip-download    # Use existing data
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
from load_euroc import SEQUENCES, download_sequence, prepare_dynamics_data


class EuRoCPINN(nn.Module):
    """
    PINN for real quadrotor data with IMU inputs.

    State (12): x, y, z, roll, pitch, yaw, p, q, r, vx, vy, vz
    Input (3): ax, ay, az (IMU accelerations as pseudo-controls)

    Unlike simulation, we don't have true thrust/torque commands,
    so we use a data-driven approach with soft physics constraints.
    """

    def __init__(self, input_size=15, hidden_size=256, output_size=12, num_layers=5, dropout=0.1):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Build network
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

        # Physical constants (for soft constraints)
        self.g = 9.81
        self.dt = 0.005  # EuRoC default timestep

    def forward(self, x):
        return self.net(x)

    def kinematic_loss(self, inputs, outputs, dt=0.005):
        """
        Enforce kinematic relationships:
        - Position derivatives should match velocities
        - Attitude derivatives should match angular rates
        """
        # Current states
        x, y, z = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        roll, pitch, yaw = inputs[:, 3], inputs[:, 4], inputs[:, 5]
        p, q, r = inputs[:, 6], inputs[:, 7], inputs[:, 8]
        vx, vy, vz = inputs[:, 9], inputs[:, 10], inputs[:, 11]

        # Predicted next states
        x_n, y_n, z_n = outputs[:, 0], outputs[:, 1], outputs[:, 2]
        roll_n, pitch_n, yaw_n = outputs[:, 3], outputs[:, 4], outputs[:, 5]

        # Position should change by velocity * dt
        x_pred = x + vx * dt
        y_pred = y + vy * dt
        z_pred = z + vz * dt

        # Attitude kinematics (simplified)
        roll_pred = roll + p * dt
        pitch_pred = pitch + q * dt
        yaw_pred = yaw + r * dt

        pos_loss = ((x_n - x_pred) ** 2 + (y_n - y_pred) ** 2 + (z_n - z_pred) ** 2).mean()
        att_loss = (
            (roll_n - roll_pred) ** 2 + (pitch_n - pitch_pred) ** 2 + (yaw_n - yaw_pred) ** 2
        ).mean()

        return pos_loss + att_loss

    def smoothness_loss(self, inputs, outputs, dt=0.005):
        """Penalize unrealistically large state changes."""
        state_change = outputs - inputs[:, :12]

        # Physical limits on state change rate
        max_rates = (
            torch.tensor(
                [
                    2.0,
                    2.0,
                    2.0,  # Position: 2m per step max
                    0.5,
                    0.5,
                    0.5,  # Attitude: 0.5 rad per step max
                    1.0,
                    1.0,
                    1.0,  # Angular rate: 1 rad/s per step max
                    2.0,
                    2.0,
                    2.0,  # Velocity: 2 m/s per step max
                ],
                device=inputs.device,
            )
            * dt
        )

        violations = torch.relu(torch.abs(state_change) - max_rates)
        return violations.pow(2).mean()


class EuRoCTrainer:
    """Trainer for EuRoC data."""

    def __init__(self, model, device="cpu", lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=15
        )
        self.criterion = nn.MSELoss()
        self.history = {"train": [], "val": [], "kinematic": [], "smooth": []}

    def train_epoch(self, loader, weights={"kinematic": 5.0, "smooth": 2.0}):
        self.model.train()
        losses = {"total": 0, "data": 0, "kinematic": 0, "smooth": 0}

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)

            # Data loss
            data_loss = self.criterion(output, target)

            # Physics-inspired losses
            kinematic_loss = self.model.kinematic_loss(data, output)
            smooth_loss = self.model.smoothness_loss(data, output)

            # Combined loss
            loss = (
                data_loss + weights["kinematic"] * kinematic_loss + weights["smooth"] * smooth_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses["total"] += loss.item()
            losses["data"] += data_loss.item()
            losses["kinematic"] += kinematic_loss.item()
            losses["smooth"] += smooth_loss.item()

        return {k: v / len(loader) for k, v in losses.items()}

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item()
        return total_loss / len(loader)

    def train(self, train_loader, val_loader, epochs=200):
        print(f"\nTraining EuRoC PINN for {epochs} epochs...")
        print(f"  Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"  Device: {self.device}")

        best_val = float("inf")

        for epoch in range(epochs):
            losses = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.scheduler.step(val_loss)

            self.history["train"].append(losses["total"])
            self.history["val"].append(val_loss)
            self.history["kinematic"].append(losses["kinematic"])
            self.history["smooth"].append(losses["smooth"])

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch

            if epoch % 10 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:3d}: Train={losses['total']:.6f}, Val={val_loss:.6f}, "
                    f"Kin={losses['kinematic']:.6f}, Smooth={losses['smooth']:.6f}, LR={lr:.2e}"
                )

        print(f"\nBest validation loss: {best_val:.6f} at epoch {best_epoch}")
        return best_val


def prepare_euroc_data(data, test_size=0.2, batch_size=64):
    """
    Convert EuRoC DataFrame to training format.

    Input: current state (12) + IMU accel (3) = 15 features
    Output: next state (12)
    """
    state_cols = [
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "p",
        "q",
        "r",
        "vx",
        "vy",
        "vz",
    ]
    control_cols = ["ax", "ay", "az"]

    # Build input/output pairs
    X = data[state_cols + control_cols].values[:-1]  # All but last
    y = data[state_cols].values[1:]  # All but first (next state)

    print(f"  Dataset: {len(X):,} samples")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=batch_size,
    )

    return train_loader, val_loader, test_loader, scaler_X, scaler_y


def evaluate_rollout(model, data, scaler_X, scaler_y, steps=100, device="cpu"):
    """Evaluate autoregressive rollout on EuRoC data."""
    model.eval()

    state_cols = [
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "p",
        "q",
        "r",
        "vx",
        "vy",
        "vz",
    ]
    control_cols = ["ax", "ay", "az"]

    # Start from middle of trajectory
    start_idx = len(data) // 2

    # Get initial state and controls for rollout
    initial_state = data[state_cols].values[start_idx]
    controls = data[control_cols].values[start_idx : start_idx + steps]
    ground_truth = data[state_cols].values[start_idx : start_idx + steps]

    # Rollout
    predictions = [initial_state.copy()]
    current_state = initial_state.copy()

    with torch.no_grad():
        for i in range(min(steps - 1, len(controls) - 1)):
            # Build input
            inp = np.concatenate([current_state, controls[i]])
            inp_scaled = scaler_X.transform(inp.reshape(1, -1))
            inp_tensor = torch.FloatTensor(inp_scaled).to(device)

            # Predict
            out_scaled = model(inp_tensor).cpu().numpy()
            out = scaler_y.inverse_transform(out_scaled)[0]

            predictions.append(out.copy())
            current_state = out.copy()

    predictions = np.array(predictions)
    ground_truth = ground_truth[: len(predictions)]

    # Compute errors
    errors = np.abs(predictions - ground_truth)
    rmse = np.sqrt(np.mean(errors**2, axis=0))

    print(f"\n{steps}-step Rollout RMSE:")
    for i, name in enumerate(state_cols):
        print(f"  {name:6s}: {rmse[i]:.4f}")

    return predictions, ground_truth, rmse


def main():
    parser = argparse.ArgumentParser(description="Train PINN on EuRoC dataset")
    parser.add_argument(
        "--sequence",
        default="MH_01_easy",
        choices=list(SEQUENCES.keys()),
        help="EuRoC sequence to use",
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip download, use existing data"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING PINN ON EuRoC MAV DATASET")
    print("=" * 70)

    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "euroc"
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    # Download and prepare data
    if not args.skip_download:
        print(f"\n[1/4] Downloading {args.sequence}...")
        seq_dir = download_sequence(args.sequence, data_dir)
    else:
        seq_dir = data_dir / args.sequence
        print(f"\n[1/4] Using existing data at {seq_dir}")

    print(f"\n[2/4] Preparing data...")
    data = prepare_dynamics_data(seq_dir, dt=0.005)

    # Prepare for training
    print(f"\n[3/4] Creating data loaders...")
    train_loader, val_loader, test_loader, scaler_X, scaler_y = prepare_euroc_data(
        data, batch_size=args.batch_size
    )

    # Train
    print(f"\n[4/4] Training model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EuRoCPINN(input_size=15, hidden_size=256, output_size=12)
    trainer = EuRoCTrainer(model, device=device)
    trainer.train(train_loader, val_loader, epochs=args.epochs)

    # Evaluate rollout
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    evaluate_rollout(model, data, scaler_X, scaler_y, steps=100, device=device)

    # Save
    save_path = model_dir / f"euroc_pinn_{args.sequence}.pth"
    torch.save(model.state_dict(), save_path)

    scaler_path = model_dir / f"euroc_scalers_{args.sequence}.pkl"
    joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, scaler_path)

    print(f"\nModel saved to: {save_path}")
    print(f"Scalers saved to: {scaler_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
