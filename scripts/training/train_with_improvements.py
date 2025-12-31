#!/usr/bin/env python3
"""
Train PINN with energy conservation loss and aggressive trajectories.
This validates the improvements for inertia parameter identification.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent))
from pinn_model import QuadrotorPINN
from train import Trainer


def main():
    PROJECT_ROOT = Path(__file__).parent.parent

    print("=" * 80)
    print("TRAINING PINN WITH IMPROVEMENTS")
    print("=" * 80)
    print("\nImprovements:")
    print("  1. Energy conservation loss (lambda=5.0)")
    print("  2. Aggressive trajectories (±45-60° angles)")
    print("  3. Combined dataset: 70,238 samples (15 trajectories)")
    print()

    # Load combined data
    df = pd.read_csv(PROJECT_ROOT / "data" / "combined_training_data.csv")
    print(f"Loaded {len(df)} samples from {df['trajectory_id'].nunique()} trajectories")

    # Define columns
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
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]
    input_features = state_cols + control_cols

    # Prepare sequences (current state + controls -> next state)
    X, y = [], []
    for traj_id in df["trajectory_id"].unique():
        df_traj = df[df["trajectory_id"] == traj_id].reset_index(drop=True)
        for i in range(len(df_traj) - 1):
            X.append(df_traj.iloc[i][input_features].values)
            y.append(df_traj.iloc[i + 1][state_cols].values)

    X = np.array(X)
    y = np.array(y)

    print(f"Prepared {len(X)} training samples")

    # Train/validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = QuadrotorPINN(input_size=16, hidden_size=256, output_size=12, num_layers=5, dropout=0.1)

    # Store true parameters
    true_params = {
        "m": 0.068,
        "Jxx": 6.86e-5,
        "Jyy": 9.2e-5,
        "Jzz": 1.366e-4,
        "kt": 0.01,
        "kq": 7.8263e-4,
    }

    print("\nTrue parameters:")
    for k, v in true_params.items():
        print(f"  {k} = {v:.2e}")

    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    trainer = Trainer(model, device=device, lr=0.0005)

    # Training with energy loss (REDUCED weight to prevent bias)
    weights = {
        "physics": 10.0,
        "temporal": 12.0,
        "stability": 5.0,
        "reg": 1.0,
        "energy": 2.0,  # REDUCED from 5.0 to prevent over-emphasis
    }

    print("\nLoss weights:")
    for k, v in weights.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    # Train for 150 epochs
    trainer.train(train_loader, val_loader, epochs=150, weights=weights)

    # Evaluate final parameters
    print("\n" + "=" * 80)
    print("PARAMETER IDENTIFICATION RESULTS")
    print("=" * 80)
    print(f"{'Parameter':<10} {'True':<15} {'Learned':<15} {'Error (%)':<10}")
    print("-" * 50)

    for k, true_val in true_params.items():
        learned_val = model.params[k].item()
        error_pct = abs(learned_val - true_val) / true_val * 100
        print(f"{k:<10} {true_val:<15.2e} {learned_val:<15.2e} {error_pct:<10.2f}")

    # Save model and scalers
    torch.save(model.state_dict(), PROJECT_ROOT / "models" / "quadrotor_pinn_improved.pth")
    joblib.dump(
        {"scaler_X": scaler_X, "scaler_y": scaler_y},
        PROJECT_ROOT / "models" / "scalers_improved.pkl",
    )

    print(f"\n" + "=" * 80)
    print("Model saved to:")
    print(f"  models/quadrotor_pinn_improved.pth")
    print(f"  models/scalers_improved.pkl")
    print("=" * 80)


if __name__ == "__main__":
    main()
