#!/usr/bin/env python3
"""
Quick Demo: Physics-Informed Neural Network for Dynamics Learning

Run:
  python demo.py           # Simulated quadrotor data
  python demo.py --real    # Real EuRoC MAV data (trained on real data!)

This demonstrates:
1. Loading a pre-trained PINN
2. Single-step and multi-step (rollout) predictions
3. Comparison with ground truth
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import joblib
import sys
from pathlib import Path

# Try new package structure first, fall back to legacy scripts
try:
    from pinn_dynamics import QuadrotorPINN
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    from pinn_model import QuadrotorPINN


class EuRoCPINN(nn.Module):
    """PINN for real EuRoC data (15 inputs: 12 states + 3 IMU accels)."""
    def __init__(self, input_size=15, hidden_size=256, output_size=12, num_layers=5, dropout=0.1):
        super().__init__()
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

    def forward(self, x):
        return self.net(x)


def load_model(use_real=False):
    """Load pre-trained model and scalers."""
    base_path = Path(__file__).parent / "models"

    if use_real:
        # EuRoC-trained model (15 inputs: states + IMU accels)
        model_path = base_path / "euroc_pinn.pth"
        scaler_path = base_path / "euroc_scalers.pkl"

        if not model_path.exists():
            print(f"EuRoC model not found at {model_path}")
            print("Run: python scripts/train_euroc.py")
            sys.exit(1)

        model = EuRoCPINN(input_size=15, hidden_size=256, output_size=12)
    else:
        # Simulation-trained model (16 inputs: states + controls)
        model_path = base_path / "quadrotor_pinn_diverse.pth"
        scaler_path = base_path / "scalers_diverse.pkl"

        if not model_path.exists():
            print(f"Model not found at {model_path}")
            print("Run: python scripts/train_with_diverse_data.py")
            sys.exit(1)

        model = QuadrotorPINN()

    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    scalers_raw = joblib.load(scaler_path)
    scalers = {
        'X': scalers_raw.get('scaler_X', scalers_raw.get('X')),
        'y': scalers_raw.get('scaler_y', scalers_raw.get('y'))
    }
    return model, scalers


def rollout(model, scalers, initial_state, controls, n_steps=100):
    """Autoregressive rollout: predict n_steps into the future."""
    predictions = []
    state = initial_state.copy()

    with torch.no_grad():
        for i in range(n_steps):
            inp = np.concatenate([state, controls[i]])
            inp_scaled = scalers['X'].transform(inp.reshape(1, -1))
            inp_tensor = torch.tensor(inp_scaled, dtype=torch.float32)
            out_scaled = model(inp_tensor).numpy()
            state = scalers['y'].inverse_transform(out_scaled).flatten()
            predictions.append(state)

    return np.array(predictions)


def load_simulated_data():
    """Load simulated quadrotor test data."""
    import pandas as pd
    data_path = Path(__file__).parent / "data" / "test_set_diverse.csv"
    df = pd.read_csv(data_path)

    traj_id = df['trajectory_id'].iloc[0]
    traj = df[df['trajectory_id'] == traj_id].iloc[:100]

    state_cols = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']

    return {
        'name': f'Simulated trajectory {traj_id}',
        'initial_state': traj[state_cols].iloc[0].values,
        'controls': traj[control_cols].values,
        'ground_truth': traj[state_cols].values,
        'state_cols': state_cols,
    }


def load_euroc_data():
    """Load real EuRoC MAV data."""
    # Try new package first
    try:
        from pinn_dynamics.data import load_euroc
        data_path = Path(__file__).parent / "data" / "euroc"
        df = load_euroc('MH_01_easy', str(data_path), dt=0.005, download=False)
    except (ImportError, FileNotFoundError):
        # Fall back to legacy loader
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        from load_euroc import prepare_dynamics_data

        data_path = Path(__file__).parent / "data" / "euroc"
        if not data_path.exists():
            print(f"EuRoC data not found at {data_path}")
            print("Download with: python scripts/load_euroc.py")
            sys.exit(1)

        df = prepare_dynamics_data(data_path, dt=0.005)

    # Take a 100-step segment from middle of flight
    start_idx = len(df) // 3
    traj = df.iloc[start_idx:start_idx + 100]

    state_cols = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    control_cols = ['ax', 'ay', 'az']  # IMU accelerations as inputs

    return {
        'name': 'EuRoC MAV (real flight data)',
        'initial_state': traj[state_cols].iloc[0].values,
        'controls': traj[control_cols].values,
        'ground_truth': traj[state_cols].values,
        'state_cols': state_cols,
        'control_cols': control_cols,
    }


def main():
    parser = argparse.ArgumentParser(description='PINN Dynamics Demo')
    parser.add_argument('--real', action='store_true', help='Use real EuRoC MAV data')
    args = parser.parse_args()

    print("=" * 60)
    print("PINN Demo: Quadrotor Dynamics Prediction")
    print("=" * 60)

    # Load model (different models for sim vs real data)
    print("\n[1/3] Loading model...")
    model, scalers = load_model(use_real=args.real)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Model loaded: {n_params:,} parameters")

    # Load data
    print("\n[2/3] Loading test data...")
    if args.real:
        data = load_euroc_data()
        print(f"      Source: {data['name']}")
        print("      Model: Trained on real EuRoC data!")
    else:
        data = load_simulated_data()
        print(f"      Source: {data['name']}")

    print(f"      Samples: {len(data['ground_truth'])} timesteps")

    # Run rollout
    print("\n[3/3] Running 100-step rollout...")
    n_steps = min(99, len(data['controls']) - 1)
    predictions = rollout(model, scalers, data['initial_state'], data['controls'], n_steps=n_steps)

    # Compute errors
    gt = data['ground_truth'][1:n_steps+1]
    pos_error = np.mean(np.abs(predictions[:, :3] - gt[:, :3]))
    att_error = np.mean(np.abs(predictions[:, 3:6] - gt[:, 3:6]))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  {n_steps}-step rollout errors:")
    print(f"    Position MAE:  {pos_error:.4f} m ({pos_error*100:.2f} cm)")
    print(f"    Attitude MAE:  {att_error:.4f} rad ({np.degrees(att_error):.2f} deg)")

    # Show state comparison at final step
    state_cols = data['state_cols']
    print(f"\n  Final state comparison (step {n_steps}):")
    print(f"    {'State':<12} {'Predicted':>12} {'Truth':>12} {'Error':>12}")
    print(f"    {'-'*48}")
    for i, name in enumerate(state_cols[:6]):
        pred = predictions[-1, i]
        true = gt[-1, i]
        err = abs(pred - true)
        print(f"    {name:<12} {pred:>12.4f} {true:>12.4f} {err:>12.4f}")

    print("\n" + "=" * 60)
    if args.real:
        print("Demo complete - running on REAL EuRoC flight data!")
    else:
        print("Demo complete. Try: python demo.py --real")
    print("=" * 60)


if __name__ == "__main__":
    main()
