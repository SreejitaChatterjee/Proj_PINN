#!/usr/bin/env python3
"""
Quick Demo: Physics-Informed Neural Network for Dynamics Learning

Run:
  python demo.py           # Simulated quadrotor data
  python demo.py --real    # Real EuRoC MAV data

This demonstrates:
1. Loading a pre-trained PINN
2. Single-step and multi-step (rollout) predictions
3. Comparison with ground truth
"""

import argparse
import torch
import numpy as np
import joblib
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from pinn_model import QuadrotorPINN


def load_model():
    """Load pre-trained model and scalers."""
    model_path = Path(__file__).parent / "models" / "quadrotor_pinn_diverse.pth"
    scaler_path = Path(__file__).parent / "models" / "scalers_diverse.pkl"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Run: python scripts/train_with_diverse_data.py")
        sys.exit(1)

    model = QuadrotorPINN(input_size=16, hidden_size=256, output_size=12)
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
    import pandas as pd
    data_path = Path(__file__).parent / "data" / "euroc_processed.csv"

    if not data_path.exists():
        print(f"EuRoC data not found at {data_path}")
        print("Run: python scripts/load_euroc.py --sequence MH_01_easy")
        sys.exit(1)

    df = pd.read_csv(data_path)

    # Take a 100-step segment from middle of flight
    start_idx = len(df) // 3
    traj = df.iloc[start_idx:start_idx + 100]

    state_cols = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    # EuRoC uses IMU accelerations as pseudo-controls (no direct motor commands)
    # Pad with zero torques to match expected input shape
    controls = np.zeros((len(traj), 4))
    controls[:, 0] = traj['az'].values  # Use z-accel as pseudo-thrust proxy

    return {
        'name': 'EuRoC MH_01_easy (real flight)',
        'initial_state': traj[state_cols].iloc[0].values,
        'controls': controls,
        'ground_truth': traj[state_cols].values,
        'state_cols': state_cols,
    }


def main():
    parser = argparse.ArgumentParser(description='PINN Dynamics Demo')
    parser.add_argument('--real', action='store_true', help='Use real EuRoC MAV data')
    args = parser.parse_args()

    print("=" * 60)
    print("PINN Demo: Quadrotor Dynamics Prediction")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    model, scalers = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Model loaded: {n_params:,} parameters")

    # Load data
    print("\n[2/3] Loading test data...")
    if args.real:
        data = load_euroc_data()
        print(f"      Source: {data['name']}")
        print("      Note: Model trained on sim data, testing on real data (domain gap expected)")
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
    print(f"    Position MAE:  {pos_error:.4f} m")
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
        print("Demo complete. Sim-to-real gap visible in errors.")
    else:
        print("Demo complete. Try: python demo.py --real")
    print("=" * 60)


if __name__ == "__main__":
    main()
