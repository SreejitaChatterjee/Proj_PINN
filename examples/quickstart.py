"""
Quick Start: Load a pre-trained model and make predictions.

This example shows how to:
1. Load a pre-trained QuadrotorPINN model
2. Make single-step predictions
3. Perform multi-step rollouts
"""

import torch
import numpy as np
from pathlib import Path

# Import from the pinn_dynamics package
from pinn_dynamics import QuadrotorPINN, Predictor
from pinn_dynamics.data import load_scalers

# Paths
MODEL_PATH = Path(__file__).parent.parent / "models" / "quadrotor_pinn_diverse.pth"
SCALER_PATH = Path(__file__).parent.parent / "models" / "diverse_scalers.pkl"


def main():
    # 1. Load model
    print("Loading model...")
    model = QuadrotorPINN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Print model summary
    print(model.summary())

    # 2. Load scalers (for proper input/output scaling)
    scaler_X, scaler_y = load_scalers(SCALER_PATH)

    # 3. Create predictor
    predictor = Predictor(model, scaler_X, scaler_y)

    # 4. Create example initial state and controls
    # State: [x, y, z, phi, theta, psi, p, q, r, vx, vy, vz]
    initial_state = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Control: [thrust, torque_x, torque_y, torque_z]
    # Hover thrust for 68g drone: m*g = 0.068 * 9.81 ≈ 0.667 N
    control = np.array([0.667, 0.0, 0.0, 0.0])

    # 5. Single-step prediction
    print("\nSingle-step prediction:")
    next_state = predictor.predict(initial_state, control)
    print(f"  Input state:  {initial_state[:3]}... (position)")
    print(f"  Output state: {next_state[:3]}... (position)")

    # 6. Multi-step rollout
    print("\nMulti-step rollout (100 steps):")
    n_steps = 100
    controls = np.tile(control, (n_steps, 1))  # Repeat control for each step

    trajectory = predictor.rollout(initial_state, controls)
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Final position: x={trajectory[-1, 0]:.3f}, y={trajectory[-1, 1]:.3f}, z={trajectory[-1, 2]:.3f}")

    # 7. With uncertainty quantification
    print("\nRollout with uncertainty (50 MC samples):")
    result = predictor.rollout_with_uncertainty(initial_state, controls, n_samples=50)
    print(f"  Mean final position: {result.mean[-1, :3]}")
    print(f"  Std final position:  {result.std[-1, :3]}")


if __name__ == "__main__":
    main()
