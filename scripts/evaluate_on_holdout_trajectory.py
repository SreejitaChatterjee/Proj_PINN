"""
PROPER Evaluation on Held-Out Test Trajectory

Uses TIME-BASED split (not random) to preserve continuous trajectories.
First 80% of data for training, last 20% for testing.
"""

import torch
import pandas as pd
import numpy as np
import joblib
from pinn_model_optimized_v2 import OptimizedPINNv2


def clip_normalized_data(x, clip_value=3.0):
    """Clip normalized data to [-3, 3]"""
    return np.clip(x, -clip_value, clip_value)


def evaluate_on_holdout(model, scaler_X, scaler_y, horizon=100):
    """
    Evaluate on held-out test trajectory (last 20% of data)

    This ensures:
    1. Continuous trajectory for autoregressive rollout
    2. Completely unseen data (temporal split)
    3. Realistic evaluation conditions
    """
    data = pd.read_csv('../data/quadrotor_training_data.csv')

    state_cols = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']

    # Time-based split: first 80% train, last 20% test
    n_total = len(data)
    train_end = int(0.8 * n_total)
    test_start = train_end

    print(f"Total data points: {n_total}")
    print(f"Training: indices 0 to {train_end} (80%)")
    print(f"Testing: indices {test_start} to {n_total} (20%)")
    print(f"Test trajectory length: {n_total - test_start} continuous steps")

    # Start evaluation at beginning of test set
    start_idx = test_start

    # Check if we have enough data
    available_steps = n_total - start_idx - 1
    if available_steps < horizon:
        print(f"\nWARNING: Only {available_steps} steps available, requested {horizon}")
        horizon = available_steps
        print(f"Using horizon={horizon} instead")

    # Initial state
    x_current = data.iloc[start_idx][state_cols + control_cols].values

    predictions = []
    ground_truth = []

    for step in range(horizon):
        # Ground truth
        y_true = data.iloc[start_idx + step + 1][state_cols].values
        ground_truth.append(y_true)

        # Normalize and clip
        x_scaled = scaler_X.transform(x_current.reshape(1, -1))
        x_scaled = clip_normalized_data(x_scaled)

        # Predict
        x_tensor = torch.FloatTensor(x_scaled)
        with torch.no_grad():
            y_pred_scaled = model(x_tensor).numpy()

        # Denormalize
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
        predictions.append(y_pred)

        # Next controls
        next_controls = data.iloc[start_idx + step + 1][control_cols].values

        # Update state (autoregressive - uses predicted state!)
        x_current = np.concatenate([y_pred, next_controls])

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # MAE per state
    mae = np.mean(np.abs(predictions - ground_truth), axis=0)

    return mae, predictions, ground_truth


def main():
    """Proper evaluation on held-out test trajectory"""
    print("="*70)
    print("EVALUATION ON HELD-OUT TEST TRAJECTORY")
    print("Time-based split: Train on first 80%, test on last 20%")
    print("="*70)

    # Load model
    scalers = joblib.load('../models/scalers_optimized_v2.pkl')
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']

    model = OptimizedPINNv2(hidden_size=256)
    model.load_state_dict(torch.load('../models/quadrotor_pinn_optimized_v2.pth'))
    model.eval()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Evaluate at multiple horizons
    state_names = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    units = ['m', 'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s', 'm/s']

    horizons = [1, 10, 50, 100]

    all_results = {}

    for horizon in horizons:
        print(f"\n{'-'*70}")
        print(f"HORIZON: {horizon} steps ({horizon * 0.1:.1f}s)")
        print(f"{'-'*70}")

        mae, preds, truth = evaluate_on_holdout(
            model, scaler_X, scaler_y, horizon
        )

        all_results[horizon] = mae

        print(f"\nPer-state MAE (averaged over {horizon} autoregressive steps):")
        for i, (name, unit) in enumerate(zip(state_names, units)):
            print(f"  {name:8s}: {mae[i]:.6f} {unit}")

    # Compare to baseline
    print("\n" + "="*70)
    print("COMPARISON TO BASELINE (100-step, held-out test)")
    print("="*70)

    baseline_100step = {
        'z': 1.49,
        'roll': 0.018,
        'pitch': 0.003,
        'yaw': 0.032,
        'p': 0.067,
        'q': 0.167,
        'r': 0.084,
        'vz': 1.55
    }

    opt_mae = all_results[100]
    print(f"\n{'State':<10} {'Baseline':<15} {'Optimized v2':<15} {'Change':<15}")
    print("-"*70)

    changes = []
    for i, name in enumerate(state_names):
        baseline_val = baseline_100step[name]
        opt_val = opt_mae[i]
        change = (baseline_val - opt_val) / baseline_val * 100

        changes.append(change)

        status = "BETTER" if change > 0 else "WORSE"
        print(f"{name:<10} {baseline_val:.4f} {units[i]:<6} {opt_val:.4f} {units[i]:<6} {change:+.1f}% [{status}]")

    avg_change = np.mean(changes)
    print("\n" + "="*70)
    print(f"Average performance change: {avg_change:+.1f}%")

    if avg_change > 10:
        print("Result: SIGNIFICANT IMPROVEMENT over baseline")
    elif avg_change > 0:
        print("Result: MARGINAL IMPROVEMENT over baseline")
    elif avg_change > -10:
        print("Result: COMPARABLE to baseline")
    else:
        print("Result: WORSE than baseline")

    print("="*70)

    print("\n" + "="*70)
    print("IMPORTANT NOTES:")
    print("="*70)
    print("1. This evaluation uses HELD-OUT test data (last 20%)")
    print("2. The model was trained ONLY on the first 80%")
    print("3. This is a time-based split preserving continuous trajectories")
    print("4. Results show TRUE generalization performance")
    print("="*70)


if __name__ == "__main__":
    main()
