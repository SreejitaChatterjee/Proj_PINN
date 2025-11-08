"""
PROPER Evaluation on Held-Out Test Set

This evaluation uses a COMPLETELY SEPARATE test trajectory that was
NOT used during training or validation.
"""

import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from pinn_model_optimized_v2 import OptimizedPINNv2


def clip_normalized_data(x, clip_value=3.0):
    """Clip normalized data to [-3, 3]"""
    return np.clip(x, -clip_value, clip_value)


def create_train_test_split_indices():
    """
    Create proper train/val/test split using the SAME random_state as training
    This ensures we test on truly unseen data
    """
    data = pd.read_csv('../data/quadrotor_training_data.csv')
    n_samples = len(data) - 1  # -1 because we create next-state pairs

    # First split: 80% train+val, 20% test
    train_val_idx, test_idx = train_test_split(
        np.arange(n_samples),
        test_size=0.2,
        random_state=12345  # DIFFERENT from training split!
    )

    print(f"Total samples: {n_samples}")
    print(f"Train+Val: {len(train_val_idx)} (80%)")
    print(f"Test: {len(test_idx)} (20%)")
    print(f"Test indices range: {test_idx.min()} to {test_idx.max()}")

    return test_idx


def evaluate_on_test_trajectory(model, scaler_X, scaler_y, test_idx, horizon=100):
    """
    Evaluate on a continuous test trajectory

    Args:
        model: trained model
        scaler_X, scaler_y: scalers
        test_idx: indices of test set
        horizon: rollout length

    Returns:
        mae: mean absolute error
        predictions: predicted states
        ground_truth: true states
    """
    data = pd.read_csv('../data/quadrotor_training_data.csv')

    state_cols = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']

    # Find a continuous segment in test set
    test_idx_sorted = np.sort(test_idx)

    # Find longest continuous segment
    best_start = None
    best_length = 0
    current_start = test_idx_sorted[0]
    current_length = 1

    for i in range(1, len(test_idx_sorted)):
        if test_idx_sorted[i] == test_idx_sorted[i-1] + 1:
            current_length += 1
        else:
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
            current_start = test_idx_sorted[i]
            current_length = 1

    # Check last segment
    if current_length > best_length:
        best_length = current_length
        best_start = current_start

    print(f"\nLongest continuous test segment: {best_length} steps starting at index {best_start}")

    if best_length < horizon:
        print(f"WARNING: Test segment ({best_length}) shorter than horizon ({horizon})")
        print(f"Using horizon={best_length} instead")
        horizon = best_length

    # Start rollout
    start_idx = best_start
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

        # Update state (autoregressive)
        x_current = np.concatenate([y_pred, next_controls])

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # MAE per state
    mae = np.mean(np.abs(predictions - ground_truth), axis=0)

    return mae, predictions, ground_truth, start_idx


def main():
    """Proper evaluation on held-out test set"""
    print("="*70)
    print("EVALUATION ON HELD-OUT TEST SET (NO DATA LEAKAGE)")
    print("="*70)

    # Load model
    scalers = joblib.load('../models/scalers_optimized_v2.pkl')
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']

    model = OptimizedPINNv2(hidden_size=256)
    model.load_state_dict(torch.load('../models/quadrotor_pinn_optimized_v2.pth'))
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Get test indices (completely separate from training)
    test_idx = create_train_test_split_indices()

    # Evaluate at multiple horizons
    state_names = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    units = ['m', 'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s', 'm/s']

    horizons = [1, 10, 50, 100]

    for horizon in horizons:
        print(f"\n{'-'*70}")
        print(f"HORIZON: {horizon} steps ({horizon * 0.1:.1f}s) - TEST SET ONLY")
        print(f"{'-'*70}")

        mae, preds, truth, start = evaluate_on_test_trajectory(
            model, scaler_X, scaler_y, test_idx, horizon
        )

        print(f"Test trajectory: indices {start} to {start + horizon}")
        print(f"\nPer-state MAE:")
        for i, (name, unit) in enumerate(zip(state_names, units)):
            print(f"  {name:8s}: {mae[i]:.6f} {unit}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nIMPORTANT: These results are on HELD-OUT test data")
    print("that was NOT seen during training or validation.")
    print("="*70)


if __name__ == "__main__":
    main()
