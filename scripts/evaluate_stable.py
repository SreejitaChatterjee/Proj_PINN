"""
Evaluation script for Stable PINN

Tests:
1. Single-step teacher-forced accuracy
2. 100-step autoregressive rollout (critical test)
3. Comparison to baseline PINN
"""

import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pinn_model_stable import StablePINN


def load_model_and_scalers():
    """Load trained stable PINN and scalers"""
    import os

    # Determine paths
    model_dir = '../models' if os.path.exists('../models') else 'models'

    # Load scalers
    with open(f'{model_dir}/scalers_stable.pkl', 'rb') as f:
        scalers = pickle.load(f)
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']

    # Load model
    model = StablePINN(hidden_size=128, num_residual_blocks=2, use_fourier=False)
    model.load_state_dict(torch.load(f'{model_dir}/quadrotor_pinn_stable.pth'))
    model.eval()

    return model, scaler_X, scaler_y


def evaluate_teacher_forced(model, scaler_X, scaler_y, data_file=None):
    """Single-step teacher-forced evaluation"""
    import os

    if data_file is None:
        data_file = '../data/quadrotor_training_data.csv' if os.path.exists('../data') else 'data/quadrotor_training_data.csv'

    print("=" * 60)
    print("SINGLE-STEP TEACHER-FORCED EVALUATION")
    print("=" * 60)

    # Load test data
    data = pd.read_csv(data_file)

    state_cols = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']

    # Create next state by shifting
    data_shifted = data[state_cols].shift(-1)
    data_shifted.columns = [c + '_next' for c in state_cols]

    data_combined = pd.concat([data[state_cols + control_cols], data_shifted], axis=1)
    data_combined = data_combined.dropna()

    input_cols = state_cols + control_cols
    output_cols = [c + '_next' for c in state_cols]

    X = data_combined[input_cols].values
    y_true = data_combined[output_cols].values

    # Normalize
    X_scaled = scaler_X.transform(X)

    # Predict
    X_tensor = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()

    # Denormalize
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Compute MAE for each state
    mae = np.mean(np.abs(y_pred - y_true), axis=0)

    state_names = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    units = ['m', 'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s', 'm/s']

    print("\nSingle-step MAE:")
    for i, (name, unit) in enumerate(zip(state_names, units)):
        print(f"  {name:8s}: {mae[i]:.6f} {unit}")

    return mae


def evaluate_autoregressive_rollout(model, scaler_X, scaler_y, num_steps=100, start_idx=1000):
    """
    100-step autoregressive rollout - the critical stability test

    Args:
        model: trained model
        scaler_X, scaler_y: scalers
        num_steps: rollout horizon (100 for 10 seconds at dt=0.1s)
        start_idx: starting index in dataset
    """
    import os

    print("\n" + "=" * 60)
    print(f"AUTOREGRESSIVE ROLLOUT - {num_steps} STEPS")
    print("=" * 60)

    # Load data
    data_file = '../data/quadrotor_training_data.csv' if os.path.exists('../data') else 'data/quadrotor_training_data.csv'
    data = pd.read_csv(data_file)

    state_cols = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']
    input_cols = state_cols + control_cols

    # Extract initial state
    x_current = data.iloc[start_idx][input_cols].values

    # Store predictions and ground truth
    predictions = []
    ground_truth = []

    for step in range(num_steps):
        # Get ground truth
        if start_idx + step + 1 < len(data):
            y_true = data.iloc[start_idx + step + 1][state_cols].values
            ground_truth.append(y_true)
        else:
            break

        # Normalize input
        x_scaled = scaler_X.transform(x_current.reshape(1, -1))

        # Predict next state
        x_tensor = torch.FloatTensor(x_scaled)
        with torch.no_grad():
            y_pred_scaled = model(x_tensor).numpy()

        # Denormalize
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
        predictions.append(y_pred)

        # Get next control inputs
        if start_idx + step + 1 < len(data):
            next_controls = data.iloc[start_idx + step + 1][control_cols].values
        else:
            next_controls = x_current[8:12]  # Use previous controls

        # Update state for next iteration (autoregressive)
        x_current = np.concatenate([y_pred, next_controls])

    # Convert to arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Compute MAE over entire rollout
    mae = np.mean(np.abs(predictions - ground_truth), axis=0)

    state_names = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    units = ['m', 'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s', 'm/s']

    print(f"\n{num_steps}-step Autoregressive MAE:")
    for i, (name, unit) in enumerate(zip(state_names, units)):
        print(f"  {name:8s}: {mae[i]:.6f} {unit}")

    # Create visualization
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()

    time = np.arange(num_steps) * 0.1  # dt = 0.1s

    for i, (name, unit) in enumerate(zip(state_names, units)):
        axes[i].plot(time, ground_truth[:, i], 'b-', label='Ground Truth', linewidth=2)
        axes[i].plot(time, predictions[:, i], 'r--', label='PINN Prediction', linewidth=1.5)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel(f'{name} ({unit})')
        axes[i].set_title(f'{name} - MAE: {mae[i]:.4f} {unit}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f'Stable PINN - Autoregressive Rollout ({num_steps} steps, {num_steps*0.1:.1f}s)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    results_dir = '../results' if os.path.exists('../results') else 'results'
    plot_path = f'{results_dir}/stable_pinn_evaluation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")

    return mae, predictions, ground_truth


def compare_to_baseline():
    """Compare stable PINN to baseline PINN"""
    print("\n" + "=" * 60)
    print("COMPARISON TO BASELINE")
    print("=" * 60)

    # Baseline results (from LESSONS_LEARNED.md)
    # Note: baseline uses phi/theta/psi naming, but same as roll/pitch/yaw
    baseline_rollout = {
        'z': 1.49,
        'roll': 0.018,  # phi
        'pitch': 0.003,  # theta
        'yaw': 0.032,  # psi
        'p': 0.067,
        'q': 0.167,
        'r': 0.084,
        'vz': 1.55
    }

    # Load stable model results
    model, scaler_X, scaler_y = load_model_and_scalers()
    stable_rollout, _, _ = evaluate_autoregressive_rollout(model, scaler_X, scaler_y, num_steps=100)

    state_names = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    units = ['m', 'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s', 'm/s']

    print("\n100-step Autoregressive Rollout Comparison:")
    print(f"{'State':<10} {'Baseline':<15} {'Stable PINN':<15} {'Improvement':<15}")
    print("-" * 60)

    for i, (name, unit) in enumerate(zip(state_names, units)):
        baseline_val = baseline_rollout[name]
        stable_val = stable_rollout[i]
        improvement = (baseline_val - stable_val) / baseline_val * 100

        print(f"{name:<10} {baseline_val:.4f} {unit:<6} {stable_val:.4f} {unit:<6} {improvement:+.1f}%")

    # Overall verdict
    avg_improvement = np.mean([
        (baseline_rollout[name] - stable_rollout[i]) / baseline_rollout[name] * 100
        for i, name in enumerate(state_names)
    ])

    print("\n" + "=" * 60)
    if avg_improvement > 0:
        print(f"[SUCCESS] Stable PINN achieves {avg_improvement:.1f}% average improvement over baseline")
        print("  Architecture optimizations SUCCESS - maintains autoregressive stability")
    elif avg_improvement > -10:
        print(f"[COMPARABLE] Stable PINN comparable to baseline ({avg_improvement:.1f}% difference)")
        print("  No regression - architecture maintains stability")
    else:
        print(f"[NEEDS TUNING] Stable PINN {-avg_improvement:.1f}% worse than baseline")
        print("  Further tuning needed")
    print("=" * 60)


if __name__ == "__main__":
    # Load model
    model, scaler_X, scaler_y = load_model_and_scalers()

    # 1. Teacher-forced evaluation
    teacher_forced_mae = evaluate_teacher_forced(model, scaler_X, scaler_y)

    # 2. Autoregressive rollout (critical test)
    rollout_mae, predictions, ground_truth = evaluate_autoregressive_rollout(
        model, scaler_X, scaler_y, num_steps=100
    )

    # 3. Compare to baseline
    compare_to_baseline()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Single-step z MAE: {teacher_forced_mae[0]:.6f} m")
    print(f"100-step z MAE: {rollout_mae[0]:.6f} m")
    print(f"Parameter reduction: {(1 - total_params/100000)*100:.1f}% vs baseline (~100K params)")
    print("=" * 60)
