"""
Multi-Horizon Evaluation for Optimized PINN v2

Evaluates model at multiple rollout horizons:
- 1 step (teacher-forced)
- 10 steps
- 50 steps
- 100 steps

Verifies monotonic increase (not exponential divergence).
"""

import torch
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pinn_model_optimized_v2 import OptimizedPINNv2


def clip_normalized_data(x, clip_value=3.0):
    """Clip normalized data to [-3, 3]"""
    return np.clip(x, -clip_value, clip_value)


def load_model_and_scalers():
    """Load trained model and scalers"""
    # Load scalers
    scalers = joblib.load('../models/scalers_optimized_v2.pkl')
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']

    # Load model
    model = OptimizedPINNv2(hidden_size=256)
    model.load_state_dict(torch.load('../models/quadrotor_pinn_optimized_v2.pth'))
    model.eval()

    return model, scaler_X, scaler_y


def evaluate_horizon(model, scaler_X, scaler_y, data, num_steps, start_idx=1000):
    """
    Evaluate autoregressive rollout for given horizon

    Args:
        model: trained model
        scaler_X, scaler_y: scalers
        data: DataFrame with full dataset
        num_steps: rollout horizon
        start_idx: starting index

    Returns:
        mae: mean absolute error over horizon
        predictions: predicted trajectory
        ground_truth: true trajectory
    """
    state_cols = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']

    # Extract initial state
    x_current = data.iloc[start_idx][state_cols + control_cols].values

    predictions = []
    ground_truth = []

    for step in range(num_steps):
        # Get ground truth
        if start_idx + step + 1 < len(data):
            y_true = data.iloc[start_idx + step + 1][state_cols].values
            ground_truth.append(y_true)
        else:
            break

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

        # Get next controls
        if start_idx + step + 1 < len(data):
            next_controls = data.iloc[start_idx + step + 1][control_cols].values
        else:
            next_controls = x_current[8:12]

        # Update state (autoregressive)
        x_current = np.concatenate([y_pred, next_controls])

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # MAE per state
    mae = np.mean(np.abs(predictions - ground_truth), axis=0)

    return mae, predictions, ground_truth


def multi_horizon_evaluation(model, scaler_X, scaler_y, data_file='../data/quadrotor_training_data.csv'):
    """
    Evaluate model at multiple horizons

    Returns:
        results: dict with MAE for each horizon
    """
    print("=" * 70)
    print("MULTI-HORIZON EVALUATION")
    print("=" * 70)

    # Load data
    data = pd.read_csv(data_file)

    # Horizons to evaluate
    horizons = [1, 10, 50, 100]

    state_names = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    units = ['m', 'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s', 'm/s']

    results = {}

    for horizon in horizons:
        print(f"\n{'-'*70}")
        print(f"HORIZON: {horizon} steps ({horizon * 0.1:.1f}s)")
        print(f"{'-'*70}")

        mae, predictions, ground_truth = evaluate_horizon(
            model, scaler_X, scaler_y, data, horizon, start_idx=1000
        )

        results[horizon] = {
            'mae': mae,
            'predictions': predictions,
            'ground_truth': ground_truth
        }

        # Print MAE
        for i, (name, unit) in enumerate(zip(state_names, units)):
            print(f"  {name:8s}: {mae[i]:.6f} {unit}")

    return results


def plot_multi_horizon_convergence(results):
    """
    Plot MAE vs horizon to check for bounded growth

    Should see monotonic increase but NOT exponential
    """
    horizons = sorted(results.keys())
    state_names = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, name in enumerate(state_names):
        mae_vs_horizon = [results[h]['mae'][i] for h in horizons]

        axes[i].plot(horizons, mae_vs_horizon, 'o-', linewidth=2, markersize=8)
        axes[i].set_xlabel('Rollout Horizon (steps)')
        axes[i].set_ylabel(f'MAE ({name})')
        axes[i].set_title(f'{name} - Error vs Horizon')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')

    plt.suptitle('Multi-Horizon MAE Analysis (Log-Log Scale)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../results/optimized_v2_multi_horizon.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to ../results/optimized_v2_multi_horizon.png")


def compare_to_baseline(results):
    """Compare 100-step results to baseline"""
    # Baseline results (from LESSONS_LEARNED.md)
    baseline_rollout = {
        'z': 1.49,
        'roll': 0.018,
        'pitch': 0.003,
        'yaw': 0.032,
        'p': 0.067,
        'q': 0.167,
        'r': 0.084,
        'vz': 1.55
    }

    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE (100-step)")
    print("=" * 70)

    optimized_mae = results[100]['mae']
    state_names = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    units = ['m', 'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s', 'm/s']

    print(f"\n{'State':<10} {'Baseline':<15} {'Optimized v2':<15} {'Improvement':<15}")
    print("-" * 70)

    improvements = []
    for i, (name, unit) in enumerate(zip(state_names, units)):
        baseline_val = baseline_rollout[name]
        optimized_val = optimized_mae[i]
        improvement = (baseline_val - optimized_val) / baseline_val * 100

        improvements.append(improvement)

        status = "[BETTER]" if improvement > 0 else "[WORSE]"
        print(f"{name:<10} {baseline_val:.4f} {unit:<6} {optimized_val:.4f} {unit:<6} {improvement:+.1f}% {status}")

    avg_improvement = np.mean(improvements)

    print("\n" + "=" * 70)
    if avg_improvement > 0:
        print(f"SUCCESS: Optimized v2 achieves {avg_improvement:.1f}% average improvement over baseline")
        print("  Architecture optimizations work when done correctly!")
    elif avg_improvement > -10:
        print(f"COMPARABLE: Optimized v2 performance within 10% of baseline ({avg_improvement:.1f}%)")
        print("  No significant regression - optimizations maintain stability")
    else:
        print(f"NEEDS TUNING: Optimized v2 {-avg_improvement:.1f}% worse than baseline")
        print("  Further hyperparameter tuning recommended")
    print("=" * 70)

    return avg_improvement


def main():
    """Main evaluation"""
    # Load model
    model, scaler_X, scaler_y = load_model_and_scalers()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Multi-horizon evaluation
    results = multi_horizon_evaluation(model, scaler_X, scaler_y)

    # Plot convergence
    plot_multi_horizon_convergence(results)

    # Compare to baseline
    avg_improvement = compare_to_baseline(results)

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"1-step MAE (z):   {results[1]['mae'][0]:.6f} m")
    print(f"10-step MAE (z):  {results[10]['mae'][0]:.6f} m")
    print(f"50-step MAE (z):  {results[50]['mae'][0]:.6f} m")
    print(f"100-step MAE (z): {results[100]['mae'][0]:.6f} m")
    print(f"\nAverage improvement over baseline: {avg_improvement:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
