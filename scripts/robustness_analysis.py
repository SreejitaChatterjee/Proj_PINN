"""
Robustness Analysis for PINN Paper

This script implements:
A. Robustness ablations (noise injection, dropout, OOD)
B. Jacobian histogram plots (sigma_max distributions)
C. Failure mode visualization (divergence plots)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append(str(Path(__file__).parent))
from pinn_architectures import BaselinePINN, ModularPINN
from run_comprehensive_ablations import PureNNBaseline, load_data, load_trajectories

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / 'data' / 'train_set_diverse.csv'
VAL_DATA = PROJECT_ROOT / 'data' / 'val_set_diverse.csv'
MODELS_DIR = PROJECT_ROOT / 'models' / 'comprehensive_ablation'
ARCH_MODELS_DIR = PROJECT_ROOT / 'models' / 'architecture_comparison'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'robustness_analysis'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def autoregressive_rollout(model, initial_state, controls, scaler_X, scaler_y, n_steps,
                           noise_std=0.0, dropout_rate=0.0):
    """
    Autoregressive rollout with optional noise injection and dropout.

    Args:
        noise_std: Gaussian noise std as fraction of feature std
        dropout_rate: Fraction of state dimensions to zero out
    """
    model.eval()
    x_mean, x_scale = scaler_X.mean_, scaler_X.scale_
    y_mean, y_scale = scaler_y.mean_, scaler_y.scale_

    states = [initial_state.copy()]
    current_state = initial_state.copy()

    with torch.no_grad():
        for i in range(min(n_steps, len(controls))):
            # Apply noise to state
            if noise_std > 0:
                noise = np.random.randn(12) * noise_std * y_scale[:12]
                current_state = current_state + noise

            # Apply dropout (zero out some state dimensions)
            if dropout_rate > 0:
                mask = np.random.rand(12) > dropout_rate
                current_state = current_state * mask

            state_controls = np.concatenate([current_state, controls[i]])
            state_controls_scaled = (state_controls - x_mean) / x_scale

            input_tensor = torch.FloatTensor(state_controls_scaled).unsqueeze(0)
            next_state_scaled = model(input_tensor).squeeze(0).numpy()
            next_state = next_state_scaled * y_scale + y_mean

            states.append(next_state)
            current_state = next_state

    return np.array(states)


def compute_jacobian_spectral_norm(model, x, scaler_X):
    """Compute spectral norm of Jacobian at point x"""
    x_scaled = (x - scaler_X.mean_) / scaler_X.scale_
    x_tensor = torch.FloatTensor(x_scaled).unsqueeze(0).requires_grad_(True)

    y = model(x_tensor)

    # Compute Jacobian via backward passes
    J = torch.zeros(12, 16)
    for i in range(12):
        if x_tensor.grad is not None:
            x_tensor.grad.zero_()
        y[0, i].backward(retain_graph=True)
        J[i] = x_tensor.grad[0].clone()

    # State-to-state Jacobian (12x12)
    J_state = J[:, :12]

    # Spectral norm
    sigma_max = torch.linalg.svdvals(J_state)[0].item()
    return sigma_max


# ============================================================================
# A. ROBUSTNESS ABLATIONS
# ============================================================================

def run_robustness_ablations(models, val_trajectories, scaler_X, scaler_y):
    """
    Test model robustness under:
    1. Noise injection (Gaussian 5% std)
    2. Sensor dropout (10% of states)
    3. OOD initial conditions
    """
    print("\n" + "="*70)
    print("A. ROBUSTNESS ABLATIONS")
    print("="*70)

    results = {}

    # Test conditions
    conditions = {
        'Clean': {'noise_std': 0.0, 'dropout_rate': 0.0},
        'Noise_5pct': {'noise_std': 0.05, 'dropout_rate': 0.0},
        'Dropout_10pct': {'noise_std': 0.0, 'dropout_rate': 0.1},
        'Both': {'noise_std': 0.05, 'dropout_rate': 0.1},
    }

    for model_name, model in models.items():
        print(f"\n{model_name}:")
        results[model_name] = {}

        for cond_name, cond_params in conditions.items():
            pos_errors = []

            for traj in val_trajectories[:10]:
                states, controls = traj['states'], traj['controls']
                if len(states) < 101:
                    continue

                predicted = autoregressive_rollout(
                    model, states[0], controls, scaler_X, scaler_y, 100,
                    noise_std=cond_params['noise_std'],
                    dropout_rate=cond_params['dropout_rate']
                )
                pos_errors.append(np.mean(np.abs(predicted[:, :3] - states[:len(predicted), :3])))

            mean_error = np.mean(pos_errors)
            results[model_name][cond_name] = mean_error
            print(f"  {cond_name}: {mean_error:.4f}m")

    # OOD initial conditions (states 50% outside training bounds)
    print("\nOOD Initial Conditions (1.5x training bounds):")
    for model_name, model in models.items():
        pos_errors = []

        for traj in val_trajectories[:10]:
            states, controls = traj['states'], traj['controls']
            if len(states) < 101:
                continue

            # Perturb initial state to be OOD
            ood_initial = states[0].copy()
            ood_initial[:3] *= 1.5  # Position 50% larger
            ood_initial[3:6] *= 1.5  # Angles 50% larger

            predicted = autoregressive_rollout(
                model, ood_initial, controls, scaler_X, scaler_y, 100
            )
            # Compare to trajectory from OOD start (use predicted as reference)
            pos_errors.append(np.mean(np.abs(predicted[1:, :3] - predicted[:-1, :3])))

        results[model_name]['OOD'] = np.mean(pos_errors)
        print(f"  {model_name}: {results[model_name]['OOD']:.4f}m (state change rate)")

    return results


# ============================================================================
# B. JACOBIAN HISTOGRAMS
# ============================================================================

def plot_jacobian_histograms(models, scaler_X, n_samples=500):
    """
    Plot sigma_max distributions for each model.
    """
    print("\n" + "="*70)
    print("B. JACOBIAN HISTOGRAM ANALYSIS")
    print("="*70)

    # Sample states from training distribution
    state_bounds = {
        'pos': (-2, 2),
        'ang': (-0.5, 0.5),
        'rate': (-2, 2),
        'vel': (-2, 2),
        'thrust': (0.5, 1.0),
        'torque': (-0.1, 0.1)
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'PureNN': 'blue', 'PINN': 'red', 'Modular': 'green'}

    all_sigmas = {}

    for idx, (model_name, model) in enumerate(models.items()):
        print(f"\nSampling Jacobians for {model_name}...")
        sigmas = []

        for _ in range(n_samples):
            x = np.zeros(16)
            x[0:3] = np.random.uniform(*state_bounds['pos'], 3)
            x[3:6] = np.random.uniform(*state_bounds['ang'], 3)
            x[6:9] = np.random.uniform(*state_bounds['rate'], 3)
            x[9:12] = np.random.uniform(*state_bounds['vel'], 3)
            x[12] = np.random.uniform(*state_bounds['thrust'])
            x[13:16] = np.random.uniform(*state_bounds['torque'], 3)

            try:
                sigma = compute_jacobian_spectral_norm(model, x, scaler_X)
                sigmas.append(sigma)
            except:
                continue

        sigmas = np.array(sigmas)
        all_sigmas[model_name] = sigmas

        # Plot histogram
        ax = axes[idx]
        ax.hist(sigmas, bins=50, alpha=0.7, color=colors.get(model_name, 'gray'),
                edgecolor='black', linewidth=0.5)
        ax.axvline(np.mean(sigmas), color='black', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(sigmas):.3f}')
        ax.axvline(np.percentile(sigmas, 95), color='red', linestyle=':', linewidth=2,
                   label=f'P95: {np.percentile(sigmas, 95):.3f}')
        ax.set_xlabel(r'$\sigma_{max}(J)$ (Lipschitz constant)')
        ax.set_ylabel('Count')
        ax.set_title(f'{model_name}\n(n={len(sigmas)})')
        ax.legend()

        print(f"  Mean: {np.mean(sigmas):.4f}, P95: {np.percentile(sigmas, 95):.4f}, Max: {np.max(sigmas):.4f}")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'jacobian_histograms.png', dpi=150, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'jacobian_histograms.pdf', bbox_inches='tight')
    print(f"\nSaved: {RESULTS_DIR / 'jacobian_histograms.png'}")

    # Combined plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, sigmas in all_sigmas.items():
        ax.hist(sigmas, bins=50, alpha=0.5, label=f'{model_name} (μ={np.mean(sigmas):.2f})',
                color=colors.get(model_name, 'gray'))
    ax.set_xlabel(r'$\sigma_{max}(J)$ (Lipschitz constant)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Jacobian Spectral Norm Distribution by Architecture', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'jacobian_histograms_combined.png', dpi=150, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'jacobian_histograms_combined.pdf', bbox_inches='tight')

    return all_sigmas


# ============================================================================
# C. FAILURE MODE VISUALIZATION
# ============================================================================

def plot_failure_modes(models, val_trajectories, scaler_X, scaler_y):
    """
    Visualize divergence vs stable rollouts.
    """
    print("\n" + "="*70)
    print("C. FAILURE MODE VISUALIZATION")
    print("="*70)

    # Pick a trajectory
    traj = val_trajectories[0]
    states, controls = traj['states'], traj['controls']
    n_steps = min(100, len(states) - 1)

    # Run rollouts for each model
    rollouts = {}
    for model_name, model in models.items():
        rollouts[model_name] = autoregressive_rollout(
            model, states[0], controls, scaler_X, scaler_y, n_steps
        )

    # Plot 1: Position error over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'PureNN': 'blue', 'PINN': 'red', 'Modular': 'green', 'Ground Truth': 'black'}

    # Position error
    ax = axes[0, 0]
    for model_name, pred in rollouts.items():
        true = states[:len(pred)]
        pos_error = np.linalg.norm(pred[:, :3] - true[:, :3], axis=1)
        ax.plot(pos_error, label=model_name, color=colors.get(model_name, 'gray'), linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error Over Autoregressive Rollout')
    ax.legend()
    ax.set_yscale('log')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='1m threshold')

    # Z trajectory
    ax = axes[0, 1]
    ax.plot(states[:n_steps+1, 2], 'k-', linewidth=2, label='Ground Truth')
    for model_name, pred in rollouts.items():
        ax.plot(pred[:, 2], '--', label=model_name, color=colors.get(model_name, 'gray'), linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Z Position (m)')
    ax.set_title('Z Position: Ground Truth vs Predictions')
    ax.legend()

    # Roll angle
    ax = axes[1, 0]
    ax.plot(states[:n_steps+1, 3], 'k-', linewidth=2, label='Ground Truth')
    for model_name, pred in rollouts.items():
        ax.plot(pred[:, 3], '--', label=model_name, color=colors.get(model_name, 'gray'), linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Roll Angle (rad)')
    ax.set_title('Roll Angle: Ground Truth vs Predictions')
    ax.legend()

    # Error growth rate
    ax = axes[1, 1]
    for model_name, pred in rollouts.items():
        true = states[:len(pred)]
        pos_error = np.linalg.norm(pred[:, :3] - true[:, :3], axis=1)
        # Compute error growth rate (ratio of consecutive errors)
        growth_rate = pos_error[1:] / (pos_error[:-1] + 1e-8)
        ax.plot(growth_rate[:50], label=model_name, color=colors.get(model_name, 'gray'), linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Error Growth Rate')
    ax.set_title('Error Amplification Factor (should be < 1 for stability)')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_ylim(0, 3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'failure_modes.png', dpi=150, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'failure_modes.pdf', bbox_inches='tight')
    print(f"\nSaved: {RESULTS_DIR / 'failure_modes.png'}")

    # Plot 2: 3D trajectory comparison
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(states[:n_steps+1, 0], states[:n_steps+1, 1], states[:n_steps+1, 2],
            'k-', linewidth=2, label='Ground Truth')
    for model_name, pred in rollouts.items():
        ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], '--',
                label=model_name, color=colors.get(model_name, 'gray'), linewidth=1.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory: Stable vs Divergent Models')
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'trajectory_3d.png', dpi=150, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'trajectory_3d.pdf', bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR / 'trajectory_3d.png'}")

    return rollouts


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("ROBUSTNESS ANALYSIS FOR PINN PAPER")
    print("="*70)

    # Load data
    print("\nLoading data...")
    X_train, y_train = load_data(TRAIN_DATA)
    X_val, y_val = load_data(VAL_DATA)
    val_trajectories = load_trajectories(VAL_DATA)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(X_train)
    scaler_y.fit(y_train)

    # Load models
    print("\nLoading models...")
    models = {}

    # PureNN
    model = PureNNBaseline()
    model_path = MODELS_DIR / 'PureNN_seed42.pth'
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        models['PureNN'] = model
        print("  Loaded PureNN")

    # PINN
    model = BaselinePINN()
    model_path = MODELS_DIR / 'PINN_Baseline_seed42.pth'
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        models['PINN'] = model
        print("  Loaded PINN")

    # Modular
    model = ModularPINN()
    model_path = ARCH_MODELS_DIR / 'modular.pth'
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        models['Modular'] = model
        print("  Loaded Modular")

    if len(models) < 2:
        print("ERROR: Need at least 2 models for comparison")
        return

    # A. Robustness ablations
    robustness_results = run_robustness_ablations(models, val_trajectories, scaler_X, scaler_y)

    # B. Jacobian histograms
    jacobian_results = plot_jacobian_histograms(models, scaler_X, n_samples=300)

    # C. Failure mode visualization
    rollout_results = plot_failure_modes(models, val_trajectories, scaler_X, scaler_y)

    # Save results
    results = {
        'robustness': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in robustness_results.items()},
        'jacobian_stats': {k: {'mean': float(np.mean(v)), 'p95': float(np.percentile(v, 95)),
                               'max': float(np.max(v))} for k, v in jacobian_results.items()}
    }

    with open(RESULTS_DIR / 'robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary tables
    print("\n" + "="*70)
    print("SUMMARY TABLES FOR PAPER")
    print("="*70)

    print("\nTable: Robustness Ablation (100-step Position MAE)")
    print("-"*60)
    print(f"{'Model':<12} {'Clean':<10} {'Noise 5%':<10} {'Drop 10%':<10} {'Both':<10}")
    print("-"*60)
    for model_name in models.keys():
        r = robustness_results[model_name]
        print(f"{model_name:<12} {r['Clean']:<10.3f} {r['Noise_5pct']:<10.3f} {r['Dropout_10pct']:<10.3f} {r['Both']:<10.3f}")

    print("\nTable: Jacobian Spectral Norm Statistics")
    print("-"*50)
    print(f"{'Model':<12} {'Mean':<10} {'P95':<10} {'Max':<10}")
    print("-"*50)
    for model_name, sigmas in jacobian_results.items():
        print(f"{model_name:<12} {np.mean(sigmas):<10.3f} {np.percentile(sigmas, 95):<10.3f} {np.max(sigmas):<10.3f}")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("Figures: jacobian_histograms.png, failure_modes.png, trajectory_3d.png")

    return results


if __name__ == '__main__':
    results = main()
