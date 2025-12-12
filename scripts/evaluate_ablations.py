"""
Quick evaluation script for ablation models
"""
import torch
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import sys

sys.path.append(str(Path(__file__).parent))
from pinn_architectures import BaselinePINN, ModularPINN
from run_comprehensive_ablations import PureNNBaseline, SmallBaseline, load_data, load_trajectories

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / 'data' / 'train_set_diverse.csv'
VAL_DATA = PROJECT_ROOT / 'data' / 'val_set_diverse.csv'
MODELS_DIR = PROJECT_ROOT / 'models' / 'comprehensive_ablation'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'comprehensive_ablation'

def autoregressive_rollout(model, initial_state, controls, scaler_X, scaler_y, n_steps):
    """Autoregressive rollout"""
    model.eval()
    x_mean, x_scale = scaler_X.mean_, scaler_X.scale_
    y_mean, y_scale = scaler_y.mean_, scaler_y.scale_

    states = [initial_state.copy()]
    current_state = initial_state.copy()

    with torch.no_grad():
        for i in range(min(n_steps, len(controls))):
            state_controls = np.concatenate([current_state, controls[i]])
            state_controls_scaled = (state_controls - x_mean) / x_scale
            input_tensor = torch.FloatTensor(state_controls_scaled).unsqueeze(0)
            next_state_scaled = model(input_tensor).squeeze(0).numpy()
            next_state = next_state_scaled * y_scale + y_mean
            states.append(next_state)
            current_state = next_state

    return np.array(states)

def evaluate_model(model, val_loader, val_trajectories, scaler_X, scaler_y):
    """Evaluate model"""
    model.eval()
    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    # Single-step
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_scaled, y_scaled, X_unscaled, y_unscaled in val_loader:
            y_pred_scaled = model(X_scaled)
            y_pred = y_pred_scaled * y_scale + y_mean
            all_preds.append(y_pred.numpy())
            all_true.append(y_unscaled.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    single_step = {
        'z_mae': float(np.mean(np.abs(all_preds[:, 2] - all_true[:, 2]))),
        'phi_mae': float(np.mean(np.abs(all_preds[:, 3] - all_true[:, 3]))),
    }

    # Rollout
    pos_errors = []
    for traj in val_trajectories[:10]:
        states, controls = traj['states'], traj['controls']
        if len(states) < 101:
            continue
        predicted = autoregressive_rollout(model, states[0], controls, scaler_X, scaler_y, 100)
        pos_errors.append(np.mean(np.abs(predicted[:, :3] - states[:len(predicted), :3])))

    rollout = {'position_mae': float(np.mean(pos_errors)) if pos_errors else float('inf')}
    return {'single_step': single_step, 'rollout': rollout}

def main():
    print("="*70)
    print("EVALUATING ABLATION MODELS")
    print("="*70)

    # Load data
    X_train, y_train = load_data(TRAIN_DATA)
    X_val, y_val = load_data(VAL_DATA)
    val_trajectories = load_trajectories(VAL_DATA)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)

    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_scaled),
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    results = {}

    # Evaluate each model
    models_to_eval = {
        'PureNN': (PureNNBaseline, 'PureNN_seed42.pth'),
        'PINN_Baseline': (BaselinePINN, 'PINN_Baseline_seed42.pth'),
        'SmallBaseline_72K': (SmallBaseline, 'SmallBaseline_72K_seed42.pth'),
        'Modular_72K': (ModularPINN, 'Modular_72K_seed42.pth'),
        'Baseline': (BaselinePINN, 'Baseline_seed42.pth'),
    }

    for name, (ModelClass, filename) in models_to_eval.items():
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            print(f"  {name}: NOT FOUND")
            continue

        print(f"\nEvaluating {name}...")
        model = ModelClass()
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

        res = evaluate_model(model, val_loader, val_trajectories, scaler_X, scaler_y)
        n_params = sum(p.numel() for p in model.parameters())
        res['n_params'] = n_params
        results[name] = res

        print(f"  Params: {n_params:,}")
        print(f"  1-step z MAE: {res['single_step']['z_mae']:.4f}")
        print(f"  100-step pos MAE: {res['rollout']['position_mae']:.4f}")

    # Least squares
    print("\nEvaluating Least Squares...")
    ls_model = Ridge(alpha=1.0)
    ls_model.fit(X_train, y_train)
    y_pred = ls_model.predict(X_val)

    ls_single = {
        'z_mae': float(np.mean(np.abs(y_pred[:, 2] - y_val[:, 2]))),
        'phi_mae': float(np.mean(np.abs(y_pred[:, 3] - y_val[:, 3]))),
    }

    # LS rollout
    pos_errors = []
    for traj in val_trajectories[:10]:
        states, controls = traj['states'], traj['controls']
        if len(states) < 101:
            continue
        predicted = [states[0]]
        current = states[0].copy()
        for i in range(min(100, len(controls))):
            inp = np.concatenate([current, controls[i]]).reshape(1, -1)
            current = ls_model.predict(inp)[0]
            predicted.append(current)
        predicted = np.array(predicted)
        pos_errors.append(np.mean(np.abs(predicted[:, :3] - states[:len(predicted), :3])))

    results['LeastSquares'] = {
        'single_step': ls_single,
        'rollout': {'position_mae': float(np.mean(pos_errors))}
    }
    print(f"  1-step z MAE: {ls_single['z_mae']:.4f}")
    print(f"  100-step pos MAE: {results['LeastSquares']['rollout']['position_mae']:.4f}")

    # Print summary tables
    print("\n" + "="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)

    print("\n--- Table 1: Physics Loss Ablation ---")
    print(f"{'Model':<20} {'Params':<10} {'1-Step z':<12} {'100-Step Pos':<12}")
    print("-"*55)
    for name in ['PureNN', 'PINN_Baseline']:
        if name in results:
            r = results[name]
            print(f"{name:<20} {r.get('n_params', 'N/A'):<10} {r['single_step']['z_mae']:<12.4f} {r['rollout']['position_mae']:<12.4f}")

    print("\n--- Table 2: Parameter-Matched Comparison ---")
    print(f"{'Model':<20} {'Params':<10} {'1-Step z':<12} {'100-Step Pos':<12}")
    print("-"*55)
    for name in ['SmallBaseline_72K', 'Modular_72K']:
        if name in results:
            r = results[name]
            print(f"{name:<20} {r.get('n_params', 'N/A'):<10} {r['single_step']['z_mae']:<12.4f} {r['rollout']['position_mae']:<12.4f}")

    print("\n--- Table 3: Comparison with Traditional Methods ---")
    print(f"{'Model':<20} {'1-Step z':<12} {'100-Step Pos':<12}")
    print("-"*45)
    for name in ['LeastSquares', 'Modular_72K']:
        if name in results:
            r = results[name]
            print(f"{name:<20} {r['single_step']['z_mae']:<12.4f} {r['rollout']['position_mae']:<12.4f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR / 'ablation_results.json'}")
    return results

if __name__ == '__main__':
    results = main()
