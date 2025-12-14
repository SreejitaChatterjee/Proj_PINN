"""
ROBUST Physics Weight Sweep - statistically valid results

Goal: Provide publication-quality evidence for physics weight effects
- 3 seeds for statistical significance
- 100 epochs for proper convergence
- 10 trajectories for rollout evaluation
- Reports mean +/- std for all metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent))
from pinn_architectures import BaselinePINN

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / 'data' / 'train_set_diverse.csv'
VAL_DATA = PROJECT_ROOT / 'data' / 'val_set_diverse.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'weight_sweep'
MODELS_DIR = PROJECT_ROOT / 'models' / 'weight_sweep_robust'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ROBUST config
BATCH_SIZE = 512
MAX_EPOCHS = 100  # Proper convergence
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 20

# Sweep config - statistical validity
PHYSICS_WEIGHTS = [0.0, 1.0, 5.0, 10.0, 20.0]  # Key weights
SEEDS = [42, 123, 456]  # 3 seeds for statistics
N_ROLLOUT_TRAJ = 10  # More trajectories


def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.rename(columns={'roll': 'phi', 'pitch': 'theta', 'yaw': 'psi'})
    state_cols = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    input_features = state_cols + ['thrust', 'torque_x', 'torque_y', 'torque_z']
    X, y = [], []
    for traj_id in df['trajectory_id'].unique():
        df_traj = df[df['trajectory_id'] == traj_id].reset_index(drop=True)
        for i in range(len(df_traj) - 1):
            X.append(df_traj.iloc[i][input_features].values)
            y.append(df_traj.iloc[i+1][state_cols].values)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_trajectories(data_path):
    df = pd.read_csv(data_path)
    df = df.rename(columns={'roll': 'phi', 'pitch': 'theta', 'yaw': 'psi'})
    state_cols = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']
    trajectories = []
    for traj_id in df['trajectory_id'].unique():
        df_traj = df[df['trajectory_id'] == traj_id].reset_index(drop=True)
        trajectories.append({
            'states': df_traj[state_cols].values.astype(np.float32),
            'controls': df_traj[control_cols].values.astype(np.float32)
        })
    return trajectories


def train_with_physics_weight(model, train_loader, val_loader, scaler_y, w_phys, seed):
    """Train model with specific physics weight, supervised-only early stopping"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    best_val_sup_loss = float('inf')
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        train_sup_loss = 0
        n_train = 0

        for X_scaled, y_scaled, X_unscaled, y_unscaled in train_loader:
            optimizer.zero_grad()
            y_pred_scaled = model(X_scaled)
            sup_loss = nn.MSELoss()(y_pred_scaled, y_scaled)
            loss = sup_loss

            if w_phys > 0 and hasattr(model, 'physics_loss'):
                y_pred_unscaled = y_pred_scaled * y_scale + y_mean
                phys_loss = model.physics_loss(X_unscaled, y_pred_unscaled)
                loss = sup_loss + w_phys * phys_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            if hasattr(model, 'constrain_parameters'):
                model.constrain_parameters()

            train_sup_loss += sup_loss.item() * X_scaled.size(0)
            n_train += X_scaled.size(0)

        train_sup_loss /= n_train

        # Validation (supervised loss only for early stopping)
        model.eval()
        val_sup_loss = 0
        n_val = 0

        with torch.no_grad():
            for X_scaled, y_scaled, X_unscaled, y_unscaled in val_loader:
                y_pred_scaled = model(X_scaled)
                sup_loss = nn.MSELoss()(y_pred_scaled, y_scaled)
                val_sup_loss += sup_loss.item() * X_scaled.size(0)
                n_val += X_scaled.size(0)

        val_sup_loss /= n_val
        scheduler.step(val_sup_loss)

        if val_sup_loss < best_val_sup_loss:
            best_val_sup_loss = val_sup_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / f'PINN_w{w_phys}_s{seed}.pth')
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"      Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(MODELS_DIR / f'PINN_w{w_phys}_s{seed}.pth'))
    return model, best_val_sup_loss


def autoregressive_rollout(model, initial_state, controls, scaler_X, scaler_y, n_steps):
    model.eval()
    x_mean, x_scale = scaler_X.mean_, scaler_X.scale_
    y_mean, y_scale = scaler_y.mean_, scaler_y.scale_
    states = [initial_state.copy()]
    current = initial_state.copy()
    with torch.no_grad():
        for i in range(min(n_steps, len(controls))):
            inp = np.concatenate([current, controls[i]])
            inp_scaled = (inp - x_mean) / x_scale
            out_scaled = model(torch.FloatTensor(inp_scaled).unsqueeze(0)).squeeze(0).numpy()
            current = out_scaled * y_scale + y_mean
            states.append(current)
    return np.array(states)


def evaluate_model(model, val_loader, val_trajectories, scaler_X, scaler_y):
    model.eval()
    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    # Single-step MAE
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_scaled, y_scaled, X_unscaled, y_unscaled in val_loader:
            y_pred_scaled = model(X_scaled)
            y_pred = y_pred_scaled * y_scale + y_mean
            all_preds.append(y_pred.numpy())
            all_true.append(y_unscaled.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    single_step_mae = float(np.mean(np.abs(all_preds - all_true)))

    # Rollout MAE (N_ROLLOUT_TRAJ trajectories)
    pos_errors = []
    for traj in val_trajectories[:N_ROLLOUT_TRAJ]:
        states, controls = traj['states'], traj['controls']
        if len(states) < 101:
            continue
        predicted = autoregressive_rollout(model, states[0], controls, scaler_X, scaler_y, 100)
        pos_errors.append(np.mean(np.abs(predicted[:, :3] - states[:len(predicted), :3])))

    rollout_mae = float(np.mean(pos_errors)) if pos_errors else float('inf')
    return single_step_mae, rollout_mae


def main():
    print("="*70)
    print("ROBUST PHYSICS WEIGHT SWEEP (Publication Quality)")
    print("="*70)
    print(f"Weights: {PHYSICS_WEIGHTS}")
    print(f"Seeds: {SEEDS}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Rollout trajectories: {N_ROLLOUT_TRAJ}")
    print(f"Total models to train: {len(PHYSICS_WEIGHTS) * len(SEEDS)}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

    # Load data
    print("\nLoading data...")
    X_train, y_train = load_data(TRAIN_DATA)
    X_val, y_val = load_data(VAL_DATA)
    val_trajectories = load_trajectories(VAL_DATA)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled),
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled),
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Store all results
    all_results = {}

    for w_phys in PHYSICS_WEIGHTS:
        print(f"\n{'='*50}")
        print(f"PHYSICS WEIGHT = {w_phys}")
        print(f"{'='*50}")

        seed_results = []

        for seed in SEEDS:
            model_path = MODELS_DIR / f'PINN_w{w_phys}_s{seed}.pth'

            # Check if model already exists (resume support)
            if model_path.exists():
                print(f"\n  Seed {seed}... LOADING (already trained)")
                model = BaselinePINN()
                model.load_state_dict(torch.load(model_path))
                # Compute validation loss for loaded model
                model.eval()
                val_sup_loss = 0
                n_val = 0
                with torch.no_grad():
                    for X_scaled, y_scaled, X_unscaled, y_unscaled in val_loader:
                        y_pred_scaled = model(X_scaled)
                        sup_loss = nn.MSELoss()(y_pred_scaled, y_scaled)
                        val_sup_loss += sup_loss.item() * X_scaled.size(0)
                        n_val += X_scaled.size(0)
                best_sup_loss = val_sup_loss / n_val
            else:
                print(f"\n  Seed {seed}... TRAINING")
                model = BaselinePINN()
                model, best_sup_loss = train_with_physics_weight(
                    model, train_loader, val_loader, scaler_y, w_phys, seed
                )

            single_mae, rollout_mae = evaluate_model(
                model, val_loader, val_trajectories, scaler_X, scaler_y
            )

            seed_results.append({
                'seed': seed,
                'sup_loss': float(best_sup_loss),
                'single_step_mae': single_mae,
                'rollout_mae': rollout_mae
            })

            print(f"    Sup loss: {best_sup_loss:.6f}")
            print(f"    1-step MAE: {single_mae:.5f}")
            print(f"    100-step MAE: {rollout_mae:.3f}m")

        # Aggregate statistics
        sup_losses = [r['sup_loss'] for r in seed_results]
        single_maes = [r['single_step_mae'] for r in seed_results]
        rollout_maes = [r['rollout_mae'] for r in seed_results]

        all_results[f'w{w_phys}'] = {
            'physics_weight': w_phys,
            'seed_results': seed_results,
            'sup_loss_mean': float(np.mean(sup_losses)),
            'sup_loss_std': float(np.std(sup_losses)),
            'single_step_mae_mean': float(np.mean(single_maes)),
            'single_step_mae_std': float(np.std(single_maes)),
            'rollout_mae_mean': float(np.mean(rollout_maes)),
            'rollout_mae_std': float(np.std(rollout_maes))
        }

        print(f"\n  AGGREGATE: Rollout = {np.mean(rollout_maes):.3f} +/- {np.std(rollout_maes):.3f}m")

    # Summary table
    print("\n" + "="*70)
    print("FINAL RESULTS (mean +/- std over 3 seeds)")
    print("="*70)
    print(f"\n{'w_phys':<10} {'Sup Loss':<20} {'1-Step MAE':<20} {'100-Step MAE':<20}")
    print("-"*70)

    for w_phys in PHYSICS_WEIGHTS:
        r = all_results[f'w{w_phys}']
        sup = f"{r['sup_loss_mean']:.6f} +/- {r['sup_loss_std']:.6f}"
        single = f"{r['single_step_mae_mean']:.4f} +/- {r['single_step_mae_std']:.4f}"
        rollout = f"{r['rollout_mae_mean']:.3f} +/- {r['rollout_mae_std']:.3f}"
        print(f"{w_phys:<10} {sup:<20} {single:<20} {rollout:<20}")

    # Statistical significance test (w=0 vs w=20)
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE")
    print("="*70)

    w0_rollouts = [r['rollout_mae'] for r in all_results['w0.0']['seed_results']]
    w20_rollouts = [r['rollout_mae'] for r in all_results['w20.0']['seed_results']]

    # Simple t-test approximation
    mean_diff = np.mean(w0_rollouts) - np.mean(w20_rollouts)
    pooled_std = np.sqrt((np.var(w0_rollouts) + np.var(w20_rollouts)) / 2)
    t_stat = mean_diff / (pooled_std * np.sqrt(2/len(SEEDS))) if pooled_std > 0 else 0

    print(f"\nw=0 rollout: {np.mean(w0_rollouts):.3f} +/- {np.std(w0_rollouts):.3f}m")
    print(f"w=20 rollout: {np.mean(w20_rollouts):.3f} +/- {np.std(w20_rollouts):.3f}m")
    print(f"Difference: {mean_diff:.3f}m")
    print(f"t-statistic: {t_stat:.2f}")

    if abs(t_stat) > 2.92:  # t-critical for df=2, alpha=0.05 (two-tailed)
        print("Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("Result: NOT statistically significant at p < 0.05")
        print("(Need more seeds or larger effect size)")

    # Find best weight
    best_w = min(PHYSICS_WEIGHTS, key=lambda w: all_results[f'w{w}']['rollout_mae_mean'])
    best_rollout = all_results[f'w{best_w}']['rollout_mae_mean']
    print(f"\nBest physics weight: w={best_w} (rollout = {best_rollout:.3f}m)")

    # Save results
    with open(RESULTS_DIR / 'weight_sweep_robust_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR / 'weight_sweep_robust_results.json'}")
    print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")

    return all_results


if __name__ == '__main__':
    results = main()
