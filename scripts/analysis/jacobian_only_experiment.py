"""
Vectorized Jacobian-Only Experiment

Only trains the Jacobian regularization condition (20 seeds).
Reuses existing baseline and physics results from weight_sweep_robust.
Uses torch.func.jacrev + vmap for efficient vectorized Jacobian computation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent))
from pinn_architectures import BaselinePINN

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / "data" / "train_set_diverse.csv"
VAL_DATA = PROJECT_ROOT / "data" / "val_set_diverse.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "jacobian_experiment"
MODELS_DIR = PROJECT_ROOT / "models" / "jacobian_experiment"
EXISTING_RESULTS = PROJECT_ROOT / "results" / "weight_sweep" / "weight_sweep_robust_results.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training config
BATCH_SIZE = 512
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 40

# Only train Jacobian condition
SEEDS = list(range(42, 62))  # 20 seeds
N_ROLLOUT_TRAJ = 10
JACOBIAN_WEIGHT = 0.1
JACOBIAN_SAMPLES = 16  # Balanced: rigorous but tractable


def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.rename(columns={"roll": "phi", "pitch": "theta", "yaw": "psi"})
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    input_features = state_cols + ["thrust", "torque_x", "torque_y", "torque_z"]
    X, y = [], []
    for traj_id in df["trajectory_id"].unique():
        df_traj = df[df["trajectory_id"] == traj_id].reset_index(drop=True)
        for i in range(len(df_traj) - 1):
            X.append(df_traj.iloc[i][input_features].values)
            y.append(df_traj.iloc[i + 1][state_cols].values)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_trajectories(data_path):
    df = pd.read_csv(data_path)
    df = df.rename(columns={"roll": "phi", "pitch": "theta", "yaw": "psi"})
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]
    trajectories = []
    for traj_id in df["trajectory_id"].unique():
        df_traj = df[df["trajectory_id"] == traj_id].reset_index(drop=True)
        trajectories.append(
            {
                "states": df_traj[state_cols].values.astype(np.float32),
                "controls": df_traj[control_cols].values.astype(np.float32),
            }
        )
    return trajectories


def jacobian_stability_loss_vectorized(model, x_batch, n_samples=JACOBIAN_SAMPLES):
    """Vectorized Jacobian regularization using Frobenius norm.

    Uses torch.func.jacrev + vmap for efficient batch Jacobian computation.
    Penalizes ||J||_F > tau where tau = sqrt(12) for marginal stability.
    """
    from copy import deepcopy

    from torch.func import functional_call, jacrev, vmap

    if x_batch.size(0) > n_samples:
        idx = torch.randperm(x_batch.size(0))[:n_samples]
        x_sample = x_batch[idx]
    else:
        x_sample = x_batch

    # Temporarily set model to eval mode to disable dropout
    was_training = model.training
    model.eval()

    # Get model parameters as a dict for functional_call
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # Create a functional version for jacrev
    def model_fn(x):
        return functional_call(model, (params, buffers), x.unsqueeze(0)).squeeze(0)

    # Compute Jacobian for each sample using vmap
    # jacrev computes dy/dx, vmap parallelizes over batch
    jacobian_fn = vmap(jacrev(model_fn))

    # J shape: (n_samples, 12, 16) - outputs x inputs
    J = jacobian_fn(x_sample)

    # Restore training mode
    if was_training:
        model.train()

    # Extract state-to-state Jacobian (12 outputs, 12 state inputs)
    J_state = J[:, :, :12]  # (n_samples, 12, 12)

    # Frobenius norm for each sample
    jac_norms = torch.sqrt((J_state**2).sum(dim=(1, 2)) + 1e-8)  # (n_samples,)

    # Penalize norms above threshold
    threshold = np.sqrt(12) * 1.0  # Marginal stability
    loss = torch.relu(jac_norms - threshold).mean()

    return loss


def train_jacobian_model(model, train_loader, val_loader, scaler_y, seed):
    """Train model with Jacobian regularization."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    best_val_sup_loss = float("inf")
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_sup_loss = 0
        n_train = 0

        for X_scaled, y_scaled, X_unscaled, y_unscaled in train_loader:
            optimizer.zero_grad()
            y_pred_scaled = model(X_scaled)
            sup_loss = nn.MSELoss()(y_pred_scaled, y_scaled)

            # Add Jacobian regularization (vectorized computation)
            jac_loss = jacobian_stability_loss_vectorized(model, X_scaled)
            loss = sup_loss + JACOBIAN_WEIGHT * jac_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            if hasattr(model, "constrain_parameters"):
                model.constrain_parameters()

            train_sup_loss += sup_loss.item() * X_scaled.size(0)
            n_train += X_scaled.size(0)

        train_sup_loss /= n_train

        # Validation
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

        model_path = MODELS_DIR / f"jacobian_s{seed}.pth"

        if val_sup_loss < best_val_sup_loss:
            best_val_sup_loss = val_sup_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"      Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(model_path))
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

    # Rollout MAE
    pos_errors = []
    for traj in val_trajectories[:N_ROLLOUT_TRAJ]:
        states, controls = traj["states"], traj["controls"]
        if len(states) < 101:
            continue
        predicted = autoregressive_rollout(model, states[0], controls, scaler_X, scaler_y, 100)
        pos_errors.append(np.mean(np.abs(predicted[:, :3] - states[: len(predicted), :3])))

    rollout_mae = float(np.mean(pos_errors)) if pos_errors else float("inf")

    return single_step_mae, rollout_mae


def main():
    print("=" * 70)
    print("JACOBIAN-ONLY EXPERIMENT (Vectorized)")
    print("=" * 70)
    print(f"Seeds: {len(SEEDS)} ({SEEDS[0]}-{SEEDS[-1]})")
    print(f"Jacobian weight: {JACOBIAN_WEIGHT}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load existing results
    if EXISTING_RESULTS.exists():
        with open(EXISTING_RESULTS) as f:
            existing = json.load(f)
        print(f"\nLoaded existing results from {EXISTING_RESULTS}")
        baseline_rollouts = [r["rollout_mae"] for r in existing["w0.0"]["seed_results"]]
        physics_rollouts = [r["rollout_mae"] for r in existing["w20.0"]["seed_results"]]
        print(
            f"Baseline (w=0): {np.mean(baseline_rollouts):.3f} ± {np.std(baseline_rollouts):.3f}m"
        )
        print(f"Physics (w=20): {np.mean(physics_rollouts):.3f} ± {np.std(physics_rollouts):.3f}m")
    else:
        print("WARNING: No existing results found!")
        baseline_rollouts = []
        physics_rollouts = []

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
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_scaled),
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_scaled),
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train Jacobian models
    print(f"\n{'=' * 50}")
    print("TRAINING JACOBIAN CONDITION")
    print(f"{'=' * 50}")

    jacobian_results = []

    for seed in SEEDS:
        model_path = MODELS_DIR / f"jacobian_s{seed}.pth"

        if model_path.exists():
            print(f"\n  Seed {seed}... LOADING")
            model = BaselinePINN()
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"\n  Seed {seed}... TRAINING")
            model = BaselinePINN()
            model, _ = train_jacobian_model(model, train_loader, val_loader, scaler_y, seed)

        single_mae, rollout_mae = evaluate_model(
            model, val_loader, val_trajectories, scaler_X, scaler_y
        )

        jacobian_results.append(
            {"seed": seed, "single_step_mae": single_mae, "rollout_mae": rollout_mae}
        )

        print(f"    1-step MAE: {single_mae:.5f}")
        print(f"    100-step MAE: {rollout_mae:.3f}m")

    jacobian_rollouts = [r["rollout_mae"] for r in jacobian_results]

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nBaseline (w=0):  {np.mean(baseline_rollouts):.3f} ± {np.std(baseline_rollouts):.3f}m")
    print(f"Physics (w=20):  {np.mean(physics_rollouts):.3f} ± {np.std(physics_rollouts):.3f}m")
    print(f"Jacobian:        {np.mean(jacobian_rollouts):.3f} ± {np.std(jacobian_rollouts):.3f}m")

    # Improvements
    baseline_mean = np.mean(baseline_rollouts)
    physics_mean = np.mean(physics_rollouts)
    jacobian_mean = np.mean(jacobian_rollouts)

    print(
        f"\nJacobian vs Baseline: {(baseline_mean - jacobian_mean) / baseline_mean * 100:.1f}% improvement"
    )
    print(
        f"Jacobian vs Physics:  {(physics_mean - jacobian_mean) / physics_mean * 100:.1f}% improvement"
    )

    # Statistical tests
    from scipy import stats

    t_bj, p_bj = stats.ttest_ind(baseline_rollouts, jacobian_rollouts)
    t_pj, p_pj = stats.ttest_ind(physics_rollouts, jacobian_rollouts)

    d_bj = (np.mean(baseline_rollouts) - np.mean(jacobian_rollouts)) / np.sqrt(
        (np.var(baseline_rollouts) + np.var(jacobian_rollouts)) / 2
    )
    d_pj = (np.mean(physics_rollouts) - np.mean(jacobian_rollouts)) / np.sqrt(
        (np.var(physics_rollouts) + np.var(jacobian_rollouts)) / 2
    )

    print(f"\nJacobian vs Baseline: t={t_bj:.3f}, p={p_bj:.4f}, d={d_bj:.3f}")
    print(f"Jacobian vs Physics:  t={t_pj:.3f}, p={p_pj:.4f}, d={d_pj:.3f}")

    # Save results
    results = {
        "jacobian": {
            "rollout_mean": float(np.mean(jacobian_rollouts)),
            "rollout_std": float(np.std(jacobian_rollouts)),
            "seed_results": jacobian_results,
        },
        "baseline": {
            "rollout_mean": float(np.mean(baseline_rollouts)),
            "rollout_std": float(np.std(baseline_rollouts)),
        },
        "physics": {
            "rollout_mean": float(np.mean(physics_rollouts)),
            "rollout_std": float(np.std(physics_rollouts)),
        },
        "stats": {
            "jacobian_vs_baseline": {
                "t": float(t_bj),
                "p": float(p_bj),
                "d": float(d_bj),
            },
            "jacobian_vs_physics": {
                "t": float(t_pj),
                "p": float(p_pj),
                "d": float(d_pj),
            },
        },
    }

    with open(RESULTS_DIR / "jacobian_only_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR / 'jacobian_only_results.json'}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == "__main__":
    results = main()
