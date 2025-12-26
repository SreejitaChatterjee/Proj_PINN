"""
Physics Weight Sweep - Essential Experiment A

Goal: Is the physics weight the cause of instability?

Train PINN with physics-loss weight w_phys in {0, 0.1, 1, 5, 20}
For each, measure:
- Single-step supervised MAE
- 100-step rollout MAE (H_epsilon proxy)
- Jacobian spectral norm (L_phi)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent))
from pinn_architectures import BaselinePINN, ModularPINN

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / "data" / "train_set_diverse.csv"
VAL_DATA = PROJECT_ROOT / "data" / "val_set_diverse.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "weight_sweep"
MODELS_DIR = PROJECT_ROOT / "models" / "weight_sweep"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training config
BATCH_SIZE = 512
MAX_EPOCHS = 150  # More epochs to ensure convergence
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 30

# Sweep config
PHYSICS_WEIGHTS = [0.0, 0.1, 1.0, 5.0, 20.0]
SEEDS = [42, 123, 456]  # 3 seeds for statistics


def load_data(data_path):
    """Load data from CSV"""
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
    """Load trajectories for rollout evaluation"""
    df = pd.read_csv(data_path)
    df = df.rename(columns={"roll": "phi", "pitch": "theta", "yaw": "psi"})

    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    trajectories = []
    for traj_id in df["trajectory_id"].unique():
        df_traj = df[df["trajectory_id"] == traj_id].reset_index(drop=True)
        states = df_traj[state_cols].values.astype(np.float32)
        controls = df_traj[control_cols].values.astype(np.float32)
        trajectories.append({"states": states, "controls": controls})

    return trajectories


def train_with_physics_weight(model, train_loader, val_loader, scaler_y, w_phys, seed, model_name):
    """Train model with specific physics weight"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15)

    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    best_val_sup_loss = float("inf")
    patience_counter = 0
    history = {"train_sup": [], "train_phys": [], "val_sup": [], "val_phys": []}

    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        train_sup_loss, train_phys_loss = 0, 0
        n_train = 0

        for X_scaled, y_scaled, X_unscaled, y_unscaled in train_loader:
            optimizer.zero_grad()

            y_pred_scaled = model(X_scaled)
            sup_loss = nn.MSELoss()(y_pred_scaled, y_scaled)

            loss = sup_loss

            # Add physics loss if w_phys > 0 and model supports it
            if w_phys > 0 and hasattr(model, "physics_loss"):
                y_pred_unscaled = y_pred_scaled * y_scale + y_mean
                phys_loss = model.physics_loss(X_unscaled, y_pred_unscaled)
                loss = sup_loss + w_phys * phys_loss
                train_phys_loss += phys_loss.item() * X_scaled.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            if hasattr(model, "constrain_parameters"):
                model.constrain_parameters()

            train_sup_loss += sup_loss.item() * X_scaled.size(0)
            n_train += X_scaled.size(0)

        train_sup_loss /= n_train
        train_phys_loss /= n_train if w_phys > 0 else 1

        # Validation (use SUPERVISED loss only for early stopping!)
        model.eval()
        val_sup_loss, val_phys_loss = 0, 0
        n_val = 0

        with torch.no_grad():
            for X_scaled, y_scaled, X_unscaled, y_unscaled in val_loader:
                y_pred_scaled = model(X_scaled)
                sup_loss = nn.MSELoss()(y_pred_scaled, y_scaled)

                if w_phys > 0 and hasattr(model, "physics_loss"):
                    y_pred_unscaled = y_pred_scaled * y_scale + y_mean
                    phys_loss = model.physics_loss(X_unscaled, y_pred_unscaled)
                    val_phys_loss += phys_loss.item() * X_scaled.size(0)

                val_sup_loss += sup_loss.item() * X_scaled.size(0)
                n_val += X_scaled.size(0)

        val_sup_loss /= n_val
        val_phys_loss /= n_val if w_phys > 0 else 1

        # Use SUPERVISED loss for scheduler and early stopping
        scheduler.step(val_sup_loss)

        history["train_sup"].append(train_sup_loss)
        history["train_phys"].append(train_phys_loss)
        history["val_sup"].append(val_sup_loss)
        history["val_phys"].append(val_phys_loss)

        if val_sup_loss < best_val_sup_loss:
            best_val_sup_loss = val_sup_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / f"{model_name}_w{w_phys}_s{seed}.pth")
        else:
            patience_counter += 1

        if epoch % 30 == 0:
            print(f"    E{epoch:3d}: sup={val_sup_loss:.5f}, phys={val_phys_loss:.5f}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / f"{model_name}_w{w_phys}_s{seed}.pth"))
    return model, history


def autoregressive_rollout(model, initial_state, controls, scaler_X, scaler_y, n_steps):
    """100-step autoregressive rollout"""
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


def compute_jacobian_spectral_norm(model, X_sample, scaler_X, n_samples=100, n_iter=10):
    """Compute empirical Jacobian spectral norms"""
    model.eval()
    x_mean = torch.FloatTensor(scaler_X.mean_)
    x_scale = torch.FloatTensor(scaler_X.scale_)

    sigmas = []
    indices = np.random.choice(len(X_sample), min(n_samples, len(X_sample)), replace=False)

    for idx in indices:
        x = torch.FloatTensor(X_sample[idx])
        x_scaled = ((x - x_mean) / x_scale).detach().requires_grad_(True)

        # Simple finite difference approximation for speed
        eps = 1e-4
        with torch.no_grad():
            y0 = model(x_scaled.unsqueeze(0)).squeeze(0)

        max_ratio = 0
        for _ in range(5):  # Random directions
            v = torch.randn_like(x_scaled)
            v = v / (v.norm() + 1e-12)

            x_plus = x_scaled + eps * v
            with torch.no_grad():
                y_plus = model(x_plus.unsqueeze(0)).squeeze(0)

            ratio = (y_plus - y0).norm().item() / eps
            max_ratio = max(max_ratio, ratio)

        sigmas.append(max_ratio)

    return {
        "mean": np.mean(sigmas),
        "p95": np.percentile(sigmas, 95),
        "max": np.max(sigmas),
    }


def evaluate_model(model, val_loader, val_trajectories, scaler_X, scaler_y, X_val):
    """Full evaluation"""
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

    single_step_mae = np.mean(np.abs(all_preds - all_true))
    z_mae = np.mean(np.abs(all_preds[:, 2] - all_true[:, 2]))

    # Rollout
    pos_errors = []
    for traj in val_trajectories[:10]:
        states, controls = traj["states"], traj["controls"]
        if len(states) < 101:
            continue
        predicted = autoregressive_rollout(model, states[0], controls, scaler_X, scaler_y, 100)
        pos_errors.append(np.mean(np.abs(predicted[:, :3] - states[: len(predicted), :3])))

    rollout_mae = np.mean(pos_errors) if pos_errors else float("inf")

    # Jacobian
    sigma_stats = compute_jacobian_spectral_norm(model, X_val, scaler_X)

    return {
        "single_step_mae": float(single_step_mae),
        "z_mae": float(z_mae),
        "rollout_mae": float(rollout_mae),
        "sigma_mean": float(sigma_stats["mean"]),
        "sigma_p95": float(sigma_stats["p95"]),
        "sigma_max": float(sigma_stats["max"]),
    }


def main():
    print("=" * 70)
    print("PHYSICS WEIGHT SWEEP EXPERIMENT")
    print("=" * 70)
    print(f"Weights: {PHYSICS_WEIGHTS}")
    print(f"Seeds: {SEEDS}")
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

    all_results = {}

    for w_phys in PHYSICS_WEIGHTS:
        print(f"\n{'='*70}")
        print(f"PHYSICS WEIGHT = {w_phys}")
        print(f"{'='*70}")

        all_results[w_phys] = []

        for seed in SEEDS:
            print(f"\n  Seed {seed}...")
            model = BaselinePINN()

            model, history = train_with_physics_weight(
                model, train_loader, val_loader, scaler_y, w_phys, seed, "PINN"
            )

            results = evaluate_model(model, val_loader, val_trajectories, scaler_X, scaler_y, X_val)
            results["seed"] = seed
            results["final_sup_loss"] = history["val_sup"][-1]
            results["final_phys_loss"] = history["val_phys"][-1] if w_phys > 0 else 0

            all_results[w_phys].append(results)

            print(f"    1-step MAE: {results['single_step_mae']:.5f}")
            print(f"    100-step MAE: {results['rollout_mae']:.3f}m")
            print(f"    sigma_max: {results['sigma_max']:.3f}")

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print("\n" + "=" * 70)
    print("WEIGHT SWEEP RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"\n{'w_phys':<10} {'1-Step MAE':<15} {'100-Step MAE':<15} {'sigma_max':<12} {'sup_loss':<12}"
    )
    print("-" * 70)

    summary = {}
    for w_phys in PHYSICS_WEIGHTS:
        results_list = all_results[w_phys]
        single_mae = np.mean([r["single_step_mae"] for r in results_list])
        rollout_mae = np.mean([r["rollout_mae"] for r in results_list])
        sigma_max = np.mean([r["sigma_max"] for r in results_list])
        sup_loss = np.mean([r["final_sup_loss"] for r in results_list])

        single_std = np.std([r["single_step_mae"] for r in results_list])
        rollout_std = np.std([r["rollout_mae"] for r in results_list])

        print(
            f"{w_phys:<10} {single_mae:.5f}+/-{single_std:.5f}  {rollout_mae:.3f}+/-{rollout_std:.3f}m  {sigma_max:.3f}       {sup_loss:.5f}"
        )

        summary[w_phys] = {
            "single_step_mae_mean": single_mae,
            "single_step_mae_std": single_std,
            "rollout_mae_mean": rollout_mae,
            "rollout_mae_std": rollout_std,
            "sigma_max_mean": sigma_max,
            "sup_loss_mean": sup_loss,
        }

    # ========================================================================
    # INTERPRETATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Check if H_epsilon decreases with physics weight
    rollouts = [summary[w]["rollout_mae_mean"] for w in PHYSICS_WEIGHTS]
    if all(rollouts[i] <= rollouts[i + 1] for i in range(len(rollouts) - 1)):
        print("\n--> H_epsilon MONOTONICALLY DECREASES with physics weight.")
        print("    Physics loss causally hurts rollout stability!")
    else:
        print("\n--> H_epsilon is NON-MONOTONIC with physics weight.")
        print("    The effect depends on tuning; intermediate weights may be optimal.")

    # Check supervised loss
    sup_losses = [summary[w]["sup_loss_mean"] for w in PHYSICS_WEIGHTS]
    if sup_losses[-1] > 2 * sup_losses[0]:
        print("\n--> High physics weight causes supervised loss to INCREASE.")
        print("    PINN is underfitting due to physics loss dominance.")

    # Save results
    with open(RESULTS_DIR / "weight_sweep_results.json", "w") as f:
        json.dump(
            {
                "all_results": {str(k): v for k, v in all_results.items()},
                "summary": {str(k): v for k, v in summary.items()},
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {RESULTS_DIR / 'weight_sweep_results.json'}")
    print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")

    return all_results, summary


if __name__ == "__main__":
    all_results, summary = main()
