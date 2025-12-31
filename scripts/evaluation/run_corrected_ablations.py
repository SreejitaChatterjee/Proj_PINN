"""
CORRECTED Ablation Study - Fixes all methodological issues

Issues fixed:
1. Uses 20 seeds (matching weight sweep) for statistical validity
2. Fixes LeastSquares evaluation bug (was using unscaled data incorrectly)
3. Consistent early stopping (supervised-only) across all models
4. Same evaluation protocol for all architectures

Run time: ~4-6 hours (20 seeds x 2 architectures x 100 epochs)
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
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent))
from pinn_architectures import BaselinePINN, ModularPINN, PhysicsLossMixin

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / "data" / "train_set_diverse.csv"
VAL_DATA = PROJECT_ROOT / "data" / "val_set_diverse.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "corrected_ablation"
MODELS_DIR = PROJECT_ROOT / "models" / "corrected_ablation"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training config - MATCHING weight sweep exactly
BATCH_SIZE = 512
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 20
PHYSICS_WEIGHT = 20.0

# KEY FIX: Use 20 seeds like weight sweep
SEEDS = list(range(20))  # Seeds 0-19
N_ROLLOUT_TRAJ = 10


def load_data(data_path):
    """Load and prepare data from CSV"""
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
    """Load data organized by trajectory for rollout evaluation"""
    df = pd.read_csv(data_path)
    df = df.rename(columns={"roll": "phi", "pitch": "theta", "yaw": "psi"})

    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    trajectories = []
    for traj_id in df["trajectory_id"].unique():
        df_traj = df[df["trajectory_id"] == traj_id].reset_index(drop=True)
        states = df_traj[state_cols].values.astype(np.float32)
        controls = df_traj[control_cols].values.astype(np.float32)
        trajectories.append({"states": states, "controls": controls, "traj_id": traj_id})

    return trajectories


def train_model_supervised_early_stop(
    model, train_loader, val_loader, scaler_y, use_physics, model_name, seed
):
    """
    Train with SUPERVISED-ONLY early stopping (matching weight sweep protocol).
    This is critical for fair comparison.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    best_val_sup_loss = float("inf")
    patience_counter = 0
    best_state = None

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

            # Add physics loss if enabled
            if use_physics and hasattr(model, "physics_loss"):
                y_pred_unscaled = y_pred_scaled * y_scale + y_mean
                phys_loss = model.physics_loss(X_unscaled, y_pred_unscaled)
                loss = sup_loss + PHYSICS_WEIGHT * phys_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            if hasattr(model, "constrain_parameters"):
                model.constrain_parameters()

            train_sup_loss += sup_loss.item() * X_scaled.size(0)
            n_train += X_scaled.size(0)

        train_sup_loss /= n_train

        # Validation - SUPERVISED LOSS ONLY for early stopping
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

        # Early stopping on SUPERVISED loss only
        if val_sup_loss < best_val_sup_loss:
            best_val_sup_loss = val_sup_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_sup_loss


def autoregressive_rollout(model, initial_state, controls, scaler_X, scaler_y, n_steps):
    """Autoregressive rollout - uses model predictions as next input"""
    model.eval()

    x_mean, x_scale = scaler_X.mean_, scaler_X.scale_
    y_mean, y_scale = scaler_y.mean_, scaler_y.scale_

    states = [initial_state.copy()]
    current_state = initial_state.copy()

    with torch.no_grad():
        for i in range(min(n_steps, len(controls))):
            # Combine state and control
            inp = np.concatenate([current_state, controls[i]])
            # Scale input
            inp_scaled = (inp - x_mean) / x_scale
            # Predict
            out_scaled = model(torch.FloatTensor(inp_scaled).unsqueeze(0)).squeeze(0).numpy()
            # Unscale output
            next_state = out_scaled * y_scale + y_mean

            states.append(next_state)
            current_state = next_state  # Use prediction for next step

    return np.array(states)


def evaluate_model(model, val_loader, val_trajectories, scaler_X, scaler_y):
    """Full evaluation with single-step and rollout metrics"""
    model.eval()

    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    # Single-step evaluation
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

    # Rollout evaluation
    pos_errors = []
    for traj in val_trajectories[:N_ROLLOUT_TRAJ]:
        states = traj["states"]
        controls = traj["controls"]

        if len(states) < 101:
            continue

        predicted = autoregressive_rollout(model, states[0], controls, scaler_X, scaler_y, 100)
        true_states = states[: len(predicted)]

        pos_errors.append(np.mean(np.abs(predicted[:, :3] - true_states[:, :3])))

    rollout_mae = float(np.mean(pos_errors)) if pos_errors else float("inf")

    return single_step_mae, rollout_mae


def least_squares_baseline_fixed(
    X_train, y_train, X_val, y_val, val_trajectories, scaler_X, scaler_y
):
    """
    FIXED Least Squares baseline.

    Bug in original: Used unscaled data for training but scaled evaluation.
    Fix: Use SCALED data consistently for both training and evaluation.
    """
    print("\n  Training Least Squares model (FIXED)...")

    # Scale the data
    X_train_scaled = scaler_X.transform(X_train)
    y_train_scaled = scaler_y.transform(y_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)

    # Fit on SCALED data
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train_scaled)

    # Single-step evaluation
    y_pred_scaled = model.predict(X_val_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    single_step_mae = float(np.mean(np.abs(y_pred - y_val)))

    # Rollout evaluation - PROPER autoregressive rollout
    pos_errors = []
    for traj in val_trajectories[:N_ROLLOUT_TRAJ]:
        states = traj["states"]
        controls = traj["controls"]

        if len(states) < 101:
            continue

        # Autoregressive rollout
        predicted = [states[0].copy()]
        current_state = states[0].copy()

        for i in range(min(100, len(controls))):
            # Combine state and control
            inp = np.concatenate([current_state, controls[i]])
            # Scale
            inp_scaled = scaler_X.transform(inp.reshape(1, -1))
            # Predict
            out_scaled = model.predict(inp_scaled)
            # Unscale
            next_state = scaler_y.inverse_transform(out_scaled)[0]

            predicted.append(next_state)
            current_state = next_state  # Use prediction for next step!

        predicted = np.array(predicted)
        true_states = states[: len(predicted)]

        pos_errors.append(np.mean(np.abs(predicted[:, :3] - true_states[:, :3])))

    rollout_mae = float(np.mean(pos_errors)) if pos_errors else float("inf")

    return single_step_mae, rollout_mae


def compute_statistics(results_list, metric_key):
    """Compute mean, std, and confidence interval"""
    values = [r[metric_key] for r in results_list]
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    n = len(values)
    se = std / np.sqrt(n)
    ci95 = 1.96 * se
    return mean, std, ci95, values


def welch_ttest(group1, group2):
    """Welch's t-test for unequal variances"""
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    # Cohen's d with pooled std
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.std(group1, ddof=1) ** 2 + (n2 - 1) * np.std(group2, ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    return t_stat, p_value, cohens_d


def main():
    print("=" * 80)
    print("CORRECTED ABLATION STUDY")
    print("=" * 80)
    print(f"Seeds: {len(SEEDS)} (matching weight sweep)")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Early stopping: SUPERVISED loss only")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[1/5] Loading data...")
    X_train, y_train = load_data(TRAIN_DATA)
    X_val, y_val = load_data(VAL_DATA)
    val_trajectories = load_trajectories(VAL_DATA)
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")

    # Prepare data
    print("\n[2/5] Preparing data...")
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

    # ========================================================================
    # ABLATION 1: Baseline vs Modular (20 seeds each)
    # ========================================================================
    print("\n[3/5] Running architecture comparison (20 seeds each)...")
    print("-" * 60)

    for arch_name, ModelClass in [("Baseline", BaselinePINN), ("Modular", ModularPINN)]:
        print(f"\n  {arch_name} architecture across {len(SEEDS)} seeds...")

        seed_results = []
        for i, seed in enumerate(SEEDS):
            model_path = MODELS_DIR / f"{arch_name}_seed{seed}.pth"

            # Check if already trained (resume support)
            if model_path.exists():
                print(f"    Seed {seed} ({i+1}/{len(SEEDS)})... LOADING")
                model = ModelClass()
                model.load_state_dict(torch.load(model_path, weights_only=True))
            else:
                print(f"    Seed {seed} ({i+1}/{len(SEEDS)})... TRAINING")
                model = ModelClass()
                model, _ = train_model_supervised_early_stop(
                    model,
                    train_loader,
                    val_loader,
                    scaler_y,
                    use_physics=True,
                    model_name=arch_name,
                    seed=seed,
                )
                torch.save(model.state_dict(), model_path)

            single_mae, rollout_mae = evaluate_model(
                model, val_loader, val_trajectories, scaler_X, scaler_y
            )

            seed_results.append(
                {
                    "seed": seed,
                    "single_step_mae": single_mae,
                    "rollout_mae": rollout_mae,
                }
            )

            print(f"      1-step: {single_mae:.4f}, 100-step: {rollout_mae:.3f}m")

        # Compute statistics
        single_mean, single_std, single_ci, single_vals = compute_statistics(
            seed_results, "single_step_mae"
        )
        rollout_mean, rollout_std, rollout_ci, rollout_vals = compute_statistics(
            seed_results, "rollout_mae"
        )

        all_results[arch_name] = {
            "seed_results": seed_results,
            "single_step_mae_mean": single_mean,
            "single_step_mae_std": single_std,
            "single_step_mae_ci95": single_ci,
            "rollout_mae_mean": rollout_mean,
            "rollout_mae_std": rollout_std,
            "rollout_mae_ci95": rollout_ci,
            "n_params": sum(p.numel() for p in ModelClass().parameters()),
        }

        print(f"\n  {arch_name} SUMMARY:")
        print(f"    1-step MAE: {single_mean:.4f} ± {single_std:.4f}")
        print(f"    100-step MAE: {rollout_mean:.3f} ± {rollout_std:.3f}m")

    # ========================================================================
    # Statistical comparison: Baseline vs Modular
    # ========================================================================
    print("\n[4/5] Statistical comparison...")
    print("-" * 60)

    baseline_rollouts = [r["rollout_mae"] for r in all_results["Baseline"]["seed_results"]]
    modular_rollouts = [r["rollout_mae"] for r in all_results["Modular"]["seed_results"]]

    t_stat, p_value, cohens_d = welch_ttest(baseline_rollouts, modular_rollouts)

    all_results["statistical_comparison"] = {
        "baseline_vs_modular": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "baseline_mean": float(np.mean(baseline_rollouts)),
            "baseline_std": float(np.std(baseline_rollouts, ddof=1)),
            "modular_mean": float(np.mean(modular_rollouts)),
            "modular_std": float(np.std(modular_rollouts, ddof=1)),
        }
    }

    print(
        f"\n  Baseline: {np.mean(baseline_rollouts):.3f} ± {np.std(baseline_rollouts, ddof=1):.3f}m"
    )
    print(f"  Modular:  {np.mean(modular_rollouts):.3f} ± {np.std(modular_rollouts, ddof=1):.3f}m")
    print(f"\n  Welch's t-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    if p_value < 0.05:
        winner = "Baseline" if np.mean(baseline_rollouts) < np.mean(modular_rollouts) else "Modular"
        print(f"\n  Result: {winner} is SIGNIFICANTLY better (p < 0.05)")
    else:
        print(f"\n  Result: NO significant difference (p = {p_value:.3f})")

    # ========================================================================
    # ABLATION 2: Least Squares baseline (FIXED)
    # ========================================================================
    print("\n[5/5] Least Squares baseline (FIXED)...")
    print("-" * 60)

    ls_single, ls_rollout = least_squares_baseline_fixed(
        X_train, y_train, X_val, y_val, val_trajectories, scaler_X, scaler_y
    )

    all_results["LeastSquares"] = {
        "single_step_mae": ls_single,
        "rollout_mae": ls_rollout,
    }

    print(f"  1-step MAE: {ls_single:.4f}")
    print(f"  100-step MAE: {ls_rollout:.3f}m")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("CORRECTED ABLATION RESULTS")
    print("=" * 80)

    print("\n" + "-" * 80)
    print(f"Architecture Comparison ({len(SEEDS)} seeds each)")
    print("-" * 80)
    print(f"{'Model':<15} {'Params':<10} {'1-Step MAE':<20} {'100-Step MAE (m)':<25}")
    print("-" * 70)

    for name in ["Baseline", "Modular"]:
        r = all_results[name]
        single_str = f"{r['single_step_mae_mean']:.4f} ± {r['single_step_mae_std']:.4f}"
        rollout_str = f"{r['rollout_mae_mean']:.3f} ± {r['rollout_mae_std']:.3f}"
        print(f"{name:<15} {r['n_params']:<10,} {single_str:<20} {rollout_str:<25}")

    r = all_results["LeastSquares"]
    print(
        f"{'LeastSquares':<15} {'N/A':<10} {r['single_step_mae']:<20.4f} {r['rollout_mae']:<25.3f}"
    )

    print("\n" + "-" * 80)
    print("Statistical Significance")
    print("-" * 80)
    comp = all_results["statistical_comparison"]["baseline_vs_modular"]
    print(f"  Welch's t-test: t = {comp['t_statistic']:.3f}")
    print(f"  p-value: {comp['p_value']:.4f}")
    print(f"  Cohen's d: {comp['cohens_d']:.3f}")

    # Interpret effect size
    d = abs(comp["cohens_d"])
    if d < 0.2:
        effect = "negligible"
    elif d < 0.5:
        effect = "small"
    elif d < 0.8:
        effect = "medium"
    else:
        effect = "large"
    print(f"  Effect size interpretation: {effect}")

    # Save results
    with open(RESULTS_DIR / "corrected_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR / 'corrected_ablation_results.json'}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results


if __name__ == "__main__":
    results = main()
