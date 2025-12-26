"""
Jacobian Regularization Experiment for ICML 2026

Tests the hypothesis: Regularizing Jacobian spectral radius improves
rollout stability more than physics loss.

Conditions:
1. w=0 (baseline NN)
2. w=20 (physics loss)
3. w=0 + Jacobian regularization (our proposal)

20 seeds per condition for statistical power.
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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training config
BATCH_SIZE = 512
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 40  # Match paper

# Experiment config
CONDITIONS = ["baseline", "physics", "jacobian"]
SEEDS = list(range(42, 62))  # 20 seeds: 42-61
N_ROLLOUT_TRAJ = 10
PHYSICS_WEIGHT = 20.0
JACOBIAN_WEIGHT = 0.1  # Tuned weight for Jacobian regularization
JACOBIAN_SAMPLES = 32  # Batch samples for Jacobian computation


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


def compute_jacobian_spectral_radius(model, x_batch, n_samples=JACOBIAN_SAMPLES):
    """
    Compute mean spectral radius of Jacobian w.r.t. state inputs.
    Uses power iteration for efficiency.
    """
    model.eval()

    # Sample subset for efficiency
    if x_batch.size(0) > n_samples:
        idx = torch.randperm(x_batch.size(0))[:n_samples]
        x_sample = x_batch[idx]
    else:
        x_sample = x_batch

    x_sample = x_sample.requires_grad_(True)

    # Forward pass
    y = model(x_sample)

    # Compute Jacobian w.r.t. state (first 12 dims of input)
    # Using power iteration to estimate spectral radius
    spectral_radii = []

    for i in range(x_sample.size(0)):
        # Initialize random vector
        v = torch.randn(12, device=x_sample.device)
        v = v / v.norm()

        # Power iteration (10 iterations)
        for _ in range(10):
            # Compute J @ v via vector-Jacobian product
            # We need d(y[i]) / d(x[i, :12])
            grad_outputs = torch.zeros_like(y[i])
            Jv = torch.zeros(12, device=x_sample.device)

            for j in range(12):  # Output dimensions (state)
                grad_outputs.zero_()
                grad_outputs[j] = 1.0

                if x_sample.grad is not None:
                    x_sample.grad.zero_()

                # Compute gradient
                grads = torch.autograd.grad(
                    y[i, j],
                    x_sample,
                    grad_outputs=torch.ones_like(y[i, j]),
                    retain_graph=True,
                    create_graph=True,
                )[0]

                Jv[j] = (grads[i, :12] * v).sum()

            # Update v
            v_new = Jv / (Jv.norm() + 1e-8)

            # Estimate eigenvalue
            eigenvalue = (Jv * v).sum()
            v = v_new

        spectral_radii.append(abs(eigenvalue.item()))

    return np.mean(spectral_radii)


def jacobian_stability_loss(model, x_batch, n_samples=JACOBIAN_SAMPLES):
    """
    Loss that penalizes spectral radius > 1.
    L_stability = E[max(0, rho(J) - 1)]

    Simplified version using Frobenius norm as proxy for computational efficiency.
    """
    # Sample subset
    if x_batch.size(0) > n_samples:
        idx = torch.randperm(x_batch.size(0))[:n_samples]
        x_sample = x_batch[idx].clone().requires_grad_(True)
    else:
        x_sample = x_batch.clone().requires_grad_(True)

    # Forward pass
    y = model(x_sample)

    # Compute Jacobian Frobenius norm as proxy for spectral radius
    # ||J||_F >= rho(J), so penalizing ||J||_F encourages smaller spectral radius

    loss = 0.0
    for i in range(min(8, x_sample.size(0))):  # Limit for memory
        jac_norm_sq = 0.0
        for j in range(12):  # Output state dimensions
            grads = torch.autograd.grad(
                y[i, j],
                x_sample,
                grad_outputs=torch.ones_like(y[i, j]),
                retain_graph=True,
                create_graph=True,
            )[0]
            # Only w.r.t. state (first 12 dims), not control
            jac_norm_sq += (grads[i, :12] ** 2).sum()

        # Frobenius norm
        jac_norm = torch.sqrt(jac_norm_sq + 1e-8)

        # Penalize if norm > sqrt(12) (equivalent to avg eigenvalue > 1)
        # Threshold chosen so spectral radius ~ 1 gives zero loss
        threshold = np.sqrt(12) * 1.0  # ~3.46
        loss += torch.relu(jac_norm - threshold)

    return loss / min(8, x_sample.size(0))


def train_model(model, train_loader, val_loader, scaler_y, condition, seed):
    """Train model with specified condition."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

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

            loss = sup_loss

            # Add physics loss for 'physics' condition
            if condition == "physics" and hasattr(model, "physics_loss"):
                y_pred_unscaled = y_pred_scaled * y_scale + y_mean
                phys_loss = model.physics_loss(X_unscaled, y_pred_unscaled)
                loss = sup_loss + PHYSICS_WEIGHT * phys_loss

            # Add Jacobian regularization for 'jacobian' condition
            elif condition == "jacobian":
                jac_loss = jacobian_stability_loss(model, X_scaled)
                loss = sup_loss + JACOBIAN_WEIGHT * jac_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            if hasattr(model, "constrain_parameters"):
                model.constrain_parameters()

            train_sup_loss += sup_loss.item() * X_scaled.size(0)
            n_train += X_scaled.size(0)

        train_sup_loss /= n_train

        # Validation (supervised loss only for fair early stopping)
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

        model_path = MODELS_DIR / f"{condition}_s{seed}.pth"

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

    # Estimate Jacobian spectral radius
    sample_X = torch.FloatTensor(
        scaler_X.transform(
            np.concatenate(
                [
                    val_trajectories[0]["states"][:100],
                    val_trajectories[0]["controls"][:100],
                ],
                axis=1,
            )
        )
    )
    spectral_radius = compute_jacobian_spectral_radius(model, sample_X, n_samples=50)

    return single_step_mae, rollout_mae, spectral_radius


def main():
    print("=" * 70)
    print("JACOBIAN REGULARIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Conditions: {CONDITIONS}")
    print(f"Seeds: {len(SEEDS)} seeds ({SEEDS[0]}-{SEEDS[-1]})")
    print(f"Physics weight: {PHYSICS_WEIGHT}")
    print(f"Jacobian weight: {JACOBIAN_WEIGHT}")
    print(f"Total models: {len(CONDITIONS) * len(SEEDS)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

    # Store results
    all_results = {cond: [] for cond in CONDITIONS}

    for condition in CONDITIONS:
        print(f"\n{'=' * 50}")
        print(f"CONDITION: {condition.upper()}")
        print(f"{'=' * 50}")

        for seed in SEEDS:
            model_path = MODELS_DIR / f"{condition}_s{seed}.pth"

            if model_path.exists():
                print(f"\n  Seed {seed}... LOADING")
                model = BaselinePINN()
                model.load_state_dict(torch.load(model_path))
            else:
                print(f"\n  Seed {seed}... TRAINING")
                model = BaselinePINN()
                model, _ = train_model(model, train_loader, val_loader, scaler_y, condition, seed)

            single_mae, rollout_mae, spectral_radius = evaluate_model(
                model, val_loader, val_trajectories, scaler_X, scaler_y
            )

            all_results[condition].append(
                {
                    "seed": seed,
                    "single_step_mae": single_mae,
                    "rollout_mae": rollout_mae,
                    "spectral_radius": spectral_radius,
                }
            )

            print(f"    1-step MAE: {single_mae:.5f}")
            print(f"    100-step MAE: {rollout_mae:.3f}m")
            print(f"    Spectral radius: {spectral_radius:.3f}")

    # Statistical analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary = {}
    for condition in CONDITIONS:
        rollouts = [r["rollout_mae"] for r in all_results[condition]]
        single_steps = [r["single_step_mae"] for r in all_results[condition]]
        spectral_radii = [r["spectral_radius"] for r in all_results[condition]]

        summary[condition] = {
            "rollout_mean": np.mean(rollouts),
            "rollout_std": np.std(rollouts),
            "single_step_mean": np.mean(single_steps),
            "single_step_std": np.std(single_steps),
            "spectral_radius_mean": np.mean(spectral_radii),
            "spectral_radius_std": np.std(spectral_radii),
            "seed_results": all_results[condition],
        }

        print(f"\n{condition.upper()}:")
        print(f"  Rollout MAE: {np.mean(rollouts):.3f} ± {np.std(rollouts):.3f}m")
        print(f"  Single-step MAE: {np.mean(single_steps):.5f} ± {np.std(single_steps):.5f}")
        print(f"  Spectral radius: {np.mean(spectral_radii):.3f} ± {np.std(spectral_radii):.3f}")

    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    from scipy import stats

    baseline_rollouts = [r["rollout_mae"] for r in all_results["baseline"]]
    physics_rollouts = [r["rollout_mae"] for r in all_results["physics"]]
    jacobian_rollouts = [r["rollout_mae"] for r in all_results["jacobian"]]

    # Baseline vs Physics
    t_bp, p_bp = stats.ttest_ind(baseline_rollouts, physics_rollouts)
    d_bp = (np.mean(baseline_rollouts) - np.mean(physics_rollouts)) / np.sqrt(
        (np.var(baseline_rollouts) + np.var(physics_rollouts)) / 2
    )

    print(f"\nBaseline vs Physics:")
    print(f"  t = {t_bp:.3f}, p = {p_bp:.4f}, d = {d_bp:.3f}")

    # Baseline vs Jacobian
    t_bj, p_bj = stats.ttest_ind(baseline_rollouts, jacobian_rollouts)
    d_bj = (np.mean(baseline_rollouts) - np.mean(jacobian_rollouts)) / np.sqrt(
        (np.var(baseline_rollouts) + np.var(jacobian_rollouts)) / 2
    )

    print(f"\nBaseline vs Jacobian:")
    print(f"  t = {t_bj:.3f}, p = {p_bj:.4f}, d = {d_bj:.3f}")

    # Physics vs Jacobian
    t_pj, p_pj = stats.ttest_ind(physics_rollouts, jacobian_rollouts)
    d_pj = (np.mean(physics_rollouts) - np.mean(jacobian_rollouts)) / np.sqrt(
        (np.var(physics_rollouts) + np.var(jacobian_rollouts)) / 2
    )

    print(f"\nPhysics vs Jacobian:")
    print(f"  t = {t_pj:.3f}, p = {p_pj:.4f}, d = {d_pj:.3f}")

    # Correlation: spectral radius vs rollout
    all_spectral = []
    all_rollout = []
    for condition in CONDITIONS:
        for r in all_results[condition]:
            all_spectral.append(r["spectral_radius"])
            all_rollout.append(r["rollout_mae"])

    r_corr, p_corr = stats.pearsonr(all_spectral, all_rollout)
    print(f"\nSpectral radius vs Rollout correlation:")
    print(f"  r = {r_corr:.3f}, p = {p_corr:.4f}")

    # Save results
    summary["statistical_tests"] = {
        "baseline_vs_physics": {"t": t_bp, "p": p_bp, "d": d_bp},
        "baseline_vs_jacobian": {"t": t_bj, "p": p_bj, "d": d_bj},
        "physics_vs_jacobian": {"t": t_pj, "p": p_pj, "d": d_pj},
        "spectral_rollout_correlation": {"r": r_corr, "p": p_corr},
    }

    with open(RESULTS_DIR / "jacobian_experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nResults saved to: {RESULTS_DIR / 'jacobian_experiment_results.json'}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING FOR ICML")
    print("=" * 70)

    best_condition = min(CONDITIONS, key=lambda c: summary[c]["rollout_mean"])
    print(f"\nBest condition: {best_condition.upper()}")
    print(f"  Rollout MAE: {summary[best_condition]['rollout_mean']:.3f}m")

    if best_condition == "jacobian":
        improvement_vs_baseline = (
            (summary["baseline"]["rollout_mean"] - summary["jacobian"]["rollout_mean"])
            / summary["baseline"]["rollout_mean"]
            * 100
        )
        improvement_vs_physics = (
            (summary["physics"]["rollout_mean"] - summary["jacobian"]["rollout_mean"])
            / summary["physics"]["rollout_mean"]
            * 100
        )
        print(f"\nJacobian regularization:")
        print(f"  {improvement_vs_baseline:.1f}% better than baseline")
        print(f"  {improvement_vs_physics:.1f}% better than physics loss")
        print("\n→ VALIDATES THE 'RIGHT INDUCTIVE BIAS' HYPOTHESIS")

    return summary


if __name__ == "__main__":
    results = main()
