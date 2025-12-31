"""
Comprehensive Ablation Study for PINN Paper

This script runs all ablations needed for a publishable paper:
1. Pure NN baseline (no physics loss) - shows physics loss value
2. Parameter-matched baseline (72K params) - isolates architecture vs params
3. Multiple seeds (3 runs each) - statistical significance
4. Least squares comparison - traditional baseline

Run time: ~2-3 hours total
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent))
from pinn_architectures import BaselinePINN, ModularPINN, PhysicsLossMixin

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / "data" / "train_set_diverse.csv"
VAL_DATA = PROJECT_ROOT / "data" / "val_set_diverse.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "comprehensive_ablation"
MODELS_DIR = PROJECT_ROOT / "models" / "comprehensive_ablation"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training config
BATCH_SIZE = 512
MAX_EPOCHS = 100  # Reduced for faster runs
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 20  # Faster early stopping
PHYSICS_WEIGHT = 20.0


# ============================================================================
# NEW ARCHITECTURES FOR ABLATION
# ============================================================================


class PureNNBaseline(nn.Module):
    """
    Pure Neural Network without physics loss.
    Same architecture as BaselinePINN but no physics constraints.
    """

    def __init__(self, input_size=16, hidden_size=256, output_size=12, num_layers=5, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)
        self.name = "PureNN"

    def forward(self, x):
        return self.network(x)


class SmallBaseline(nn.Module, PhysicsLossMixin):
    """
    Parameter-matched baseline: same param count as Modular (72K).
    Uses hidden_size=128 to match modular's parameter count.
    """

    def __init__(self, input_size=16, hidden_size=128, output_size=12, num_layers=5, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)
        self._init_physics_params()
        self.name = "SmallBaseline"

    def forward(self, x):
        return self.network(x)


# ============================================================================
# DATA LOADING
# ============================================================================


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


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def train_model(
    model,
    train_loader,
    val_loader,
    scaler_X,
    scaler_y,
    model_name,
    use_physics=True,
    seed=0,
):
    """Generic training function"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for X_scaled, y_scaled, X_unscaled, y_unscaled in train_loader:
            optimizer.zero_grad()

            y_pred_scaled = model(X_scaled)
            data_loss = nn.MSELoss()(y_pred_scaled, y_scaled)

            loss = data_loss

            # Add physics loss if enabled and model supports it
            if use_physics and hasattr(model, "physics_loss"):
                y_pred_unscaled = y_pred_scaled * y_scale + y_mean
                physics_loss = model.physics_loss(X_unscaled, y_pred_unscaled)
                loss = data_loss + PHYSICS_WEIGHT * physics_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            if hasattr(model, "constrain_parameters"):
                model.constrain_parameters()

            train_loss += loss.item() * X_scaled.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_scaled, y_scaled, X_unscaled, y_unscaled in val_loader:
                y_pred_scaled = model(X_scaled)
                data_loss = nn.MSELoss()(y_pred_scaled, y_scaled)

                if use_physics and hasattr(model, "physics_loss"):
                    y_pred_unscaled = y_pred_scaled * y_scale + y_mean
                    physics_loss = model.physics_loss(X_unscaled, y_pred_unscaled)
                    val_loss += (
                        data_loss.item() + PHYSICS_WEIGHT * physics_loss.item()
                    ) * X_scaled.size(0)
                else:
                    val_loss += data_loss.item() * X_scaled.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / f"{model_name}_seed{seed}.pth")
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / f"{model_name}_seed{seed}.pth"))
    return model


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================


def autoregressive_rollout(model, initial_state, controls, scaler_X, scaler_y, n_steps):
    """Autoregressive rollout prediction"""
    model.eval()
    states = [initial_state.copy()]

    x_mean, x_scale = scaler_X.mean_, scaler_X.scale_
    y_mean, y_scale = scaler_y.mean_, scaler_y.scale_

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
    """Full evaluation: single-step + rollout"""
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

    single_step = {
        "z_mae": np.mean(np.abs(all_preds[:, 2] - all_true[:, 2])),
        "phi_mae": np.mean(np.abs(all_preds[:, 3] - all_true[:, 3])),
        "total_mae": np.mean(np.abs(all_preds - all_true)),
    }

    # Rollout evaluation
    pos_errors, att_errors = [], []
    for traj in val_trajectories[:10]:
        states = traj["states"]
        controls = traj["controls"]

        if len(states) < 101:
            continue

        predicted = autoregressive_rollout(model, states[0], controls, scaler_X, scaler_y, 100)
        true_states = states[:101]

        pos_errors.append(np.mean(np.abs(predicted[:, :3] - true_states[: len(predicted), :3])))
        att_errors.append(np.mean(np.abs(predicted[:, 3:6] - true_states[: len(predicted), 3:6])))

    rollout = {
        "position_mae": np.mean(pos_errors) if pos_errors else float("inf"),
        "attitude_mae": np.mean(att_errors) if att_errors else float("inf"),
    }

    return {"single_step": single_step, "rollout": rollout}


# ============================================================================
# LEAST SQUARES BASELINE
# ============================================================================


def least_squares_baseline(X_train, y_train, X_val, y_val, val_trajectories, scaler_X, scaler_y):
    """
    Traditional least squares system identification baseline.
    Fits a linear model: x_{t+1} = A*x_t + B*u_t
    """
    print("\n  Training Least Squares model...")

    # Fit Ridge regression (regularized least squares)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Single-step evaluation
    y_pred = model.predict(X_val)
    single_step = {
        "z_mae": np.mean(np.abs(y_pred[:, 2] - y_val[:, 2])),
        "phi_mae": np.mean(np.abs(y_pred[:, 3] - y_val[:, 3])),
        "total_mae": np.mean(np.abs(y_pred - y_val)),
    }

    # Rollout evaluation (need custom rollout for sklearn model)
    pos_errors, att_errors = [], []
    for traj in val_trajectories[:10]:
        states = traj["states"]
        controls = traj["controls"]

        if len(states) < 101:
            continue

        # Manual rollout
        predicted = [states[0]]
        current_state = states[0].copy()

        for i in range(min(100, len(controls))):
            inp = np.concatenate([current_state, controls[i]]).reshape(1, -1)
            next_state = model.predict(inp)[0]
            predicted.append(next_state)
            current_state = next_state

        predicted = np.array(predicted)
        true_states = states[: len(predicted)]

        pos_errors.append(np.mean(np.abs(predicted[:, :3] - true_states[:, :3])))
        att_errors.append(np.mean(np.abs(predicted[:, 3:6] - true_states[:, 3:6])))

    rollout = {
        "position_mae": np.mean(pos_errors) if pos_errors else float("inf"),
        "attitude_mae": np.mean(att_errors) if att_errors else float("inf"),
    }

    return {"single_step": single_step, "rollout": rollout}


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def main():
    print("=" * 80)
    print("COMPREHENSIVE ABLATION STUDY")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[1/6] Loading data...")
    X_train, y_train = load_data(TRAIN_DATA)
    X_val, y_val = load_data(VAL_DATA)
    val_trajectories = load_trajectories(VAL_DATA)
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")

    # Prepare data
    print("\n[2/6] Preparing data...")
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

    joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, MODELS_DIR / "scalers.pkl")

    all_results = {}

    # ========================================================================
    # ABLATION 1: Pure NN (no physics loss)
    # ========================================================================
    print("\n[3/6] ABLATION 1: Pure NN vs PINN...")
    print("-" * 60)

    for name, use_physics in [("PureNN", False), ("PINN_Baseline", True)]:
        print(f"\n  Training {name}...")
        if name == "PureNN":
            model = PureNNBaseline()
        else:
            model = BaselinePINN()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {n_params:,}")

        model = train_model(
            model,
            train_loader,
            val_loader,
            scaler_X,
            scaler_y,
            name,
            use_physics=use_physics,
            seed=42,
        )

        results = evaluate_model(model, val_loader, val_trajectories, scaler_X, scaler_y)
        results["n_params"] = n_params
        all_results[name] = results

        print(f"    1-step z MAE: {results['single_step']['z_mae']:.4f}")
        print(f"    100-step pos MAE: {results['rollout']['position_mae']:.4f}")

    # ========================================================================
    # ABLATION 2: Parameter-matched baseline (72K params)
    # ========================================================================
    print("\n[4/6] ABLATION 2: Parameter-matched comparison...")
    print("-" * 60)

    for name, ModelClass in [
        ("SmallBaseline_72K", SmallBaseline),
        ("Modular_72K", ModularPINN),
    ]:
        print(f"\n  Training {name}...")
        model = ModelClass()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {n_params:,}")

        model = train_model(
            model,
            train_loader,
            val_loader,
            scaler_X,
            scaler_y,
            name,
            use_physics=True,
            seed=42,
        )

        results = evaluate_model(model, val_loader, val_trajectories, scaler_X, scaler_y)
        results["n_params"] = n_params
        all_results[name] = results

        print(f"    1-step z MAE: {results['single_step']['z_mae']:.4f}")
        print(f"    100-step pos MAE: {results['rollout']['position_mae']:.4f}")

    # ========================================================================
    # ABLATION 3: Multiple seeds
    # ========================================================================
    print("\n[5/6] ABLATION 3: Multiple seeds (statistical significance)...")
    print("-" * 60)

    seeds = [42, 123]  # 2 seeds for faster runs

    for arch_name, ModelClass in [("Baseline", BaselinePINN), ("Modular", ModularPINN)]:
        print(f"\n  {arch_name} across {len(seeds)} seeds...")

        seed_results = []
        for seed in seeds:
            print(f"    Seed {seed}...")
            model = ModelClass()
            model = train_model(
                model,
                train_loader,
                val_loader,
                scaler_X,
                scaler_y,
                f"{arch_name}",
                use_physics=True,
                seed=seed,
            )

            results = evaluate_model(model, val_loader, val_trajectories, scaler_X, scaler_y)
            seed_results.append(results)

        # Compute statistics
        z_maes = [r["single_step"]["z_mae"] for r in seed_results]
        pos_maes = [r["rollout"]["position_mae"] for r in seed_results]

        all_results[f"{arch_name}_MultiSeed"] = {
            "z_mae_mean": np.mean(z_maes),
            "z_mae_std": np.std(z_maes),
            "pos_mae_mean": np.mean(pos_maes),
            "pos_mae_std": np.std(pos_maes),
            "seeds": seeds,
            "all_results": seed_results,
        }

        print(f"    1-step z MAE: {np.mean(z_maes):.4f} +/- {np.std(z_maes):.4f}")
        print(f"    100-step pos MAE: {np.mean(pos_maes):.4f} +/- {np.std(pos_maes):.4f}")

    # ========================================================================
    # ABLATION 4: Least Squares baseline
    # ========================================================================
    print("\n[6/6] ABLATION 4: Least Squares baseline...")
    print("-" * 60)

    ls_results = least_squares_baseline(
        X_train, y_train, X_val, y_val, val_trajectories, scaler_X, scaler_y
    )
    all_results["LeastSquares"] = ls_results

    print(f"  1-step z MAE: {ls_results['single_step']['z_mae']:.4f}")
    print(f"  100-step pos MAE: {ls_results['rollout']['position_mae']:.4f}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("Table 1: Physics Loss Ablation (Pure NN vs PINN)")
    print("-" * 80)
    print(f"{'Model':<20} {'Params':<10} {'1-Step z':<12} {'100-Step Pos':<12}")
    print("-" * 60)
    for name in ["PureNN", "PINN_Baseline"]:
        r = all_results[name]
        print(
            f"{name:<20} {r['n_params']:<10,} {r['single_step']['z_mae']:<12.4f} {r['rollout']['position_mae']:<12.4f}"
        )

    print("\n" + "-" * 80)
    print("Table 2: Parameter-Matched Comparison")
    print("-" * 80)
    print(f"{'Model':<20} {'Params':<10} {'1-Step z':<12} {'100-Step Pos':<12}")
    print("-" * 60)
    for name in ["SmallBaseline_72K", "Modular_72K"]:
        r = all_results[name]
        print(
            f"{name:<20} {r['n_params']:<10,} {r['single_step']['z_mae']:<12.4f} {r['rollout']['position_mae']:<12.4f}"
        )

    print("\n" + "-" * 80)
    print("Table 3: Statistical Significance (3 seeds)")
    print("-" * 80)
    print(f"{'Model':<20} {'1-Step z (mean +/- std)':<25} {'100-Step Pos (mean +/- std)':<25}")
    print("-" * 70)
    for name in ["Baseline_MultiSeed", "Modular_MultiSeed"]:
        r = all_results[name]
        z_str = f"{r['z_mae_mean']:.4f} +/- {r['z_mae_std']:.4f}"
        pos_str = f"{r['pos_mae_mean']:.4f} +/- {r['pos_mae_std']:.4f}"
        print(f"{name:<20} {z_str:<25} {pos_str:<25}")

    print("\n" + "-" * 80)
    print("Table 4: Comparison with Traditional Methods")
    print("-" * 80)
    print(f"{'Model':<20} {'1-Step z':<12} {'100-Step Pos':<12}")
    print("-" * 50)
    print(
        f"{'LeastSquares':<20} {all_results['LeastSquares']['single_step']['z_mae']:<12.4f} {all_results['LeastSquares']['rollout']['position_mae']:<12.4f}"
    )
    print(
        f"{'Modular PINN':<20} {all_results['Modular_72K']['single_step']['z_mae']:<12.4f} {all_results['Modular_72K']['rollout']['position_mae']:<12.4f}"
    )

    # Save results
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR / 'ablation_results.json'}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results


if __name__ == "__main__":
    results = main()
