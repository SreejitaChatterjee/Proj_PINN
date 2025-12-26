"""
Ablation Study for Curriculum PINN

This script validates the ablation table in the papers:
1. Baseline (no improvements)
2. + Curriculum learning
3. + Scheduled sampling
4. + Dropout regularization
5. + Energy conservation loss (FULL METHOD)

Each component is added incrementally to show its contribution.
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pinn_architectures import BaselinePINN, PhysicsLossMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / "data" / "train_set_diverse.csv"
VAL_DATA = PROJECT_ROOT / "data" / "val_set_diverse.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "ablation_study"
MODELS_DIR = PROJECT_ROOT / "models" / "ablation_study"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 512
MAX_EPOCHS = 250
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 35
PHYSICS_WEIGHT = 20.0
ENERGY_WEIGHT = 5.0


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


def autoregressive_rollout(model, initial_state, controls, scaler_X, scaler_y, n_steps):
    """Autoregressive rollout"""
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


def evaluate_rollout(model, trajectories, scaler_X, scaler_y, n_steps=100):
    """Evaluate 100-step rollout performance"""
    position_errors = []

    for traj in trajectories[:10]:
        states = traj["states"]
        controls = traj["controls"]

        if len(states) < n_steps + 1:
            continue

        initial_state = states[0]
        predicted = autoregressive_rollout(
            model, initial_state, controls, scaler_X, scaler_y, n_steps
        )
        true_states = states[: n_steps + 1]

        # Position error (x, y, z) - this is what we report in the paper
        pos_error = np.mean(np.abs(predicted[:, :3] - true_states[: len(predicted), :3]))
        position_errors.append(pos_error)

    return np.mean(position_errors) if position_errors else float("inf")


def curriculum_horizon(epoch):
    """Curriculum: 5 -> 10 -> 25 -> 50 steps"""
    if epoch < 50:
        return 5
    elif epoch < 100:
        return 10
    elif epoch < 150:
        return 25
    else:
        return 50


def scheduled_sampling_prob(epoch, max_epochs, final_prob=0.3):
    """Scheduled sampling probability"""
    return min(final_prob, final_prob * epoch / (max_epochs * 0.7))


class AblationConfig:
    """Configuration for each ablation variant"""

    def __init__(
        self,
        name,
        use_curriculum=False,
        use_scheduled_sampling=False,
        use_dropout=False,
        use_energy_loss=False,
    ):
        self.name = name
        self.use_curriculum = use_curriculum
        self.use_scheduled_sampling = use_scheduled_sampling
        self.use_dropout = use_dropout
        self.use_energy_loss = use_energy_loss
        self.dropout_rate = 0.3 if use_dropout else 0.0


def train_ablation_variant(
    config,
    train_loader,
    val_loader,
    val_trajectories,
    scaler_X,
    scaler_y,
    max_epochs=MAX_EPOCHS,
):
    """Train a single ablation variant"""
    print(f"\n{'='*60}")
    print(f"Training: {config.name}")
    print(f"  Curriculum: {config.use_curriculum}")
    print(f"  Scheduled sampling: {config.use_scheduled_sampling}")
    print(f"  Dropout: {config.use_dropout} ({config.dropout_rate})")
    print(f"  Energy loss: {config.use_energy_loss}")
    print(f"{'='*60}")

    model = BaselinePINN(dropout=config.dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if config.use_curriculum:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - 30)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15
        )

    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "rollout_error": []}

    for epoch in range(max_epochs):
        # Get curriculum horizon if enabled
        horizon = curriculum_horizon(epoch) if config.use_curriculum else 1
        ss_prob = scheduled_sampling_prob(epoch, max_epochs) if config.use_scheduled_sampling else 0

        # Training
        model.train()
        train_loss = 0

        for X_scaled, y_scaled, X_unscaled, y_unscaled in train_loader:
            optimizer.zero_grad()

            y_pred_scaled = model(X_scaled)
            data_loss = nn.MSELoss()(y_pred_scaled, y_scaled)

            y_pred_unscaled = y_pred_scaled * y_scale + y_mean
            physics_loss = model.physics_loss(X_unscaled, y_pred_unscaled)

            loss = data_loss + PHYSICS_WEIGHT * physics_loss

            # Energy conservation if enabled
            if config.use_energy_loss:
                energy_loss = model.energy_conservation_loss(X_unscaled, y_pred_unscaled)
                loss += ENERGY_WEIGHT * energy_loss

            # Scheduled sampling if enabled
            if config.use_scheduled_sampling and np.random.random() < ss_prob:
                with torch.no_grad():
                    noisy_input = X_scaled.clone()
                    noisy_input[:, :12] += 0.01 * torch.randn_like(noisy_input[:, :12])
                y_pred_noisy = model(noisy_input)
                ss_loss = nn.MSELoss()(y_pred_noisy, y_scaled)
                loss = 0.7 * loss + 0.3 * ss_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
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
                y_pred_unscaled = y_pred_scaled * y_scale + y_mean
                physics_loss = model.physics_loss(X_unscaled, y_pred_unscaled)
                val_loss += (
                    data_loss.item() + PHYSICS_WEIGHT * physics_loss.item()
                ) * X_scaled.size(0)

        val_loss /= len(val_loader.dataset)

        # Scheduler step
        if config.use_curriculum and epoch < max_epochs - 30:
            scheduler.step()
        elif not config.use_curriculum:
            scheduler.step(val_loss)

        # Track rollout error periodically
        rollout_error = 0
        if epoch % 25 == 0 or epoch == max_epochs - 1:
            rollout_error = evaluate_rollout(
                model, val_trajectories, scaler_X, scaler_y, n_steps=100
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["rollout_error"].append(rollout_error)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                MODELS_DIR / f'{config.name.replace(" ", "_").replace("+", "plus")}.pth',
            )
        else:
            patience_counter += 1

        if epoch % 25 == 0:
            print(
                f"  Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f}, rollout={rollout_error:.4f}m"
            )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best model and evaluate final rollout
    model.load_state_dict(
        torch.load(MODELS_DIR / f'{config.name.replace(" ", "_").replace("+", "plus")}.pth')
    )
    final_rollout = evaluate_rollout(model, val_trajectories, scaler_X, scaler_y, n_steps=100)

    return model, final_rollout, history


def main():
    print("=" * 80)
    print("ABLATION STUDY EXPERIMENTS")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[1/3] Loading data...")
    X_train, y_train = load_data(TRAIN_DATA)
    X_val, y_val = load_data(VAL_DATA)
    val_trajectories = load_trajectories(VAL_DATA)
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")

    # Create scalers and dataloaders
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

    # Define ablation configurations (cumulative)
    ablation_configs = [
        AblationConfig(
            "Baseline",
            use_curriculum=False,
            use_scheduled_sampling=False,
            use_dropout=False,
            use_energy_loss=False,
        ),
        AblationConfig(
            "+Curriculum",
            use_curriculum=True,
            use_scheduled_sampling=False,
            use_dropout=False,
            use_energy_loss=False,
        ),
        AblationConfig(
            "+Sched_Sampling",
            use_curriculum=True,
            use_scheduled_sampling=True,
            use_dropout=False,
            use_energy_loss=False,
        ),
        AblationConfig(
            "+Dropout",
            use_curriculum=True,
            use_scheduled_sampling=True,
            use_dropout=True,
            use_energy_loss=False,
        ),
        AblationConfig(
            "+Energy_Cons",
            use_curriculum=True,
            use_scheduled_sampling=True,
            use_dropout=True,
            use_energy_loss=True,
        ),
    ]

    # Run ablation experiments
    print("\n[2/3] Running ablation experiments...")
    results = {}

    for config in ablation_configs:
        model, rollout_error, history = train_ablation_variant(
            config, train_loader, val_loader, val_trajectories, scaler_X, scaler_y
        )
        results[config.name] = {
            "rollout_mae": rollout_error,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        }

    # Print ablation table
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"{'Configuration':<25} {'100-Step MAE (m)':<18} {'Improvement':<15}")
    print("-" * 60)

    baseline_mae = results["Baseline"]["rollout_mae"]
    for config in ablation_configs:
        mae = results[config.name]["rollout_mae"]
        if config.name == "Baseline":
            improvement = "--"
        else:
            improvement = f"{(1 - mae/baseline_mae)*100:.0f}%"
        print(f"{config.name:<25} {mae:<18.3f} {improvement:<15}")

    # Calculate improvement factor
    final_mae = results["+Energy_Cons"]["rollout_mae"]
    improvement_factor = baseline_mae / final_mae if final_mae > 0 else float("inf")
    print("-" * 60)
    print(f"Total improvement: {improvement_factor:.1f}x")

    # Save results
    print("\n[3/3] Saving results...")
    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
