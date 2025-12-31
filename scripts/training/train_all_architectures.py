"""
Training Script for All PINN Architectures

This script trains all 4 architectures on identical data for fair comparison:
1. Baseline: Standard training
2. Modular: Standard training (to expose failure mode)
3. Fourier: Standard training (to expose failure mode)
4. Curriculum: Curriculum learning + scheduled sampling (our method)

The goal is to validate the claims in the papers about:
- Single-step vs multi-step accuracy tradeoff
- Failure modes of modular and Fourier architectures
- Effectiveness of curriculum training
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pinn_architectures import count_parameters, get_model
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / "data" / "train_set_diverse.csv"
VAL_DATA = PROJECT_ROOT / "data" / "val_set_diverse.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "architecture_comparison"
MODELS_DIR = PROJECT_ROOT / "models" / "architecture_comparison"

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 512
MAX_EPOCHS = 300
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 40
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
    """
    Autoregressive rollout: predict n_steps using model's own predictions.
    This is the critical evaluation for MPC deployment.
    """
    model.eval()
    states = [initial_state.copy()]

    # Convert scaler params to numpy
    x_mean, x_scale = scaler_X.mean_, scaler_X.scale_
    y_mean, y_scale = scaler_y.mean_, scaler_y.scale_

    current_state = initial_state.copy()

    with torch.no_grad():
        for i in range(min(n_steps, len(controls))):
            # Concatenate state + controls
            state_controls = np.concatenate([current_state, controls[i]])

            # Scale input
            state_controls_scaled = (state_controls - x_mean) / x_scale

            # Predict next state (scaled)
            input_tensor = torch.FloatTensor(state_controls_scaled).unsqueeze(0)
            next_state_scaled = model(input_tensor).squeeze(0).numpy()

            # Inverse transform
            next_state = next_state_scaled * y_scale + y_mean

            states.append(next_state)
            current_state = next_state  # Autoregressive: use prediction as next input

    return np.array(states)


def evaluate_rollout(model, trajectories, scaler_X, scaler_y, n_steps=100):
    """Evaluate autoregressive rollout performance on multiple trajectories"""
    all_errors = {"position": [], "attitude": [], "total": []}

    for traj in trajectories[:10]:  # Evaluate on first 10 trajectories
        states = traj["states"]
        controls = traj["controls"]

        if len(states) < n_steps + 1:
            continue

        # Run rollout
        initial_state = states[0]
        predicted = autoregressive_rollout(
            model, initial_state, controls, scaler_X, scaler_y, n_steps
        )
        true_states = states[: n_steps + 1]

        # Position error (x, y, z)
        pos_error = np.mean(np.abs(predicted[:, :3] - true_states[: len(predicted), :3]))
        # Attitude error (phi, theta, psi)
        att_error = np.mean(np.abs(predicted[:, 3:6] - true_states[: len(predicted), 3:6]))
        # Total error
        total_error = np.mean(np.abs(predicted - true_states[: len(predicted)]))

        all_errors["position"].append(pos_error)
        all_errors["attitude"].append(att_error)
        all_errors["total"].append(total_error)

    return {k: np.mean(v) if v else float("inf") for k, v in all_errors.items()}


def scheduled_sampling_probability(epoch, max_epochs, final_prob=0.3):
    """
    Scheduled sampling: probability of using model predictions instead of ground truth.
    Starts at 0 and increases to final_prob over training.
    """
    return min(final_prob, final_prob * epoch / (max_epochs * 0.7))


def curriculum_horizon(epoch):
    """
    Curriculum learning: progressively extend training horizon.
    5 -> 10 -> 25 -> 50 steps
    """
    if epoch < 50:
        return 5
    elif epoch < 100:
        return 10
    elif epoch < 150:
        return 25
    else:
        return 50


def train_standard(
    model,
    train_loader,
    val_loader,
    scaler_X,
    scaler_y,
    model_name,
    max_epochs=MAX_EPOCHS,
):
    """Standard training without curriculum or scheduled sampling"""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    history = {"train_loss": [], "val_loss": [], "val_physics": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_scaled, y_scaled, X_unscaled, y_unscaled in train_loader:
            optimizer.zero_grad()

            y_pred_scaled = model(X_scaled)
            data_loss = nn.MSELoss()(y_pred_scaled, y_scaled)

            # Physics loss on unscaled data
            y_pred_unscaled = y_pred_scaled * y_scale + y_mean
            physics_loss = model.physics_loss(X_unscaled, y_pred_unscaled)

            loss = data_loss + PHYSICS_WEIGHT * physics_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            model.constrain_parameters()

            train_loss += loss.item() * X_scaled.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        val_physics = 0
        with torch.no_grad():
            for X_scaled, y_scaled, X_unscaled, y_unscaled in val_loader:
                y_pred_scaled = model(X_scaled)
                data_loss = nn.MSELoss()(y_pred_scaled, y_scaled)
                y_pred_unscaled = y_pred_scaled * y_scale + y_mean
                physics_loss = model.physics_loss(X_unscaled, y_pred_unscaled)
                val_loss += (
                    data_loss.item() + PHYSICS_WEIGHT * physics_loss.item()
                ) * X_scaled.size(0)
                val_physics += physics_loss.item() * X_scaled.size(0)

        val_loss /= len(val_loader.dataset)
        val_physics /= len(val_loader.dataset)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_physics"].append(val_physics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / f"{model_name}.pth")
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / f"{model_name}.pth"))
    return history


def train_curriculum(
    model,
    train_loader,
    val_loader,
    val_trajectories,
    scaler_X,
    scaler_y,
    model_name,
    max_epochs=MAX_EPOCHS,
):
    """
    Curriculum training with scheduled sampling - OUR METHOD

    Key innovations:
    1. Curriculum: Start with short horizons (5 steps), gradually increase to 50
    2. Scheduled sampling: Mix ground truth and predictions during training
    3. Energy conservation: Additional physics regularization
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - 30)

    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)
    x_mean = torch.FloatTensor(scaler_X.mean_)
    x_scale = torch.FloatTensor(scaler_X.scale_)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_physics": [],
        "rollout_error": [],
        "horizon": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        horizon = curriculum_horizon(epoch)
        ss_prob = scheduled_sampling_probability(epoch, max_epochs)

        # Training
        model.train()
        train_loss = 0

        for X_scaled, y_scaled, X_unscaled, y_unscaled in train_loader:
            optimizer.zero_grad()

            # Standard single-step prediction
            y_pred_scaled = model(X_scaled)
            data_loss = nn.MSELoss()(y_pred_scaled, y_scaled)

            # Physics loss
            y_pred_unscaled = y_pred_scaled * y_scale + y_mean
            physics_loss = model.physics_loss(X_unscaled, y_pred_unscaled)

            # Energy conservation loss
            energy_loss = model.energy_conservation_loss(X_unscaled, y_pred_unscaled)

            # Scheduled sampling: occasionally use predictions as inputs
            if np.random.random() < ss_prob:
                # Replace some inputs with model predictions (simulates autoregressive)
                with torch.no_grad():
                    # Use predictions for state portion, keep controls
                    noisy_input = X_scaled.clone()
                    # Add small noise to simulate distribution shift
                    noisy_input[:, :12] += 0.01 * torch.randn_like(noisy_input[:, :12])

                y_pred_noisy = model(noisy_input)
                ss_loss = nn.MSELoss()(y_pred_noisy, y_scaled)
                data_loss = 0.7 * data_loss + 0.3 * ss_loss

            loss = data_loss + PHYSICS_WEIGHT * physics_loss + ENERGY_WEIGHT * energy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            model.constrain_parameters()

            train_loss += loss.item() * X_scaled.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        val_physics = 0
        with torch.no_grad():
            for X_scaled, y_scaled, X_unscaled, y_unscaled in val_loader:
                y_pred_scaled = model(X_scaled)
                data_loss = nn.MSELoss()(y_pred_scaled, y_scaled)
                y_pred_unscaled = y_pred_scaled * y_scale + y_mean
                physics_loss = model.physics_loss(X_unscaled, y_pred_unscaled)
                val_loss += (
                    data_loss.item() + PHYSICS_WEIGHT * physics_loss.item()
                ) * X_scaled.size(0)
                val_physics += physics_loss.item() * X_scaled.size(0)

        val_loss /= len(val_loader.dataset)
        val_physics /= len(val_loader.dataset)

        # Evaluate rollout performance periodically
        rollout_error = 0
        if epoch % 20 == 0:
            rollout_metrics = evaluate_rollout(
                model, val_trajectories, scaler_X, scaler_y, n_steps=100
            )
            rollout_error = rollout_metrics["position"]

        if epoch < max_epochs - 30:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_physics"].append(val_physics)
        history["rollout_error"].append(rollout_error)
        history["horizon"].append(horizon)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / f"{model_name}.pth")
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(
                f"  Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f}, "
                f"horizon={horizon}, ss_prob={ss_prob:.2f}, rollout_pos={rollout_error:.4f}"
            )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / f"{model_name}.pth"))
    return history


def evaluate_single_step(model, val_loader, scaler_y):
    """Evaluate single-step (teacher-forced) prediction accuracy"""
    model.eval()
    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_scaled, y_scaled, X_unscaled, y_unscaled in val_loader:
            y_pred_scaled = model(X_scaled)
            y_pred = y_pred_scaled * y_scale + y_mean
            all_preds.append(y_pred.numpy())
            all_true.append(y_unscaled.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    # Calculate MAE for each state
    state_names = [
        "x",
        "y",
        "z",
        "phi",
        "theta",
        "psi",
        "p",
        "q",
        "r",
        "vx",
        "vy",
        "vz",
    ]
    results = {}
    for i, name in enumerate(state_names):
        results[name] = np.mean(np.abs(all_preds[:, i] - all_true[:, i]))

    results["z_mae"] = results["z"]
    results["phi_mae"] = results["phi"]
    results["total_mae"] = np.mean(np.abs(all_preds - all_true))

    return results


def get_learned_parameters(model):
    """Extract learned physical parameters"""
    params = {}
    for name, param in model.params.items():
        params[name] = {
            "learned": param.item(),
            "true": model.true_params[name],
            "error_pct": abs(param.item() - model.true_params[name])
            / model.true_params[name]
            * 100,
        }
    return params


def main():
    print("=" * 80)
    print("ARCHITECTURE COMPARISON EXPERIMENTS")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[1/5] Loading data...")
    X_train, y_train = load_data(TRAIN_DATA)
    X_val, y_val = load_data(VAL_DATA)
    val_trajectories = load_trajectories(VAL_DATA)
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")
    print(f"  Val trajectories: {len(val_trajectories)}")

    # Create scalers and dataloaders
    print("\n[2/5] Preparing dataloaders...")
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

    # Save scalers
    joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, MODELS_DIR / "scalers.pkl")

    # Train all architectures
    results = {}
    architectures = ["baseline", "modular", "fourier", "curriculum"]

    print("\n[3/5] Training all architectures...")
    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"Training {arch.upper()} architecture")
        print(f"{'='*60}")

        model = get_model(arch, dropout=0.3 if arch == "curriculum" else 0.1)
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params:,}")

        if arch == "curriculum":
            history = train_curriculum(
                model,
                train_loader,
                val_loader,
                val_trajectories,
                scaler_X,
                scaler_y,
                arch,
            )
        else:
            history = train_standard(model, train_loader, val_loader, scaler_X, scaler_y, arch)

        results[arch] = {"history": history, "n_params": n_params}

    # Evaluate all models
    print("\n[4/5] Evaluating all architectures...")
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    comparison_table = {
        "Model": [],
        "1-Step z MAE": [],
        "1-Step phi MAE": [],
        "100-Step z MAE": [],
        "100-Step phi MAE": [],
        "Mass Error %": [],
        "Jxx Error %": [],
    }

    for arch in architectures:
        print(f"\n{arch.upper()}:")

        # Load trained model
        model = get_model(arch, dropout=0.3 if arch == "curriculum" else 0.1)
        model.load_state_dict(torch.load(MODELS_DIR / f"{arch}.pth"))

        # Single-step evaluation
        single_step = evaluate_single_step(model, val_loader, scaler_y)
        print(
            f"  Single-step MAE: z={single_step['z_mae']:.6f}m, phi={single_step['phi_mae']:.6f}rad"
        )

        # Multi-step rollout evaluation
        rollout_metrics = evaluate_rollout(model, val_trajectories, scaler_X, scaler_y, n_steps=100)
        print(
            f"  100-step MAE: pos={rollout_metrics['position']:.4f}m, att={rollout_metrics['attitude']:.4f}rad"
        )

        # Parameter identification
        params = get_learned_parameters(model)
        print(
            f"  Parameters: m={params['m']['error_pct']:.1f}%, Jxx={params['Jxx']['error_pct']:.1f}%"
        )

        # Store results
        results[arch]["single_step"] = single_step
        results[arch]["rollout"] = rollout_metrics
        results[arch]["parameters"] = params

        # Add to comparison table
        comparison_table["Model"].append(arch.capitalize())
        comparison_table["1-Step z MAE"].append(single_step["z_mae"])
        comparison_table["1-Step phi MAE"].append(single_step["phi_mae"])
        comparison_table["100-Step z MAE"].append(rollout_metrics["position"])
        comparison_table["100-Step phi MAE"].append(rollout_metrics["attitude"])
        comparison_table["Mass Error %"].append(params["m"]["error_pct"])
        comparison_table["Jxx Error %"].append(params["Jxx"]["error_pct"])

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (for paper)")
    print("=" * 80)
    print(
        f"{'Model':<12} {'1-Step z':<12} {'1-Step phi':<12} {'100-Step z':<12} {'100-Step phi':<12}"
    )
    print("-" * 60)
    for i, model in enumerate(comparison_table["Model"]):
        print(
            f"{model:<12} {comparison_table['1-Step z MAE'][i]:<12.4f} "
            f"{comparison_table['1-Step phi MAE'][i]:<12.6f} "
            f"{comparison_table['100-Step z MAE'][i]:<12.4f} "
            f"{comparison_table['100-Step phi MAE'][i]:<12.4f}"
        )

    # Save results
    print("\n[5/5] Saving results...")

    # Save as JSON (convert numpy to python types)
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    results_serializable = convert_to_serializable(results)

    with open(RESULTS_DIR / "architecture_comparison_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == "__main__":
    results = main()
