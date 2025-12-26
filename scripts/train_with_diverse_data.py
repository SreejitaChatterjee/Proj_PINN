"""
Train PINN with 100 diverse trajectories and relaxed parameter bounds
Improvements:
1. 10× more training data (70 vs 7 trajectories)
2. Much higher diversity in maneuvers
3. Relaxed parameter bounds (±60% for inertias, ±40% for mass)
4. Proper regularization maintained
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pinn_model import QuadrotorPINN
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / "data" / "train_set_diverse.csv"
VAL_DATA = PROJECT_ROOT / "data" / "val_set_diverse.csv"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "quadrotor_pinn_diverse.pth"
SCALER_SAVE_PATH = PROJECT_ROOT / "models" / "scalers_diverse.pkl"

# Hyperparameters (keeping successful regularization from previous training)
HIDDEN_SIZE = 256
NUM_LAYERS = 5
DROPOUT = 0.3
BATCH_SIZE = 512  # Larger batch size for more data
MAX_EPOCHS = 1000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 50

# Loss weights
PHYSICS_WEIGHT = 20.0
TEMPORAL_WEIGHT = 2.0
STABILITY_WEIGHT = 0.05


def load_data(data_path):
    """Load and prepare data from CSV"""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Map column names
    df = df.rename(columns={"roll": "phi", "pitch": "theta", "yaw": "psi"})

    # State and input features
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    input_features = state_cols + ["thrust", "torque_x", "torque_y", "torque_z"]

    # Build sequences respecting trajectory boundaries
    X, y = [], []
    for traj_id in df["trajectory_id"].unique():
        df_traj = df[df["trajectory_id"] == traj_id].reset_index(drop=True)
        for i in range(len(df_traj) - 1):
            X.append(df_traj.iloc[i][input_features].values)
            y.append(df_traj.iloc[i + 1][state_cols].values)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"  Loaded {len(X):,} samples from {df['trajectory_id'].nunique()} trajectories")
    return X, y


def create_dataloaders(X_train, y_train, X_val, y_val, scaler_X, scaler_y):
    """Scale data and create DataLoaders - keep both scaled and unscaled for physics loss"""
    # Scale data
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)

    # Create datasets with BOTH scaled and unscaled data
    # [scaled_X, scaled_y, unscaled_X, unscaled_y]
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_scaled),
        torch.FloatTensor(X_train),  # Unscaled for physics loss
        torch.FloatTensor(y_train),  # Unscaled for physics loss
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_scaled),
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


def evaluate_model(model, dataloader, scaler_y):
    """Evaluate model on validation/test set"""
    model.eval()
    total_data_loss = 0
    total_physics_loss = 0
    total_samples = 0

    # Convert scaler params to tensors for inverse transform
    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    with torch.no_grad():
        for X_scaled, y_scaled, X_unscaled, y_unscaled in dataloader:
            # Predictions (scaled)
            y_pred_scaled = model(X_scaled)

            # Data loss (on scaled data - this is fine)
            data_loss = nn.MSELoss()(y_pred_scaled, y_scaled)

            # Inverse transform predictions for physics loss
            y_pred_unscaled = y_pred_scaled * y_scale + y_mean

            # Physics loss on UNSCALED data (actual physical units)
            physics_loss = model.physics_loss(X_unscaled, y_pred_unscaled)

            batch_size = X_scaled.size(0)
            total_data_loss += data_loss.item() * batch_size
            total_physics_loss += physics_loss.item() * batch_size
            total_samples += batch_size

    avg_data_loss = total_data_loss / total_samples
    avg_physics_loss = total_physics_loss / total_samples
    total_loss = avg_data_loss + PHYSICS_WEIGHT * avg_physics_loss

    return total_loss, avg_data_loss, avg_physics_loss


def train_model():
    print("=" * 80)
    print("TRAINING PINN WITH DIVERSE DATASET (100 TRAJECTORIES)")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading diverse train and validation data...")
    X_train, y_train = load_data(TRAIN_DATA)
    X_val, y_val = load_data(VAL_DATA)

    # Create scalers and dataloaders
    print("\n[2/6] Creating dataloaders with scaling...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, scaler_X, scaler_y
    )

    # Initialize model
    print(f"\n[3/6] Initializing model with relaxed parameter bounds...")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Relaxed bounds: mass +/-40%, inertias +/-60%")
    model = QuadrotorPINN(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)

    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay (L2): {WEIGHT_DECAY}")
    print(f"  Gradient clipping: {GRADIENT_CLIP}")
    print(f"  Early stopping patience: {EARLY_STOP_PATIENCE}")

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_data_loss": [],
        "val_physics_loss": [],
        "learning_rate": [],
        "param_m": [],
        "param_Jxx": [],
        "param_Jyy": [],
        "param_Jzz": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n[4/6] Starting training for max {MAX_EPOCHS} epochs...")
    print(f"{'='*80}")
    print(
        f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Data':<12} {'Val Physics':<12} {'LR':<10} {'Status'}"
    )
    print(f"{'='*80}")

    # Convert scaler params to tensors for inverse transform during training
    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for X_scaled, y_scaled, X_unscaled, y_unscaled in train_loader:
            optimizer.zero_grad()

            # Forward pass (on scaled data)
            y_pred_scaled = model(X_scaled)

            # Data loss (scaled space - this is fine for learning)
            data_loss = nn.MSELoss()(y_pred_scaled, y_scaled)

            # Inverse transform predictions for physics loss
            y_pred_unscaled = y_pred_scaled * y_scale + y_mean

            # Physics loss on UNSCALED data (actual physical units!)
            physics_loss = model.physics_loss(X_unscaled, y_pred_unscaled)

            # Temporal loss (scaled space)
            temporal_loss = nn.MSELoss()(y_pred_scaled[:, :3], y_scaled[:, :3])

            # Stability loss (scaled space)
            stability_loss = torch.mean(torch.sum(y_pred_scaled**2, dim=1))

            # Combined loss
            loss = (
                data_loss
                + PHYSICS_WEIGHT * physics_loss
                + TEMPORAL_WEIGHT * temporal_loss
                + STABILITY_WEIGHT * stability_loss
            )

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            # Constrain parameters
            model.constrain_parameters()

            train_loss += loss.item() * X_scaled.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        val_loss, val_data_loss, val_physics_loss = evaluate_model(model, val_loader, scaler_y)

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_data_loss"].append(val_data_loss)
        history["val_physics_loss"].append(val_physics_loss)
        history["learning_rate"].append(current_lr)

        # Track parameter evolution
        history["param_m"].append(model.params["m"].item())
        history["param_Jxx"].append(model.params["Jxx"].item())
        history["param_Jyy"].append(model.params["Jyy"].item())
        history["param_Jzz"].append(model.params["Jzz"].item())

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            status = "BEST"
            # Save best model
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, SCALER_SAVE_PATH)
        else:
            patience_counter += 1
            status = f"Patience: {patience_counter}/{EARLY_STOP_PATIENCE}"

        # Print progress
        if epoch % 10 == 0 or patience_counter == 0:
            print(
                f"{epoch:<8} {train_loss:<12.6f} {val_loss:<12.6f} {val_data_loss:<12.6f} {val_physics_loss:<12.6f} {current_lr:<10.2e} {status}"
            )

        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(
                f"\n[EARLY STOP] No improvement for {EARLY_STOP_PATIENCE} epochs. Best val loss: {best_val_loss:.6f}"
            )
            break

    print(f"{'='*80}")
    print(f"\n[5/6] Training completed!")
    print(f"  Total epochs: {epoch+1}")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Final learning rate: {current_lr:.2e}")

    # Print learned parameters
    print(f"\n[LEARNED PARAMETERS]")
    print(f"{'Parameter':<10} {'Learned':<15} {'True':<15} {'Error %'}")
    print("-" * 55)
    for param_name, param in model.params.items():
        learned = param.item()
        true = model.true_params[param_name]
        error = abs(learned - true) / true * 100
        print(f"{param_name:<10} {learned:<15.6e} {true:<15.6e} {error:<.2f}%")

    # Plot training history
    print(f"\n[6/6] Saving training history plot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss curves
    axes[0, 0].plot(history["train_loss"], label="Train Loss", alpha=0.7)
    axes[0, 0].plot(history["val_loss"], label="Val Loss", alpha=0.7)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Training vs Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale("log")

    # Validation components
    axes[0, 1].plot(history["val_data_loss"], label="Data Loss", alpha=0.7)
    axes[0, 1].plot(history["val_physics_loss"], label="Physics Loss", alpha=0.7)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Validation Loss Components")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale("log")

    # Learning rate
    axes[0, 2].plot(history["learning_rate"])
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Learning Rate")
    axes[0, 2].set_title("Learning Rate Schedule")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale("log")

    # Parameter evolution - Mass
    axes[1, 0].plot(history["param_m"], label="Learned")
    axes[1, 0].axhline(y=model.true_params["m"], color="r", linestyle="--", label="True")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Mass (kg)")
    axes[1, 0].set_title("Mass Parameter Evolution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Parameter evolution - Inertias
    axes[1, 1].plot(history["param_Jxx"], label="Jxx")
    axes[1, 1].plot(history["param_Jyy"], label="Jyy")
    axes[1, 1].plot(history["param_Jzz"], label="Jzz")
    axes[1, 1].axhline(y=model.true_params["Jxx"], color="r", linestyle="--", alpha=0.3)
    axes[1, 1].axhline(y=model.true_params["Jyy"], color="g", linestyle="--", alpha=0.3)
    axes[1, 1].axhline(y=model.true_params["Jzz"], color="b", linestyle="--", alpha=0.3)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Inertia (kg·m²)")
    axes[1, 1].set_title("Inertia Parameters Evolution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale("log")

    # Generalization gap
    gap = [v - t for v, t in zip(history["val_loss"], history["train_loss"])]
    axes[1, 2].plot(gap)
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Val Loss - Train Loss")
    axes[1, 2].set_title("Generalization Gap")
    axes[1, 2].axhline(y=0, color="r", linestyle="--", alpha=0.3)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "results" / "training_history_diverse.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {plot_path}")

    print(f"\n{'='*80}")
    print("SUCCESS: Model trained with diverse dataset!")
    print(f"{'='*80}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Scalers saved to: {SCALER_SAVE_PATH}")
    print(f"\nKey improvements:")
    print(f"  1. 10x more training data (70 vs 7 trajectories)")
    print(f"  2. Much higher diversity in maneuvers")
    print(f"  3. Relaxed parameter bounds")
    print(f"\nNext: Evaluate on test set (15 held-out trajectories)")
    print(f"{'='*80}")

    return model, history


if __name__ == "__main__":
    train_model()
