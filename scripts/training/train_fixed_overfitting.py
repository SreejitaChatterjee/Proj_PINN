"""
Retrain PINN with proper regularization to fix overfitting
Key improvements:
1. Proper train/val/test split (by trajectory)
2. Increased dropout (0.3 instead of 0.1)
3. L2 weight decay
4. Early stopping
5. Gradient clipping
6. Better monitoring
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
TRAIN_DATA = PROJECT_ROOT / "data" / "train_set.csv"
VAL_DATA = PROJECT_ROOT / "data" / "val_set.csv"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "quadrotor_pinn_fixed.pth"
SCALER_SAVE_PATH = PROJECT_ROOT / "models" / "scalers_fixed.pkl"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters (with regularization)
HIDDEN_SIZE = 256
NUM_LAYERS = 5
DROPOUT = 0.3  # Increased from 0.1
BATCH_SIZE = 256
MAX_EPOCHS = 1000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # L2 regularization
GRADIENT_CLIP = 1.0  # Gradient clipping
EARLY_STOP_PATIENCE = 50  # Stop if no improvement for 50 epochs

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

    print(f"  Loaded {len(X)} samples from {df['trajectory_id'].nunique()} trajectories")
    return X, y


def create_dataloaders(X_train, y_train, X_val, y_val, scaler_X, scaler_y):
    """Scale data and create DataLoaders"""
    # Scale data
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled)
    )
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled))

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

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # Predictions
            y_pred = model(X_batch)

            # Data loss
            data_loss = nn.MSELoss()(y_pred, y_batch)

            # Physics loss
            physics_loss = model.physics_loss(X_batch, y_pred)

            batch_size = X_batch.size(0)
            total_data_loss += data_loss.item() * batch_size
            total_physics_loss += physics_loss.item() * batch_size
            total_samples += batch_size

    avg_data_loss = total_data_loss / total_samples
    avg_physics_loss = total_physics_loss / total_samples
    total_loss = avg_data_loss + PHYSICS_WEIGHT * avg_physics_loss

    return total_loss, avg_data_loss, avg_physics_loss


def train_model():
    print("=" * 80)
    print("RETRAINING PINN WITH PROPER REGULARIZATION")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading train and validation data...")
    X_train, y_train = load_data(TRAIN_DATA)
    X_val, y_val = load_data(VAL_DATA)

    # Create scalers and dataloaders
    print("\n[2/6] Creating dataloaders with scaling...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, scaler_X, scaler_y
    )

    # Initialize model with increased dropout
    print(f"\n[3/6] Initializing model...")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT} (increased from 0.1)")
    model = QuadrotorPINN(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)

    # Optimizer with weight decay (L2 regularization)
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
    }

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n[4/6] Starting training for max {MAX_EPOCHS} epochs...")
    print(f"{'='*80}")
    print(
        f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Data':<12} {'Val Physics':<12} {'LR':<10} {'Status'}"
    )
    print(f"{'='*80}")

    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(X_batch)

            # Losses
            data_loss = nn.MSELoss()(y_pred, y_batch)
            physics_loss = model.physics_loss(X_batch, y_pred)
            temporal_loss = nn.MSELoss()(y_pred[:, :3], y_batch[:, :3])  # Position continuity
            stability_loss = torch.mean(torch.sum(y_pred**2, dim=1))  # Prevent explosion

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

            train_loss += loss.item() * X_batch.size(0)

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

        # Print progress every 10 epochs or on improvements
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

    # Plot training history
    print(f"\n[6/6] Saving training history plot...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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
    axes[1, 0].plot(history["learning_rate"])
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale("log")

    # Generalization gap
    gap = [v - t for v, t in zip(history["val_loss"], history["train_loss"])]
    axes[1, 1].plot(gap)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Val Loss - Train Loss")
    axes[1, 1].set_title("Generalization Gap (Should be small)")
    axes[1, 1].axhline(y=0, color="r", linestyle="--", alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "results" / "training_history_fixed.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {plot_path}")

    print(f"\n{'='*80}")
    print("SUCCESS: Model retrained with proper regularization!")
    print(f"{'='*80}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Scalers saved to: {SCALER_SAVE_PATH}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate on test set to verify generalization")
    print(f"  2. Compare test performance with previous overfitted model")
    print(f"  3. Generate new trajectory plots for paper")
    print(f"{'='*80}")

    return model, history


if __name__ == "__main__":
    train_model()
