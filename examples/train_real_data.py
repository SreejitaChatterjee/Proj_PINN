"""
Train on Real Data: Use EuRoC MAV dataset.

This example shows how to:
1. Download real flight data from EuRoC
2. Preprocess it for dynamics learning
3. Train a model when control inputs are unknown
"""

import torch
import numpy as np
from pathlib import Path

from pinn_dynamics import QuadrotorPINN, Trainer
from pinn_dynamics.data import load_euroc
from pinn_dynamics.data.loaders import save_scalers

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main():
    # Configuration
    SEQUENCE = "MH_01_easy"  # Easy flight sequence
    MODEL_PATH = Path(__file__).parent.parent / "models" / "euroc_pinn.pth"
    SCALER_PATH = Path(__file__).parent.parent / "models" / "euroc_scalers.pkl"

    # 1. Load EuRoC data
    print(f"Loading EuRoC sequence: {SEQUENCE}")
    print("(This will download ~1GB if not cached)")
    data = load_euroc(SEQUENCE, dt=0.005)
    print(f"  Loaded {len(data)} samples")

    # 2. Prepare training data
    print("\nPreparing training data...")

    # State columns (12 states)
    state_cols = ["x", "y", "z", "roll", "pitch", "yaw", "p", "q", "r", "vx", "vy", "vz"]

    # Use IMU accelerations as pseudo-controls (since we don't have true thrust/torque)
    control_cols = ["ax", "ay", "az"]

    # Create input-output pairs
    features = state_cols + control_cols
    X = data[features].values[:-1]  # All but last
    y = data[state_cols].values[1:]  # Next states

    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {y.shape}")

    # Split and scale
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=64,
    )

    # 3. Create model
    # Note: Using 15 inputs (12 states + 3 IMU accels) instead of 16
    print("\nCreating model...")
    model = QuadrotorPINN(hidden_size=256, num_layers=5)

    # Override input dimension for EuRoC data
    model.input_dim = 15
    model.control_dim = 3

    # Rebuild network with correct input size
    model.network = torch.nn.Sequential(
        torch.nn.Linear(15, 256),
        torch.nn.SiLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 256),
        torch.nn.SiLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 256),
        torch.nn.SiLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 256),
        torch.nn.SiLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 12),
    )

    print(model.summary())

    # 4. Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    trainer = Trainer(model, device=device, lr=0.001)

    # For real data, we use lower physics weight since we don't have true controls
    loss_weights = {
        "physics": 0.0,  # Disable physics loss (no true controls)
        "temporal": 10.0,
        "stability": 5.0,
        "reg": 0.0,
        "energy": 0.0,
    }

    print("\nTraining (data-driven mode, no physics loss)...")
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=100,
        weights=loss_weights,
        scheduled_sampling_final=0.3,
        verbose=True,
    )

    # 5. Save
    print("\nSaving model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    save_scalers(scaler_X, scaler_y, str(SCALER_PATH))
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  Scalers saved to: {SCALER_PATH}")

    # 6. Evaluate rollout performance
    print("\nEvaluating rollout performance...")

    # Take a segment from validation data
    test_idx = np.random.randint(0, len(X_val) - 100)
    test_X = scaler_X.inverse_transform(X_val[test_idx : test_idx + 100])
    test_y = scaler_y.inverse_transform(y_val[test_idx : test_idx + 100])

    # Rollout
    model.eval()
    predictions = []
    state = torch.FloatTensor(test_X[0, :12]).unsqueeze(0)

    with torch.no_grad():
        for i in range(100):
            control = torch.FloatTensor(test_X[i, 12:]).unsqueeze(0)
            inp = torch.cat([state, control], dim=-1)

            # Scale for model
            inp_scaled = torch.FloatTensor(scaler_X.transform(inp.numpy()))
            pred_scaled = model(inp_scaled)
            pred = torch.FloatTensor(scaler_y.inverse_transform(pred_scaled.numpy()))

            predictions.append(pred.squeeze(0).numpy())
            state = pred[:, :12]

    predictions = np.array(predictions)

    # Compute errors
    errors = np.abs(predictions - test_y)
    position_error = errors[:, :3].mean()
    total_error = errors.mean()

    print(f"  100-step rollout position error: {position_error*100:.2f} cm")
    print(f"  100-step rollout total error: {total_error:.4f}")


if __name__ == "__main__":
    main()
