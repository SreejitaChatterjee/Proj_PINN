"""
Train Model: Train a PINN on your own data.

This example shows how to:
1. Load data from CSV
2. Configure the trainer
3. Train with physics constraints
4. Save the trained model
"""

import torch
from pathlib import Path

from pinn_dynamics import QuadrotorPINN, Trainer
from pinn_dynamics.data import prepare_data, save_scalers


def main():
    # Paths
    DATA_PATH = Path(__file__).parent.parent / "data" / "quadrotor_training_data.csv"
    MODEL_PATH = Path(__file__).parent.parent / "models" / "my_trained_model.pth"
    SCALER_PATH = Path(__file__).parent.parent / "models" / "my_scalers.pkl"

    # Check if data exists
    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}")
        print("Please generate training data first, or use your own CSV.")
        return

    # 1. Prepare data
    print("Loading data...")
    train_loader, val_loader, scaler_X, scaler_y = prepare_data(
        str(DATA_PATH),
        batch_size=64,
        test_size=0.2,
        val_size=0.2,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # 2. Create model
    print("\nCreating model...")
    model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)
    print(model.summary())

    # 3. Configure trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    trainer = Trainer(model, device=device, lr=0.001)

    # 4. Train!
    print("\nTraining...")
    loss_weights = {
        "physics": 10.0,     # Physics constraint weight
        "temporal": 12.0,    # Temporal smoothness
        "stability": 5.0,    # State bounds
        "reg": 1.0,          # Parameter regularization
        "energy": 5.0,       # Energy conservation
    }

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=50,  # Increase for better results
        weights=loss_weights,
        scheduled_sampling_final=0.3,  # Helps autoregressive stability
        verbose=True,
    )

    # 5. Save model and scalers
    print("\nSaving model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    save_scalers(scaler_X, scaler_y, str(SCALER_PATH))
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  Scalers saved to: {SCALER_PATH}")

    # 6. Print final learned parameters
    print("\nLearned physics parameters:")
    for name, param in model.params.items():
        true_val = model.true_params.get(name, param.item())
        error = abs(param.item() - true_val) / true_val * 100
        print(f"  {name}: {param.item():.6e} (error: {error:.1f}%)")


if __name__ == "__main__":
    main()
