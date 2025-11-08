"""Quick 20-epoch test of Optimized PINN v2"""
import sys
sys.path.append('.')

from train_optimized_v2 import *

if __name__ == "__main__":
    # Quick test - 20 epochs
    print("Loading data...")
    data = pd.read_csv('../data/quadrotor_training_data.csv')

    state_cols = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']

    data_shifted = data[state_cols].shift(-1)
    data_shifted.columns = [c + '_next' for c in state_cols]

    data_combined = pd.concat([data[state_cols + control_cols], data_shifted], axis=1)
    data_combined = data_combined.dropna()

    X = data_combined[state_cols + control_cols].values
    y = data_combined[[c + '_next' for c in state_cols]].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    model = OptimizedPINNv2(hidden_size=256)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    trainer = OptimizedTrainer(model, device='cpu', lr=0.001)

    # Override train method for quick test (20 epochs, no L-BFGS)
    base_weights = {
        'physics': 10.0,
        'temporal': 20.0,
        'stability': 5.0,
        'rollout': 1.0,
        'reg': 1.0
    }

    best_val = float('inf')

    for epoch in range(20):
        train_losses, rollout_horizon = trainer.train_epoch(train_loader, epoch, base_weights)
        val_loss = trainer.validate(val_loader)
        trainer.scheduler.step()

        if val_loss < best_val:
            best_val = val_loss

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d}: Train={train_losses['total']:.4f}, Val={val_loss:.6f}, Best={best_val:.6f}")
            print(f"  Data={train_losses['data']:.6f}, Rollout={train_losses['rollout']:.4f} (K={rollout_horizon})")

    print(f"\nTest complete! Best val: {best_val:.6f}")

    # Save test model
    torch.save(model.state_dict(), '../models/quadrotor_pinn_optimized_v2.pth')
    joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, '../models/scalers_optimized_v2.pkl')
    print("Test model and scalers saved.")
