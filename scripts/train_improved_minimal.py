"""
Train minimal improved PINN - baseline hyperparameters + residual connections

Uses EXACT baseline settings:
- 250 epochs
- Adam optimizer, lr=0.001
- physics_weight=10.0, temporal=20.0, stability=5.0, reg=1.0
- scheduled sampling 0% → 30%
- ReduceLROnPlateau scheduler

Only change: Add residual connections to architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pinn_model import QuadrotorPINN  # Import baseline for loss functions
from pinn_model_improved_minimal import QuadrotorPINNMinimal
import sys
import os


def load_data(batch_size=128):
    """Load data (same as baseline)"""
    print(f"Loading data...")

    data_path = '../data/quadrotor_training_data.csv' if os.path.exists('../data') else 'data/quadrotor_training_data.csv'
    data = pd.read_csv(data_path)

    state_cols = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']

    # Create next state
    data_shifted = data[state_cols].shift(-1)
    data_shifted.columns = [c + '_next' for c in state_cols]

    data_combined = pd.concat([data[state_cols + control_cols], data_shifted], axis=1)
    data_combined = data_combined.dropna()

    X = data_combined[state_cols + control_cols].values
    y = data_combined[[c + '_next' for c in state_cols]].values

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # Convert to tensors
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler_X, scaler_y


def train_minimal_improved(epochs=250, batch_size=128, lr=0.001):
    """Train with baseline hyperparameters"""
    device = torch.device('cpu')
    print(f"Device: {device}\n")

    # Load data
    train_loader, val_loader, scaler_X, scaler_y = load_data(batch_size)

    # Create model
    model = QuadrotorPINNMinimal(hidden_size=250).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # Optimizer and scheduler (same as baseline)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    # Loss weights (same as baseline)
    weights = {'physics': 10.0, 'temporal': 20.0, 'stability': 5.0, 'reg': 1.0}

    print(f"Training for {epochs} epochs")
    print(f"Architecture: Baseline + Residual Connections")
    print(f"Loss weights: physics={weights['physics']}, temporal={weights['temporal']}, stability={weights['stability']}")
    print(f"Scheduled sampling: 0% -> 30%\n")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        # Scheduled sampling (same as baseline: 0% -> 30%)
        ss_prob = 0.3 * (epoch / epochs)

        # Training
        model.train()
        train_loss = 0.0
        train_data_loss = 0.0
        train_physics_loss = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Scheduled sampling
            if ss_prob > 0 and torch.rand(1).item() < ss_prob:
                with torch.no_grad():
                    pred = model(x_batch)
                    x_batch = torch.cat([pred[:, :8].detach(), x_batch[:, 8:]], dim=1)

            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x_batch)

            # Losses
            data_loss = nn.MSELoss()(y_pred, y_batch)
            physics_loss = model.physics_loss(x_batch, y_pred)

            # Combined loss
            total_loss = data_loss + weights['physics'] * physics_loss

            # Backprop
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Parameter constraints
            with torch.no_grad():
                model.params['Jxx'].clamp_(5e-5, 1e-4)
                model.params['Jyy'].clamp_(5e-5, 1.5e-4)
                model.params['Jzz'].clamp_(5e-5, 2e-4)
                model.params['m'].clamp_(0.05, 0.1)

            optimizer.step()

            train_loss += total_loss.item()
            train_data_loss += data_loss.item()
            train_physics_loss += physics_loss.item()

        train_loss /= len(train_loader)
        train_data_loss /= len(train_loader)
        train_physics_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_val_pred = model(x_val)
                val_loss += nn.MSELoss()(y_val_pred, y_val).item()
        val_loss /= len(val_loader)

        # Scheduler step
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model_dir = '../models' if os.path.exists('../models') else 'models'
            torch.save(model.state_dict(), f'{model_dir}/quadrotor_pinn_improved_minimal.pth')

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            params = {k: v.item() for k, v in model.params.items()}
            print(f"Epoch {epoch:03d}: Train={train_loss:.6f}, Val={val_loss:.6f}")
            print(f"  Data={train_data_loss:.6f}, Physics={train_physics_loss:.6f}")
            print(f"  SS_prob={ss_prob:.2f}, LR={optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Params: m={params['m']:.2e}, Jxx={params['Jxx']:.2e}")

    print(f"\nBest validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # Save final model and scalers
    model_dir = '../models' if os.path.exists('../models') else 'models'
    with open(f'{model_dir}/scalers_improved_minimal.pkl', 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

    print(f"\nModel saved to {model_dir}/quadrotor_pinn_improved_minimal.pth")
    print(f"Scalers saved to {model_dir}/scalers_improved_minimal.pkl")

    # Final parameters
    params = {k: v.item() for k, v in model.params.items()}
    true_params = {'Jxx': 6.86e-5, 'Jyy': 9.20e-5, 'Jzz': 1.366e-4, 'm': 0.068}

    print("\nFinal parameters:")
    for name in ['Jxx', 'Jyy', 'Jzz', 'm']:
        learned = params[name]
        true_val = true_params[name]
        error = abs(learned - true_val) / true_val * 100
        print(f"{name}: {learned:.6e} (true: {true_val:.6e}, error: {error:.1f}%)")


if __name__ == "__main__":
    train_minimal_improved(epochs=250, batch_size=128, lr=0.001)
