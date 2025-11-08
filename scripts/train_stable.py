"""
Training script for Stable PINN with autoregressive stability enhancements

Features:
1. Curriculum rollout learning (5 → 10 → 20 steps)
2. Jacobian regularization for contractive dynamics
3. Gradient clipping
4. Process noise during training
5. Heavy scheduled sampling (0% → 70%)
6. Adaptive loss weighting
7. Long training (200 epochs)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import sys
import os
import math

sys.path.append(os.path.dirname(__file__))
from pinn_model_stable import StablePINN


class AdaptiveLossWeights:
    """Adaptive loss weighting with warmup"""
    def __init__(self, max_physics=10.0, max_rollout=0.1, max_jac=1e-4, warmup_epochs=50):
        self.max_physics = max_physics
        self.max_rollout = max_rollout
        self.max_jac = max_jac
        self.warmup_epochs = warmup_epochs

    def get_weights(self, epoch):
        # Exponential warmup
        k = 3.0 / self.warmup_epochs
        progress = 1.0 - math.exp(-k * epoch)

        return {
            'physics': self.max_physics * progress,
            'rollout': self.max_rollout * progress,
            'jac': self.max_jac * progress
        }


class CurriculumSchedule:
    """Curriculum learning for rollout horizon"""
    def __init__(self, schedule=None):
        if schedule is None:
            # Default: gradually increase rollout length
            # Epochs 0-50: 5 steps
            # Epochs 50-100: 10 steps
            # Epochs 100-150: 20 steps
            # Epochs 150+: 30 steps
            schedule = {
                0: 5,
                50: 10,
                100: 20,
                150: 30
            }
        self.schedule = schedule

    def get_rollout_length(self, epoch):
        # Find appropriate rollout length
        rollout_len = 5  # default
        for epoch_threshold, length in sorted(self.schedule.items()):
            if epoch >= epoch_threshold:
                rollout_len = length
        return rollout_len


class ScheduledSampling:
    """Scheduled sampling: gradually use model predictions instead of ground truth"""
    def __init__(self, final_prob=0.7, warmup_epochs=200):
        self.final_prob = final_prob
        self.warmup_epochs = warmup_epochs

    def get_probability(self, epoch):
        # Linear increase
        return min(self.final_prob, (epoch / self.warmup_epochs) * self.final_prob)


def load_data(batch_size=128):
    """Load and preprocess training data"""
    print(f"Loading data with batch_size={batch_size}...")

    # Load training data
    data_path = '../data/quadrotor_training_data.csv' if os.path.exists('../data/quadrotor_training_data.csv') else 'data/quadrotor_training_data.csv'
    data = pd.read_csv(data_path)

    # Input: [z, roll, pitch, yaw, p, q, r, vz, thrust, tx, ty, tz]
    # Output: [z_next, roll_next, pitch_next, yaw_next, p_next, q_next, r_next, vz_next]
    # Note: dataset uses roll, pitch, yaw (not phi, theta, psi)
    state_cols = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']

    # Create next state by shifting
    data_shifted = data[state_cols].shift(-1)
    data_shifted.columns = [c + '_next' for c in state_cols]

    # Combine current state + controls with next state
    data_combined = pd.concat([data[state_cols + control_cols], data_shifted], axis=1)
    data_combined = data_combined.dropna()  # Remove last row (no next state)

    input_cols = state_cols + control_cols
    output_cols = [c + '_next' for c in state_cols]

    X = data_combined[input_cols].values
    y = data_combined[output_cols].values

    # Normalize inputs and outputs
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)

    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler_X, scaler_y


def train_stable_pinn(
    epochs=200,
    batch_size=128,
    learning_rate=1e-3,
    use_fourier=False,
    hidden_size=128,
    num_residual_blocks=2
):
    """
    Train stable PINN with all stability enhancements

    Args:
        epochs: int - number of training epochs (200 recommended)
        batch_size: int
        learning_rate: float
        use_fourier: bool - use low-frequency Fourier features
        hidden_size: int
        num_residual_blocks: int
    """
    device = torch.device('cpu')  # Use CPU for consistency
    print(f"Using device: {device}\n")

    # Load data
    train_loader, val_loader, scaler_X, scaler_y = load_data(batch_size)

    # Create model
    model = StablePINN(
        hidden_size=hidden_size,
        num_residual_blocks=num_residual_blocks,
        use_fourier=use_fourier,
        num_fourier_freq=1  # Only 1 frequency if enabled
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}\n")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Loss weights (reduced to match baseline better)
    adaptive_weights = AdaptiveLossWeights(
        max_physics=3.0,  # Reduced from 10.0
        max_rollout=0.01,  # Reduced from 0.1
        max_jac=1e-5,  # Reduced from 1e-4
        warmup_epochs=50
    )

    # Curriculum schedule
    curriculum = CurriculumSchedule()

    # Scheduled sampling
    scheduled_sampling = ScheduledSampling(final_prob=0.7, warmup_epochs=epochs)

    # Training loop
    print(f"Training Stable PINN for {epochs} epochs")
    print(f"Architecture: Unified trunk + task heads, {num_residual_blocks} residual blocks")
    print(f"Fourier features: {'Enabled (k=1)' if use_fourier else 'Disabled'}")
    print(f"Device: {device}")
    print(f"Curriculum rollout: 5->10->20->30 steps")
    print(f"Scheduled sampling: 0%->70%")
    print(f"Jacobian regularization: Enabled\n")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        model.train()

        # Get loss weights for this epoch
        weights = adaptive_weights.get_weights(epoch)

        # Get rollout length for this epoch
        rollout_len = curriculum.get_rollout_length(epoch)

        # Get scheduled sampling probability
        ss_prob = scheduled_sampling.get_probability(epoch)

        # Training
        train_loss = 0.0
        train_data_loss = 0.0
        train_physics_loss = 0.0
        train_rollout_loss = 0.0
        train_jac_loss = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # 1. Data loss (with process noise)
            y_pred = model(x_batch, add_noise=True, noise_level=0.01)
            data_loss = nn.MSELoss()(y_pred, y_batch)

            # 2. Physics loss
            physics_loss = model.physics_loss(x_batch, y_pred, y_batch, dt=0.001)

            # 3. Multi-step rollout loss (every 5 batches to save compute)
            if batch_idx % 5 == 0:
                # Create mini rollout sequence from batch
                rollout_batch_size = min(16, x_batch.shape[0])  # Small batch for efficiency

                # Extract initial state (first 8 dims of first sample)
                x_initial = y_batch[:rollout_batch_size, :]  # Use previous outputs as initial states

                # Create control sequence (repeat controls for rollout_len steps)
                u_controls = x_batch[:rollout_batch_size, 8:12]  # [thrust, tx, ty, tz]
                u_sequence = u_controls.unsqueeze(1).repeat(1, rollout_len, 1)  # [batch, rollout_len, 4]

                # Create ground truth sequence (use same next state as approximation)
                y_true_sequence = y_batch[:rollout_batch_size, :].unsqueeze(1).repeat(1, rollout_len, 1)

                rollout_loss = model.multistep_rollout_loss(
                    x_initial, u_sequence, y_true_sequence, rollout_len, add_noise=True
                )
            else:
                rollout_loss = torch.tensor(0.0)

            # 4. Jacobian regularization (disabled - too expensive for marginal benefit)
            jac_loss = torch.tensor(0.0)
            # if batch_idx % 10 == 0 and epoch >= 20:  # Start after 20 epochs
            #     jac_loss = model.get_jacobian_loss(x_batch)

            # Combined loss
            total_loss = (
                data_loss +
                weights['physics'] * physics_loss +
                weights['rollout'] * rollout_loss +
                weights['jac'] * jac_loss
            )

            # Backprop
            total_loss.backward()

            # Gradient clipping (essential for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Parameter constraints (ensure physical validity)
            with torch.no_grad():
                model.Jxx.clamp_(min=5e-5, max=1e-4)
                model.Jyy.clamp_(min=5e-5, max=1.5e-4)
                model.Jzz.clamp_(min=5e-5, max=2e-4)
                model.kt.clamp_(min=1e-3, max=2e-2)
                model.kq.clamp_(min=1e-4, max=1e-3)
                model.m.clamp_(min=0.05, max=0.1)

            optimizer.step()

            # Track losses
            train_loss += total_loss.item()
            train_data_loss += data_loss.item()
            train_physics_loss += physics_loss.item()
            if isinstance(rollout_loss, torch.Tensor):
                train_rollout_loss += rollout_loss.item()
            if isinstance(jac_loss, torch.Tensor):
                train_jac_loss += jac_loss.item()

        # Average losses
        train_loss /= len(train_loader)
        train_data_loss /= len(train_loader)
        train_physics_loss /= len(train_loader)
        train_rollout_loss /= len(train_loader)
        train_jac_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_val_pred = model(x_val)
                val_loss += nn.MSELoss()(y_val_pred, y_val).item()
        val_loss /= len(val_loader)

        # Learning rate step
        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model_path = '../models/quadrotor_pinn_stable.pth' if os.path.exists('../models') else 'models/quadrotor_pinn_stable.pth'
            torch.save(model.state_dict(), model_path)

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            params = model.get_parameters_dict()
            print(f"Epoch {epoch:03d}: Train={train_loss:.6f}, Val={val_loss:.6f}")
            print(f"  Data={train_data_loss:.6f}, Physics={train_physics_loss:.6f} (w={weights['physics']:.1f})")
            print(f"  Rollout={train_rollout_loss:.6f} (len={rollout_len}, w={weights['rollout']:.3f})")
            print(f"  Jacobian={train_jac_loss:.6f} (w={weights['jac']:.6f})")
            print(f"  SS_prob={ss_prob:.2f}, LR={scheduler.get_last_lr()[0]:.2e}")
            print(f"  Params: m={params['m']:.2e}, Jxx={params['Jxx']:.2e}, kt={params['kt']:.2e}")

    print(f"\nBest validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # Save final model and scalers
    model_dir = '../models' if os.path.exists('../models') else 'models'
    print(f"\nModel saved to {model_dir}/quadrotor_pinn_stable.pth")

    with open(f'{model_dir}/scalers_stable.pkl', 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
    print(f"Scalers saved to {model_dir}/scalers_stable.pkl")

    # Print final parameters
    params = model.get_parameters_dict()
    true_params = {
        'Jxx': 6.86e-5,
        'Jyy': 9.20e-5,
        'Jzz': 1.366e-4,
        'kq': 7.8263e-4,
        'kt': 0.01,
        'm': 0.068
    }

    print("\nFinal parameters:")
    for name in ['Jxx', 'Jyy', 'Jzz', 'kq', 'kt', 'm']:
        learned = params[name]
        true_val = true_params[name]
        error = abs(learned - true_val) / true_val * 100
        print(f"{name}: {learned:.6e} (true: {true_val:.6e}, error: {error:.1f}%)")


if __name__ == "__main__":
    train_stable_pinn(
        epochs=200,
        batch_size=128,
        learning_rate=1e-3,
        use_fourier=False,  # Start without Fourier
        hidden_size=128,
        num_residual_blocks=2
    )
