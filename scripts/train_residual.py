"""
Train baseline + residual connections

EXACT baseline training procedure, only swap model
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from pinn_model_residual import QuadrotorPINN  # Use residual version

class Trainer:
    def __init__(self, model, device='cpu', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                               factor=0.5, patience=20)
        self.criterion = torch.nn.MSELoss()
        self.history = {'train': [], 'val': [], 'physics': [], 'temporal': [], 'stability': [], 'reg': []}

    def train_epoch(self, loader, weights={'physics': 10.0, 'temporal': 20.0, 'stability': 5.0, 'reg': 1.0},
                    scheduled_sampling_prob=0.0):
        self.model.train()
        losses = {'total': 0, 'physics': 0, 'temporal': 0, 'stability': 0, 'reg': 0}

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Scheduled sampling
            if scheduled_sampling_prob > 0 and torch.rand(1).item() < scheduled_sampling_prob:
                with torch.no_grad():
                    pred = self.model(data)
                    data = torch.cat([pred[:, :8].detach(), data[:, 8:]], dim=1)

            output = self.model(data)
            data_loss = self.criterion(output, target)
            physics_loss = self.model.physics_loss(data, output)
            temporal_loss = self.model.temporal_smoothness_loss(data, output)
            stability_loss = self.model.stability_loss(data, output)

            # Parameter regularization
            param_loss = sum(((self.model.params[k] - self.model.true_params[k])/self.model.true_params[k])**2
                           for k in self.model.params.keys())

            loss = (data_loss +
                    weights['physics'] * physics_loss +
                    weights['temporal'] * temporal_loss +
                    weights['stability'] * stability_loss +
                    weights['reg'] * param_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.model.constrain_parameters()

            losses['total'] += loss.item()
            losses['physics'] += physics_loss.item()
            losses['temporal'] += temporal_loss.item()
            losses['stability'] += stability_loss.item()
            losses['reg'] += param_loss.item()

        return {k: v/len(loader) for k, v in losses.items()}

    def validate(self, loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
        return val_loss / len(loader)

    def train(self, train_loader, val_loader, epochs=250, weights=None):
        if weights is None:
            weights = {'physics': 10.0, 'temporal': 20.0, 'stability': 5.0, 'reg': 1.0}

        print(f"Training for {epochs} epochs with Residual architecture:")
        print(f"  Loss weights: {weights}")

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Scheduled sampling: 0% -> 30%
            scheduled_sampling_prob = 0.3 * (epoch / epochs)

            train_losses = self.train_epoch(train_loader, weights, scheduled_sampling_prob)
            val_loss = self.validate(val_loader)

            self.scheduler.step(val_loss)

            for k, v in train_losses.items():
                if k in self.history:
                    self.history[k].append(v)
            self.history['val'].append(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                Path('../models').mkdir(exist_ok=True)
                torch.save(self.model.state_dict(), '../models/quadrotor_pinn_residual.pth')

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}: Train={train_losses['total']:.4f}, Val={val_loss:.4f}, "
                      f"Phys={train_losses['physics']:.4f}, LR={self.optimizer.param_groups[0]['lr']:.2e}")

        print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
        return self.history


def main():
    # Load data
    data = pd.read_csv('../data/quadrotor_training_data.csv')

    state_cols = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']

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

    # DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    # Train
    model = QuadrotorPINN(hidden_size=256)
    trainer = Trainer(model)
    trainer.train(train_loader, val_loader, epochs=250)

    # Save scalers
    joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, '../models/scalers_residual.pkl')
    print("Scalers saved to ../models/scalers_residual.pkl")


if __name__ == "__main__":
    main()
