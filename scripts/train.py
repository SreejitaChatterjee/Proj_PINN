"""Unified training script for quadrotor PINN"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from pinn_model import QuadrotorPINN

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
        """
        Train for one epoch with scheduled sampling.

        scheduled_sampling_prob: probability of using model's prediction instead of ground truth
                                increases over training to improve autoregressive performance
        """
        self.model.train()
        losses = {'total': 0, 'physics': 0, 'temporal': 0, 'stability': 0, 'reg': 0}

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Scheduled sampling: sometimes use model's own prediction
            if scheduled_sampling_prob > 0 and torch.rand(1).item() < scheduled_sampling_prob:
                with torch.no_grad():
                    # Use model's prediction as input for next step (autoregressive)
                    pred = self.model(data)
                    # Replace states (first 8 dims) with predictions, keep controls
                    data = torch.cat([pred[:, :8].detach(), data[:, 8:]], dim=1)

            output = self.model(data)
            data_loss = self.criterion(output, target)
            physics_loss = self.model.physics_loss(data, output)
            temporal_loss = self.model.temporal_smoothness_loss(data, output)
            stability_loss = self.model.stability_loss(data, output)
            reg_loss = self.model.regularization_loss()

            # Rebalanced: reduced physics (20->10), increased temporal (15->20), added stability
            loss = (data_loss +
                    weights['physics']*physics_loss +
                    weights['temporal']*temporal_loss +
                    weights['stability']*stability_loss +
                    weights['reg']*reg_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.model.constrain_parameters()

            losses['total'] += loss.item()
            losses['physics'] += physics_loss.item()
            losses['temporal'] += temporal_loss.item()
            losses['stability'] += stability_loss.item()
            losses['reg'] += reg_loss.item()

        return {k: v/len(loader) for k, v in losses.items()}

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                total_loss += self.criterion(self.model(data), target).item()
        return total_loss / len(loader)

    def train(self, train_loader, val_loader, epochs=250, weights=None):
        weights = weights or {'physics': 10.0, 'temporal': 20.0, 'stability': 5.0, 'reg': 1.0}
        print(f"Training for {epochs} epochs with IMPROVED architecture:")
        print(f"  - Model: 256 neurons, 5 layers, dropout=0.1")
        print(f"  - Loss weights: {weights}")
        print(f"  - Learning rate: {self.optimizer.param_groups[0]['lr']} (with scheduling)")
        print(f"  - Scheduled sampling: 0% -> 30% (gradual increase)")

        for epoch in range(epochs):
            # Gradually increase scheduled sampling probability: 0% -> 30% over training
            # This forces model to learn from its own predictions (autoregressive)
            scheduled_sampling_prob = 0.3 * (epoch / epochs)

            losses = self.train_epoch(train_loader, weights, scheduled_sampling_prob)
            val_loss = self.validate(val_loader)

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)

            for k in self.history:
                self.history[k].append(val_loss if k == 'val' else losses.get(k.replace('train', 'total'), 0))

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: Train={losses['total']:.4f}, Val={val_loss:.6f}, "
                      f"Physics={losses['physics']:.4f}, Temporal={losses['temporal']:.4f}, "
                      f"Stability={losses['stability']:.4f}, Reg={losses['reg']:.4f}, "
                      f"SS_prob={scheduled_sampling_prob:.2f}")
                if epoch % 30 == 0:
                    print(f"  Params: " + ", ".join(f"{k}={v.item():.2e}" for k, v in self.model.params.items()))

def prepare_data(csv_path, test_size=0.2):
    df = pd.read_csv(csv_path)
    # Rename columns to match expected names (data generator uses roll/pitch/yaw)
    df = df.rename(columns={'roll': 'phi', 'pitch': 'theta', 'yaw': 'psi'})
    # FIXED: Removed p_dot, q_dot, r_dot - these are NOT valid inputs (cannot be measured in real deployment)
    features = ['z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vz', 'thrust', 'torque_x', 'torque_y', 'torque_z']

    X, y = [], []
    for traj_id in df['trajectory_id'].unique():
        traj = df[df['trajectory_id'] == traj_id].sort_values('timestamp')
        # Vectorized: convert to numpy once, then slice
        traj_data = traj[features].values
        X.append(traj_data[:-1])  # All but last
        y.append(traj_data[1:, :8])  # All but first, only first 8 features

    X, y = np.vstack(X), np.vstack(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train, y_train = scaler_X.fit_transform(X_train), scaler_y.fit_transform(y_train)
    X_val, y_val = scaler_X.transform(X_val), scaler_y.transform(y_val)

    return (DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=64, shuffle=True),
            DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=64),
            scaler_X, scaler_y)

if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data' / 'quadrotor_training_data.csv'
    train_loader, val_loader, scaler_X, scaler_y = prepare_data(data_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = QuadrotorPINN()  # Now: 256 neurons, 5 layers, dropout
    trainer = Trainer(model, device)
    trainer.train(train_loader, val_loader, epochs=250)

    save_path = Path(__file__).parent.parent / 'models' / 'quadrotor_pinn.pth'
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)

    # Save scalers for evaluation
    scaler_path = save_path.parent / 'scalers.pkl'
    joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, scaler_path)

    print(f"\nModel saved to {save_path}")
    print(f"Scalers saved to {scaler_path}")
    print("\nFinal parameters:")
    for k, v in model.params.items():
        print(f"{k}: {v.item():.6e} (true: {model.true_params[k]:.6e}, error: {abs(v.item()-model.true_params[k])/model.true_params[k]*100:.1f}%)")
