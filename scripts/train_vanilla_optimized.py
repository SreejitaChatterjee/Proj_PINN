"""Training script for Vanilla Optimized PINN (No Fourier Features)

This version removes Fourier features to fix autoregressive instability
while keeping all other optimizations.
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
from pinn_model_vanilla_optimized import QuadrotorPINNVanillaOptimized
import math

class AdaptiveLossWeights:
    """Adaptive weighting scheme for multi-objective losses"""
    def __init__(self, max_physics=15.0, max_energy=0.05, max_temporal=10.0,
                 max_stability=5.0, warmup_epochs=50):
        self.max_physics = max_physics
        self.max_energy = max_energy  # Low energy weight (soft constraint)
        self.max_temporal = max_temporal
        self.max_stability = max_stability
        self.warmup_epochs = warmup_epochs

    def get_weights(self, epoch):
        """Adaptive weighting: start small, increase as data loss converges"""
        k = 3.0 / self.warmup_epochs
        progress = 1.0 - math.exp(-k * epoch)

        return {
            'physics': self.max_physics * progress,
            'energy': self.max_energy * progress,
            'temporal': self.max_temporal * progress,
            'stability': self.max_stability * progress,
            'reg': 1.0
        }

class VanillaOptimizedTrainer:
    def __init__(self, model, device='cpu', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-6
        )

        self.lbfgs_optimizer = None
        self.criterion = torch.nn.MSELoss()
        self.adaptive_weights = AdaptiveLossWeights()

        self.history = {
            'train': [], 'val': [], 'physics': [], 'energy': [],
            'temporal': [], 'stability': [], 'reg': [], 'rollout': []
        }

    def train_epoch_adam(self, loader, epoch, total_epochs,
                         scheduled_sampling_prob=0.0, use_rollout=True):
        """Train one epoch with Adam optimizer"""
        self.model.train()
        losses = {
            'total': 0, 'physics': 0, 'energy': 0, 'temporal': 0,
            'stability': 0, 'reg': 0, 'rollout': 0
        }

        weights = self.adaptive_weights.get_weights(epoch)

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Scheduled sampling
            if scheduled_sampling_prob > 0 and torch.rand(1).item() < scheduled_sampling_prob:
                with torch.no_grad():
                    pred = self.model(data)
                    data = torch.cat([pred[:, :8].detach(), data[:, 8:]], dim=1)

            output = self.model(data)

            # Individual losses
            data_loss = self.criterion(output, target)
            physics_loss = self.model.physics_loss(data, output)
            energy_loss = self.model.energy_loss(data, output)
            temporal_loss = self.model.temporal_smoothness_loss(data, output)
            stability_loss = self.model.stability_loss(data, output)
            reg_loss = self.model.regularization_loss()

            # Multi-step rollout loss (every 5 epochs)
            rollout_loss = torch.tensor(0.0, device=self.device)
            if use_rollout and epoch % 5 == 0:
                rollout_loss = self.model.multistep_rollout_loss(data, num_steps=5)

            # Combined loss
            loss = (data_loss +
                    weights['physics'] * physics_loss +
                    weights['energy'] * energy_loss +
                    weights['temporal'] * temporal_loss +
                    weights['stability'] * stability_loss +
                    weights['reg'] * reg_loss +
                    0.3 * rollout_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.model.constrain_parameters()

            # Accumulate losses
            losses['total'] += loss.item()
            losses['physics'] += physics_loss.item()
            losses['energy'] += energy_loss.item()
            losses['temporal'] += temporal_loss.item()
            losses['stability'] += stability_loss.item()
            losses['reg'] += reg_loss.item()
            losses['rollout'] += rollout_loss.item()

        return {k: v/len(loader) for k, v in losses.items()}, weights

    def train_epoch_lbfgs(self, loader):
        """Fine-tune with L-BFGS"""
        self.model.train()

        # Collect all data
        all_data, all_target = [], []
        for data, target in loader:
            all_data.append(data)
            all_target.append(target)
        all_data = torch.cat(all_data).to(self.device)
        all_target = torch.cat(all_target).to(self.device)

        # Create L-BFGS optimizer
        if self.lbfgs_optimizer is None:
            self.lbfgs_optimizer = optim.LBFGS(
                self.model.parameters(),
                lr=0.1,
                max_iter=20,
                history_size=10,
                line_search_fn='strong_wolfe'
            )

        def closure():
            self.lbfgs_optimizer.zero_grad()
            output = self.model(all_data)

            data_loss = self.criterion(output, all_target)
            physics_loss = self.model.physics_loss(all_data, output)
            energy_loss = self.model.energy_loss(all_data, output)

            loss = data_loss + 15.0*physics_loss + 0.05*energy_loss
            loss.backward()
            return loss

        loss = self.lbfgs_optimizer.step(closure)
        self.model.constrain_parameters()
        return loss.item()

    def validate(self, loader):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item()
        return total_loss / len(loader)

    def train(self, train_loader, val_loader, epochs=100, lbfgs_epochs=10):
        """Hybrid training: Adam + L-BFGS"""
        print(f"Training VANILLA OPTIMIZED PINN for {epochs} Adam + {lbfgs_epochs} L-BFGS epochs")
        print(f"Architecture: Residual MLP + Modular (NO Fourier features)")
        print(f"Device: {self.device}")
        print(f"Adaptive loss weighting: physics 0->{self.adaptive_weights.max_physics}")
        print()

        best_val_loss = float('inf')
        best_epoch = 0

        # Phase 1: Adam training
        for epoch in range(epochs):
            scheduled_sampling_prob = 0.3 * (epoch / epochs)

            losses, weights = self.train_epoch_adam(
                train_loader, epoch, epochs, scheduled_sampling_prob
            )
            val_loss = self.validate(val_loader)

            self.scheduler.step()

            for k in self.history:
                if k == 'val':
                    self.history[k].append(val_loss)
                else:
                    self.history[k].append(losses.get(k.replace('train', 'total'), 0))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: Train={losses['total']:.4f}, Val={val_loss:.6f}")
                print(f"  Physics={losses['physics']:.4f} (w={weights['physics']:.1f}), "
                      f"Energy={losses['energy']:.4f} (w={weights['energy']:.2f}), "
                      f"Temporal={losses['temporal']:.4f}, Stability={losses['stability']:.4f}")
                print(f"  Rollout={losses['rollout']:.4f}, SS_prob={scheduled_sampling_prob:.2f}, "
                      f"LR={self.optimizer.param_groups[0]['lr']:.2e}")

                if epoch % 30 == 0:
                    print(f"  Params: " + ", ".join(
                        f"{k}={v.item():.2e}" for k, v in self.model.params.items()
                    ))

        print(f"\nBest validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

        # Phase 2: L-BFGS fine-tuning
        print(f"\nPhase 2: L-BFGS fine-tuning for {lbfgs_epochs} epochs...")
        for epoch in range(lbfgs_epochs):
            train_loss = self.train_epoch_lbfgs(train_loader)
            val_loss = self.validate(val_loader)

            if epoch % 5 == 0:
                print(f"L-BFGS Epoch {epoch:02d}: Train={train_loss:.6f}, Val={val_loss:.6f}")

        print(f"\nTraining complete!")
        return self.history

def prepare_data(csv_path, test_size=0.2, batch_size=128):
    """Prepare data for training"""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'roll': 'phi', 'pitch': 'theta', 'yaw': 'psi'})
    features = ['z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vz',
                'thrust', 'torque_x', 'torque_y', 'torque_z']

    X, y = [], []
    for traj_id in df['trajectory_id'].unique():
        traj = df[df['trajectory_id'] == traj_id].sort_values('timestamp')
        traj_data = traj[features].values
        X.append(traj_data[:-1])
        y.append(traj_data[1:, :8])

    X, y = np.vstack(X), np.vstack(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train, y_train = scaler_X.fit_transform(X_train), scaler_y.fit_transform(y_train)
    X_val, y_val = scaler_X.transform(X_val), scaler_y.transform(y_val)

    return (
        DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                   batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                   batch_size=batch_size),
        scaler_X, scaler_y
    )

if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data' / 'quadrotor_training_data.csv'
    print("Loading data with batch_size=128...")
    train_loader, val_loader, scaler_X, scaler_y = prepare_data(data_path, batch_size=128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create vanilla optimized model (NO Fourier features)
    model = QuadrotorPINNVanillaOptimized(hidden_size=128, dropout=0.1)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}\n")

    # Train with hybrid approach
    trainer = VanillaOptimizedTrainer(model, device, lr=0.001)
    trainer.train(train_loader, val_loader, epochs=100, lbfgs_epochs=10)

    # Save model
    save_path = Path(__file__).parent.parent / 'models' / 'quadrotor_pinn_vanilla_optimized.pth'
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)

    # Save scalers
    scaler_path = save_path.parent / 'scalers_vanilla_optimized.pkl'
    joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, scaler_path)

    print(f"\nModel saved to {save_path}")
    print(f"Scalers saved to {scaler_path}")
    print("\nFinal parameters:")
    for k, v in model.params.items():
        true_val = model.true_params[k]
        error = abs(v.item() - true_val) / true_val * 100
        print(f"{k}: {v.item():.6e} (true: {true_val:.6e}, error: {error:.1f}%)")
