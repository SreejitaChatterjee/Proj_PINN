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
                    # Replace states (first 12 dims) with predictions, keep controls
                    data = torch.cat([pred[:, :12].detach(), data[:, 12:]], dim=1)

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
        weights = weights or {'physics': 10.0, 'temporal': 12.0, 'stability': 5.0, 'reg': 1.0}
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

def create_sequences(traj_data, seq_len=10):
    """
    Create sliding window sequences from trajectory data.

    Args:
        traj_data: (T, 16) array with [states(12) + controls(4)]
        seq_len: sequence length for LSTM (default 10)

    Returns:
        X_seq: (T-1, seq_len, 12) - state sequences
        X_ctrl: (T-1, 4) - current controls
        y: (T-1, 12) - next states
    """
    T = len(traj_data)
    X_seq, X_ctrl, y = [], [], []

    for i in range(T - 1):
        # State sequence: states from (i-seq_len+1) to i (inclusive)
        start_idx = max(0, i - seq_len + 1)
        state_window = traj_data[start_idx:i+1, :12]  # Get states only (12 states)

        # Pad with first state if at trajectory start
        if len(state_window) < seq_len:
            padding = np.tile(traj_data[0:1, :12], (seq_len - len(state_window), 1))
            state_window = np.vstack([padding, state_window])

        X_seq.append(state_window)
        X_ctrl.append(traj_data[i, 12:])  # Current controls (last 4 dims)
        y.append(traj_data[i+1, :12])  # Next state (first 12 dims)

    return np.array(X_seq), np.array(X_ctrl), np.array(y)

def prepare_data(csv_path, test_size=0.2, use_sequences=False, seq_len=10):
    """
    Prepare data for training.

    Args:
        csv_path: path to CSV file
        test_size: fraction of data for testing
        use_sequences: if True, create sequence data for LSTM
        seq_len: sequence length for LSTM

    Returns:
        If use_sequences=False (original):
            train_loader, val_loader, scaler_X, scaler_y
        If use_sequences=True (hybrid PINN+LSTM):
            train_loader, val_loader, scaler_seq, scaler_ctrl, scaler_y
    """
    df = pd.read_csv(csv_path)
    # Rename columns to match expected names (data generator uses roll/pitch/yaw)
    df = df.rename(columns={'roll': 'phi', 'pitch': 'theta', 'yaw': 'psi'})
    # Full 12-state model: positions (x,y,z), attitudes (phi,theta,psi), rates (p,q,r), velocities (vx,vy,vz)
    features = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz', 'thrust', 'torque_x', 'torque_y', 'torque_z']

    if not use_sequences:
        # Original single-timestep preparation
        X, y = [], []
        for traj_id in df['trajectory_id'].unique():
            traj = df[df['trajectory_id'] == traj_id].sort_values('timestamp')
            traj_data = traj[features].values
            X.append(traj_data[:-1])  # All but last
            y.append(traj_data[1:, :12])  # All but first, only first 12 features (states)

        X, y = np.vstack(X), np.vstack(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_train, y_train = scaler_X.fit_transform(X_train), scaler_y.fit_transform(y_train)
        X_val, y_val = scaler_X.transform(X_val), scaler_y.transform(y_val)

        return (DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=64, shuffle=True),
                DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=64),
                scaler_X, scaler_y)

    else:
        # Sequence-based preparation for hybrid PINN+LSTM
        X_seq_all, X_ctrl_all, y_all = [], [], []

        for traj_id in df['trajectory_id'].unique():
            traj = df[df['trajectory_id'] == traj_id].sort_values('timestamp')
            traj_data = traj[features].values

            X_seq, X_ctrl, y = create_sequences(traj_data, seq_len)
            X_seq_all.append(X_seq)
            X_ctrl_all.append(X_ctrl)
            y_all.append(y)

        X_seq_all = np.vstack(X_seq_all)  # (N, seq_len, 12)
        X_ctrl_all = np.vstack(X_ctrl_all)  # (N, 4)
        y_all = np.vstack(y_all)  # (N, 12)

        # Split into train/val
        indices = np.arange(len(X_seq_all))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

        X_seq_train, X_ctrl_train, y_train = X_seq_all[train_idx], X_ctrl_all[train_idx], y_all[train_idx]
        X_seq_val, X_ctrl_val, y_val = X_seq_all[val_idx], X_ctrl_all[val_idx], y_all[val_idx]

        # Normalize: fit scalers on flattened training data
        scaler_seq = StandardScaler()
        scaler_ctrl = StandardScaler()
        scaler_y = StandardScaler()

        # Reshape for scaling: (N*seq_len, 8) for sequences
        N_train, seq_len_dim, state_dim = X_seq_train.shape
        X_seq_train_flat = X_seq_train.reshape(-1, state_dim)
        scaler_seq.fit(X_seq_train_flat)
        X_seq_train_scaled = scaler_seq.transform(X_seq_train_flat).reshape(N_train, seq_len_dim, state_dim)

        N_val = X_seq_val.shape[0]
        X_seq_val_flat = X_seq_val.reshape(-1, state_dim)
        X_seq_val_scaled = scaler_seq.transform(X_seq_val_flat).reshape(N_val, seq_len_dim, state_dim)

        # Scale controls and targets
        X_ctrl_train = scaler_ctrl.fit_transform(X_ctrl_train)
        X_ctrl_val = scaler_ctrl.transform(X_ctrl_val)
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)

        # Create DataLoaders with tuple datasets (seq, ctrl, target)
        train_dataset = [(torch.FloatTensor(seq), torch.FloatTensor(ctrl), torch.FloatTensor(tgt))
                         for seq, ctrl, tgt in zip(X_seq_train_scaled, X_ctrl_train, y_train)]
        val_dataset = [(torch.FloatTensor(seq), torch.FloatTensor(ctrl), torch.FloatTensor(tgt))
                       for seq, ctrl, tgt in zip(X_seq_val_scaled, X_ctrl_val, y_val)]

        # Custom collate function to handle tuple data
        def collate_fn(batch):
            seqs = torch.stack([item[0] for item in batch])
            ctrls = torch.stack([item[1] for item in batch])
            targets = torch.stack([item[2] for item in batch])
            return seqs, ctrls, targets

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)

        return train_loader, val_loader, scaler_seq, scaler_ctrl, scaler_y

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
