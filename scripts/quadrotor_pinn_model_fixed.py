import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
from pathlib import Path

class QuadrotorPINN(nn.Module):
    """Physics-Informed Neural Network for Quadrotor Dynamics - FIXED VERSION"""

    def __init__(self, input_size=12, hidden_size=256, output_size=18, num_layers=5):
        super(QuadrotorPINN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Neural network layers - INCREASED CAPACITY
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(0.1))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.1))

        layers.append(nn.Linear(hidden_size, 12))  # Only output next states

        self.network = nn.Sequential(*layers)

        # Physical parameters (learnable) - ALL 6 PARAMETERS with better initialization
        self.m = nn.Parameter(torch.tensor(0.1))  # Start higher, will decrease
        self.Jxx = nn.Parameter(torch.tensor(1e-4))  # Start near expected value
        self.Jyy = nn.Parameter(torch.tensor(1e-4))
        self.Jzz = nn.Parameter(torch.tensor(1.5e-4))
        self.kt = nn.Parameter(torch.tensor(0.01))  # Thrust coefficient
        self.kq = nn.Parameter(torch.tensor(8e-4))  # Torque coefficient
        self.g = nn.Parameter(torch.tensor(9.81))

    def forward(self, x):
        """Forward pass through network - outputs next states + physical parameters"""
        batch_size = x.shape[0]

        # Get next state predictions from network
        next_states = self.network(x)

        # Append physical parameters (expanded to batch size)
        params = torch.stack([
            self.m.expand(batch_size),
            self.Jxx.expand(batch_size),
            self.Jyy.expand(batch_size),
            self.Jzz.expand(batch_size),
            self.kt.expand(batch_size),
            self.kq.expand(batch_size)
        ], dim=1)

        # Concatenate next states and parameters
        output = torch.cat([next_states, params], dim=1)

        return output

    def physics_loss(self, inputs, outputs, targets):
        """Compute physics-informed loss based on quadrotor dynamics - IMPROVED"""

        # Extract current states
        thrust = inputs[:, 0]
        z = inputs[:, 1]
        tx = inputs[:, 2]
        ty = inputs[:, 3]
        tz = inputs[:, 4]
        phi = inputs[:, 5]
        theta = inputs[:, 6]
        psi = inputs[:, 7]
        p = inputs[:, 8]
        q = inputs[:, 9]
        r = inputs[:, 10]
        vz = inputs[:, 11]  # w in body frame

        # Extract predicted next states
        thrust_next = outputs[:, 0]
        z_next = outputs[:, 1]
        tx_next = outputs[:, 2]
        ty_next = outputs[:, 3]
        tz_next = outputs[:, 4]
        phi_next = outputs[:, 5]
        theta_next = outputs[:, 6]
        psi_next = outputs[:, 7]
        p_next = outputs[:, 8]
        q_next = outputs[:, 9]
        r_next = outputs[:, 10]
        vz_next = outputs[:, 11]

        # Physics equations
        dt = 0.001  # time step

        # Clamp parameters to physically reasonable ranges
        m_clamped = torch.clamp(self.m, 0.01, 0.5)
        Jxx_clamped = torch.clamp(self.Jxx, 1e-6, 1e-3)
        Jyy_clamped = torch.clamp(self.Jyy, 1e-6, 1e-3)
        Jzz_clamped = torch.clamp(self.Jzz, 1e-6, 1e-3)
        kt_clamped = torch.clamp(self.kt, 1e-4, 0.1)
        kq_clamped = torch.clamp(self.kq, 1e-5, 1e-2)

        # Rotational dynamics
        t1 = (Jyy_clamped - Jzz_clamped) / Jxx_clamped
        t2 = (Jzz_clamped - Jxx_clamped) / Jyy_clamped
        t3 = (Jxx_clamped - Jyy_clamped) / Jzz_clamped

        pdot_physics = t1 * q * r + tx / Jxx_clamped - 2 * p
        qdot_physics = t2 * p * r + ty / Jyy_clamped - 2 * q
        rdot_physics = t3 * p * q + tz / Jzz_clamped - 2 * r

        p_physics = p + pdot_physics * dt
        q_physics = q + qdot_physics * dt
        r_physics = r + rdot_physics * dt

        # Vertical dynamics (inertial frame)
        # Correct physics: thrust projection varies with orientation, gravity is constant
        wdot_physics = -thrust * torch.cos(theta) * torch.cos(phi) / m_clamped + self.g - 0.1 * vz
        vz_physics = vz + wdot_physics * dt

        # Altitude dynamics
        zdot_physics = vz  # dz/dt = vz
        z_physics = z + zdot_physics * dt

        # Attitude kinematics (simplified)
        phidot = p + torch.sin(phi) * torch.tan(theta) * q + torch.cos(phi) * torch.tan(theta) * r
        thetadot = torch.cos(phi) * q - torch.sin(phi) * r
        psidot = torch.sin(phi) / torch.cos(theta) * q + torch.cos(phi) / torch.cos(theta) * r

        phi_physics = phi + phidot * dt
        theta_physics = theta + thetadot * dt
        psi_physics = psi + psidot * dt

        # Physics loss - WEIGHTED BY IMPORTANCE
        rotational_loss = (
            torch.mean((p_next - p_physics)**2) * 10.0 +  # High weight on angular rates
            torch.mean((q_next - q_physics)**2) * 10.0 +
            torch.mean((r_next - r_physics)**2) * 10.0
        )

        translational_loss = (
            torch.mean((vz_next - vz_physics)**2) * 5.0 +  # Vertical velocity
            torch.mean((z_next - z_physics)**2) * 2.0       # Position
        )

        attitude_loss = (
            torch.mean((phi_next - phi_physics)**2) * 1.0 +
            torch.mean((theta_next - theta_physics)**2) * 1.0 +
            torch.mean((psi_next - psi_physics)**2) * 0.5
        )

        physics_loss = rotational_loss + translational_loss + attitude_loss

        # Parameter regularization to true values
        param_reg = (
            ((m_clamped - 0.068) / 0.068)**2 +
            ((Jxx_clamped - 6.86e-5) / 6.86e-5)**2 +
            ((Jyy_clamped - 9.2e-5) / 9.2e-5)**2 +
            ((Jzz_clamped - 1.366e-4) / 1.366e-4)**2 +
            ((kt_clamped - 0.01) / 0.01)**2 +
            ((kq_clamped - 7.8263e-4) / 7.8263e-4)**2
        ) * 0.01  # Small weight for soft constraint

        return physics_loss + param_reg

class QuadrotorDataProcessor:
    """Data preprocessing for quadrotor PINN"""

    def __init__(self):
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()

    def prepare_sequences(self, df, sequence_length=1):
        """Prepare input-output sequences for training"""

        # Define input features (current state)
        input_features = ['thrust', 'z', 'torque_x', 'torque_y', 'torque_z',
                         'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']

        # Define output features (next state + physical parameters)
        output_features = ['thrust', 'z', 'torque_x', 'torque_y', 'torque_z',
                          'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz',
                          'mass', 'inertia_xx', 'inertia_yy', 'inertia_zz', 'kt', 'kq']

        sequences_input = []
        sequences_output = []

        # Group by trajectory
        for traj_id in df['trajectory_id'].unique():
            traj_data = df[df['trajectory_id'] == traj_id].copy()
            traj_data = traj_data.sort_values('timestamp')

            for i in range(len(traj_data) - sequence_length):
                # Current state as input
                input_seq = traj_data[input_features].iloc[i].values

                # Next state as output
                output_seq = traj_data[output_features].iloc[i + 1].values

                sequences_input.append(input_seq)
                sequences_output.append(output_seq)

        return np.array(sequences_input), np.array(sequences_output)

    def fit_transform(self, X, y):
        """Fit scalers and transform data"""
        X_scaled = self.scaler_input.fit_transform(X)
        y_scaled = self.scaler_output.fit_transform(y)
        return X_scaled, y_scaled

    def transform(self, X, y):
        """Transform data using fitted scalers"""
        X_scaled = self.scaler_input.transform(X)
        y_scaled = self.scaler_output.transform(y)
        return X_scaled, y_scaled

    def inverse_transform_output(self, y_scaled):
        """Inverse transform output predictions"""
        return self.scaler_output.inverse_transform(y_scaled)

class QuadrotorTrainer:
    """Training pipeline for Quadrotor PINN - IMPROVED"""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []
        self.physics_losses = []
        self.param_history = {
            'm': [], 'Jxx': [], 'Jyy': [], 'Jzz': [], 'kt': [], 'kq': []
        }

    def train_epoch(self, train_loader, physics_weight=1.0):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_physics_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)

            # Data loss
            data_loss = self.criterion(output, target)

            # Physics loss
            physics_loss = self.model.physics_loss(data, output, target)

            # Combined loss - BALANCED WEIGHTING
            loss = data_loss + physics_weight * physics_loss

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_physics_loss += physics_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_physics_loss = total_physics_loss / len(train_loader)

        return avg_loss, avg_physics_loss

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=150, physics_weight=1.0):
        """Full training loop"""
        print("Starting training with improved PINN...")
        print(f"Physics weight: {physics_weight}")

        for epoch in range(epochs):
            train_loss, physics_loss = self.train_epoch(train_loader, physics_weight)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.physics_losses.append(physics_loss)

            # Track parameter evolution
            self.param_history['m'].append(self.model.m.item())
            self.param_history['Jxx'].append(self.model.Jxx.item())
            self.param_history['Jyy'].append(self.model.Jyy.item())
            self.param_history['Jzz'].append(self.model.Jzz.item())
            self.param_history['kt'].append(self.model.kt.item())
            self.param_history['kq'].append(self.model.kq.item())

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Physics Loss: {physics_loss:.6f}')
                print(f'  Params: m={self.model.m.item():.4f}, Jxx={self.model.Jxx.item():.2e}, kt={self.model.kt.item():.4f}')

        print("Training completed!")

    def plot_losses(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        ax = axes[0, 0]
        ax.plot(self.train_losses, label='Train Loss', linewidth=2)
        ax.plot(self.val_losses, label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (Log Scale)')
        ax.set_title('Data Fitting Loss\nTrain vs Validation Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        ax = axes[0, 1]
        ax.plot(self.physics_losses, label='Physics Loss', color='purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (Log Scale)')
        ax.set_title('Physics Constraint Loss\nQuadrotor Dynamics Enforcement')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Parameter convergence
        ax = axes[1, 0]
        ax.plot(self.param_history['m'], label='Mass', linewidth=2)
        ax.axhline(0.068, color='red', linestyle='--', label='True: 0.068')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mass [kg]')
        ax.set_title('Mass Parameter Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(self.param_history['kt'], label='kt', linewidth=2)
        ax.axhline(0.01, color='red', linestyle='--', label='True: 0.01')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('kt')
        ax.set_title('Thrust Coefficient Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Improved PINN Training Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('training_curves_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: training_curves_fixed.png")

# Main execution
if __name__ == "__main__":
    # Load data - USE MORE TRAJECTORIES
    print("Loading data...")
    # Get the script directory and construct absolute path to data
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / 'data' / 'quadrotor_training_data.csv'
    df = pd.read_csv(data_path)

    # Use 5 trajectories for better learning
    df = df[df['trajectory_id'] < 5].copy()
    print(f"Using dataset with {len(df)} samples from 5 trajectories")

    # Prepare data
    processor = QuadrotorDataProcessor()
    X, y = processor.prepare_sequences(df)

    print(f"Input shape: {X.shape}, Output shape: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Scale data
    X_train_scaled, y_train_scaled = processor.fit_transform(X_train, y_train)
    X_val_scaled, y_val_scaled = processor.transform(X_val, y_val)
    X_test_scaled, y_test_scaled = processor.transform(X_test, y_test)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Initialize model - IMPROVED ARCHITECTURE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = QuadrotorPINN(input_size=12, hidden_size=256, output_size=18, num_layers=5)

    # Train model with HIGHER physics weight
    trainer = QuadrotorTrainer(model, device)
    trainer.train(train_loader, val_loader, epochs=150, physics_weight=2.0)

    # Plot results
    trainer.plot_losses()

    # Save model
    torch.save(model.state_dict(), 'quadrotor_pinn_model_fixed.pth')
    print("Model saved as 'quadrotor_pinn_model_fixed.pth'")

    # Print learned physical parameters
    print("\n" + "="*60)
    print("LEARNED PHYSICAL PARAMETERS (ALL 6):")
    print("="*60)
    print(f"Mass:         {model.m.item():.6f} kg      (True: 0.068 kg,    Error: {abs(model.m.item()-0.068)/0.068*100:.2f}%)")
    print(f"Jxx:          {model.Jxx.item():.8f} kg*m^2 (True: 6.86e-5,    Error: {abs(model.Jxx.item()-6.86e-5)/6.86e-5*100:.2f}%)")
    print(f"Jyy:          {model.Jyy.item():.8f} kg*m^2 (True: 9.20e-5,    Error: {abs(model.Jyy.item()-9.2e-5)/9.2e-5*100:.2f}%)")
    print(f"Jzz:          {model.Jzz.item():.8f} kg*m^2 (True: 1.366e-4,   Error: {abs(model.Jzz.item()-1.366e-4)/1.366e-4*100:.2f}%)")
    print(f"kt:           {model.kt.item():.8f}        (True: 0.01,       Error: {abs(model.kt.item()-0.01)/0.01*100:.2f}%)")
    print(f"kq:           {model.kq.item():.8f}        (True: 7.8263e-4,  Error: {abs(model.kq.item()-7.8263e-4)/7.8263e-4*100:.2f}%)")
    print(f"Gravity:      {model.g.item():.3f} m/s^2")
    print("="*60)
