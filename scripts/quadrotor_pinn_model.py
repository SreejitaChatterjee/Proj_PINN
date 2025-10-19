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
    """Physics-Informed Neural Network for Quadrotor Dynamics"""
    
    def __init__(self, input_size=12, hidden_size=128, output_size=18, num_layers=4):
        super(QuadrotorPINN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Neural network layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_size, 12))  # Only output next states

        self.network = nn.Sequential(*layers)

        # Physical parameters (learnable) - ALL 6 PARAMETERS
        self.m = nn.Parameter(torch.tensor(0.068))
        self.Jxx = nn.Parameter(torch.tensor(6.86e-5))
        self.Jyy = nn.Parameter(torch.tensor(9.2e-5))
        self.Jzz = nn.Parameter(torch.tensor(1.366e-4))
        self.kt = nn.Parameter(torch.tensor(0.01))  # Thrust coefficient (LEARNABLE)
        self.kq = nn.Parameter(torch.tensor(7.8263e-4))  # Torque coefficient (LEARNABLE)
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
        """Compute physics-informed loss based on quadrotor dynamics"""
        
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
        
        # Physics equations (simplified for key dynamics)
        dt = 0.001  # time step
        
        # Rotational dynamics
        t1 = (self.Jyy - self.Jzz) / self.Jxx
        t2 = (self.Jzz - self.Jxx) / self.Jyy
        t3 = (self.Jxx - self.Jyy) / self.Jzz
        
        pdot_physics = t1 * q * r + tx / self.Jxx - 2 * p
        qdot_physics = t2 * p * r + ty / self.Jyy - 2 * q  
        rdot_physics = t3 * p * q + tz / self.Jzz - 2 * r
        
        p_physics = p + pdot_physics * dt
        q_physics = q + qdot_physics * dt
        r_physics = r + rdot_physics * dt
        
        # Vertical dynamics (inertial frame)
        # Correct physics: thrust projection varies with orientation, gravity is constant
        wdot_physics = -thrust * torch.cos(theta) * torch.cos(phi) / self.m + self.g - 0.1 * vz
        vz_physics = vz + wdot_physics * dt
        
        # Physics loss
        physics_loss = torch.mean((p_next - p_physics)**2 + 
                                 (q_next - q_physics)**2 + 
                                 (r_next - r_physics)**2 +
                                 (vz_next - vz_physics)**2)
        
        return physics_loss

class QuadrotorDataProcessor:
    """Data preprocessing for quadrotor PINN"""
    
    def __init__(self):
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()
        
    def prepare_sequences(self, df, sequence_length=1):
        """Prepare input-output sequences for training"""
        
        # Define input features (current state) - using actual column names from CSV
        input_features = ['thrust', 'z', 'torque_x', 'torque_y', 'torque_z', 
                         'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
        
        # Define output features (next state + physical parameters) - using actual column names
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
    """Training pipeline for Quadrotor PINN"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.physics_losses = []
        
    def train_epoch(self, train_loader, physics_weight=0.1):
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
            
            # Combined loss
            loss = data_loss + physics_weight * physics_loss
            
            loss.backward()
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
    
    def train(self, train_loader, val_loader, epochs=100, physics_weight=0.1):
        """Full training loop"""
        print("Starting training...")
        
        for epoch in range(epochs):
            train_loss, physics_loss = self.train_epoch(train_loader, physics_weight)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.physics_losses.append(physics_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Physics Loss: {physics_loss:.6f}')
                
        print("Training completed!")
    
    def plot_losses(self):
        """Plot training curves"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Data Fitting Loss\nTrain vs Validation Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.physics_losses, label='Physics Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Physics Constraint Loss\nQuadrotor Dynamics Enforcement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.subplot(1, 3, 3)
        total_loss = np.array(self.train_losses) + 0.1 * np.array(self.physics_losses)
        plt.plot(total_loss, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Combined Loss\nWeighted Data + Physics Terms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.suptitle('PINN Training Analysis: Basic Model Loss Evolution\nPhysics-Informed Neural Network Training Progress', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

# Main execution
if __name__ == "__main__":
    # Load data with reduced size to avoid memory issues
    print("Loading data...")
    # Get the script directory and construct absolute path to data
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / 'data' / 'quadrotor_training_data.csv'
    df = pd.read_csv(data_path)
    
    # Use only first 3 trajectories to reduce memory usage
    df = df[df['trajectory_id'] < 3].copy()
    print(f"Using reduced dataset with {len(df)} samples")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QuadrotorPINN(input_size=12, hidden_size=128, output_size=18, num_layers=4)
    
    # Train model
    trainer = QuadrotorTrainer(model, device)
    trainer.train(train_loader, val_loader, epochs=50, physics_weight=0.1)
    
    # Plot results
    trainer.plot_losses()
    
    # Save model
    torch.save(model.state_dict(), 'quadrotor_pinn_model.pth')
    print("Model saved as 'quadrotor_pinn_model.pth'")
    
    # Print learned physical parameters
    print("\nLearned Physical Parameters (6 total):")
    print(f"Mass: {model.m.item():.6f} kg")
    print(f"Jxx: {model.Jxx.item():.8f} kg*m^2")
    print(f"Jyy: {model.Jyy.item():.8f} kg*m^2")
    print(f"Jzz: {model.Jzz.item():.8f} kg*m^2")
    print(f"kt (thrust coefficient): {model.kt.item():.8f}")
    print(f"kq (torque coefficient): {model.kq.item():.8f}")
    print(f"Gravity: {model.g.item():.3f} m/s^2")