#!/usr/bin/env python3
"""
Extreme optimization with very high physics weights and relaxed constraints
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from enhanced_pinn_model import EnhancedQuadrotorPINN
from quadrotor_pinn_model import QuadrotorDataProcessor

class ExtremeOptimizedPINN(EnhancedQuadrotorPINN):
    """PINN with relaxed constraints and extreme physics focus"""
    
    def constrain_parameters(self):
        """Apply relaxed parameter constraints for better convergence"""
        with torch.no_grad():
            # More relaxed mass constraints
            self.m.data = torch.clamp(self.m.data, 0.050, 0.100)
            
            # Relaxed inertia constraints (allow more variation)
            self.Jxx.data = torch.clamp(self.Jxx.data, 3e-5, 15e-5)
            self.Jyy.data = torch.clamp(self.Jyy.data, 5e-5, 20e-5)
            self.Jzz.data = torch.clamp(self.Jzz.data, 8e-5, 25e-5)
            
            # Gravity constraints
            self.g.data = torch.clamp(self.g.data, 9.0, 10.5)
    
    def extreme_physics_loss(self, inputs, outputs, targets):
        """Enhanced physics loss with additional terms"""
        
        # Get base physics loss
        base_physics_loss = self.enhanced_physics_loss(inputs, outputs, targets)
        
        # Extract states for additional constraints
        tx = inputs[:, 2]  
        ty = inputs[:, 3] 
        tz = inputs[:, 4] 
        p = inputs[:, 8]   
        q = inputs[:, 9]  
        r = inputs[:, 10]  
        
        # Extract next step values
        p_next = targets[:, 8]
        q_next = targets[:, 9]
        r_next = targets[:, 10]
        
        dt = 0.001
        
        # Additional physics constraint: Direct inertia ratios
        # From the physics: J_xx < J_yy < J_zz for typical quadrotors
        inertia_ordering_loss = (
            torch.relu(self.Jxx - self.Jyy) +  # Jxx should be < Jyy
            torch.relu(self.Jyy - self.Jzz)    # Jyy should be < Jzz
        )
        
        # Energy conservation constraint
        # Angular kinetic energy should be consistent
        ke_current = 0.5 * (self.Jxx * p**2 + self.Jyy * q**2 + self.Jzz * r**2)
        ke_next = 0.5 * (self.Jxx * p_next**2 + self.Jyy * q_next**2 + self.Jzz * r_next**2)
        
        # Work done by torques
        work_done = tx * p * dt + ty * q * dt + tz * r * dt
        energy_conservation_loss = torch.mean((ke_next - ke_current - work_done)**2)
        
        # Total enhanced physics loss
        total_physics_loss = (base_physics_loss + 
                             10.0 * inertia_ordering_loss + 
                             1.0 * energy_conservation_loss)
        
        return total_physics_loss

class ExtremeTrainer:
    """Extreme trainer with very high physics emphasis"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        # Even lower learning rate for precision
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.physics_losses = []
        self.param_reg_losses = []
        self.direct_id_losses = []
        
    def train_epoch(self, train_loader, physics_weight=50.0, reg_weight=20.0, direct_id_weight=100.0):
        """Extreme training with very high physics weights"""
        self.model.train()
        total_loss = 0
        total_physics_loss = 0
        total_reg_loss = 0
        total_id_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            
            # Individual losses
            data_loss = self.criterion(output, target)
            
            # Use extreme physics loss
            physics_loss = self.model.extreme_physics_loss(data, output, target)
            
            # Direct parameter identification
            direct_id_loss = self.model.direct_parameter_identification_loss(data, target)
            
            # Strong parameter regularization
            reg_loss = self.model.parameter_regularization_loss()
            
            # Extreme weighting: Physics dominates everything
            loss = (0.1 * data_loss +  # Minimize data loss influence
                   physics_weight * physics_loss + 
                   reg_weight * reg_loss +
                   direct_id_weight * direct_id_loss)
            
            loss.backward()
            
            # Stronger gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            self.model.constrain_parameters()
            
            total_loss += loss.item()
            total_physics_loss += physics_loss.item()
            total_reg_loss += reg_loss.item()
            total_id_loss += direct_id_loss.item()
            
        return (total_loss / len(train_loader), 
                total_physics_loss / len(train_loader),
                total_reg_loss / len(train_loader),
                total_id_loss / len(train_loader))
    
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
    
    def train(self, train_loader, val_loader, epochs=300):
        """Extreme training with high epochs for convergence"""
        print("Starting EXTREME physics-dominated training...")
        print("Weights: Physics=50, Regularization=20, Direct ID=100, Data=0.1")
        
        for epoch in range(epochs):
            train_loss, physics_loss, reg_loss, id_loss = self.train_epoch(train_loader)
            
            self.train_losses.append(train_loss)
            self.physics_losses.append(physics_loss)
            self.param_reg_losses.append(reg_loss)
            self.direct_id_losses.append(id_loss)
            
            if epoch % 25 == 0:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f'Epoch {epoch:03d}: Train: {train_loss:.2f}, Val: {val_loss:.6f}, '
                      f'Physics: {physics_loss:.2f}, Reg: {reg_loss:.2f}, ID: {id_loss:.2f}')
                
                # Print current parameters
                if epoch % 50 == 0:
                    print(f'  Current params - m: {self.model.m.item():.6f}, '
                          f'Jxx: {self.model.Jxx.item():.2e}, '
                          f'Jyy: {self.model.Jyy.item():.2e}, '
                          f'Jzz: {self.model.Jzz.item():.2e}')
                
        print("Extreme training completed!")

def run_extreme_optimization():
    """Run extreme optimization experiment"""
    
    print("EXTREME PHYSICS-DOMINATED OPTIMIZATION")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('quadrotor_training_data.csv')
    df = df[df['trajectory_id'] < 3].copy()
    
    processor = QuadrotorDataProcessor()
    X, y = processor.prepare_sequences(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    X_train_scaled, y_train_scaled = processor.fit_transform(X_train, y_train)
    X_val_scaled, y_val_scaled = processor.transform(X_val, y_val)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Smaller batch size for more stable training
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create extreme model
    model = ExtremeOptimizedPINN(input_size=12, hidden_size=128, output_size=16, num_layers=4)
    
    print(f"Initial Parameters:")
    print(f"Mass: {model.m.item():.6f} kg")
    print(f"Jxx: {model.Jxx.item():.2e} kg*m^2") 
    print(f"Jyy: {model.Jyy.item():.2e} kg*m^2")
    print(f"Jzz: {model.Jzz.item():.2e} kg*m^2")
    print(f"Gravity: {model.g.item():.3f} m/s^2")
    
    # Train with extreme settings
    trainer = ExtremeTrainer(model, device)
    trainer.train(train_loader, val_loader, epochs=200)
    
    # Evaluate final results
    true_values = {
        'Mass': 0.068,
        'Jxx': 6.86e-5,
        'Jyy': 9.2e-5,
        'Jzz': 1.366e-4,
        'Gravity': 9.81
    }
    
    learned_values = {
        'Mass': model.m.item(),
        'Jxx': model.Jxx.item(),
        'Jyy': model.Jyy.item(),
        'Jzz': model.Jzz.item(),
        'Gravity': model.g.item()
    }
    
    print("\nFINAL EXTREME OPTIMIZATION RESULTS:")
    print("-" * 50)
    print(f"{'Parameter':<12} {'True':<12} {'Learned':<12} {'Accuracy':<10}")
    print("-" * 50)
    
    total_accuracy = 0
    for param in true_values:
        true_val = true_values[param]
        learned_val = learned_values[param]
        error_pct = abs(learned_val - true_val) / true_val * 100
        accuracy_pct = max(0, 100 - error_pct)  # Cap at 0% minimum
        total_accuracy += accuracy_pct
        
        if param == 'Mass':
            print(f"{param:<12} {true_val:<12.6f} {learned_val:<12.6f} {accuracy_pct:<9.1f}%")
        elif param == 'Gravity':
            print(f"{param:<12} {true_val:<12.3f} {learned_val:<12.3f} {accuracy_pct:<9.1f}%")
        else:
            print(f"{param:<12} {true_val:<12.2e} {learned_val:<12.2e} {accuracy_pct:<9.1f}%")
    
    avg_accuracy = total_accuracy / len(true_values)
    print("-" * 50)
    print(f"{'AVERAGE':<12} {'':<12} {'':<12} {avg_accuracy:<9.1f}%")
    
    # Save extreme model
    torch.save(model.state_dict(), 'extreme_optimized_pinn_model.pth')
    print(f"\nExtreme model saved as 'extreme_optimized_pinn_model.pth'")
    
    if avg_accuracy > 78.4:
        print(f"SUCCESS! Achieved {avg_accuracy:.1f}% accuracy (improved from 78.4%)")
    else:
        print(f"Result: {avg_accuracy:.1f}% accuracy (no improvement from 78.4%)")
        
    return model, avg_accuracy

if __name__ == "__main__":
    model, accuracy = run_extreme_optimization()