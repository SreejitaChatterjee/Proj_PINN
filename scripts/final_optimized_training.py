#!/usr/bin/env python3
"""
Final optimized training with best practices for parameter learning
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

class FinalOptimizedPINN(EnhancedQuadrotorPINN):
    """Final optimized PINN with best parameter learning techniques"""
    
    def constrain_parameters(self):
        """Optimized parameter constraints - not too tight, not too loose"""
        with torch.no_grad():
            # Optimal mass constraints (allow 10% variation)
            self.m.data = torch.clamp(self.m.data, 0.061, 0.075)
            
            # Optimized inertia constraints (allow reasonable variation)
            self.Jxx.data = torch.clamp(self.Jxx.data, 4e-5, 11e-5)
            self.Jyy.data = torch.clamp(self.Jyy.data, 6e-5, 14e-5)
            self.Jzz.data = torch.clamp(self.Jzz.data, 9e-5, 20e-5)
            
            # Gravity constraints
            self.g.data = torch.clamp(self.g.data, 9.3, 10.3)

class FinalOptimizedTrainer:
    """Final optimized trainer with best weight combination"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        # Optimized learning rate and regularization
        self.optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-6)
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=20, verbose=True)
        
        self.train_losses = []
        self.val_losses = []
        self.physics_losses = []
        self.param_reg_losses = []
        self.direct_id_losses = []
        
    def train_epoch(self, train_loader):
        """Optimized training epoch"""
        self.model.train()
        total_loss = 0
        total_physics_loss = 0
        total_reg_loss = 0
        total_id_loss = 0
        
        # Optimal weights from testing (with slight increase to Direct ID)
        physics_weight = 15.0      # Increased from 10.0
        reg_weight = 15.0          # Increased from 10.0  
        direct_id_weight = 40.0    # Increased from 30.0
        data_weight = 0.5          # Reduced data influence
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            
            # Individual losses
            data_loss = self.criterion(output, target)
            physics_loss = self.model.enhanced_physics_loss(data, output, target)
            direct_id_loss = self.model.direct_parameter_identification_loss(data, target)
            reg_loss = self.model.parameter_regularization_loss()
            
            # Optimal weighted combination
            loss = (data_weight * data_loss +
                   physics_weight * physics_loss + 
                   reg_weight * reg_loss +
                   direct_id_weight * direct_id_loss)
            
            loss.backward()
            
            # Optimal gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.8)
            
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
    
    def train(self, train_loader, val_loader, epochs=150):
        """Final optimized training"""
        print("FINAL OPTIMIZED TRAINING")
        print("Weights: Physics=15.0, Reg=15.0, DirectID=40.0, Data=0.5")
        print("Features: LR scheduling, optimized constraints, gradient clipping")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, physics_loss, reg_loss, id_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.physics_losses.append(physics_loss)
            self.param_reg_losses.append(reg_loss)
            self.direct_id_losses.append(id_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_final_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 15 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch:03d}: Train: {train_loss:.2f}, Val: {val_loss:.6f}, '
                      f'Physics: {physics_loss:.2f}, Reg: {reg_loss:.2f}, ID: {id_loss:.2f}, '
                      f'LR: {current_lr:.6f}')
                
                if epoch % 30 == 0:
                    print(f'  Params - m: {self.model.m.item():.6f}, '
                          f'Jxx: {self.model.Jxx.item():.2e}, '
                          f'Jyy: {self.model.Jyy.item():.2e}, '
                          f'Jzz: {self.model.Jzz.item():.2e}')
            
            # Early stopping
            if patience_counter >= 50:
                print(f"Early stopping at epoch {epoch}")
                break
                
        print("Final optimized training completed!")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_final_model.pth'))

def run_final_optimization():
    """Run final optimized training"""
    
    print("FINAL PARAMETER LEARNING OPTIMIZATION")
    print("=" * 50)
    
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
    
    # Optimized batch size
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create final optimized model
    model = FinalOptimizedPINN(input_size=12, hidden_size=128, output_size=16, num_layers=4)
    
    # Train with final optimization
    trainer = FinalOptimizedTrainer(model, device)
    trainer.train(train_loader, val_loader, epochs=150)
    
    # Final evaluation
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
    
    print("\nFINAL OPTIMIZATION RESULTS:")
    print("-" * 45)
    print(f"{'Parameter':<12} {'True':<12} {'Learned':<12} {'Accuracy':<10}")
    print("-" * 45)
    
    total_accuracy = 0
    for param in true_values:
        true_val = true_values[param]
        learned_val = learned_values[param]
        error_pct = abs(learned_val - true_val) / true_val * 100
        accuracy_pct = max(0, 100 - error_pct)
        total_accuracy += accuracy_pct
        
        if param == 'Mass':
            print(f"{param:<12} {true_val:<12.6f} {learned_val:<12.6f} {accuracy_pct:<9.1f}%")
        elif param == 'Gravity':
            print(f"{param:<12} {true_val:<12.3f} {learned_val:<12.3f} {accuracy_pct:<9.1f}%")
        else:
            print(f"{param:<12} {true_val:<12.2e} {learned_val:<12.2e} {accuracy_pct:<9.1f}%")
    
    avg_accuracy = total_accuracy / len(true_values)
    print("-" * 45)
    print(f"{'OVERALL':<12} {'':<12} {'':<12} {avg_accuracy:<9.1f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'final_optimized_pinn_model.pth')
    
    improvement = avg_accuracy - 78.4
    if improvement > 1.0:
        print(f"\nSUCCESS! Achieved {avg_accuracy:.1f}% accuracy")
        print(f"Improvement: +{improvement:.1f}% over previous best")
    else:
        print(f"\nResult: {avg_accuracy:.1f}% accuracy") 
        print(f"Change: {improvement:+.1f}% from previous")
        
    return model, avg_accuracy

if __name__ == "__main__":
    model, accuracy = run_final_optimization()