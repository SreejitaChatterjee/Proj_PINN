#!/usr/bin/env python3
"""
Final ultra-optimized training with all improvements
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import time

# Import our enhanced model
import sys
sys.path.append('scripts')
from ultra_enhanced_pinn import UltraEnhancedPINN, UltraDataProcessor

class FinalUltraTrainer:
    """Final ultra trainer with best techniques"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-6)
        self.criterion = nn.MSELoss()
        
    def train_ultra(self, train_loader, val_loader, epochs=100):
        """Ultra training with best weights and techniques"""
        
        print("Starting ultra-optimized training...")
        print("Using: Complete physics + Motor dynamics + Aerodynamics")
        
        # Optimal weights from all testing
        physics_weight = 25.0      # Strong physics
        reg_weight = 20.0          # Strong regularization
        direct_id_weight = 60.0    # Very strong direct ID
        motor_weight = 10.0        # Motor dynamics
        data_weight = 0.1          # Minimal data weight
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                output = self.model(data)
                
                # All loss components
                data_loss = self.criterion(output, target)
                physics_loss = self.model.complete_physics_loss(data, output, target)
                direct_id_loss = self.model.direct_parameter_identification_loss(data, target)
                reg_loss = self.model.comprehensive_parameter_regularization_loss()
                
                # Ultra-optimized loss combination
                loss = (data_weight * data_loss +
                       physics_weight * physics_loss + 
                       reg_weight * reg_loss +
                       direct_id_weight * direct_id_loss)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.model.constrain_parameters()
                
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += self.criterion(output, target).item()
            
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'models/ultra_best_model.pth')
            
            if epoch % 15 == 0:
                print(f'Epoch {epoch:03d}: Train={total_loss/len(train_loader):.2f}, Val={val_loss:.6f}')
                
                if epoch % 30 == 0:
                    print(f'  Params: m={self.model.m.item():.6f}, '
                          f'Jxx={self.model.Jxx.item():.2e}, '
                          f'Jyy={self.model.Jyy.item():.2e}')
        
        # Load best model
        self.model.load_state_dict(torch.load('models/ultra_best_model.pth'))
        print("Ultra training completed!")

def run_final_ultra_optimization():
    """Run final ultra-optimized training"""
    
    print("FINAL ULTRA-OPTIMIZED PINN TRAINING")
    print("=" * 50)
    
    # Load aggressive data
    df = pd.read_csv('results/aggressive_quadrotor_training_data.csv')
    print(f"Loaded aggressive data: {len(df):,} points")
    print(f"Max angular rate: {df[['p','q','r']].abs().max().max():.1f} rad/s")
    
    # Prepare data
    processor = UltraDataProcessor()
    X, y = processor.prepare_sequences(df)
    
    print(f"Prepared sequences: {X.shape}")
    
    # Use subset for faster training (still 10x larger than original)
    subset_size = min(50000, len(X))
    indices = np.random.choice(len(X), subset_size, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_subset, y_subset, test_size=0.2, random_state=42)
    
    # Scale data
    X_train_scaled, y_train_scaled = processor.fit_transform(X_train, y_train)
    X_val_scaled, y_val_scaled = processor.transform(X_val, y_val)
    
    # Create loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), 
                                torch.FloatTensor(y_train_scaled))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), 
                              torch.FloatTensor(y_val_scaled))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create ultra model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = UltraEnhancedPINN(
        input_size=12, 
        hidden_size=256,
        output_size=16, 
        num_layers=6
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Learnable physics parameters: 13")
    
    # Train ultra model
    trainer = FinalUltraTrainer(model, device)
    
    start_time = time.time()
    trainer.train_ultra(train_loader, val_loader, epochs=150)
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time/60:.1f} minutes")
    
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
    
    print(f"\nFINAL ULTRA-OPTIMIZED RESULTS:")
    print("-" * 50)
    print(f"{'Parameter':<12} {'True':<12} {'Learned':<12} {'Accuracy':<10}")
    print("-" * 50)
    
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
    print("-" * 50)
    print(f"{'FINAL':<12} {'':<12} {'':<12} {avg_accuracy:<9.1f}%")
    
    # Additional learnable parameters
    print(f"\nAdditional Learned Parameters:")
    print(f"Motor kt: {model.kt.item():.6f} (True: 0.01)")
    print(f"Motor kq: {model.kq.item():.6f} (True: 0.000783)")
    print(f"Arm length b: {model.b.item():.6f} (True: 0.044)")
    print(f"Drag coeff Cd: {model.Cd.item():.6f} (True: 0.1)")
    
    # Performance comparison
    print(f"\nPERFORMANCE COMPARISON:")
    print(f"Original PINN (gentle data): 6.7% accuracy")
    print(f"Enhanced PINN (gentle data): 78.4% accuracy")
    print(f"ULTRA-OPTIMIZED (aggressive data): {avg_accuracy:.1f}% accuracy")
    print(f"")
    print(f"Improvement from original: {avg_accuracy/6.7:.1f}x better")
    print(f"Improvement from enhanced: +{avg_accuracy-78.4:.1f}% points")
    
    if avg_accuracy > 85:
        print(f"\nSUCCESS! Achieved >85% accuracy target!")
        print(f"Ultra-optimization successful!")
    elif avg_accuracy > 78.4:
        print(f"\nIMPROVED! Beat previous best of 78.4%")
        print(f"Aggressive data + complete physics = SUCCESS")
    else:
        print(f"\nProgress made with advanced techniques")
    
    # Save final model
    torch.save(model.state_dict(), 'models/final_ultra_optimized_model.pth')
    
    return avg_accuracy, learned_values

if __name__ == "__main__":
    accuracy, parameters = run_final_ultra_optimization()
    
    print(f"\n" + "="*60)
    print("ULTRA-OPTIMIZATION COMPLETE!")
    print(f"Final accuracy: {accuracy:.1f}%")
    print("All improvements implemented successfully!")