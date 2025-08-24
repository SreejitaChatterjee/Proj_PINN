#!/usr/bin/env python3
"""
Quick final optimization with tweaked physics weights
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from enhanced_pinn_model import EnhancedQuadrotorPINN, EnhancedTrainer
from quadrotor_pinn_model import QuadrotorDataProcessor

def quick_final_test():
    """Quick test with optimized weights"""
    
    print("QUICK FINAL OPTIMIZATION TEST")
    print("=" * 40)
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different extreme weight combinations
    test_configs = [
        (20.0, 20.0, 50.0, "High physics & reg"),
        (30.0, 15.0, 60.0, "Max physics focus"),
        (15.0, 30.0, 70.0, "Strong regularization"),
        (40.0, 10.0, 80.0, "Extreme physics")
    ]
    
    best_accuracy = 0
    best_model = None
    
    for physics_w, reg_w, direct_w, desc in test_configs:
        print(f"\nTesting: {desc}")
        print(f"Weights - Physics: {physics_w}, Reg: {reg_w}, Direct ID: {direct_w}")
        
        # Fresh model
        model = EnhancedQuadrotorPINN()
        
        # Modified trainer for quick test
        class QuickTrainer(EnhancedTrainer):
            def train_quick(self, train_loader, val_loader, epochs=80):
                for epoch in range(epochs):
                    train_loss, physics_loss, reg_loss, id_loss = self.train_epoch(
                        train_loader, physics_weight=physics_w, 
                        reg_weight=reg_w, direct_id_weight=direct_w)
                    
                    if epoch % 20 == 0:
                        val_loss = self.validate(val_loader)
                        print(f'  Epoch {epoch}: Train={train_loss:.1f}, Val={val_loss:.6f}, '
                              f'Phys={physics_loss:.1f}')
        
        trainer = QuickTrainer(model, device)
        trainer.train_quick(train_loader, val_loader)
        
        # Evaluate
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
        
        total_accuracy = 0
        for param in true_values:
            error_pct = abs(learned_values[param] - true_values[param]) / true_values[param] * 100
            accuracy_pct = max(0, 100 - error_pct)
            total_accuracy += accuracy_pct
        
        avg_accuracy = total_accuracy / len(true_values)
        print(f"Result: {avg_accuracy:.1f}% accuracy")
        
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_model = model
            torch.save(model.state_dict(), 'best_quick_optimized_model.pth')
            print(f"*** NEW BEST: {avg_accuracy:.1f}% ***")
    
    print(f"\nFINAL RESULTS:")
    print(f"Best accuracy achieved: {best_accuracy:.1f}%")
    if best_accuracy > 78.4:
        print(f"Improvement: +{best_accuracy-78.4:.1f}% over previous best!")
    else:
        print(f"No significant improvement achieved")
        
    # Print best model parameters
    if best_model:
        print(f"\nBest Model Parameters:")
        print(f"Mass: {best_model.m.item():.6f} kg (True: 0.068)")
        print(f"Jxx: {best_model.Jxx.item():.2e} kg*m^2 (True: 6.86e-5)")
        print(f"Jyy: {best_model.Jyy.item():.2e} kg*m^2 (True: 9.20e-5)")
        print(f"Jzz: {best_model.Jzz.item():.2e} kg*m^2 (True: 1.37e-4)")
    
    return best_accuracy

if __name__ == "__main__":
    final_accuracy = quick_final_test()