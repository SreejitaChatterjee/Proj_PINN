#!/usr/bin/env python3
"""
Retrain quadrotor PINN with improved physics constraints and parameter learning
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
from improved_pinn_model import ImprovedQuadrotorPINN, ImprovedTrainer
from quadrotor_pinn_model import QuadrotorDataProcessor

def compare_models():
    """Compare old vs new model parameters"""
    print("\n" + "="*60)
    print("MODEL COMPARISON - OLD VS NEW")
    print("="*60)
    
    # Load old model
    old_model = torch.load('quadrotor_pinn_model.pth')
    
    # True values
    true_values = {
        'Mass': 0.068,
        'Jxx': 6.86e-5,
        'Jyy': 9.2e-5,
        'Jzz': 1.366e-4,
        'Gravity': 9.81
    }
    
    print(f"{'Parameter':<12} {'True':<12} {'Old Model':<12} {'New Model':<12} {'Old Error %':<12} {'New Error %':<12}")
    print("-" * 80)
    
    return true_values

if __name__ == "__main__":
    print("RETRAINING WITH IMPROVED PINN MODEL")
    print("="*50)
    
    # Load data (same reduced dataset)
    print("Loading training data...")
    df = pd.read_csv('quadrotor_training_data.csv')
    df = df[df['trajectory_id'] < 3].copy()
    print(f"Using dataset with {len(df)} samples")
    
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
    
    # Initialize improved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ImprovedQuadrotorPINN(input_size=12, hidden_size=128, output_size=16, num_layers=4)
    
    # Print initial parameters
    print("\nInitial Parameters:")
    print(f"Mass: {model.m.item():.6f} kg")
    print(f"Jxx: {model.Jxx.item():.8f} kg*m^2")
    print(f"Jyy: {model.Jyy.item():.8f} kg*m^2") 
    print(f"Jzz: {model.Jzz.item():.8f} kg*m^2")
    print(f"Gravity: {model.g.item():.3f} m/s^2")
    
    # Train improved model with stronger physics constraints
    trainer = ImprovedTrainer(model, device)
    trainer.train(train_loader, val_loader, 
                  epochs=100, 
                  physics_weight=2.0,  # Even stronger physics constraint
                  reg_weight=0.5)      # Strong regularization
    
    # Plot improved training curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(trainer.train_losses, label='Train Loss', color='blue')
    plt.plot(trainer.val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Data Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(trainer.physics_losses, label='Physics Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Physics Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(trainer.param_reg_losses, label='Parameter Regularization', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Parameter Regularization Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    total_loss = np.array(trainer.train_losses) + 2.0 * np.array(trainer.physics_losses) + 0.5 * np.array(trainer.param_reg_losses)
    plt.plot(total_loss, label='Total Combined Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Combined Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot parameter evolution during training
    plt.subplot(2, 3, 5)
    epochs = range(len(trainer.train_losses))
    plt.axhline(y=0.068, color='blue', linestyle='--', label='True Mass', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Mass (kg)')
    plt.title('Mass Parameter Evolution')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    plt.axhline(y=9.81, color='blue', linestyle='--', label='True Gravity', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Gravity (m/s^2)')
    plt.title('Gravity Parameter Evolution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save improved model
    torch.save(model.state_dict(), 'improved_quadrotor_pinn_model.pth')
    print("Improved model saved as 'improved_quadrotor_pinn_model.pth'")
    
    # Print final learned parameters
    print("\nFINAL LEARNED PARAMETERS:")
    print("-" * 40)
    
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
    
    print(f"{'Parameter':<12} {'True':<12} {'Learned':<12} {'Error %':<10}")
    print("-" * 50)
    
    for param in true_values:
        true_val = true_values[param]
        learned_val = learned_values[param]
        error_pct = abs(learned_val - true_val) / true_val * 100
        
        print(f"{param:<12} {true_val:<12.6f} {learned_val:<12.6f} {error_pct:>8.1f}%")
    
    print(f"\nModel training completed with improved physics constraints!")
    print(f"Physics weight: 2.0 (vs 0.1 in original)")
    print(f"Parameter regularization weight: 0.5")
    print(f"Parameter constraints: Applied")