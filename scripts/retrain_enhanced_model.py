#!/usr/bin/env python3
"""
Retrain with enhanced PINN including direct parameter identification
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from enhanced_pinn_model import EnhancedQuadrotorPINN, EnhancedTrainer
from quadrotor_pinn_model import QuadrotorDataProcessor

if __name__ == "__main__":
    print("ENHANCED PINN TRAINING WITH DIRECT PARAMETER IDENTIFICATION")
    print("=" * 65)
    
    # Load data
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
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Smaller batch size
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize enhanced model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EnhancedQuadrotorPINN(input_size=12, hidden_size=128, output_size=16, num_layers=4)
    
    # Print initial parameters
    print("\nInitial Parameters:")
    print(f"Mass: {model.m.item():.6f} kg (True: 0.068000)")
    print(f"Jxx: {model.Jxx.item():.2e} kg*m^2 (True: 6.86e-05)")
    print(f"Jyy: {model.Jyy.item():.2e} kg*m^2 (True: 9.20e-05)") 
    print(f"Jzz: {model.Jzz.item():.2e} kg*m^2 (True: 1.37e-04)")
    print(f"Gravity: {model.g.item():.3f} m/s^2 (True: 9.810)")
    
    # Train enhanced model
    trainer = EnhancedTrainer(model, device)
    trainer.train(train_loader, val_loader, epochs=200)
    
    # Plot comprehensive training curves
    plt.figure(figsize=(20, 12))
    
    # Loss curves
    plt.subplot(2, 4, 1)
    plt.plot(trainer.train_losses, label='Train Loss', color='blue')
    plt.plot(trainer.val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Data Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(2, 4, 2)
    plt.plot(trainer.physics_losses, label='Physics Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Enhanced Physics Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(2, 4, 3)
    plt.plot(trainer.param_reg_losses, label='Parameter Regularization', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Parameter Regularization')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(2, 4, 4)
    plt.plot(trainer.direct_id_losses, label='Direct Identification', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Direct Parameter ID Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Parameter evolution (if we tracked it)
    plt.subplot(2, 4, 5)
    epochs = range(len(trainer.train_losses))
    plt.axhline(y=0.068, color='blue', linestyle='--', label='True Mass', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Mass (kg)')
    plt.title('Mass Parameter Target')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.060, 0.076)
    
    plt.subplot(2, 4, 6)
    plt.axhline(y=6.86e-5, color='blue', linestyle='--', label='True Jxx', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Jxx (kg*m^2)')
    plt.title('Jxx Parameter Target')
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.subplot(2, 4, 7)
    total_loss = (np.array(trainer.train_losses) + 
                  5.0 * np.array(trainer.physics_losses) + 
                  2.0 * np.array(trainer.param_reg_losses) +
                  10.0 * np.array(trainer.direct_id_losses))
    plt.plot(total_loss, label='Total Combined Loss', color='black', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Combined Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(2, 4, 8)
    plt.axhline(y=9.81, color='blue', linestyle='--', label='True Gravity', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Gravity (m/s^2)')
    plt.title('Gravity Parameter Target')
    plt.legend()
    plt.grid(True)
    plt.ylim(9.5, 10.1)
    
    plt.tight_layout()
    plt.savefig('enhanced_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save enhanced model
    torch.save(model.state_dict(), 'enhanced_quadrotor_pinn_model.pth')
    print("Enhanced model saved as 'enhanced_quadrotor_pinn_model.pth'")
    
    # Print final results
    print("\nFINAL ENHANCED MODEL PARAMETERS:")
    print("-" * 50)
    
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
    
    total_error = 0
    for param in true_values:
        true_val = true_values[param]
        learned_val = learned_values[param]
        error_pct = abs(learned_val - true_val) / true_val * 100
        total_error += error_pct
        
        if param == 'Mass':
            print(f"{param:<12} {true_val:<12.6f} {learned_val:<12.6f} {error_pct:<9.1f}%")
        elif param == 'Gravity':
            print(f"{param:<12} {true_val:<12.3f} {learned_val:<12.3f} {error_pct:<9.1f}%")
        else:
            print(f"{param:<12} {true_val:<12.2e} {learned_val:<12.2e} {error_pct:<9.1f}%")
    
    avg_error = total_error / len(true_values)
    print(f"\nAverage parameter error: {avg_error:.1f}%")
    
    print(f"\nEnhanced Model Features Used:")
    print(f"• Direct parameter identification from torque/acceleration")
    print(f"• Complete rotational kinematics (Euler dynamics)")
    print(f"• Cross-coupling terms in rotational equations")
    print(f"• Tight parameter constraints")
    print(f"• Strong regularization (100x weight)")
    print(f"• Multi-loss optimization (data + physics + regularization + direct ID)")
    
    if avg_error < 50:
        print(f"\n SUCCESS: Average parameter error < 50%!")
    else:
        print(f"\n NEED MORE WORK: Average error still high at {avg_error:.1f}%")