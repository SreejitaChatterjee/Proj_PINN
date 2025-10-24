#!/usr/bin/env python3
"""
Train enhanced PINN model with realistic quadrotor data
Addresses all 7 critical anomalies identified in the report
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Import the enhanced PINN model
from enhanced_pinn_model import EnhancedQuadrotorPINN, EnhancedTrainer

def load_and_prepare_data(data_path):
    """Load and prepare training data"""
    print("Loading realistic quadrotor training data...")
    df = pd.read_csv(data_path)

    print(f"  Loaded {len(df)} samples")
    print(f"  Trajectories: {df['trajectory_id'].nunique()}")
    print(f"  Thrust range: [{df['thrust'].min():.3f}, {df['thrust'].max():.3f}] N")
    print(f"  Altitude range: [{df['z'].min():.3f}, {df['z'].max():.3f}] m")
    print(f"  Vertical velocity range: [{df['vz'].min():.3f}, {df['vz'].max():.3f}] m/s")

    # Select input and output columns
    input_cols = ['thrust', 'z', 'torque_x', 'torque_y', 'torque_z',
                  'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    output_cols = input_cols  # Same columns for next state prediction

    # Create sequential data (t -> t+1)
    X = df[input_cols].values[:-1]
    y = df[output_cols].values[1:]

    print(f"  Created {len(X)} time-step pairs (t -> t+1)")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    # Standardize data
    print("Standardizing data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    return train_loader, val_loader, scaler_X, scaler_y

def main():
    """Main training function"""
    print("=" * 80)
    print("TRAINING ENHANCED PINN WITH REALISTIC QUADROTOR DATA")
    print("=" * 80)
    print("\nANOMALY FIXES IMPLEMENTED:")
    print("  1. Filtered reference trajectories (smooth transitions)")
    print("  2. Motor dynamics with time constants (80ms)")
    print("  3. Slew rate limits (thrust: 15 N/s, torque: 0.5 N·m/s)")
    print("  4. Reduced controller gains (50% reduction)")
    print("  5. State derivative constraints in physics loss")
    print("  6. Increased physics loss weight (5.0 -> 15.0, 3x increase)")
    print("  7. Realistic vertical velocity control")
    print("=" * 80)
    print()

    # Paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / 'data' / 'quadrotor_training_data.csv'
    model_save_path = project_root / 'models' / 'enhanced_pinn_realistic.pth'
    model_save_path.parent.mkdir(exist_ok=True, parents=True)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Load data
    train_loader, val_loader, scaler_X, scaler_y = load_and_prepare_data(data_path)

    # Create model
    print("Initializing enhanced PINN model...")
    model = EnhancedQuadrotorPINN(
        input_size=12,
        hidden_size=128,
        output_size=12,
        num_layers=4
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Learnable physical parameters: 6 (m, Jxx, Jyy, Jzz, kt, kq)")
    print(f"  Fixed constants: g = {model.g} m/s²")
    print()

    # Create trainer
    trainer = EnhancedTrainer(model, device=device)

    # Train model
    print("Starting training with enhanced physics constraints...")
    print("  Physics weight: 15.0 (increased from 5.0)")
    print("  Derivative constraint weight: 8.0 (NEW)")
    print("  Direct identification weight: 10.0")
    print("  Regularization weight: 2.0")
    print()

    trainer.train(train_loader, val_loader, epochs=150)

    # Save model
    print()
    print(f"Saving model to {model_save_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'learned_params': {
            'm': model.m.item(),
            'Jxx': model.Jxx.item(),
            'Jyy': model.Jyy.item(),
            'Jzz': model.Jzz.item(),
            'kt': model.kt.item(),
            'kq': model.kq.item(),
            'g': model.g  # Fixed constant
        },
        'true_params': {
            'm': model.true_m,
            'Jxx': model.true_Jxx,
            'Jyy': model.true_Jyy,
            'Jzz': model.true_Jzz,
            'kt': model.true_kt,
            'kq': model.true_kq,
            'g': model.g  # Fixed constant
        }
    }, model_save_path)

    # Print final parameter comparison
    print()
    print("=" * 80)
    print("FINAL PARAMETER COMPARISON")
    print("=" * 80)

    params = [
        ('Mass (kg)', model.m.item(), model.true_m),
        ('Jxx (kg·m²)', model.Jxx.item(), model.true_Jxx),
        ('Jyy (kg·m²)', model.Jyy.item(), model.true_Jyy),
        ('Jzz (kg·m²)', model.Jzz.item(), model.true_Jzz),
        ('kt (N/rpm²)', model.kt.item(), model.true_kt),
        ('kq (N·m/rpm²)', model.kq.item(), model.true_kq),
    ]

    print(f"{'Parameter':<20} {'Learned':<15} {'True':<15} {'Error %':<10}")
    print("-" * 60)
    for name, learned, true in params:
        error_pct = abs((learned - true) / true) * 100
        print(f"{name:<20} {learned:<15.6e} {true:<15.6e} {error_pct:<10.2f}")

    print()
    print(f"Fixed constant: g = {model.g} m/s² (not learned)")
    print()
    print("=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
