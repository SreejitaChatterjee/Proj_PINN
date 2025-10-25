#!/usr/bin/env python3
"""
Generate ACTUAL parameter convergence plots from real training run
NOT idealized - these show the real noisy training behavior
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from quadrotor_pinn_model_fixed import QuadrotorPINN, QuadrotorDataProcessor, QuadrotorTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load data and setup model
print("Loading data and setting up model...")
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_path = project_root / 'data' / 'quadrotor_training_data.csv'
df = pd.read_csv(data_path)

# Use 5 trajectories
df = df[df['trajectory_id'] < 5].copy()
print(f"Using dataset with {len(df)} samples from 5 trajectories")

# Prepare data
processor = QuadrotorDataProcessor()
X, y = processor.prepare_sequences(df)

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
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Initialize model and trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = QuadrotorPINN(input_size=12, hidden_size=256, output_size=18, num_layers=5)

# Train model to get ACTUAL convergence history
print("\nTraining model to capture ACTUAL parameter evolution...")
trainer = QuadrotorTrainer(model, device)
trainer.train(train_loader, val_loader, epochs=150, physics_weight=5.0)

# Generate ACTUAL convergence plots
print("\nGenerating ACTUAL (not idealized) convergence plots...")

fig, axes = plt.subplots(3, 2, figsize=(15, 18))
fig.suptitle('ACTUAL Parameter Convergence (Real Training Data - NOT Idealized)',
             fontsize=16, fontweight='bold', y=0.995)

# True values
true_values = {
    'm': 0.068,
    'Jxx': 6.86e-5,
    'Jyy': 9.2e-5,
    'Jzz': 1.366e-4,
    'kt': 0.01,
    'kq': 7.8263e-4
}

# Final values
final_values = {
    'm': model.m.item(),
    'Jxx': model.Jxx.item(),
    'Jyy': model.Jyy.item(),
    'Jzz': model.Jzz.item(),
    'kt': model.kt.item(),
    'kq': model.kq.item()
}

# Calculate errors
errors = {
    param: abs(final_values[param] - true_values[param]) / true_values[param] * 100
    for param in true_values.keys()
}

param_info = [
    ('m', 'Mass (kg)', axes[0, 0], 'mass'),
    ('kt', 'Thrust Coefficient kt', axes[0, 1], 'kt'),
    ('kq', 'Torque Coefficient kq', axes[1, 0], 'kq'),
    ('Jxx', 'Inertia Jxx (kg·m²)', axes[1, 1], 'Jxx'),
    ('Jyy', 'Inertia Jyy (kg·m²)', axes[2, 0], 'Jyy'),
    ('Jzz', 'Inertia Jzz (kg·m²)', axes[2, 1], 'Jzz')
]

for param, ylabel, ax, key in param_info:
    epochs = np.arange(len(trainer.param_history[param]))
    history = trainer.param_history[param]

    # Plot actual noisy training curve
    ax.plot(epochs, history, 'b-', linewidth=1.5, alpha=0.7, label='Actual Training')

    # Plot true value
    ax.axhline(true_values[param], color='red', linestyle='--', linewidth=2,
               label=f'True: {true_values[param]:.6g}')

    # Plot final learned value
    ax.axhline(final_values[param], color='green', linestyle=':', linewidth=2,
               label=f'Learned: {final_values[param]:.6g}')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{param} Evolution - Error: {errors[param]:.2f}%',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = project_root / 'visualizations' / 'actual_parameter_convergence.png'
output_path.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")

# Generate individual parameter plots for report
for param, ylabel, _, key in param_info:
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = np.arange(len(trainer.param_history[param]))
    history = trainer.param_history[param]

    ax.plot(epochs, history, 'b-', linewidth=2, alpha=0.8, label='Actual Training Curve')
    ax.axhline(true_values[param], color='red', linestyle='--', linewidth=2.5,
               label=f'True Value: {true_values[param]:.6g}')
    ax.axhline(final_values[param], color='green', linestyle=':', linewidth=2.5,
               label=f'Final Learned: {final_values[param]:.6g}')

    ax.set_xlabel('Training Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')

    if errors[param] < 1.0:
        status = "EXCELLENT"
        color = 'green'
    elif errors[param] < 10.0:
        status = "GOOD"
        color = 'orange'
    else:
        status = "OBSERVABILITY PROBLEM"
        color = 'red'

    ax.set_title(f'ACTUAL {param} Convergence - Error: {errors[param]:.2f}% ({status})',
                fontsize=14, fontweight='bold', color=color)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_file = project_root / 'visualizations' / 'detailed' / f'actual_{param.lower()}_convergence.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

print("\n" + "="*70)
print("ACTUAL PARAMETER CONVERGENCE RESULTS:")
print("="*70)
for param in true_values.keys():
    status = "✓ EXCELLENT" if errors[param] < 1.0 else "⚠ OBSERVABILITY PROBLEM"
    print(f"{param:4s}: True={true_values[param]:.6g}, Learned={final_values[param]:.6g}, "
          f"Error={errors[param]:7.2f}% {status}")
print("="*70)
