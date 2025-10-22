"""
Generate Motor Coefficient Convergence Plots

This script generates the missing Figures 17-18 showing kt and kq convergence
during PINN training, addressing reviewer Issue #3.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_parameter_convergence(history, param_name, true_value, title, save_path):
    """
    Plot parameter convergence over training epochs

    Args:
        history: Training history dictionary
        param_name: Name of parameter ('kt', 'kq', 'mass', etc.)
        true_value: Ground truth value
        title: Plot title
        save_path: Path to save the figure
    """
    epochs = history['epoch']
    param_values = history[param_name]

    # Convert to numpy if tensors
    if isinstance(param_values[0], torch.Tensor):
        param_values = [p.item() for p in param_values]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot learned parameter evolution
    ax.plot(epochs, param_values, 'b-', linewidth=2, label=f'Learned {param_name}')

    # Plot true value as horizontal line
    ax.axhline(y=true_value, color='r', linestyle='--', linewidth=2, label=f'True {param_name}')

    # Calculate final error
    final_value = param_values[-1]
    error_pct = abs((final_value - true_value) / true_value) * 100

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(f'{param_name}', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add text box with final values
    textstr = f'True: {true_value:.6f}\nLearned: {final_value:.6f}\nError: {error_pct:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_all_parameters_grid(history, true_values, save_path):
    """
    Create a 2x3 grid showing all 6 parameters

    Args:
        history: Training history dictionary
        true_values: Dictionary of true parameter values
        save_path: Path to save the comprehensive figure
    """
    params = ['mass', 'Jxx', 'Jyy', 'Jzz', 'kt', 'kq']
    param_labels = ['Mass (kg)', 'Jxx (kg·m²)', 'Jyy (kg·m²)', 'Jzz (kg·m²)',
                    'kt (thrust coeff)', 'kq (torque coeff)']

    epochs = history['epoch']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('PINN Parameter Convergence During Training (Mixed Dataset)',
                 fontsize=16, fontweight='bold')

    for idx, (param, label) in enumerate(zip(params, param_labels)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Get parameter values
        param_values = history[param]
        if isinstance(param_values[0], torch.Tensor):
            param_values = [p.item() for p in param_values]

        true_value = true_values[param]

        # Plot
        ax.plot(epochs, param_values, 'b-', linewidth=2, label='Learned')
        ax.axhline(y=true_value, color='r', linestyle='--', linewidth=2, label='True')

        # Calculate error
        final_value = param_values[-1]
        error_pct = abs((final_value - true_value) / true_value) * 100

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f'{param} (Error: {error_pct:.2f}%)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        # Add final value annotation
        ax.annotate(f'{final_value:.6f}',
                   xy=(epochs[-1], final_value),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, color='blue',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved comprehensive grid: {save_path}")
    plt.close()

def main():
    print("="*80)
    print("GENERATING MOTOR COEFFICIENT CONVERGENCE PLOTS")
    print("="*80)

    # Load model with training history
    model_path = Path("../models/pinn_model_improved_mixed.pth")
    print(f"\nLoading model: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    history = checkpoint['training_history']

    print(f"  Training epochs: {len(history['epoch'])}")
    print(f"  Parameters tracked: {[k for k in history.keys() if k not in ['epoch', 'train_loss', 'physics_loss', 'reg_loss']]}")

    # True parameter values (from simulation)
    true_values = {
        'mass': 0.068,           # kg
        'Jxx': 6.86e-5,          # kg·m²
        'Jyy': 9.20e-5,          # kg·m²
        'Jzz': 1.366e-4,         # kg·m²
        'kt': 0.01,              # thrust coefficient
        'kq': 7.826e-4           # torque coefficient
    }

    # Create output directory
    output_dir = Path("../visualizations/detailed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING INDIVIDUAL PARAMETER PLOTS")
    print("="*80)

    # Generate individual plots for motor coefficients (missing Figures 17-18)
    print("\nMotor Coefficients:")
    plot_parameter_convergence(
        history, 'kt', true_values['kt'],
        'Thrust Coefficient (kt) Convergence During Training',
        output_dir / 'kt_convergence.png'
    )

    plot_parameter_convergence(
        history, 'kq', true_values['kq'],
        'Torque Coefficient (kq) Convergence During Training',
        output_dir / 'kq_convergence.png'
    )

    # Generate plots for all other parameters for completeness
    print("\nPhysical Parameters:")
    plot_parameter_convergence(
        history, 'mass', true_values['mass'],
        'Mass Convergence During Training',
        output_dir / 'mass_convergence.png'
    )

    plot_parameter_convergence(
        history, 'Jxx', true_values['Jxx'],
        'Moment of Inertia Jxx Convergence During Training',
        output_dir / 'Jxx_convergence.png'
    )

    plot_parameter_convergence(
        history, 'Jyy', true_values['Jyy'],
        'Moment of Inertia Jyy Convergence During Training',
        output_dir / 'Jyy_convergence.png'
    )

    plot_parameter_convergence(
        history, 'Jzz', true_values['Jzz'],
        'Moment of Inertia Jzz Convergence During Training',
        output_dir / 'Jzz_convergence.png'
    )

    # Generate comprehensive grid plot
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE GRID PLOT")
    print("="*80)
    plot_all_parameters_grid(
        history, true_values,
        output_dir / 'all_parameters_convergence_grid.png'
    )

    # Print final summary
    print("\n" + "="*80)
    print("FINAL PARAMETER VALUES AND ERRORS")
    print("="*80)

    print(f"\n{'Parameter':<12} {'True':<12} {'Learned':<12} {'Error (%)':<10}")
    print("-" * 50)

    for param in ['mass', 'Jxx', 'Jyy', 'Jzz', 'kt', 'kq']:
        param_values = history[param]
        if isinstance(param_values[-1], torch.Tensor):
            final_value = param_values[-1].item()
        else:
            final_value = param_values[-1]

        true_value = true_values[param]
        error_pct = abs((final_value - true_value) / true_value) * 100

        print(f"{param:<12} {true_value:<12.6e} {final_value:<12.6e} {error_pct:<10.2f}")

    print("\n" + "="*80)
    print("COMPLETE - All motor coefficient plots generated!")
    print("="*80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - kt_convergence.png (Figure 17 for report)")
    print("  - kq_convergence.png (Figure 18 for report)")
    print("  - all_parameters_convergence_grid.png (comprehensive overview)")
    print("  - Individual plots for mass, Jxx, Jyy, Jzz")

if __name__ == "__main__":
    main()
