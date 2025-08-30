#!/usr/bin/env python3
"""
Generate clean parameter comparison visualization without previous versions
"""

import torch
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.improved_pinn_model import ImprovedQuadrotorPINN

def plot_parameter_comparison():
    """Compare true vs improved model parameters"""
    
    # Load improved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    improved_model = ImprovedQuadrotorPINN(input_size=12, hidden_size=128, output_size=16, num_layers=4)
    improved_model.load_state_dict(torch.load('../models/improved_quadrotor_pinn_model.pth', map_location=device))
    improved_model.eval()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    # True values
    true_params = {
        'Mass': 0.068,
        'Jxx': 6.86e-5,
        'Jyy': 9.2e-5,
        'Jzz': 1.366e-4,
        'Gravity': 9.81
    }
    
    # Improved model parameters
    improved_params = {
        'Mass': improved_model.m.item(),
        'Jxx': improved_model.Jxx.item(),
        'Jyy': improved_model.Jyy.item(),
        'Jzz': improved_model.Jzz.item(),
        'Gravity': improved_model.g.item()
    }
    
    param_names = list(true_params.keys())
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        x = ['True', 'Improved\\nModel']
        y = [true_params[param], improved_params[param]]
        colors = ['blue', 'green']
        
        bars = ax.bar(x, y, color=colors, alpha=0.7)
        ax.set_title(f'{param}')
        ax.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.6f}', ha='center', va='bottom', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        
        # Calculate and show error percentage
        impr_error = abs(improved_params[param] - true_params[param]) / true_params[param] * 100
        
        ax.text(0.5, 0.95, f'Error: {impr_error:.1f}%', 
               transform=ax.transAxes, ha='center', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('parameter_comparison_old_vs_new.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print results
    print("IMPROVED MODEL RESULTS")
    print("=" * 40)
    print("Parameter Accuracy:")
    
    for param in param_names:
        impr_error = abs(improved_params[param] - true_params[param]) / true_params[param] * 100
        print(f"{param:<10}: {impr_error:>6.1f}% error")
    
    return true_params, improved_params

if __name__ == "__main__":
    plot_parameter_comparison()
    print("\\nFile generated: parameter_comparison_old_vs_new.png")