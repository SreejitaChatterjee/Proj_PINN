"""
Generate PINN architecture diagram for paper.

Shows:
1. Input layer (12 states + 4 controls)
2. Hidden layers (5 x 256)
3. Output layer (12 state changes)
4. Physics loss computation
5. Total loss = prediction loss + physics loss
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10))

# Colors
color_input = '#E8F4F8'
color_hidden = '#B8E6F0'
color_output = '#88D8E8'
color_physics = '#FFE6CC'
color_loss = '#FFCCCC'

# Layer positions
x_input = 1
x_hidden = np.linspace(3, 9, 5)
x_output = 11
x_physics = 13
x_loss = 15

y_center = 5

# Input layer
input_box = FancyBboxPatch((x_input-0.4, y_center-1.5), 0.8, 3,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color_input,
                          linewidth=2)
ax.add_patch(input_box)
ax.text(x_input, y_center+0.8, 'Input', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(x_input, y_center+0.3, '12 states', ha='center', va='center', fontsize=10)
ax.text(x_input, y_center, '(x, y, z, phi,', ha='center', va='center', fontsize=9)
ax.text(x_input, y_center-0.3, 'theta, psi,', ha='center', va='center', fontsize=9)
ax.text(x_input, y_center-0.6, 'p, q, r,', ha='center', va='center', fontsize=9)
ax.text(x_input, y_center-0.9, 'vx, vy, vz)', ha='center', va='center', fontsize=9)
ax.text(x_input, y_center-1.2, '+ 4 controls', ha='center', va='center', fontsize=10)

# Hidden layers
for i, x in enumerate(x_hidden):
    hidden_box = FancyBboxPatch((x-0.3, y_center-1.5), 0.6, 3,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=color_hidden,
                                linewidth=2)
    ax.add_patch(hidden_box)
    ax.text(x, y_center+0.5, f'Hidden {i+1}', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(x, y_center, '256 units', ha='center', va='center', fontsize=10)
    ax.text(x, y_center-0.5, 'tanh', ha='center', va='center', fontsize=9, style='italic')
    ax.text(x, y_center-1, 'dropout=0.1', ha='center', va='center', fontsize=8)

# Output layer
output_box = FancyBboxPatch((x_output-0.4, y_center-1.5), 0.8, 3,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor=color_output,
                           linewidth=2)
ax.add_patch(output_box)
ax.text(x_output, y_center+0.5, 'Output', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(x_output, y_center, '12 states', ha='center', va='center', fontsize=10)
ax.text(x_output, y_center-0.5, 'Predicted', ha='center', va='center', fontsize=10)
ax.text(x_output, y_center-1, 'next state', ha='center', va='center', fontsize=10)

# Arrows between layers
arrow_props = dict(arrowstyle='->', lw=2, color='black')

# Input to first hidden
ax.annotate('', xy=(x_hidden[0]-0.3, y_center), xytext=(x_input+0.4, y_center),
            arrowprops=arrow_props)

# Between hidden layers
for i in range(len(x_hidden)-1):
    ax.annotate('', xy=(x_hidden[i+1]-0.3, y_center), xytext=(x_hidden[i]+0.3, y_center),
                arrowprops=arrow_props)

# Last hidden to output
ax.annotate('', xy=(x_output-0.4, y_center), xytext=(x_hidden[-1]+0.3, y_center),
            arrowprops=arrow_props)

# Physics loss computation
physics_box = FancyBboxPatch((x_physics-0.5, y_center+2), 1, 2,
                            boxstyle="round,pad=0.1",
                            edgecolor='orange', facecolor=color_physics,
                            linewidth=2, linestyle='--')
ax.add_patch(physics_box)
ax.text(x_physics, y_center+3.5, 'Physics Loss', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(x_physics, y_center+3, 'Newton-Euler', ha='center', va='center', fontsize=9)
ax.text(x_physics, y_center+2.6, 'Equations', ha='center', va='center', fontsize=9)

# Arrow from output to physics
ax.annotate('', xy=(x_physics-0.3, y_center+2.2), xytext=(x_output+0.2, y_center+1),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='orange', linestyle='--'))

# Prediction loss computation
pred_loss_box = FancyBboxPatch((x_physics-0.5, y_center-2.5), 1, 1.5,
                              boxstyle="round,pad=0.1",
                              edgecolor='red', facecolor=color_loss,
                              linewidth=2)
ax.add_patch(pred_loss_box)
ax.text(x_physics, y_center-1.3, 'Prediction', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(x_physics, y_center-1.8, 'Loss (MSE)', ha='center', va='center', fontsize=9)

# Arrow from output to prediction loss
ax.annotate('', xy=(x_physics-0.3, y_center-1.2), xytext=(x_output+0.2, y_center-0.8),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))

# Total loss
total_loss_box = FancyBboxPatch((x_loss-0.5, y_center-0.75), 1, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='darkred', facecolor='#FFB3B3',
                               linewidth=3)
ax.add_patch(total_loss_box)
ax.text(x_loss, y_center+0.3, 'Total Loss', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(x_loss, y_center-0.2, 'L = L_pred', ha='center', va='center', fontsize=10)
ax.text(x_loss, y_center-0.5, '+ w * L_phys', ha='center', va='center', fontsize=10)

# Arrows to total loss
ax.annotate('', xy=(x_loss-0.5, y_center+0.2), xytext=(x_physics+0.5, y_center+2.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='darkred', linestyle='--'))
ax.annotate('', xy=(x_loss-0.5, y_center-0.2), xytext=(x_physics+0.5, y_center-1.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='darkred'))

# Model architecture annotation
ax.text(6, y_center-3, 'Model Architecture', ha='center', va='center',
        fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.text(6, y_center-3.6, '204,818 parameters | 0.79 MB | 0.34 ms inference',
        ha='center', va='center', fontsize=10)

# Training variants annotation
variant_y = y_center+5
ax.text(6, variant_y, 'Training Variants', ha='center', va='center',
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(3, variant_y-0.6, 'w=0 (Pure Data-Driven)', ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.text(3, variant_y-1.1, 'Best for fault detection', ha='center', va='center', fontsize=8, style='italic')
ax.text(9, variant_y-0.6, 'w=20 (Physics-Informed)', ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
ax.text(9, variant_y-1.1, 'Physics constraints hurt', ha='center', va='center', fontsize=8, style='italic')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input/Output'),
    mpatches.Patch(facecolor=color_hidden, edgecolor='black', label='Hidden Layers'),
    mpatches.Patch(facecolor=color_physics, edgecolor='orange', label='Physics Loss (optional)'),
    mpatches.Patch(facecolor=color_loss, edgecolor='red', label='Prediction Loss'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Title
ax.text(8, y_center+7, 'Physics-Informed Neural Network for UAV Fault Detection',
        ha='center', va='center', fontsize=16, fontweight='bold')

# Clean up axes
ax.set_xlim(0, 17)
ax.set_ylim(-4, 8)
ax.axis('off')

plt.tight_layout()
plt.savefig('research/security/figures/pinn_architecture.png', dpi=300, bbox_inches='tight')
plt.savefig('research/security/figures/pinn_architecture.pdf', bbox_inches='tight')
print("Architecture diagram saved: pinn_architecture.png/pdf")
plt.close()

print("PINN architecture diagram created successfully!")
