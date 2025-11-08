"""Generate comprehensive comparison plot including Optimized v2"""
import matplotlib.pyplot as plt
import numpy as np

# Data from all models (100-step MAE)
models = ['Baseline', 'Optimized\n(Modular)', 'Vanilla\nOptimized', 'Stable\nv1', 'Optimized\nv2']

# 100-step errors
z_errors = [1.49, 17.69, 177.0, 2.63, 0.030]  # m
roll_errors = [0.018, 0.112, 1.180, 0.065, 0.002]  # rad
pitch_errors = [0.003, 0.095, 1.174, 0.022, 0.0007]  # rad
vz_errors = [1.55, 3.60, 35.67, 4.22, 0.064]  # m/s

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('100-Step Autoregressive Prediction: All Models Comparison',
             fontsize=16, fontweight='bold')

colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#2ca02c']
bar_width = 0.6

# Plot 1: Z Position Error (log scale)
ax1 = axes[0, 0]
bars1 = ax1.bar(models, z_errors, color=colors, alpha=0.7, width=bar_width)
ax1.set_ylabel('MAE (m)', fontsize=12, fontweight='bold')
ax1.set_title('Z Position Error', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars1, z_errors)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height * 1.3,
             f'{val:.3f}' if val < 10 else f'{val:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)
ax1.tick_params(axis='x', labelsize=10)
ax1.axhline(y=1.49, color='gray', linestyle='--', alpha=0.5, label='Baseline')

# Plot 2: Roll Angle Error (log scale)
ax2 = axes[0, 1]
bars2 = ax2.bar(models, roll_errors, color=colors, alpha=0.7, width=bar_width)
ax2.set_ylabel('MAE (rad)', fontsize=12, fontweight='bold')
ax2.set_title('Roll Angle Error', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars2, roll_errors)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height * 1.3,
             f'{val:.3f}' if val < 1 else f'{val:.2f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)
ax2.tick_params(axis='x', labelsize=10)
ax2.axhline(y=0.018, color='gray', linestyle='--', alpha=0.5, label='Baseline')

# Plot 3: Pitch Angle Error (log scale)
ax3 = axes[1, 0]
bars3 = ax3.bar(models, pitch_errors, color=colors, alpha=0.7, width=bar_width)
ax3.set_ylabel('MAE (rad)', fontsize=12, fontweight='bold')
ax3.set_title('Pitch Angle Error', fontsize=13, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars3, pitch_errors)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height * 1.3,
             f'{val:.4f}' if val < 0.1 else (f'{val:.3f}' if val < 1 else f'{val:.2f}'),
             ha='center', va='bottom', fontweight='bold', fontsize=10)
ax3.tick_params(axis='x', labelsize=10)
ax3.axhline(y=0.003, color='gray', linestyle='--', alpha=0.5, label='Baseline')

# Plot 4: VZ Velocity Error (log scale)
ax4 = axes[1, 1]
bars4 = ax4.bar(models, vz_errors, color=colors, alpha=0.7, width=bar_width)
ax4.set_ylabel('MAE (m/s)', fontsize=12, fontweight='bold')
ax4.set_title('VZ Velocity Error', fontsize=13, fontweight='bold')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars4, vz_errors)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height * 1.3,
             f'{val:.3f}' if val < 10 else f'{val:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)
ax4.tick_params(axis='x', labelsize=10)
ax4.axhline(y=1.55, color='gray', linestyle='--', alpha=0.5, label='Baseline')

# Add improvement annotations
improvement_z = (1.49 - 0.030) / 1.49 * 100
improvement_vz = (1.55 - 0.064) / 1.55 * 100

fig.text(0.5, 0.02,
         f'Optimized v2 Achievements: Z error reduced by 98.0% (49x better) | '
         f'VZ error reduced by 95.9% (24x better) | Average improvement: 91.4%',
         ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig('../results/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("Comprehensive comparison plot saved to results/comprehensive_comparison.png")

# Multi-horizon error growth plot
fig2, ax = plt.subplots(figsize=(10, 6))

horizons = [1, 10, 50, 100]
baseline_z = [0.087, 0.162, 0.521, 1.49]  # From baseline evaluation
optimized_v2_z = [0.010, 0.024, 0.030, 0.030]  # From optimized v2

ax.plot(horizons, baseline_z, 'o-', linewidth=2.5, markersize=10,
        label='Baseline PINN', color='#1f77b4')
ax.plot(horizons, optimized_v2_z, 's-', linewidth=2.5, markersize=10,
        label='Optimized PINN v2', color='#2ca02c')

ax.set_xlabel('Prediction Horizon (steps)', fontsize=13, fontweight='bold')
ax.set_ylabel('Z Position MAE (m)', fontsize=13, fontweight='bold')
ax.set_title('Multi-Horizon Error Growth: Baseline vs Optimized v2',
             fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='upper left')

# Add annotations
ax.annotate('49x improvement\nat 100 steps',
            xy=(100, 0.030), xytext=(60, 0.15),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'),
            fontsize=11, fontweight='bold', color='green')

ax.annotate('Error plateaus\n(dynamic stability)',
            xy=(50, 0.030), xytext=(20, 0.08),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'),
            fontsize=10, fontweight='bold', color='darkgreen')

plt.tight_layout()
plt.savefig('../results/error_growth_comparison.png', dpi=300, bbox_inches='tight')
print("Error growth comparison plot saved to results/error_growth_comparison.png")

# Summary metrics table
fig3, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Model', 'Z (m)', 'Roll (rad)', 'Pitch (rad)', 'VZ (m/s)', 'Avg Improv'],
    ['Baseline', '1.490', '0.018', '0.003', '1.550', '—'],
    ['Optimized (Modular)', '17.69', '0.112', '0.095', '3.600', '-431%'],
    ['Vanilla Optimized', '177.0', '1.180', '1.174', '35.67', '-7653%'],
    ['Stable v1', '2.630', '0.065', '0.022', '4.220', '-122%'],
    ['Optimized v2', '0.030', '0.002', '0.0007', '0.064', '+91.4%']
]

colors_table = [['lightgray']*6] + \
               [['white']*5 + ['white']] + \
               [['mistyrose']*5 + ['mistyrose']] + \
               [['mistyrose']*5 + ['mistyrose']] + \
               [['mistyrose']*5 + ['mistyrose']] + \
               [['lightgreen']*5 + ['lightgreen']]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 cellColours=colors_table, bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Bold header row
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold', fontsize=12)
    if j == 0:
        cell.set_text_props(weight='bold')
    if i == 5:  # Optimized v2 row
        cell.set_text_props(weight='bold')

plt.title('100-Step Prediction Performance: All Models',
          fontsize=14, fontweight='bold', pad=20)
plt.savefig('../results/performance_table.png', dpi=300, bbox_inches='tight')
print("Performance table saved to results/performance_table.png")

print("\nAll plots updated successfully!")
print("\nFinal Results Summary:")
print("=" * 60)
print("Optimized PINN v2 (250 epochs with all 10 techniques)")
print("=" * 60)
print(f"Z position (100-step):   0.030m (baseline: 1.49m) → 49x better")
print(f"VZ velocity (100-step):  0.064 m/s (baseline: 1.55 m/s) → 24x better")
print(f"Average improvement:     91.4% across all 8 states")
print(f"Error stability:         Plateaus at 50-100 steps (proves stability)")
print("=" * 60)
