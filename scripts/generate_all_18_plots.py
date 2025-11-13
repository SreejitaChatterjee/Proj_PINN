"""Generate complete time-series plots for all 19 state variables"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from pinn_model import QuadrotorPINN

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'quadrotor_pinn.pth'
DATA_PATH = PROJECT_ROOT / 'data' / 'quadrotor_training_data.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'detailed'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable definitions with proper names and units
VARIABLES = [
    ('x', 'X Position', 'm', 'position'),
    ('y', 'Y Position', 'm', 'position'),
    ('z', 'Altitude (Z Position)', 'm', 'position'),
    ('roll', 'Roll Angle (φ)', 'rad', 'angle'),
    ('pitch', 'Pitch Angle (θ)', 'rad', 'angle'),
    ('yaw', 'Yaw Angle (ψ)', 'rad', 'angle'),
    ('p', 'Roll Rate', 'rad/s', 'rate'),
    ('q', 'Pitch Rate', 'rad/s', 'rate'),
    ('r', 'Yaw Rate', 'rad/s', 'rate'),
    ('vx', 'X Velocity', 'm/s', 'velocity'),
    ('vy', 'Y Velocity', 'm/s', 'velocity'),
    ('vz', 'Z Velocity (Vertical)', 'm/s', 'velocity'),
    ('p_dot', 'Roll Acceleration', 'rad/s²', 'acceleration'),
    ('q_dot', 'Pitch Acceleration', 'rad/s²', 'acceleration'),
    ('r_dot', 'Yaw Acceleration', 'rad/s²', 'acceleration'),
    ('thrust', 'Total Thrust', 'N', 'control'),
    ('torque_x', 'Roll Torque (τ_x)', 'N·m', 'control'),
    ('torque_y', 'Pitch Torque (τ_y)', 'N·m', 'control'),
    ('torque_z', 'Yaw Torque (τ_z)', 'N·m', 'control'),
]

def plot_variable(ax, time, true_vals, pred_vals, var_name, var_label, var_unit, mae, rmse):
    """Plot a single variable with true vs predicted"""
    ax.plot(time, true_vals, 'b-', linewidth=1.5, label='True', alpha=0.8)
    ax.plot(time, pred_vals, 'r--', linewidth=1.2, label='Predicted', alpha=0.8)
    ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{var_label} ({var_unit})', fontsize=11, fontweight='bold')
    ax.set_title(f'{var_label} vs Time\nMAE: {mae:.4f} {var_unit}, RMSE: {rmse:.4f} {var_unit}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

def main():
    print("="*80)
    print("GENERATING ALL 19 TIME-SERIES PLOTS")
    print("="*80)

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = QuadrotorPINN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load scalers
    scaler_path = MODEL_PATH.parent / 'scalers.pkl'
    print(f"Loading scalers from: {scaler_path}")
    scalers = joblib.load(scaler_path)
    scaler_X, scaler_y = scalers['scaler_X'], scalers['scaler_y']

    # Load data
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Map column names
    df = df.rename(columns={'roll': 'phi', 'pitch': 'theta', 'yaw': 'psi'})

    # Model predicts 12 core states: x, y, z, phi, theta, psi, p, q, r, vx, vy, vz
    predicted_states = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    input_features = predicted_states + ['thrust', 'torque_x', 'torque_y', 'torque_z']

    # Generate predictions for all data
    print(f"\nGenerating predictions for {len(df)-1} timesteps...")
    predictions = []
    with torch.no_grad():
        for idx in range(len(df) - 1):
            input_data = df.iloc[idx][input_features].values
            input_scaled = scaler_X.transform(input_data.reshape(1, -1))
            pred_scaled = model(torch.FloatTensor(input_scaled)).squeeze(0)[:12].numpy()
            pred = scaler_y.inverse_transform(pred_scaled.reshape(1, -1)).flatten()
            predictions.append(pred)

    # Create predictions DataFrame
    df_pred = pd.DataFrame(predictions, columns=predicted_states)
    df_pred['timestamp'] = df['timestamp'].iloc[1:].values

    # For control inputs and derivatives, use ground truth (these are not predicted by model)
    for col in ['thrust', 'torque_x', 'torque_y', 'torque_z', 'p_dot', 'q_dot', 'r_dot']:
        if col in df.columns:
            df_pred[col] = df[col].iloc[1:].values

    # Rename back for plotting
    df = df.rename(columns={'phi': 'roll', 'theta': 'pitch', 'psi': 'yaw'})
    df_pred = df_pred.rename(columns={'phi': 'roll', 'theta': 'pitch', 'psi': 'yaw'})

    print(f"\nGenerating individual plots...")
    # Generate plots for all 19 variables
    plot_num = 1
    for var_name, var_label, var_unit, var_type in VARIABLES:
        if var_name not in df.columns:
            print(f"  WARNING: Variable '{var_name}' not found in data, skipping...")
            continue

        print(f"  [{plot_num:02d}/19] Generating plot for {var_name}...")

        # Get data
        time = df['timestamp'].iloc[1:].values
        true_vals = df[var_name].iloc[1:].values

        # For predicted states, use model predictions; for others, use ground truth
        # Current model predicts 12 states: x, y, z, roll, pitch, yaw, p, q, r, vx, vy, vz
        if var_name in predicted_states or var_name in ['roll', 'pitch', 'yaw']:
            pred_vals = df_pred[var_name].values
            is_predicted = True
        else:
            # Control inputs and derivatives are measurements/inputs, not predicted
            pred_vals = df_pred[var_name].values  # Ground truth
            is_predicted = False

        # Calculate errors
        mae = np.mean(np.abs(true_vals - pred_vals)) if is_predicted else 0.0
        rmse = np.sqrt(np.mean((true_vals - pred_vals)**2)) if is_predicted else 0.0

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_variable(ax, time, true_vals, pred_vals, var_name, var_label, var_unit, mae, rmse)

        # Add note for non-predicted variables
        if not is_predicted:
            if var_type == 'control':
                note_text = 'NOTE: Control input (not predicted by PINN)'
            elif var_type in ['acceleration', 'velocity'] and var_name in ['vx', 'vy', 'p_dot', 'q_dot', 'r_dot']:
                note_text = 'NOTE: Measured quantity used for physics constraints (not predicted)'
            elif var_type == 'position' and var_name in ['x', 'y']:
                note_text = 'NOTE: Computed from velocity integration (not currently predicted by PINN)'
            else:
                note_text = 'NOTE: This is a measurement, not a predicted state'
            ax.text(0.02, 0.98, note_text,
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        plt.tight_layout()

        # Save plot
        output_file = OUTPUT_DIR / f'{plot_num:02d}_{var_name}_time_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"      Saved: {output_file}")
        if is_predicted:
            print(f"      MAE: {mae:.6f} {var_unit}, RMSE: {rmse:.6f} {var_unit}")

        plot_num += 1

    print(f"\n{'='*80}")
    print(f"SUCCESS: Generated {plot_num-1} plots in {OUTPUT_DIR}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
