"""Generate comparative plots showing DIVERSE TEST SET trajectories (15 held-out trajectories)"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from pinn_model import QuadrotorPINN

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'quadrotor_pinn_diverse.pth'
TEST_DATA_PATH = PROJECT_ROOT / 'data' / 'test_set_diverse.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'test_set_trajectories_diverse'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable definitions with proper names and units
VARIABLES = [
    ('x', 'X Position', 'm', 'position'),
    ('y', 'Y Position', 'm', 'position'),
    ('z', 'Altitude (Z Position)', 'm', 'position'),
    ('roll', 'Roll Angle (phi)', 'rad', 'angle'),
    ('pitch', 'Pitch Angle (theta)', 'rad', 'angle'),
    ('yaw', 'Yaw Angle (psi)', 'rad', 'angle'),
    ('p', 'Roll Rate', 'rad/s', 'rate'),
    ('q', 'Pitch Rate', 'rad/s', 'rate'),
    ('r', 'Yaw Rate', 'rad/s', 'rate'),
    ('vx', 'X Velocity', 'm/s', 'velocity'),
    ('vy', 'Y Velocity', 'm/s', 'velocity'),
    ('vz', 'Z Velocity (Vertical)', 'm/s', 'velocity'),
    ('thrust', 'Total Thrust', 'N', 'control'),
    ('torque_x', 'Roll Torque (tau_x)', 'Nm', 'control'),
    ('torque_y', 'Pitch Torque (tau_y)', 'Nm', 'control'),
    ('torque_z', 'Yaw Torque (tau_z)', 'Nm', 'control'),
]

def generate_predictions_for_trajectory(model, df_traj, scaler_X, scaler_y, predicted_states, input_features):
    """Generate predictions for a single trajectory"""
    predictions = []
    with torch.no_grad():
        for idx in range(len(df_traj) - 1):
            input_data = df_traj.iloc[idx][input_features].values
            input_scaled = scaler_X.transform(input_data.reshape(1, -1))
            pred_scaled = model(torch.FloatTensor(input_scaled)).squeeze(0)[:12].numpy()
            pred = scaler_y.inverse_transform(pred_scaled.reshape(1, -1)).flatten()
            predictions.append(pred)

    # Create predictions DataFrame
    df_pred = pd.DataFrame(predictions, columns=predicted_states)
    df_pred['timestamp'] = df_traj['timestamp'].iloc[1:].values

    # For control inputs, use ground truth
    for col in ['thrust', 'torque_x', 'torque_y', 'torque_z']:
        if col in df_traj.columns:
            df_pred[col] = df_traj[col].iloc[1:].values

    return df_pred

def main():
    print("="*80)
    print("GENERATING DIVERSE TEST SET TRAJECTORY PLOTS")
    print("(15 HELD-OUT TRAJECTORIES - NOT SEEN DURING TRAINING)")
    print("="*80)

    # Check if test data exists
    if not TEST_DATA_PATH.exists():
        print(f"\n[ERROR] Test data not found at: {TEST_DATA_PATH}")
        print("Please run 'split_diverse_data.py' first to create train/test split")
        return

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = QuadrotorPINN(dropout=0.3)  # Match training dropout
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load scalers
    scaler_path = MODEL_PATH.parent / 'scalers_diverse.pkl'
    print(f"Loading scalers from: {scaler_path}")
    scalers = joblib.load(scaler_path)
    scaler_X, scaler_y = scalers['scaler_X'], scalers['scaler_y']

    # Load TEST data only
    print(f"Loading TEST data from: {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH)
    print(f"  Test samples: {len(df)}")

    # Map column names
    df = df.rename(columns={'roll': 'phi', 'pitch': 'theta', 'yaw': 'psi'})

    # Model predicts 12 core states
    predicted_states = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    input_features = predicted_states + ['thrust', 'torque_x', 'torque_y', 'torque_z']

    # Get unique trajectory IDs in test set
    trajectory_ids = sorted(df['trajectory_id'].unique())
    print(f"  Test trajectories: {len(trajectory_ids)}")
    print(f"  Trajectory IDs (first 5): {trajectory_ids[:5]}")

    # Sample a subset of trajectories for visualization (first 10 out of 15)
    sample_traj_ids = trajectory_ids[:10]
    print(f"  Plotting first 10 trajectories for visualization")

    # Pre-process sampled trajectories
    print("\nPre-processing test trajectories...")
    trajectory_data = {}
    for traj_id in sample_traj_ids:
        df_traj = df[df['trajectory_id'] == traj_id].copy()
        df_traj = df_traj.reset_index(drop=True)

        if len(df_traj) < 2:
            print(f"  [SKIP] Trajectory {traj_id}: Only {len(df_traj)} timesteps - skipping")
            continue

        # Generate predictions
        df_pred = generate_predictions_for_trajectory(
            model, df_traj, scaler_X, scaler_y, predicted_states, input_features
        )

        # Rename back for plotting
        df_traj = df_traj.rename(columns={'phi': 'roll', 'theta': 'pitch', 'psi': 'yaw'})
        df_pred = df_pred.rename(columns={'phi': 'roll', 'theta': 'pitch', 'psi': 'yaw'})

        trajectory_data[traj_id] = {
            'true': df_traj,
            'pred': df_pred
        }
        print(f"  [OK] Trajectory {traj_id}: {len(df_traj)} timesteps")

    if len(trajectory_data) == 0:
        print("\n[ERROR] No valid test trajectories found!")
        return

    # Determine grid size based on number of trajectories
    n_traj = len(trajectory_data)
    if n_traj <= 4:
        nrows, ncols = 2, 2
        figsize = (12, 12)
    elif n_traj <= 6:
        nrows, ncols = 2, 3
        figsize = (15, 10)
    elif n_traj <= 9:
        nrows, ncols = 3, 3
        figsize = (15, 15)
    else:
        nrows, ncols = 5, 2
        figsize = (16, 20)

    # Generate comparative plots for each variable
    print(f"\nGenerating test set plots ({nrows}x{ncols} grid)...")
    plot_num = 1
    overall_metrics = {}

    for var_name, var_label, var_unit, var_type in VARIABLES:
        print(f"  [{plot_num:02d}/16] Plotting {var_name}...")

        # Create figure with subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Check if variable is predicted
        is_predicted = var_name in predicted_states or var_name in ['roll', 'pitch', 'yaw']

        # Collect all errors for overall statistics
        all_errors = []

        # Plot each trajectory
        for idx, traj_id in enumerate(trajectory_data.keys()):
            if idx >= len(axes):
                break
            ax = axes[idx]
            df_traj = trajectory_data[traj_id]['true']
            df_pred = trajectory_data[traj_id]['pred']

            if var_name not in df_traj.columns:
                ax.text(0.5, 0.5, f'Variable {var_name}\nnot found',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Trajectory {traj_id}', fontsize=10, fontweight='bold')
                continue

            # Get data
            time = df_traj['timestamp'].iloc[1:].values
            true_vals = df_traj[var_name].iloc[1:].values
            pred_vals = df_pred[var_name].values

            # Plot with INCREASED LINE WIDTH for visibility
            ax.plot(time, true_vals, 'b-', linewidth=2.5, label='Ground Truth', alpha=0.9)
            if is_predicted:
                ax.plot(time, pred_vals, 'r--', linewidth=2.0, label='PINN Prediction', alpha=0.8)

            # Calculate errors
            if is_predicted:
                mae = np.mean(np.abs(true_vals - pred_vals))
                rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
                all_errors.append(mae)
                title = f'Test Traj {traj_id}\nMAE: {mae:.4f}, RMSE: {rmse:.4f}'
            else:
                title = f'Test Traj {traj_id}\n(Control Input)'

            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel(f'{var_unit}', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # Add legend to all subplots for clarity
            if is_predicted:
                ax.legend(fontsize=7, loc='best')

        # Hide unused subplots
        for idx in range(len(trajectory_data), len(axes)):
            axes[idx].axis('off')

        # Overall title with test set indicator
        if is_predicted and len(all_errors) > 0:
            overall_mae = np.mean(all_errors)
            overall_metrics[var_name] = overall_mae
            title = f'DIVERSE TEST SET (15 Held-Out Trajectories): {var_label} - Overall MAE: {overall_mae:.4f} {var_unit}'
        else:
            title = f'DIVERSE TEST SET (15 Held-Out Trajectories): {var_label}'

        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.tight_layout(rect=[0, 0, 1, 0.993])

        # Save plot
        output_file = OUTPUT_DIR / f'{plot_num:02d}_{var_name}_diverse_test.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"      Saved: {output_file.name}")
        plot_num += 1

    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"DIVERSE TEST SET PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Number of test trajectories plotted: {len(trajectory_data)}")
    print(f"Total test trajectories: {len(trajectory_ids)}")
    print(f"Test trajectory IDs (plotted): {list(trajectory_data.keys())}")
    print(f"\nOverall MAE per variable (from plotted trajectories):")
    for var_name, mae in overall_metrics.items():
        var_label = next(v[1] for v in VARIABLES if v[0] == var_name)
        var_unit = next(v[2] for v in VARIABLES if v[0] == var_name)
        print(f"  {var_label:30s}: {mae:.6f} {var_unit}")

    print(f"\n{'='*80}")
    print(f"SUCCESS: Generated {plot_num-1} diverse test set plots")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  [NOTE] These are 15 HELD-OUT test trajectories from 100 diverse maneuvers")
    print(f"  Model trained on 70 trajectories, validated on 15, tested on 15")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
