"""Generate separate plots for each of the 10 training trajectories"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pinn_model import QuadrotorPINN

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "quadrotor_pinn.pth"
DATA_PATH = PROJECT_ROOT / "data" / "quadrotor_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "individual_trajectories"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable definitions with proper names and units
VARIABLES = [
    ("x", "X Position", "m", "position"),
    ("y", "Y Position", "m", "position"),
    ("z", "Altitude (Z Position)", "m", "position"),
    ("roll", "Roll Angle (φ)", "rad", "angle"),
    ("pitch", "Pitch Angle (θ)", "rad", "angle"),
    ("yaw", "Yaw Angle (ψ)", "rad", "angle"),
    ("p", "Roll Rate", "rad/s", "rate"),
    ("q", "Pitch Rate", "rad/s", "rate"),
    ("r", "Yaw Rate", "rad/s", "rate"),
    ("vx", "X Velocity", "m/s", "velocity"),
    ("vy", "Y Velocity", "m/s", "velocity"),
    ("vz", "Z Velocity (Vertical)", "m/s", "velocity"),
    ("thrust", "Total Thrust", "N", "control"),
    ("torque_x", "Roll Torque (τ_x)", "N·m", "control"),
    ("torque_y", "Pitch Torque (τ_y)", "N·m", "control"),
    ("torque_z", "Yaw Torque (τ_z)", "N·m", "control"),
]


def plot_single_trajectory_variable(
    ax,
    time,
    true_vals,
    pred_vals,
    var_name,
    var_label,
    var_unit,
    trajectory_id,
    mae,
    rmse,
    is_predicted,
):
    """Plot a single variable for one trajectory"""
    ax.plot(time, true_vals, "b-", linewidth=1.5, label="True", alpha=0.8)
    if is_predicted:
        ax.plot(time, pred_vals, "r--", linewidth=1.2, label="Predicted", alpha=0.8)
    ax.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{var_label} ({var_unit})", fontsize=11, fontweight="bold")

    title = f"{var_label} vs Time - Trajectory {trajectory_id}"
    if is_predicted:
        title += f"\nMAE: {mae:.4f} {var_unit}, RMSE: {rmse:.4f} {var_unit}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    # Add note for non-predicted variables
    if not is_predicted:
        note_text = "NOTE: Control input (not predicted by PINN)"
        ax.text(
            0.02,
            0.98,
            note_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
        )


def generate_predictions_for_trajectory(
    model, df_traj, scaler_X, scaler_y, predicted_states, input_features
):
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
    df_pred["timestamp"] = df_traj["timestamp"].iloc[1:].values

    # For control inputs, use ground truth
    for col in ["thrust", "torque_x", "torque_y", "torque_z"]:
        if col in df_traj.columns:
            df_pred[col] = df_traj[col].iloc[1:].values

    return df_pred


def main():
    print("=" * 80)
    print("GENERATING INDIVIDUAL TRAJECTORY PLOTS (10 Trajectories × 16 Variables)")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = QuadrotorPINN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load scalers
    scaler_path = MODEL_PATH.parent / "scalers.pkl"
    print(f"Loading scalers from: {scaler_path}")
    scalers = joblib.load(scaler_path)
    scaler_X, scaler_y = scalers["scaler_X"], scalers["scaler_y"]

    # Load data
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Map column names
    df = df.rename(columns={"roll": "phi", "pitch": "theta", "yaw": "psi"})

    # Model predicts 12 core states
    predicted_states = [
        "x",
        "y",
        "z",
        "phi",
        "theta",
        "psi",
        "p",
        "q",
        "r",
        "vx",
        "vy",
        "vz",
    ]
    input_features = predicted_states + ["thrust", "torque_x", "torque_y", "torque_z"]

    # Get unique trajectory IDs
    trajectory_ids = sorted(df["trajectory_id"].unique())
    print(f"\nFound {len(trajectory_ids)} trajectories: {trajectory_ids}")

    # Process each trajectory
    total_plots = 0
    for traj_id in trajectory_ids:
        print(f"\n{'='*80}")
        print(f"Processing Trajectory {traj_id}")
        print(f"{'='*80}")

        # Extract trajectory data
        df_traj = df[df["trajectory_id"] == traj_id].copy()
        df_traj = df_traj.reset_index(drop=True)
        print(f"  Trajectory {traj_id}: {len(df_traj)} timesteps")

        # Generate predictions for this trajectory
        print(f"  Generating predictions...")
        df_pred = generate_predictions_for_trajectory(
            model, df_traj, scaler_X, scaler_y, predicted_states, input_features
        )

        # Rename back for plotting
        df_traj = df_traj.rename(columns={"phi": "roll", "theta": "pitch", "psi": "yaw"})
        df_pred = df_pred.rename(columns={"phi": "roll", "theta": "pitch", "psi": "yaw"})

        # Create output directory for this trajectory
        traj_output_dir = OUTPUT_DIR / f"trajectory_{traj_id}"
        traj_output_dir.mkdir(exist_ok=True)

        # Generate plots for all 16 variables
        plot_num = 1
        for var_name, var_label, var_unit, var_type in VARIABLES:
            if var_name not in df_traj.columns:
                print(f"    WARNING: Variable '{var_name}' not found, skipping...")
                continue

            # Get data
            time = df_traj["timestamp"].iloc[1:].values
            true_vals = df_traj[var_name].iloc[1:].values

            # Check if this variable is predicted
            if var_name in predicted_states or var_name in ["roll", "pitch", "yaw"]:
                pred_vals = df_pred[var_name].values
                is_predicted = True
            else:
                pred_vals = df_pred[var_name].values  # Ground truth for control inputs
                is_predicted = False

            # Calculate errors
            mae = np.mean(np.abs(true_vals - pred_vals)) if is_predicted else 0.0
            rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2)) if is_predicted else 0.0

            # Create figure
            fig, ax = plt.subplots(figsize=(14, 6))
            plot_single_trajectory_variable(
                ax,
                time,
                true_vals,
                pred_vals,
                var_name,
                var_label,
                var_unit,
                traj_id,
                mae,
                rmse,
                is_predicted,
            )

            plt.tight_layout()

            # Save plot
            output_file = traj_output_dir / f"{plot_num:02d}_{var_name}_trajectory_{traj_id}.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

            if plot_num == 1:  # Only print first one to reduce clutter
                print(f"    Plotting variables...")

            plot_num += 1
            total_plots += 1

        print(f"  [OK] Generated {plot_num-1} plots for Trajectory {traj_id}")
        print(f"    Saved to: {traj_output_dir}")

    print(f"\n{'='*80}")
    print(f"SUCCESS: Generated {total_plots} total plots")
    print(f"  • {len(trajectory_ids)} trajectories")
    print(f"  • {len(VARIABLES)} variables per trajectory")
    print(f"  • Output directory: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
