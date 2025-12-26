"""Evaluate the diverse-data trained model on test set"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from pinn_model import QuadrotorPINN

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "quadrotor_pinn_diverse.pth"
SCALER_PATH = PROJECT_ROOT / "models" / "scalers_diverse.pkl"
TEST_DATA = PROJECT_ROOT / "data" / "test_set_diverse.csv"


def load_data(data_path):
    """Load and prepare data from CSV"""
    df = pd.read_csv(data_path)
    df = df.rename(columns={"roll": "phi", "pitch": "theta", "yaw": "psi"})

    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    input_features = state_cols + ["thrust", "torque_x", "torque_y", "torque_z"]

    X, y = [], []
    for traj_id in df["trajectory_id"].unique():
        df_traj = df[df["trajectory_id"] == traj_id].reset_index(drop=True)
        for i in range(len(df_traj) - 1):
            X.append(df_traj.iloc[i][input_features].values)
            y.append(df_traj.iloc[i + 1][state_cols].values)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def main():
    print("=" * 80)
    print("EVALUATING DIVERSE-DATA TRAINED MODEL ON TEST SET")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = QuadrotorPINN(dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load scalers
    print(f"Loading scalers from: {SCALER_PATH}")
    scalers = joblib.load(SCALER_PATH)
    scaler_X, scaler_y = scalers["scaler_X"], scalers["scaler_y"]

    # Load test data
    print(f"Loading test data from: {TEST_DATA}")
    X_test, y_test = load_data(TEST_DATA)
    print(f"  Test samples: {len(X_test):,}")

    # Scale test data
    X_test_scaled = scaler_X.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled)

    # Make predictions
    print("\nGenerating predictions...")
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("TEST SET PERFORMANCE (15 HELD-OUT TRAJECTORIES)")
    print("=" * 80)

    state_names = [
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
    state_labels = [
        "X Position",
        "Y Position",
        "Z Position",
        "Roll",
        "Pitch",
        "Yaw",
        "Roll Rate",
        "Pitch Rate",
        "Yaw Rate",
        "X Vel",
        "Y Vel",
        "Z Vel",
    ]
    units = [
        "m",
        "m",
        "m",
        "rad",
        "rad",
        "rad",
        "rad/s",
        "rad/s",
        "rad/s",
        "m/s",
        "m/s",
        "m/s",
    ]

    print(f"\n{'Variable':<15} {'MAE':<12} {'RMSE':<12} {'Max Error':<12} {'Unit'}")
    print("-" * 65)

    results = {}
    for i, (name, label, unit) in enumerate(zip(state_names, state_labels, units)):
        mae = np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
        rmse = np.sqrt(np.mean((y_test[:, i] - y_pred[:, i]) ** 2))
        max_err = np.max(np.abs(y_test[:, i] - y_pred[:, i]))
        results[name] = {"mae": mae, "rmse": rmse, "max": max_err}
        print(f"{label:<15} {mae:<12.6f} {rmse:<12.6f} {max_err:<12.6f} {unit}")

    # Overall metrics
    overall_mae = np.mean(np.abs(y_test - y_pred))
    overall_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    print("-" * 65)
    print(f"{'OVERALL':<15} {overall_mae:<12.6f} {overall_rmse:<12.6f}")

    # Physics loss
    with torch.no_grad():
        physics_loss = model.physics_loss(X_test_tensor, torch.FloatTensor(y_pred_scaled))
    print(f"\nPhysics Loss: {physics_loss.item():.6f}")

    # Learned parameters
    print(f"\n{'='*80}")
    print("LEARNED PHYSICAL PARAMETERS")
    print(f"{'='*80}")
    print(f"{'Parameter':<10} {'Learned':<15} {'True':<15} {'Error %':<10} {'Status'}")
    print("-" * 70)

    param_status = {}
    for param_name, param in model.params.items():
        learned = param.item()
        true = model.true_params[param_name]
        error = abs(learned - true) / true * 100

        # Check if hitting bounds
        if param_name == "m":
            bounds = (0.0408, 0.0952)
        elif param_name in ["Jxx", "Jyy", "Jzz"]:
            if param_name == "Jxx":
                bounds = (2.74e-5, 1.10e-4)
            elif param_name == "Jyy":
                bounds = (3.68e-5, 1.47e-4)
            else:
                bounds = (5.46e-5, 2.19e-4)
        else:
            bounds = (None, None)

        status = ""
        if bounds[0] and abs(learned - bounds[0]) / bounds[0] < 0.01:
            status = "[AT LOWER BOUND]"
        elif bounds[1] and abs(learned - bounds[1]) / bounds[1] < 0.01:
            status = "[AT UPPER BOUND]"
        elif error < 5:
            status = "[EXCELLENT]"
        elif error < 15:
            status = "[GOOD]"
        else:
            status = "[NEEDS WORK]"

        param_status[param_name] = status
        print(f"{param_name:<10} {learned:<15.6e} {true:<15.6e} {error:<10.2f} {status}")

    # Comparison with previous model
    print(f"\n{'='*80}")
    print("COMPARISON WITH PREVIOUS MODEL (2 test trajectories)")
    print(f"{'='*80}")

    # Previous results from limited data
    prev_results = {
        "x": 1.35,
        "y": 1.85,
        "z": 5.44,
        "phi": 0.045,
        "theta": 0.022,
        "psi": 0.064,
        "p": 0.085,
        "q": 0.048,
        "r": 0.104,
        "vx": 0.49,
        "vy": 0.74,
        "vz": 2.47,
    }

    print(f"{'Variable':<15} {'Previous MAE':<15} {'New MAE':<15} {'Change':<15}")
    print("-" * 65)

    improvements = []
    for name in state_names:
        prev = prev_results.get(name, None)
        new = results[name]["mae"]
        if prev:
            change = (new - prev) / prev * 100
            improvements.append(change)
            change_str = f"{change:+.1f}%"
            if change < -10:
                change_str += " [BETTER]"
            elif change > 10:
                change_str += " [WORSE]"
            print(f"{name:<15} {prev:<15.6f} {new:<15.6f} {change_str}")

    avg_improvement = np.mean(improvements)
    print("-" * 65)
    print(f"Average change: {avg_improvement:+.1f}%")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Test set: 15 held-out trajectories ({len(X_test):,} samples)")
    print(f"Overall MAE: {overall_mae:.6f}")
    print(f"Overall RMSE: {overall_rmse:.6f}")

    # Count parameter status
    excellent = sum(1 for s in param_status.values() if "EXCELLENT" in s)
    good = sum(1 for s in param_status.values() if "GOOD" in s)
    at_bounds = sum(1 for s in param_status.values() if "BOUND" in s)

    print(f"\nParameter learning:")
    print(f"  Excellent (<5% error): {excellent}/6")
    print(f"  Good (<15% error): {good}/6")
    print(f"  At bounds (problematic): {at_bounds}/6")

    if avg_improvement < -5:
        print(f"\n[SUCCESS] Model improved significantly vs previous!")
    elif avg_improvement < 5:
        print(f"\n[NEUTRAL] Model performance similar to previous")
    else:
        print(f"\n[CAUTION] Model degraded vs previous - may need more work")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
