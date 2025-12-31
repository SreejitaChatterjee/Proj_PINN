"""Evaluate the retrained model on the held-out test set"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from pinn_model import QuadrotorPINN

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "quadrotor_pinn_fixed.pth"
SCALER_PATH = PROJECT_ROOT / "models" / "scalers_fixed.pkl"
TEST_DATA = PROJECT_ROOT / "data" / "test_set.csv"


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
    print("EVALUATING RETRAINED MODEL ON TEST SET")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = QuadrotorPINN(dropout=0.3)  # Match training dropout
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load scalers
    print(f"Loading scalers from: {SCALER_PATH}")
    scalers = joblib.load(SCALER_PATH)
    scaler_X, scaler_y = scalers["scaler_X"], scalers["scaler_y"]

    # Load test data
    print(f"Loading test data from: {TEST_DATA}")
    X_test, y_test = load_data(TEST_DATA)
    print(f"  Test samples: {len(X_test)}")

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
    print("TEST SET PERFORMANCE")
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

    for i, (name, label, unit) in enumerate(zip(state_names, state_labels, units)):
        mae = np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
        rmse = np.sqrt(np.mean((y_test[:, i] - y_pred[:, i]) ** 2))
        max_err = np.max(np.abs(y_test[:, i] - y_pred[:, i]))
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
    print(f"{'Parameter':<10} {'Learned':<15} {'True':<15} {'Error %'}")
    print("-" * 55)

    for param_name, param in model.params.items():
        learned = param.item()
        true = model.true_params[param_name]
        error = abs(learned - true) / true * 100
        print(f"{param_name:<10} {learned:<15.6e} {true:<15.6e} {error:<.2f}%")

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
