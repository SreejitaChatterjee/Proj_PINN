"""Model evaluation and prediction script"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from pinn_model import QuadrotorPINN
from plot_utils import PlotGenerator


def rollout_predictions(model, initial_state, controls, scaler_X, scaler_y, n_steps):
    """
    Autoregressive rollout: predict n_steps into future using model's own predictions
    This exposes compounding errors and true multi-step performance
    """
    model.eval()
    states = [initial_state.cpu().numpy()]  # Store as numpy for scaling

    with torch.no_grad():
        current_state = initial_state.cpu().numpy()
        for i in range(n_steps):
            # Concatenate state + controls
            state_controls = np.concatenate([current_state, controls[i].cpu().numpy()])
            # Scale input
            state_controls_scaled = scaler_X.transform(state_controls.reshape(1, -1))
            # Predict next state (scaled)
            next_state_scaled = (
                model(torch.FloatTensor(state_controls_scaled)).squeeze(0)[:12].numpy()
            )
            # Inverse transform to get actual next state
            next_state = scaler_y.inverse_transform(next_state_scaled.reshape(1, -1)).flatten()
            states.append(next_state)
            current_state = next_state  # Use prediction as next input (autoregressive)

    return np.array(states)


def evaluate_model(model_path, data_path, output_dir="results"):
    """Evaluate model and generate visualizations"""
    model = QuadrotorPINN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Load scalers
    scaler_path = model_path.parent / "scalers.pkl"
    scalers = joblib.load(scaler_path)
    scaler_X, scaler_y = scalers["scaler_X"], scalers["scaler_y"]

    df = pd.read_csv(data_path)
    # Rename columns to match expected names (data generator uses roll/pitch/yaw)
    df = df.rename(columns={"roll": "phi", "pitch": "theta", "yaw": "psi"})
    states = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]

    # Compute predictions
    predictions = []
    with torch.no_grad():
        for idx in range(len(df) - 1):
            # Input: 12 states + 4 controls = 16 features
            input_data = df.iloc[idx][
                [
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
                    "thrust",
                    "torque_x",
                    "torque_y",
                    "torque_z",
                ]
            ].values
            # Apply input scaling
            input_scaled = scaler_X.transform(input_data.reshape(1, -1))
            pred_scaled = model(torch.FloatTensor(input_scaled)).squeeze(0)[:12].numpy()
            # Inverse transform predictions
            pred = scaler_y.inverse_transform(pred_scaled.reshape(1, -1)).flatten()
            predictions.append(pred)

    df_pred = pd.DataFrame(predictions, columns=states)
    df_pred["timestamp"] = df["timestamp"].iloc[1:].values

    # Calculate errors (teacher-forced)
    errors = {}
    for state in states:
        true_vals = df[state].iloc[1:].values
        pred_vals = df_pred[state].values
        errors[state] = {
            "mae": np.mean(np.abs(true_vals - pred_vals)),
            "rmse": np.sqrt(np.mean((true_vals - pred_vals) ** 2)),
            "mape": np.mean(np.abs((true_vals - pred_vals) / (true_vals + 1e-8))) * 100,
        }

    # Evaluate autoregressive rollout on first trajectory (100 steps = 0.1s)
    traj_0 = df[df["trajectory_id"] == 0].iloc[:101]  # First 100 steps - realistic horizon
    initial_state = torch.FloatTensor(traj_0.iloc[0][states].values)
    controls = torch.FloatTensor(traj_0[["thrust", "torque_x", "torque_y", "torque_z"]].values[:-1])

    rollout_states = rollout_predictions(
        model, initial_state, controls, scaler_X, scaler_y, len(controls)
    )

    # Calculate rollout errors
    rollout_errors = {}
    for i, state in enumerate(states):
        true_vals = traj_0[state].values
        pred_vals = rollout_states[:, i]
        rollout_errors[state] = {
            "mae": np.mean(np.abs(true_vals - pred_vals)),
            "rmse": np.sqrt(np.mean((true_vals - pred_vals) ** 2)),
        }

    # Generate plots
    plotter = PlotGenerator(output_dir)
    plotter.plot_summary({"true": df.iloc[1:], "pred": df_pred})
    plotter.plot_state_comparison(df.iloc[1:], df_pred, states)

    print("\n" + "=" * 80)
    print("TEACHER-FORCED (One-Step-Ahead) Results:")
    print("=" * 80)
    for state, metrics in errors.items():
        print(
            f"{state:8s}: MAE={metrics['mae']:8.4f}, RMSE={metrics['rmse']:8.4f}, MAPE={metrics['mape']:6.2f}%"
        )

    print("\n" + "=" * 80)
    print("AUTOREGRESSIVE ROLLOUT (0.1s, 100 steps) Results:")
    print("=" * 80)
    for state, metrics in rollout_errors.items():
        print(f"{state:8s}: MAE={metrics['mae']:8.4f}, RMSE={metrics['rmse']:8.4f}")

    print(f"\n" + "=" * 80)
    print(f"Model Parameters:")
    print("=" * 80)
    for k, v in model.params.items():
        error = abs(v.item() - model.true_params[k]) / model.true_params[k] * 100
        print(f"{k:4s}: {v.item():.6e} (true: {model.true_params[k]:.6e}, error: {error:5.1f}%)")

    return errors


if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / "models" / "quadrotor_pinn.pth"
    data_path = Path(__file__).parent.parent / "data" / "quadrotor_training_data.csv"
    evaluate_model(model_path, data_path)
