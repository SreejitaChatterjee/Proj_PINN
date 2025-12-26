#!/usr/bin/env python3
"""
Generate residual (prediction error) analysis plots.

This analysis helps identify:
- Temporal patterns in errors (autocorrelation)
- Heteroscedasticity (variance changing with predicted values)
- Systematic biases as function of state magnitude
- Error drift over time
"""

# Import model
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import signal

sys.path.append(str(Path(__file__).parent))
from pinn_model import QuadrotorPINN


def load_model_and_data():
    """Load trained model, scalers, and data"""
    PROJECT_ROOT = Path(__file__).parent.parent

    # Load model
    model = QuadrotorPINN(input_size=16, hidden_size=256, output_size=12, num_layers=5, dropout=0.1)
    model.load_state_dict(
        torch.load(PROJECT_ROOT / "models" / "quadrotor_pinn.pth", weights_only=True)
    )
    model.eval()

    # Load scalers
    scalers = joblib.load(PROJECT_ROOT / "models" / "scalers.pkl")
    scaler_X = scalers["scaler_X"]
    scaler_y = scalers["scaler_y"]

    # Load data (use trajectory 5 for analysis)
    df_all = pd.read_csv(PROJECT_ROOT / "data" / "quadrotor_training_data.csv")
    df_test = df_all[df_all["trajectory_id"] == 5].reset_index(drop=True)

    return model, scaler_X, scaler_y, df_test


def generate_predictions(model, df, scaler_X, scaler_y):
    """Generate predictions for data"""

    state_cols = [
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "p",
        "q",
        "r",
        "vx",
        "vy",
        "vz",
    ]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]
    input_features = state_cols + control_cols

    predictions = []
    actuals = []
    timestamps = []

    with torch.no_grad():
        for idx in range(len(df) - 1):
            # Get input
            input_data = df.iloc[idx][input_features].values
            input_scaled = scaler_X.transform(input_data.reshape(1, -1))

            # Predict
            pred_scaled = model(torch.FloatTensor(input_scaled)).squeeze(0)[:12].numpy()
            pred = scaler_y.inverse_transform(pred_scaled.reshape(1, -1)).flatten()

            # Get actual next state
            actual = df.iloc[idx + 1][state_cols].values

            predictions.append(pred)
            actuals.append(actual)
            timestamps.append(df.iloc[idx]["timestamp"])

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    timestamps = np.array(timestamps)

    return predictions, actuals, timestamps


def plot_residuals_vs_time(errors, timestamps, save_dir):
    """Plot residuals vs time for all states"""

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
    state_units = [
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

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    axes = axes.flatten()

    for i, (name, unit) in enumerate(zip(state_names, state_units)):
        ax = axes[i]
        error = errors[:, i]

        # Plot residuals
        ax.scatter(timestamps, error, alpha=0.3, s=10, color="steelblue")
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5, label="Zero error")

        # Add running mean
        window = 50
        if len(error) > window:
            running_mean = np.convolve(error, np.ones(window) / window, mode="same")
            ax.plot(
                timestamps,
                running_mean,
                color="orange",
                linewidth=2,
                label=f"Running mean ({window} samples)",
            )

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel(f"Residual ({unit})", fontsize=10)
        ax.set_title(f"{name.upper()}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(
        "Residuals vs Time - Temporal Error Patterns",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "residuals_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Residuals vs Time")


def plot_residuals_vs_predicted(errors, predictions, save_dir):
    """Plot residuals vs predicted values (heteroscedasticity check)"""

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
    state_units = [
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

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    axes = axes.flatten()

    for i, (name, unit) in enumerate(zip(state_names, state_units)):
        ax = axes[i]
        error = errors[:, i]
        pred = predictions[:, i]

        # Scatter plot
        ax.scatter(pred, error, alpha=0.3, s=10, color="steelblue")
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5)

        # Fit linear trend to check for systematic bias
        if len(pred) > 10:
            z = np.polyfit(pred, error, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(pred.min(), pred.max(), 100)
            ax.plot(
                x_trend,
                p(x_trend),
                color="orange",
                linewidth=2,
                linestyle="--",
                label=f"Trend: {z[0]:.2e}x + {z[1]:.2e}",
            )

        ax.set_xlabel(f"Predicted {name} ({unit})", fontsize=10)
        ax.set_ylabel(f"Residual ({unit})", fontsize=10)
        ax.set_title(f"{name.upper()}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(
        "Residuals vs Predicted Values - Heteroscedasticity Check",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "residuals_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Residuals vs Predicted")


def plot_autocorrelation(errors, save_dir):
    """Plot autocorrelation function for residuals"""

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

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    axes = axes.flatten()

    max_lag = 200

    for i, name in enumerate(state_names):
        ax = axes[i]
        error = errors[:, i]

        # Compute autocorrelation
        lags = range(max_lag)
        acf = [
            np.corrcoef(error[: -lag or None], error[lag:])[0, 1] if lag > 0 else 1.0
            for lag in lags
        ]

        # Plot
        ax.bar(lags, acf, color="steelblue", alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

        # Add confidence interval (95%)
        conf_int = 1.96 / np.sqrt(len(error))
        ax.axhline(y=conf_int, color="r", linestyle="--", linewidth=1, label="95% confidence")
        ax.axhline(y=-conf_int, color="r", linestyle="--", linewidth=1)

        ax.set_xlabel("Lag", fontsize=10)
        ax.set_ylabel("Autocorrelation", fontsize=10)
        ax.set_title(f"{name.upper()}", fontsize=12, fontweight="bold")
        ax.set_ylim([-1.1, 1.1])
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    plt.suptitle(
        "Residual Autocorrelation - Temporal Dependencies",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "autocorrelation.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Autocorrelation Analysis")


def plot_rolling_statistics(errors, timestamps, save_dir):
    """Plot rolling mean and std of residuals"""

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
    state_units = [
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

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    axes = axes.flatten()

    window = 100

    for i, (name, unit) in enumerate(zip(state_names, state_units)):
        ax = axes[i]
        error = errors[:, i]

        if len(error) > window:
            # Compute rolling statistics
            rolling_mean = np.convolve(error, np.ones(window) / window, mode="same")
            rolling_std = np.array(
                [
                    np.std(error[max(0, i - window // 2) : min(len(error), i + window // 2)])
                    for i in range(len(error))
                ]
            )

            # Plot
            ax.plot(
                timestamps,
                rolling_mean,
                color="blue",
                linewidth=2,
                label="Rolling Mean",
            )
            ax.fill_between(
                timestamps,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                color="blue",
                alpha=0.2,
                label="Rolling Std",
            )
            ax.axhline(y=0, color="r", linestyle="--", linewidth=1)

            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel(f"Residual ({unit})", fontsize=10)
            ax.set_title(f"{name.upper()}", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    plt.suptitle(
        f"Rolling Statistics (window={window}) - Error Stability Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "rolling_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Rolling Statistics")


def main():
    """Generate all residual analysis plots"""

    print("=" * 80)
    print("GENERATING RESIDUAL ANALYSIS PLOTS")
    print("=" * 80)
    print()

    print("Loading model and data...")
    model, scaler_X, scaler_y, df_test = load_model_and_data()

    print(f"Generating predictions on {len(df_test)} samples...")
    predictions, actuals, timestamps = generate_predictions(model, df_test, scaler_X, scaler_y)
    errors = predictions - actuals

    # Create output directory
    PROJECT_ROOT = Path(__file__).parent.parent
    save_dir = PROJECT_ROOT / "results" / "residual_analysis"
    save_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    # 1. Residuals vs time
    plot_residuals_vs_time(errors, timestamps, save_dir)

    # 2. Residuals vs predicted
    plot_residuals_vs_predicted(errors, predictions, save_dir)

    # 3. Autocorrelation
    plot_autocorrelation(errors, save_dir)

    # 4. Rolling statistics
    plot_rolling_statistics(errors, timestamps, save_dir)

    print()
    print("=" * 80)
    print(f"SUCCESS: Generated 4 residual analysis plots")
    print(f"Saved to: {save_dir}")
    print()
    print("Key insights from residual analysis:")
    print("  - Residuals vs time: temporal error patterns and drift")
    print("  - Residuals vs predicted: heteroscedasticity and systematic biases")
    print("  - Autocorrelation: temporal dependencies in errors")
    print("  - Rolling statistics: error stability over trajectory")
    print("=" * 80)


if __name__ == "__main__":
    main()
