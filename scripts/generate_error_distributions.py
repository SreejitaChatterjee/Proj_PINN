#!/usr/bin/env python3
"""
Generate error distribution histogram plots for all 12 states.

This analysis helps identify:
- Whether prediction errors are normally distributed
- Presence of systematic biases (non-zero mean)
- Outliers and heavy tails in error distributions
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
from scipy import stats

sys.path.append(str(Path(__file__).parent))
from pinn_model import QuadrotorPINN


def load_model_and_data():
    """Load trained model, scalers, and test data"""
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

    # Load training data (use last trajectory as test set)
    df_all = pd.read_csv(PROJECT_ROOT / "data" / "quadrotor_training_data.csv")
    # Use last trajectory (ID=9) as test data
    df_test = df_all[df_all["trajectory_id"] == 9].reset_index(drop=True)

    return model, scaler_X, scaler_y, df_test


def generate_predictions(model, test_data, scaler_X, scaler_y):
    """Generate predictions for all test data"""

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

    # Single DataFrame
    df = test_data

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

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = predictions - actuals

    return errors, predictions, actuals


def plot_error_distributions(errors, save_dir):
    """Generate histogram plots for all 12 state errors"""

    state_names = [
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

    # Create output directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING ERROR DISTRIBUTION HISTOGRAMS")
    print("=" * 80)
    print(f"Output directory: {save_dir}")
    print()

    # Plot each state's error distribution
    for i, (name, unit) in enumerate(zip(state_names, state_units)):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        error = errors[:, i]

        # Left: Histogram with normal distribution overlay
        ax = axes[0]
        n, bins, patches = ax.hist(
            error,
            bins=50,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )

        # Fit and plot normal distribution
        mu, sigma = error.mean(), error.std()
        x = np.linspace(error.min(), error.max(), 100)
        ax.plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            "r-",
            linewidth=2,
            label=f"Normal({mu:.2e}, {sigma:.2e})",
        )

        ax.set_xlabel(f"Prediction Error ({unit})", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_title(f"{name.upper()} Error Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        textstr = "\n".join(
            [
                f"Mean: {mu:.2e} {unit}",
                f"Std: {sigma:.2e} {unit}",
                f"Min: {error.min():.2e} {unit}",
                f"Max: {error.max():.2e} {unit}",
                f"MAE: {np.abs(error).mean():.2e} {unit}",
            ]
        )
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Right: Q-Q plot (quantile-quantile plot for normality check)
        ax = axes[1]
        stats.probplot(error, dist="norm", plot=ax)
        ax.set_title(f"{name.upper()} Q-Q Plot (Normality Check)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Compute normality test statistic
        _, p_value = stats.shapiro(
            error[:5000] if len(error) > 5000 else error
        )  # Shapiro-Wilk test
        ax.text(
            0.02,
            0.98,
            f"Shapiro-Wilk p-value: {p_value:.4f}\n"
            + ("Normal" if p_value > 0.05 else "Non-normal"),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
        )

        plt.tight_layout()
        plt.savefig(save_dir / f"{name}_error_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[OK] {name}: mean={mu:.2e}, std={sigma:.2e}, MAE={np.abs(error).mean():.2e} {unit}")

    print()
    print("=" * 80)
    print(f"SUCCESS: Generated {len(state_names)} error distribution plots")
    print(f"Saved to: {save_dir}")
    print("=" * 80)


def plot_combined_error_overview(errors, save_dir):
    """Create a single overview figure with all 12 error distributions"""

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

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, (name, unit) in enumerate(zip(state_names, state_units)):
        ax = axes[i]
        error = errors[:, i]

        # Histogram
        ax.hist(
            error,
            bins=40,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )

        # Normal distribution overlay
        mu, sigma = error.mean(), error.std()
        x = np.linspace(error.min(), error.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2)

        ax.set_xlabel(f"Error ({unit})", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{name.upper()}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Statistics
        textstr = f"Mean: {mu:.2e}\nStd: {sigma:.2e}\nMAE: {np.abs(error).mean():.2e}"
        ax.text(
            0.98,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.suptitle(
        "Prediction Error Distributions - All States",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "all_states_error_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Combined overview plot saved")


def main():
    """Generate all error distribution plots"""

    print("Loading model and data...")
    model, scaler_X, scaler_y, df_test = load_model_and_data()

    print(f"Generating predictions on {len(df_test)} test samples...")
    errors, predictions, actuals = generate_predictions(model, df_test, scaler_X, scaler_y)

    # Create output directory
    PROJECT_ROOT = Path(__file__).parent.parent
    save_dir = PROJECT_ROOT / "results" / "error_distributions"

    # Generate individual error distribution plots
    plot_error_distributions(errors, save_dir)

    # Generate combined overview plot
    plot_combined_error_overview(errors, save_dir)

    print()
    print("=" * 80)
    print("Analysis complete! Error distributions reveal:")
    print("  - Normality of prediction errors (Q-Q plots)")
    print("  - Presence of systematic biases (mean != 0)")
    print("  - Error magnitude and spread (std, MAE)")
    print("=" * 80)


if __name__ == "__main__":
    main()
