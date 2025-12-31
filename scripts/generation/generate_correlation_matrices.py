#!/usr/bin/env python3
"""
Generate correlation matrices between quadrotor states.

This analysis helps identify:
- Strong dependencies between states
- Decoupled vs coupled dynamics
- Multicollinearity in state variables
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

sys.path.append(str(Path(__file__).parent))
from pinn_model import QuadrotorPINN


def load_model_and_data():
    """Load trained model, scalers, and training data"""
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

    # Load full training data for correlation analysis
    df_all = pd.read_csv(PROJECT_ROOT / "data" / "quadrotor_training_data.csv")

    return model, scaler_X, scaler_y, df_all


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

    return predictions, actuals


def plot_correlation_matrix(data, title, save_path, cmap="coolwarm"):
    """Plot correlation matrix heatmap"""

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

    # Compute correlation matrix
    df = pd.DataFrame(data, columns=state_names)
    corr_matrix = df.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
        ax=ax,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("States", fontsize=12)
    ax.set_ylabel("States", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] {title}")


def plot_error_correlation_matrix(errors, save_path):
    """Plot correlation matrix for prediction errors"""

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

    # Compute correlation matrix
    df = pd.DataFrame(errors, columns=state_names)
    corr_matrix = df.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
        ax=ax,
    )

    ax.set_title("Prediction Error Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("States", fontsize=12)
    ax.set_ylabel("States", fontsize=12)

    # Add interpretation text
    textstr = "Strong correlations in errors indicate:\n- Systematic model biases\n- Missing physics constraints\n- Coupled error propagation"
    plt.gcf().text(
        0.5,
        -0.02,
        textstr,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Prediction Error Correlation Matrix")


def plot_cross_correlation_actual_vs_predicted(actuals, predictions, save_path):
    """Plot cross-correlation between actual and predicted states"""

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

    # Compute cross-correlation
    n_states = len(state_names)
    cross_corr = np.zeros((n_states, n_states))

    for i in range(n_states):
        for j in range(n_states):
            cross_corr[i, j] = np.corrcoef(actuals[:, i], predictions[:, j])[0, 1]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        cross_corr,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=0,
        vmax=1,
        ax=ax,
        xticklabels=[f"{name}_pred" for name in state_names],
        yticklabels=[f"{name}_actual" for name in state_names],
    )

    ax.set_title(
        "Cross-Correlation: Actual vs Predicted States",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Predicted States", fontsize=12)
    ax.set_ylabel("Actual States", fontsize=12)

    # Add interpretation text
    textstr = "Diagonal values near 1.0 indicate high prediction accuracy.\nOff-diagonal values show cross-state prediction patterns."
    plt.gcf().text(
        0.5,
        -0.02,
        textstr,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Cross-Correlation Matrix (Actual vs Predicted)")


def plot_grouped_correlation_analysis(actuals, save_path):
    """Plot correlation analysis grouped by physical subsystems"""

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

    # Create DataFrame
    df = pd.DataFrame(actuals, columns=state_names)

    # Define subsystems
    subsystems = {
        "Position": ["x", "y", "z"],
        "Orientation": ["phi", "theta", "psi"],
        "Angular Rates": ["p", "q", "r"],
        "Velocities": ["vx", "vy", "vz"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (subsystem_name, states) in enumerate(subsystems.items()):
        ax = axes[idx]
        subset_df = df[states]
        corr_matrix = subset_df.corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            vmin=-1,
            vmax=1,
            ax=ax,
        )

        ax.set_title(f"{subsystem_name} Subsystem Correlation", fontsize=14, fontweight="bold")

    plt.suptitle(
        "Correlation Analysis by Physical Subsystems",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Grouped Correlation Analysis")


def main():
    """Generate all correlation matrix plots"""

    print("=" * 80)
    print("GENERATING CORRELATION MATRICES")
    print("=" * 80)
    print()

    print("Loading model and data...")
    model, scaler_X, scaler_y, df_all = load_model_and_data()

    # Subsample for efficiency (use every 10th sample)
    df_sample = df_all[::10].reset_index(drop=True)
    print(f"Using {len(df_sample)} samples for correlation analysis...")

    print("Generating predictions...")
    predictions, actuals = generate_predictions(model, df_sample, scaler_X, scaler_y)
    errors = predictions - actuals

    # Create output directory
    PROJECT_ROOT = Path(__file__).parent.parent
    save_dir = PROJECT_ROOT / "results" / "correlation_analysis"
    save_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    # 1. Actual states correlation
    plot_correlation_matrix(
        actuals,
        "Actual States Correlation Matrix",
        save_dir / "actual_states_correlation.png",
    )

    # 2. Predicted states correlation
    plot_correlation_matrix(
        predictions,
        "Predicted States Correlation Matrix",
        save_dir / "predicted_states_correlation.png",
    )

    # 3. Error correlation
    plot_error_correlation_matrix(errors, save_dir / "error_correlation.png")

    # 4. Cross-correlation (actual vs predicted)
    plot_cross_correlation_actual_vs_predicted(
        actuals, predictions, save_dir / "cross_correlation_actual_vs_predicted.png"
    )

    # 5. Grouped correlation analysis
    plot_grouped_correlation_analysis(actuals, save_dir / "grouped_correlation_analysis.png")

    print()
    print("=" * 80)
    print(f"SUCCESS: Generated 5 correlation analysis plots")
    print(f"Saved to: {save_dir}")
    print()
    print("Key insights from correlation analysis:")
    print("  - Diagonal dominance in cross-correlation => high prediction accuracy")
    print("  - Position/velocity coupling => kinematic relationships captured")
    print("  - Attitude/angular rate coupling => rotational dynamics captured")
    print("  - Error correlations => systematic model limitations")
    print("=" * 80)


if __name__ == "__main__":
    main()
