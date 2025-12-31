"""Unified plotting utilities for quadrotor PINN analysis"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotGenerator:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.detailed_dir = self.output_dir / "detailed"
        self.detailed_dir.mkdir(exist_ok=True)

    def plot_training_curves(self, history, save_name="training_curves.png"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(history["train"], label="Train")
        axes[0].plot(history["val"], label="Val")
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Data Loss", yscale="log")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(history["physics"])
        axes[1].set(xlabel="Epoch", ylabel="Loss", title="Physics Loss", yscale="log")
        axes[1].grid(alpha=0.3)

        axes[2].plot(history["reg"])
        axes[2].set(xlabel="Epoch", ylabel="Loss", title="Regularization Loss", yscale="log")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_time_series(self, data, states, title_prefix="", n_cols=4):
        n_states = len(states)
        n_rows = (n_states + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, state in enumerate(states):
            if state in data.columns:
                axes[i].plot(data["timestamp"], data[state])
                axes[i].set(
                    xlabel="Time (s)",
                    ylabel=state.title(),
                    title=f"{title_prefix}{state}",
                )
                axes[i].grid(alpha=0.3)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        return fig

    def plot_state_comparison(self, df_true, df_pred, states, save_prefix="comparison"):
        for i, state in enumerate(states, 1):
            if state not in df_true.columns or state not in df_pred.columns:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_true["timestamp"], df_true[state], "b-", label="True", alpha=0.7)
            ax.plot(
                df_pred["timestamp"],
                df_pred[state],
                "r--",
                label="Predicted",
                alpha=0.7,
            )
            ax.set(
                xlabel="Time (s)",
                ylabel=state.title(),
                title=f"{state.title()} Prediction",
            )
            ax.legend()
            ax.grid(alpha=0.3)
            plt.savefig(
                self.detailed_dir / f"{i:02d}_{state}_time_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def plot_parameter_convergence(
        self, param_history, true_values, save_name="param_convergence.png"
    ):
        n_params = len(param_history)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, (name, values) in enumerate(param_history.items()):
            axes[i].plot(values, label="Learned")
            axes[i].axhline(true_values[name], color="r", linestyle="--", label="True")
            axes[i].set(xlabel="Epoch", ylabel=name, title=f"{name} Convergence")
            axes[i].legend()
            axes[i].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_summary(self, results, save_name="summary.png"):
        states = ["z", "roll", "pitch", "yaw", "p", "q", "r", "vz"]
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        axes = axes.flatten()

        for i, state in enumerate(states):
            if state in results["true"].columns:
                axes[i].plot(
                    results["true"]["timestamp"],
                    results["true"][state],
                    "b-",
                    label="True",
                    alpha=0.6,
                )
                axes[i].plot(
                    results["pred"]["timestamp"],
                    results["pred"][state],
                    "r--",
                    label="Pred",
                    alpha=0.6,
                )
                axes[i].set(xlabel="Time (s)", ylabel=state.title(), title=state.title())
                axes[i].legend(loc="best")
                axes[i].grid(alpha=0.3)

        plt.suptitle(
            "Quadrotor State Predictions vs Ground Truth",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()


def quick_plot_training(history_dict):
    """Quick training visualization"""
    plotter = PlotGenerator()
    plotter.plot_training_curves(history_dict)
    print("Training curves saved to visualizations/training_curves.png")


def quick_plot_states(df, states=None):
    """Quick state visualization"""
    plotter = PlotGenerator()
    states = states or ["z", "roll", "pitch", "yaw", "p", "q", "r", "vz"]
    fig = plotter.plot_time_series(df, states)
    plt.savefig(plotter.output_dir / "states.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("State plots saved to visualizations/states.png")
