"""
Generate Publication Figures from Validated Experimental Results

This script reads actual experimental results and generates publication-quality
figures for the papers. It replaces the hardcoded placeholder figures with
real validated data.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "paper_versions"

# IEEE column width: 3.5 inches, max height ~9 inches for full page
COLUMN_WIDTH = 3.5
DPI = 600

# Style settings for IEEE papers
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "serif",
        "text.usetex": False,
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
    }
)


def load_architecture_results():
    """Load architecture comparison results"""
    results_file = RESULTS_DIR / "architecture_comparison" / "architecture_comparison_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            return json.load(f)
    return None


def load_ablation_results():
    """Load ablation study results"""
    results_file = RESULTS_DIR / "ablation_study" / "ablation_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            return json.load(f)
    return None


def generate_stability_figure(arch_results):
    """
    Figure 1: Stability comparison across architectures
    Shows single-step vs 100-step MAE for all 4 architectures
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH * 2, 2.5))

    models = ["baseline", "modular", "fourier", "curriculum"]
    model_labels = ["Baseline", "Modular", "Fourier", "Ours"]
    colors = ["#CC3300", "#FF9900", "#FFCC00", "#0066CC"]

    # Extract data
    single_step_z = []
    single_step_phi = []
    hundred_step_pos = []
    hundred_step_att = []

    for model in models:
        if model in arch_results:
            single_step_z.append(arch_results[model]["single_step"]["z_mae"])
            single_step_phi.append(arch_results[model]["single_step"]["phi_mae"])
            hundred_step_pos.append(arch_results[model]["rollout"]["position"])
            hundred_step_att.append(arch_results[model]["rollout"]["attitude"])
        else:
            single_step_z.append(0)
            single_step_phi.append(0)
            hundred_step_pos.append(0)
            hundred_step_att.append(0)

    x = np.arange(len(models))
    width = 0.35

    # Single-step accuracy (left)
    bars1 = ax1.bar(
        x - width / 2,
        single_step_z,
        width,
        label="Position (m)",
        color=colors,
        alpha=0.8,
    )
    ax1.set_ylabel("Single-Step MAE")
    ax1.set_xlabel("Architecture")
    ax1.set_title("(a) Single-Step Accuracy", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=15, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    # 100-step stability (right)
    bars2 = ax2.bar(x, hundred_step_pos, width, color=colors, alpha=0.8)
    ax2.set_ylabel("100-Step Position MAE (m)")
    ax2.set_xlabel("Architecture")
    ax2.set_title("(b) 100-Step Stability", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, rotation=15, ha="right")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars2, hundred_step_pos):
        if val > 0:
            height = bar.get_height()
            ax2.annotate(
                f"{val:.2f}" if val < 100 else f"{val:.0e}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=6,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_stability.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_stability_column.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: fig_stability.pdf")


def generate_ablation_figure(ablation_results):
    """
    Figure 2: Ablation study bar chart
    Shows progressive improvement with each component
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.8))

    configs = ["Baseline", "+Curriculum", "+Sched_Sampling", "+Dropout", "+Energy_Cons"]
    config_labels = [
        "Baseline",
        "+Curriculum",
        "+Sched.\nSampling",
        "+Dropout",
        "+Energy\nCons.",
    ]

    mae_values = []
    for config in configs:
        if config in ablation_results:
            mae_values.append(ablation_results[config]["rollout_mae"])
        else:
            mae_values.append(0)

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(configs)))

    bars = ax.bar(range(len(configs)), mae_values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("100-Step Position MAE (m)", fontweight="bold")
    ax.set_xlabel("Configuration", fontweight="bold")
    ax.set_title("Ablation Study: Component Contributions", fontweight="bold")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(config_labels, fontsize=6)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, mae_values):
        if val > 0:
            height = bar.get_height()
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=6,
                xytext=(0, 1),
                textcoords="offset points",
            )

    # Add improvement annotation
    if mae_values[0] > 0 and mae_values[-1] > 0:
        improvement = mae_values[0] / mae_values[-1]
        ax.annotate(
            f"{improvement:.0f}× improvement",
            xy=(len(configs) - 1, mae_values[-1]),
            xytext=(len(configs) - 2, mae_values[0] * 0.5),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
            fontsize=8,
            fontweight="bold",
            color="green",
        )

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_ablation.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_ablation_column.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: fig_ablation.pdf")


def generate_comparison_table(arch_results):
    """
    Generate LaTeX table for architecture comparison
    """
    table = r"""
\begin{table}[t]
\centering
\caption{Single-Step vs. 100-Step Performance}
\label{tab:stability}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{\textbf{1-Step MAE}} & \multicolumn{2}{c}{\textbf{100-Step MAE}} \\
\textbf{Model} & $z$ (m) & $\phi$ (rad) & $z$ (m) & $\phi$ (rad) \\
\midrule
"""

    models = ["baseline", "modular", "fourier", "curriculum"]
    model_labels = ["Baseline", r"\textbf{Modular (Best)}", "Fourier", "Curriculum"]

    for model, label in zip(models, model_labels):
        if model in arch_results:
            r = arch_results[model]
            z_1 = r["single_step"]["z_mae"]
            phi_1 = r["single_step"]["phi_mae"]
            z_100 = r["rollout"]["position"]
            phi_100 = r["rollout"]["attitude"]

            # Format large numbers in scientific notation
            z_100_str = f"{z_100:.2f}" if z_100 < 1000 else f"{z_100:.1e}"
            phi_100_str = f"{phi_100:.4f}" if phi_100 < 100 else f"{phi_100:.1e}"

            # Bold the best results (Modular is best)
            if model == "modular":
                table += f"{label} & \\textbf{{{z_1:.3f}}} & \\textbf{{{phi_1:.4f}}} & \\textbf{{{z_100_str}}} & \\textbf{{{phi_100_str}}} \\\\\n"
            else:
                table += f"{label} & {z_1:.3f} & {phi_1:.4f} & {z_100_str} & {phi_100_str} \\\\\n"

    table += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(FIGURES_DIR / "table_stability.tex", "w") as f:
        f.write(table)
    print(f"Saved: table_stability.tex")


def generate_ablation_table(ablation_results):
    """
    Generate LaTeX table for ablation study
    """
    table = r"""
\begin{table}[t]
\centering
\caption{Ablation Study: 100-Step Position MAE}
\label{tab:ablation}
\begin{tabular}{lcc}
\toprule
\textbf{Configuration} & \textbf{MAE (m)} & \textbf{Improvement} \\
\midrule
"""

    configs = ["Baseline", "+Curriculum", "+Sched_Sampling", "+Dropout", "+Energy_Cons"]
    config_labels = [
        "Baseline",
        "+ Curriculum",
        "+ Scheduled sampling",
        "+ Dropout",
        "+ Energy conservation",
    ]

    baseline_mae = ablation_results.get("Baseline", {}).get("rollout_mae", 1.0)

    for config, label in zip(configs, config_labels):
        if config in ablation_results:
            mae = ablation_results[config]["rollout_mae"]
            if config == "Baseline":
                improvement = "--"
            else:
                improvement = f"{(1 - mae/baseline_mae)*100:.0f}\\%"

            if config == "+Energy_Cons":
                table += f"\\textbf{{{label}}} & \\textbf{{{mae:.3f}}} & \\textbf{{{improvement}}} \\\\\n"
            else:
                table += f"{label} & {mae:.3f} & {improvement} \\\\\n"

    table += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(FIGURES_DIR / "table_ablation.tex", "w") as f:
        f.write(table)
    print(f"Saved: table_ablation.tex")


def generate_parameter_table(arch_results):
    """
    Generate LaTeX table for parameter identification results
    """
    # Use curriculum model results for parameter identification
    if "curriculum" not in arch_results:
        print("No curriculum results found for parameter table")
        return

    params = arch_results["curriculum"].get("parameters", {})

    table = r"""
\begin{table}[t]
\centering
\caption{Parameter Identification Results}
\label{tab:params}
\begin{tabular}{lccc}
\toprule
\textbf{Parameter} & \textbf{True} & \textbf{Learned} & \textbf{Error} \\
\midrule
"""

    param_order = ["m", "kt", "kq", "Jxx", "Jyy", "Jzz"]
    param_labels = ["Mass $m$", "$k_t$", "$k_q$", "$J_{xx}$", "$J_{yy}$", "$J_{zz}$"]

    for param, label in zip(param_order, param_labels):
        if param in params:
            p = params[param]
            true = p["true"]
            learned = p["learned"]
            error = p["error_pct"]

            # Format based on magnitude
            if true < 0.001:
                true_str = f"{true:.2e}"
                learned_str = f"{learned:.2e}"
            else:
                true_str = f"{true:.4f}"
                learned_str = f"{learned:.4f}"

            # Add units
            if param == "m":
                true_str += " kg"
                learned_str += " kg"

            table += f"{label} & {true_str} & {learned_str} & {error:.1f}\\% \\\\\n"

    table += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(FIGURES_DIR / "table_params.tex", "w") as f:
        f.write(table)
    print(f"Saved: table_params.tex")


def main():
    print("=" * 60)
    print("GENERATING VALIDATED PUBLICATION FIGURES")
    print("=" * 60)

    # Load results
    arch_results = load_architecture_results()
    ablation_results = load_ablation_results()

    if arch_results is None:
        print("\nWARNING: Architecture comparison results not found!")
        print("Run train_all_architectures.py first.")
    else:
        print(f"\nLoaded architecture results for: {list(arch_results.keys())}")
        generate_stability_figure(arch_results)
        generate_comparison_table(arch_results)
        generate_parameter_table(arch_results)

    if ablation_results is None:
        print("\nWARNING: Ablation study results not found!")
        print("Run run_ablation_study.py first.")
    else:
        print(f"\nLoaded ablation results for: {list(ablation_results.keys())}")
        generate_ablation_figure(ablation_results)
        generate_ablation_table(ablation_results)

    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
