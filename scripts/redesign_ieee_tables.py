#!/usr/bin/env python3
"""
Redesign tables for IEEE format - make them concise, professional, and readable.
"""

import re
from pathlib import Path


def redesign_ieee_tables(input_file, output_file):
    """Redesign tables to be concise and IEEE-appropriate."""

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Strategy: Replace verbose tables with compact, professional IEEE tables
    # Use normalsize font and increased row spacing for readability

    # Replace Phase 1 table
    phase1_replacement = r"""\subsection{Phase 1: Data Generation \& Preparation}

\begin{table*}[!t]
\centering
\caption{Data Generation Pipeline}
\label{tab:phase1}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{clp{11cm}}
\toprule
& \textbf{Step} & \textbf{Description} \\
\midrule
1 & Model Design & 12-state quadrotor dynamics (thrust, position, torques, angles, rates) \\
2 & Trajectories & 10 diverse sequences with square wave inputs, PID control, 600ms LPF (50,000 samples) \\
3 & Simulation & Newton-Euler equations with precisely known parameters \\
4 & Data Structure & Sequential state pairs: current\_state $\rightarrow$ next\_state \\
5 & Validation & Physics consistency and trajectory realism verification \\
\bottomrule
\end{tabular}
\end{table*}

\textit{Note: PID tracking achieves transient response <0.5s.}
"""

    phase1_pattern = r"\\subsection\{Phase 1: Data Generation.*?\\textit\{Note:.*?\}"
    content = re.sub(phase1_pattern, phase1_replacement, content, flags=re.DOTALL)

    # Replace Phase 2 table
    phase2_replacement = r"""\subsection{Phase 2: PINN Architecture Development}

\begin{table*}[!t]
\centering
\caption{PINN Architecture Development Steps}
\label{tab:phase2}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{clp{11cm}}
\toprule
& \textbf{Step} & \textbf{Implementation} \\
\midrule
6 & Network Design & 5-layer architecture, 256 neurons/layer, 16$\rightarrow$12 mapping (204,818 parameters) \\
7 & Physics Integration & Newton-Euler equations embedded in loss function \\
8 & Parameter Learning & 6 trainable parameters: m, $J_{xx}$, $J_{yy}$, $J_{zz}$, $k_t$, $k_q$ \\
9 & Loss Function & Multi-objective: data + physics + temporal + stability + regularization \\
10 & Constraints & Parameter bounds and physics law enforcement \\
\bottomrule
\end{tabular}
\end{table*}
"""

    phase2_pattern = r"\\subsection\{Phase 2: PINN Architecture Development\}.*?\\end\{table\*\}"
    content = re.sub(phase2_pattern, phase2_replacement, content, flags=re.DOTALL)

    # Replace Phase 3 table
    phase3_replacement = r"""\subsection{Phase 3: Model Evolution \& Optimization}

\begin{table*}[!t]
\centering
\caption{Progressive Model Optimization}
\label{tab:phase3}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{clp{7cm}r}
\toprule
& \textbf{Step} & \textbf{Implementation} & \textbf{Result} \\
\midrule
11 & Foundation Model & Basic PINN with standard physics weighting & 14.8\% error \\
12 & Enhanced Physics & 10× physics loss increase & 8.9\% error \\
13 & Direct Parameter ID & Torque/acceleration-based identification & 5.8\% error \\
14 & Training Optimization & Gradient clipping, advanced regularization & <100 epochs \\
15 & Hyperparameter Tuning & Learning rates, batch sizes, loss weights & Optimized \\
16 & Autoregressive Stability & Dropout, scheduled sampling (0$\rightarrow$30\%) & 500-step rollout \\
\bottomrule
\end{tabular}
\end{table*}
"""

    phase3_pattern = r"\\subsection\{Phase 3: Model Evolution.*?\\end\{table\*\}"
    content = re.sub(phase3_pattern, phase3_replacement, content, flags=re.DOTALL)

    # Replace Phase 4 table
    phase4_replacement = r"""\subsection{Phase 4: Comprehensive Evaluation}

\begin{table*}[!t]
\centering
\caption{Validation and Evaluation Methodology}
\label{tab:phase4}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{clp{11cm}}
\toprule
& \textbf{Step} & \textbf{Description} \\
\midrule
17 & Cross-Validation & 10-fold strategy across diverse trajectory groups \\
18 & Generalization Testing & Hold-out trajectory evaluation (<10\% accuracy degradation) \\
19 & Physics Compliance & Physics loss convergence (2 orders of magnitude reduction) \\
20 & Performance Metrics & Comprehensive MAE, RMSE, and correlation analysis \\
21 & Comparative Analysis & Benchmarking across all model evolutionary variants \\
\bottomrule
\end{tabular}
\end{table*}
"""

    phase4_pattern = r"\\subsection\{Phase 4: Comprehensive Evaluation\}.*?\\end\{table\*\}"
    content = re.sub(phase4_pattern, phase4_replacement, content, flags=re.DOTALL)

    # Replace Phase 5 table
    phase5_replacement = r"""\subsection{Phase 5: Results Visualization \& Documentation}

\begin{table*}[!t]
\centering
\caption{Visualization and Documentation}
\label{tab:phase5}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{clp{11cm}}
\toprule
& \textbf{Step} & \textbf{Output} \\
\midrule
22 & Comprehensive Plotting & 16 output visualizations + 5 analysis plots \\
23 & Clean Visualization & 10 trajectory subplots with professional styling \\
24 & Performance Metrics & Complete MAE, RMSE, and correlation analysis \\
25 & Physics Validation & Parameter convergence and constraint satisfaction plots \\
26 & Documentation & Publication-ready LaTeX technical report \\
\bottomrule
\end{tabular}
\end{table*}
"""

    phase5_pattern = r"\\subsection\{Phase 5: Results Visualization.*?\\end\{table\*\}"
    content = re.sub(phase5_pattern, phase5_replacement, content, flags=re.DOTALL)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] Redesigned tables for IEEE format")
    print(f"[OK] Created compact, professional two-column tables")
    print(f"[OK] All tables use table* (span both columns)")
    print(f"[OK] Normal font size with 1.4x row spacing for readability")


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    input_file = PROJECT_ROOT / "reports" / "quadrotor_pinn_report_IEEE.tex"
    output_file = PROJECT_ROOT / "reports" / "quadrotor_pinn_report_IEEE_redesigned.tex"

    print("=" * 80)
    print("REDESIGNING TABLES FOR IEEE FORMAT")
    print("=" * 80)
    print("\nImprovements:")
    print("  - Compact two-column spanning tables (table*)")
    print("  - Normal font size (readable)")
    print("  - Increased row spacing (1.4x)")
    print("  - Concise professional content")
    print()

    redesign_ieee_tables(input_file, output_file)

    # Replace original with redesigned version
    import shutil

    shutil.move(str(output_file), str(input_file))
    print(f"[OK] Updated {input_file.name}")

    print("\n" + "=" * 80)
    print("TABLE REDESIGN COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
