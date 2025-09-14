# LaTeX Conversion Instructions

## What to Include in LaTeX Document

### 1. Include All Generated Plots as Figures
**When converting to LaTeX, include these 16 plots as figures:**

**State Variable Time-Series (Figures 1-12):**
- Figure 1: Thrust vs Time (all trajectories)
- Figure 2: Altitude vs Time (all trajectories)
- Figure 3: Roll Torque vs Time (all trajectories)
- Figure 4: Pitch Torque vs Time (all trajectories)
- Figure 5: Yaw Torque vs Time (all trajectories)
- Figure 6: Roll Angle vs Time (all trajectories)
- Figure 7: Pitch Angle vs Time (all trajectories)
- Figure 8: Yaw Angle vs Time (all trajectories)
- Figure 9: Roll Rate vs Time (all trajectories)
- Figure 10: Pitch Rate vs Time (all trajectories)
- Figure 11: Yaw Rate vs Time (all trajectories)
- Figure 12: Vertical Velocity vs Time (all trajectories)

**Parameter Convergence (Figures 13-16):**
- Figure 13: Mass Parameter Convergence vs Training Epochs
- Figure 14: Jxx Inertia Convergence vs Training Epochs
- Figure 15: Jyy Inertia Convergence vs Training Epochs
- Figure 16: Jzz Inertia Convergence vs Training Epochs

**Summary Visualizations (Figures 17-21):**
- Figure 17: Complete Trajectory Analysis (from 01_all_outputs_complete_analysis.png)
- Figure 18: Key Flight Variables (from 02_key_flight_variables.png)
- Figure 19: Physical Parameters Analysis (from 03_physical_parameters.png)
- Figure 20: Control Inputs Analysis (from 04_control_inputs.png)
- Figure 21: Model Performance Statistics (from 05_model_summary_statistics.png)

### 2. LaTeX Figure Reference Format
Use this format for each figure:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figure_name.png}
\caption{Descriptive caption for the plot}
\label{fig:figure_name}
\end{figure}
```

### 3. Table Conversion
Convert all README tables to professional LaTeX tables using:
- `\begin{table}[htbp]` environment
- `\begin{tabular}{|c|c|c|}` with proper column formatting
- Bold headers using `\textbf{}`
- Proper mathematical notation using `$...$`

### 4. Mathematical Equations
Convert physics equations to proper LaTeX math:
- Use `\begin{equation}` for numbered equations
- Use proper Greek letters: `\tau`, `\phi`, `\theta`, `\psi`
- Use subscripts: `J_{xx}`, `J_{yy}`, `J_{zz}`
- Use proper notation for derivatives: `\dot{p}`, `\ddot{z}`

### 5. Document Structure
Organize as academic paper:
```latex
\documentclass{article}
\title{Quadrotor Physics-Informed Neural Network (PINN) Project}
\author{[Your Name]}

\begin{document}
\maketitle
\tableofcontents

\section{Introduction}
\section{Model Architecture}
\section{Implementation Methodology}
\section{Results and Analysis}
\section{Conclusion}

\end{document}
```

### 6. Important Notes for AI Conversion
- **Include ALL 21 figures** (16 individual + 5 summary)
- **Convert ALL tables** to LaTeX format
- **Use proper mathematical notation** throughout
- **Reference figures properly** in text (e.g., "as shown in Figure 1...")
- **Maintain academic writing style**
- **Include figure captions** that explain what each plot shows
- **Number all sections and subsections**
- **Use professional formatting** throughout

The final LaTeX document should be completely self-contained with all plots embedded as figures, allowing you to share just the PDF with your professor.