# Research Artifacts

This directory contains research experiments and analysis from the PINN dynamics study.

## Contents

### `/paper/`
LaTeX sources and compiled PDF for the research paper.

### `/ablation/`
Ablation study results comparing different components:
- Curriculum learning
- Scheduled sampling
- Dropout
- Energy conservation loss

### `/weight_sweep/`
Physics loss weight experiments showing the relationship between
physics loss weight and autoregressive rollout stability.

**Observation**: In our experiments (w=20, lr=1e-3, 20 seeds), w=0 outperformed w=20.
This does NOT imply physics loss is harmful in general---it highlights the need
for careful hyperparameter tuning. See the paper for full caveats.

### `/experiments/`
Experimental scripts for architecture comparison and other research.

## Note

These artifacts are for reproducibility and documentation purposes.
For the production framework, see the `pinn_dynamics/` package.
