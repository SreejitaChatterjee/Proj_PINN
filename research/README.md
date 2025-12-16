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

**Key Finding**: Physics loss doesn't improve (and may hurt) autoregressive
rollout stability. Training regime and architecture matter more.

### `/experiments/`
Experimental scripts for architecture comparison and other research.

## Note

These artifacts are for reproducibility and documentation purposes.
For the production framework, see the `pinn_dynamics/` package.
