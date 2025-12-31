# Modular Architecture Design for Quadrotor Dynamics Learning

**Venue:** ICRA 2026 (IEEE International Conference on Robotics and Automation)
**Status:** Needs Revision

## Core Contribution

Why separating translation and rotation dynamics improves stability and reduces model complexity.

## Key Results

| Metric | Modular | Baseline | Improvement |
|--------|---------|----------|-------------|
| 100-step MAE | 1.11m | 5.09m | **4.6x** |
| Parameters | 72K | 205K | **65% fewer** |
| Training Time | Faster | Slower | ~2x |

## Files

- `ICRA_2026_submission.tex` - Main paper source

## Abstract

We demonstrate that modular neural network architectures, which separately model translation and rotation dynamics, achieve superior stability and efficiency for quadrotor dynamics learning. This separation exploits the natural structure of rigid body dynamics.

## Build

```bash
pdflatex ICRA_2026_submission.tex
bibtex ICRA_2026_submission
pdflatex ICRA_2026_submission.tex
pdflatex ICRA_2026_submission.tex
```

## Related Code

- Framework: `../../pinn_dynamics/`
- Modular architecture: `../../pinn_dynamics/systems/`
