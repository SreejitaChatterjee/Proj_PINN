# Systematic Comparison of Neural Network Architectures for Dynamics Learning

**Venue:** NeurIPS 2025
**Status:** Needs Revision

## Core Contribution

Systematic comparison of monolithic vs modular vs Fourier architectures for learning dynamical systems.

## Key Results

| Architecture | 100-step MAE | Parameters |
|--------------|--------------|------------|
| Baseline (Monolithic) | 5.09m | 205K |
| **Modular** | **1.11m** | **72K** |
| Fourier | 5.09m | 205K |

**Key Finding:** Fourier features show NO improvement over baseline, disproving prior claims.

## Files

- `NeurIPS_2025_submission.tex` - Main paper source
- `neurips_2025.sty` - NeurIPS style file

## Abstract

We present a systematic comparison of neural network architectures for learning dynamical systems. Our experiments reveal that modular architectures separating translation and rotation dynamics achieve 4.6x better stability with 65% fewer parameters, while Fourier feature networks provide no measurable benefit.

## Build

```bash
pdflatex NeurIPS_2025_submission.tex
bibtex NeurIPS_2025_submission
pdflatex NeurIPS_2025_submission.tex
pdflatex NeurIPS_2025_submission.tex
```

## Related Code

- Framework: `../../pinn_dynamics/`
- Ablation studies: `../../research/ablation/`
