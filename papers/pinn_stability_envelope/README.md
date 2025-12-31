# Stability Envelope Framework for Physics-Informed Neural Networks

**Venue:** ACC/CDC (American Control Conference / Conference on Decision and Control)
**Status:** Needs Revision

## Core Contribution

Formal definition of the stability envelope H_ε for physics-informed neural network predictions.

## Key Results

| Metric | Value |
|--------|-------|
| Autoregressive Stability | 4.6x better than baseline |
| Architecture | Modular (translation/rotation separation) |

## Files

- `ACC_CDC_submission.tex` - Main paper source
- `fig_stability_envelope.pdf` - Stability envelope visualization
- `fig_stability.pdf` - Stability comparison plots

## Abstract

This paper introduces a formal framework for characterizing the stability envelope of physics-informed neural networks for dynamics prediction. We define H_ε as the region in state space where autoregressive predictions remain stable within error bounds.

## Build

```bash
pdflatex ACC_CDC_submission.tex
bibtex ACC_CDC_submission
pdflatex ACC_CDC_submission.tex
pdflatex ACC_CDC_submission.tex
```

## Related Code

- Framework: `../../pinn_dynamics/`
- Experiments: `../../research/`
