# Curriculum Training for Physics-Informed Dynamics Models

**Venue:** RA-L (IEEE Robotics and Automation Letters)
**Status:** Needs Revision

## Core Contribution

Practical curriculum training methodology for physics-informed neural networks.

## Key Results

| Method | Improvement | Notes |
|--------|-------------|-------|
| Curriculum Training | **+25%** | Progressive difficulty |
| Dropout | No improvement | Failed approach |
| Energy Constraints | No improvement | Failed approach |

## Files

- `RAL_submission.tex` - Main paper source

## Abstract

We present a curriculum training methodology for physics-informed dynamics models that progressively increases prediction horizon during training. This approach achieves 25% improvement in long-horizon stability while being simple to implement.

## Honest Limitations

- Dropout regularization does NOT help
- Energy-based constraints do NOT improve generalization
- Benefits are primarily from curriculum, not other techniques

## Build

```bash
pdflatex RAL_submission.tex
bibtex RAL_submission
pdflatex RAL_submission.tex
pdflatex RAL_submission.tex
```

## Related Code

- Training pipeline: `../../pinn_dynamics/training/`
