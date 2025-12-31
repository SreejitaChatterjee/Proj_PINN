# Architecture-Dependent Parameter Identification in Learned Dynamics

**Venue:** CDC/L4DC (Conference on Decision and Control / Learning for Dynamics and Control)
**Status:** Draft

## Core Contribution

How neural network architecture affects the identifiability of physical parameters.

## Key Results

| Parameter | Identification Error | Notes |
|-----------|---------------------|-------|
| Motor Coefficients | **0%** | All architectures |
| Inertias | **50-60%** | Observability-limited |

## Files

- `new_novelty.tex` - Main paper source

## Abstract

We investigate how neural network architecture choices affect the identifiability of physical parameters when learning dynamics models. We find that certain parameters (motor coefficients) are perfectly identifiable across all architectures, while others (inertias) remain fundamentally limited by observability constraints.

## Key Insight

Parameter identification is architecture-dependent but ultimately bounded by observability, not model capacity.

## Build

```bash
pdflatex new_novelty.tex
bibtex new_novelty
pdflatex new_novelty.tex
pdflatex new_novelty.tex
```

## Related Code

- Parameter identification: `../../research/`
