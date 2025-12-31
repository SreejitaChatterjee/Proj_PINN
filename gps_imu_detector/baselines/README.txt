# Baselines: What We Compare Against

**Hierarchy Position:** Reference implementations for comparison

## Purpose

This folder contains baseline detectors that ICI improves upon.
These are NOT contributions—they demonstrate the limitation that motivates ICI.

## Baseline Detectors

### 1. Residual Detector (Forward-Only)

```python
score = ||f_θ(x_t) - x_{t+1}||
```

**Expected AUROC:** 0.5 (random guessing for consistent spoofing)

**Why it fails:** Residual Equivalence Class theorem

### 2. EKF-NIS Detector

```python
NIS = r_t^T @ S_t^{-1} @ r_t
```

**Expected AUROC:** 0.65-0.75 (better for jumps, worse for drifts)

**Why it's limited:** Assumes Gaussian innovation, not learned

### 3. Threshold-Based Detector

```python
alarm = (position_change > fixed_threshold)
```

**Expected AUROC:** Variable (depends on threshold tuning)

**Why it's limited:** Cannot adapt to trajectory dynamics

## File Organization

```
baselines/
├── residual_detector.py    # Forward-only baseline
├── ekf_nis_detector.py     # Classical EKF approach
├── threshold_detector.py   # Naive fixed-threshold
└── evaluate_baselines.py   # Compare all baselines
```

## Usage

```bash
# Run all baselines
python baselines/evaluate_baselines.py --output results/baseline_comparison.json
```

## Results Summary

| Baseline | AUROC (50m offset) | AUROC (drift) | Status |
|----------|-------------------|---------------|--------|
| Residual | 0.500 | 0.500 | Fails (REC) |
| EKF-NIS | 0.720 | 0.580 | Marginal |
| Threshold | 0.650 | 0.550 | Marginal |
| **ICI** | **1.000** | **1.000** | **Detectable** |

## Why Keep Baselines?

1. **Comparison:** Reviewers need to see improvement over prior art
2. **Ablation:** Shows each component's contribution
3. **Fairness:** Demonstrates we gave baselines a fair chance

## Note

These baselines are **frozen**. Do not tune them to improve performance.
The point is to show ICI's advantage, not to make baselines look artificially weak.
