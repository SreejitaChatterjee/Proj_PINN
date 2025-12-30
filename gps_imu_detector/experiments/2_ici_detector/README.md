# 2. ICI Detector: The New Primitive

**Hierarchy Position:** Core contribution

**Prerequisite:** Understand `1_impossibility/` first

## Inverse-Cycle Instability (ICI)

ICI exploits the inverse model to detect spoofing:

```
ICI(x_t) = ||x_t - g_φ(f_θ(x_t))||
```

**Key Insight:** The inverse model g_φ was NOT trained to be consistent with spoofed data.
When x_t is spoofed, the round-trip f→g fails to return to x_t.

## Why ICI Works When Residuals Fail

| Detector | What It Measures | Blind To |
|----------|------------------|----------|
| Residual | Forward consistency | Position offset (REC) |
| ICI | Inverse consistency | Nothing (breaks REC) |

The inverse model sees the **absolute position**, not just the dynamics.

## Evidence

| Attack | Magnitude | Residual AUROC | ICI AUROC |
|--------|-----------|----------------|-----------|
| Constant offset | 50m | 0.500 | 1.000 |
| Constant offset | 25m | 0.500 | 1.000 |
| Consistent drift | 0.5m/s | 0.500 | 1.000 |

## Experiments in This Folder

- `train_ici_detector.py` - Train forward + inverse models
- `evaluate_ici.py` - Per-attack AUROC evaluation
- `ici_threshold_calibration.py` - Set detection threshold

## What This Proves

1. ICI breaks the Residual Equivalence Class
2. Perfect separation (AUROC=1.0) for offsets >= 25m
3. The inverse model is the key to detection

## Next Step

See `3_scaling_law/` for how detection scales with attack magnitude.
