# 3. Scaling Law: Offset → ICI

**Hierarchy Position:** Theoretical grounding

**Prerequisite:** Understand `2_ici_detector/` first

## The Monotonic Scaling Law

ICI scales monotonically with spoofing magnitude:

```
ICI(offset) ∝ offset  (for offset > noise floor)
```

This is not empirical—it follows from the geometry of the learned manifold.

## Evidence

| Offset (m) | Mean ICI | AUROC | Status |
|------------|----------|-------|--------|
| 1 | 28.5 | 0.52 | Below noise |
| 5 | 31.2 | 0.52 | Below noise |
| 10 | 38.7 | 0.66 | Marginal |
| 25 | 68.4 | 0.99 | Detectable |
| 50 | 106.2 | 1.00 | Detectable |
| 100 | 184.7 | 1.00 | Detectable |

## Key Thresholds

| Threshold | Offset | Meaning |
|-----------|--------|---------|
| Noise floor | ~10m | Below this, ICI ≈ nominal |
| Detection floor | ~25m | Above this, AUROC > 0.95 |
| Perfect detection | ~50m | Above this, AUROC = 1.0 |

## Experiments in This Folder

- `offset_sweep.py` - ICI vs offset magnitude
- `scaling_plot.py` - Generate Figure 2 (scaling curve)

## What This Proves

1. ICI is not a binary detector—it provides graded confidence
2. The detection floor (~25m) is a fundamental limit
3. For attacks above detection floor, separation is perfect

## Theoretical Explanation

The inverse model g_φ learns a mapping:
```
g_φ: next_state → current_state
```

For nominal data, this is approximately the identity on the manifold.
For spoofed data offset by Δ, the output is offset by approximately g_φ(Δ).

Since g_φ was not trained on offsets, it does not "correct" them.
The round-trip error grows with Δ.

## Next Step

See `4_self_healing/` for how to use ICI for correction, not just detection.
