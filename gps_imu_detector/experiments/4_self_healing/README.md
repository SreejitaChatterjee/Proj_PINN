# 4. Self-Healing: IASP

**Hierarchy Position:** From detection to defense

**Prerequisite:** Understand `3_scaling_law/` first

## Inverse-Anchored State Projection (IASP)

IASP extends ICI from detection-only to closed-loop defense:

```
x_healed = (1 - α) * x_t + α * g_φ(f_θ(x_t))
```

Where:
```
α = min(1, (ICI - threshold) / C)
```

**Key Insight:** The same inverse model used for detection can also correct spoofed states.

## Why IASP Works

The composition g_φ ∘ f_θ defines a projection operator:
- Nominal states are (approximate) fixed points: g(f(x)) ≈ x
- Spoofed states are repelled toward the manifold

This is a **contractive map** on the nominal manifold.

## Evidence

| Metric | Value | Meaning |
|--------|-------|---------|
| Position error reduction | 77.1% | 100m spoof → 23m error |
| ICI reduction | 99.9% | Manifold restored |
| Stability | PASS | No oscillation |
| Quiescence | 0.8% | <1% false healing on nominal |

## Experiments in This Folder

- `iasp_healing_demo.py` - Full validation experiment
- `stability_analysis.py` - Check for oscillation
- `quiescence_test.py` - Verify <1% false triggers

## What This Proves

1. Detection and correction use the same learned structure
2. IASP is stable (no divergence, no oscillation)
3. IASP is quiescent (does not interfere with nominal operation)

## What IASP Does NOT Do

- Restore "true" state (only model-consistent state)
- Work without detection (requires ICI threshold)
- Guarantee zero error (reduces, does not eliminate)

## Next Step

See `5_limits/` for attacks that defeat ICI and IASP.
