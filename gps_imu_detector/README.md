# GPS-IMU Anomaly Detector v2.0.0

**Status:** HONEST EVALUATION | **Date:** 2026-01-01 | **Version:** 2.0.0

---

## Executive Summary

A GPS-IMU spoofing detector with **honest, rigorously validated** results.

### Honest Results (v2.0.0)

**Only 2 of 6 attack types are reliably detectable:**

| Attack Type | AUROC @ 10x Noise | Classification |
|-------------|-------------------|----------------|
| noise_injection | **99.6%** [99.5%, 99.6%] | RELIABLE |
| intermittent | **94.0%** [93.4%, 94.5%] | RELIABLE |
| bias | 50.3% [49.7%, 51.0%] | UNDETECTABLE |
| drift | 49.5% [48.8%, 50.2%] | UNDETECTABLE |
| coordinated | 52.1% [51.6%, 52.7%] | UNDETECTABLE |
| step | 48.2% [47.6%, 48.9%] | UNDETECTABLE |

**Note:** 10x noise = 15m offset (GPS noise std = 1.5m)

### Why 4 Attack Types Are Undetectable

Physics-consistent attacks (bias, drift, coordinated) **don't violate any observable physical constraint**. They are fundamentally undetectable by passive monitoring.

| Attack Property | Detectable? | Why |
|-----------------|-------------|-----|
| Increased variance | YES | Changes statistical signature |
| Discontinuities | YES | Creates sharp transitions |
| Constant offset | NO | Same as GPS bias |
| Slow drift | NO | Same as GPS error walk |
| Physics-consistent | NO | Maintains all invariants |

---

## Comparison: Exaggerated vs Honest

| Metric | Phase 3 (Exaggerated) | Honest (v2.0.0) |
|--------|----------------------|-----------------|
| Overall AUROC | 99.8% | **~66%** |
| Bias AUROC | 100% | **50.3%** |
| Noise AUROC | 100% | **99.6%** |
| Intermittent AUROC | 98.7% | **94.0%** |
| FPR | 0.21% | **1-2%** |

### What Went Wrong in Phase 3

| Issue | Impact |
|-------|--------|
| Trivially separable attacks (50m vs 0.5m noise) | 100% AUROC on all attacks |
| Threshold tuning on test data | Inflated metrics |
| No bootstrap CIs | Unknown uncertainty |
| Non-monotonic curves | Evaluation artifacts |

---

## Quick Start

```bash
# Run honest evaluation
cd gps_imu_detector/scripts
python honest_evaluation.py

# Results saved to: results/honest/honest_results.json
```

---

## Key Documentation

| Document | Description |
|----------|-------------|
| `docs/HONEST_RESULTS.md` | Full honest evaluation with CIs |
| `docs/CORRECTED_EVALUATION.txt` | Identified mistakes and fixes |
| `results/honest/honest_results.json` | Raw evaluation data |

---

## What Works (Honest Claims)

| Claim | Evidence |
|-------|----------|
| Detects noise injection at 94%+ AUROC | CI: [94.2%, 94.7%] at 1x noise |
| Detects intermittent at >90% AUROC | CI: [90.6%, 91.6%] at 5x noise |
| Monotonic sensitivity curves | 6/6 attacks pass |
| Realistic GPS noise (1.5m) | Industry-standard |

## What Doesn't Work (Honest Limitations)

| Limitation | Implication |
|------------|-------------|
| Bias attacks undetectable | Can't detect constant GPS offset |
| Drift attacks undetectable | Can't detect slow GPS drift |
| Coordinated attacks undetectable | Can't detect physics-consistent spoofing |
| Passive monitoring only | Needs active probing or redundancy |

---

## The Protective Sentence

> **"The reported results apply to attacks that violate statistical or temporal properties (noise injection, intermittent). Physics-consistent attacks (bias, drift, coordinated) are fundamentally undetectable by passive monitoring and require active probing or analytical redundancy."**

---

## Architecture

```
gps_imu_detector/
├── src/                       # Core detection algorithms
│   ├── inverse_model.py       # Core ICI detector
│   ├── temporal_ici.py        # Temporal aggregation
│   ├── industry_aligned.py    # Two-stage decision logic
│   └── ...
├── scripts/                   # Evaluation scripts
│   ├── honest_evaluation.py   # Honest evaluation (v2.0.0)
│   └── ...
├── docs/                      # Documentation
│   ├── HONEST_RESULTS.md      # Honest results
│   └── CORRECTED_EVALUATION.txt # Corrections
├── results/                   # Evaluation outputs
│   └── honest/                # Honest evaluation results
└── tests/                     # Unit tests
```

---

## Test Coverage

```
Total tests: 206
All passing
```

---

## Version History

| Version | Date | Focus | Key Change |
|---------|------|-------|------------|
| 0.1-1.1 | 2025 | Development | Various improvements |
| **2.0.0** | **2026-01-01** | **Honest Evaluation** | **Fixed exaggerated results** |

---

## Reproducibility

```python
# Configuration (frozen)
GPS_NOISE_STD = 1.5  # meters (realistic)
ATTACK_MAGNITUDES = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  # x noise floor
N_TRAIN_SEQUENCES = 100
N_TEST_SEQUENCES = 100
N_BOOTSTRAP = 200
RANDOM_SEED = 42
```

---

*Version 2.0.0 - 2026-01-01 (Honest Evaluation)*
