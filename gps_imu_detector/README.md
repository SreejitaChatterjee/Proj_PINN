# GPS-IMU Anomaly Detector v3.0.0

**Status:** VALIDATED ON EUROC | **Date:** 2026-01-01 | **Version:** 3.0.0

---

## Executive Summary

A GPS-IMU spoofing detector with **real-world validated** results on EuRoC MAV dataset.

### Key Results (v3.0.0 - EuRoC Validated)

**ALL 6/6 attack types are detectable on real flight data:**

| Attack Type | AUROC @ 10x (15m) | Status |
|-------------|-------------------|--------|
| noise_injection | **100.0%** | DETECTABLE |
| drift | **99.9%** | DETECTABLE |
| bias | **99.4%** | DETECTABLE |
| coordinated | **99.4%** | DETECTABLE |
| step | **99.4%** | DETECTABLE |
| intermittent | **84.2%** | DETECTABLE |
| **Mean** | **97.2%** | — |

**Self-Healing Performance:**

| Metric | Value |
|--------|-------|
| Error reduction (100m spoof) | **91.9%** |
| Stability | **100%** |
| Real-time latency | **0.31ms** |
| Mode switch required | **0%** |

---

## Methodology

### Data
- **Dataset:** EuRoC MAV (real UAV flights)
- **Sequences:** MH_01, MH_02, MH_03, V1_01, V1_02
- **Samples:** 138,088 total

### Evaluation Protocol
- **Training:** Unsupervised (normal flights only)
- **Validation:** LOSO-CV (Leave-One-Sequence-Out)
- **Attack injection:** Synthetic GPS spoofing on real trajectories
- **GPS noise:** 1.5m std (realistic)

### Why Real Data Matters

| Data Type | Physics-Consistent Attacks | Reason |
|-----------|---------------------------|--------|
| Synthetic (simulated) | ~50% AUROC | Simple random walks lack structure |
| **EuRoC (real)** | **99.4% AUROC** | Real flights have learnable patterns |

Real flight data contains:
- Complex maneuver signatures
- Sensor correlations
- Flight envelope constraints

The Mahalanobis detector learns these patterns, making even "physics-consistent" attacks detectable.

---

## Quick Start

```bash
# Run detection evaluation on EuRoC
python gps_imu_detector/scripts/evaluate_euroc_gps.py

# Run self-healing evaluation on EuRoC
python gps_imu_detector/scripts/evaluate_euroc_healing.py

# Results saved to:
#   results/euroc_gps_results.json
#   results/euroc_healing_results.json
```

---

## Detection Results by Magnitude

| Attack | 1x | 2x | 5x | 10x | 20x |
|--------|-----|-----|-----|-----|-----|
| bias | 62.6% | 82.1% | 97.5% | 99.4% | 99.8% |
| drift | 99.3% | 99.7% | 99.9% | 99.9% | 99.9% |
| noise_injection | 100% | 100% | 100% | 100% | 100% |
| coordinated | 62.7% | 82.2% | 97.6% | 99.4% | 99.8% |
| intermittent | 84.4% | 84.1% | 84.4% | 84.2% | 84.0% |
| step | 62.6% | 82.1% | 97.5% | 99.4% | 99.8% |

**Note:** 1x = 1.5m, 10x = 15m offset

---

## Self-Healing Results by Magnitude

| Spoof | Error Reduction | Stability | Latency |
|-------|-----------------|-----------|---------|
| 25m | 67.6% | 100% | 0.30ms |
| 50m | 83.8% | 100% | 0.29ms |
| 100m | **91.9%** | 100% | 0.31ms |
| 200m | 95.9% | 100% | 0.28ms |

---

## Architecture

```
gps_imu_detector/
├── src/                       # Core algorithms
│   ├── inverse_model.py       # ICI detector (cycle consistency)
│   ├── ekf.py                 # EKF baseline
│   └── hybrid.py              # Hybrid fusion
├── scripts/                   # Evaluation scripts
│   ├── evaluate_euroc_gps.py  # Detection evaluation
│   ├── evaluate_euroc_healing.py # Self-healing evaluation
│   └── honest_evaluation.py   # Synthetic evaluation
├── docs/                      # Documentation
│   └── HONEST_RESULTS.md      # Full results
├── results/                   # Outputs
│   ├── euroc_gps_results.json
│   └── euroc_healing_results.json
└── data/                      # Datasets
    └── euroc/                 # EuRoC MAV data
```

---

## Comparison: Synthetic vs Real Data

| Metric | Synthetic Data | EuRoC (Real) |
|--------|----------------|--------------|
| Bias AUROC | 50.3% | **99.4%** |
| Drift AUROC | 49.5% | **99.9%** |
| Coordinated AUROC | 52.1% | **99.4%** |
| Mean AUROC | ~66% | **97.2%** |
| Detectable attacks | 2/6 | **6/6** |

---

## Reproducibility

```python
# Configuration (frozen)
DATA_PATH = "data/euroc/all_sequences.csv"
GPS_NOISE_STD = 1.5  # meters
ATTACK_MAGNITUDES = [1.0, 2.0, 5.0, 10.0, 20.0]  # x noise
LOSO_CV = True  # Leave-One-Sequence-Out
RANDOM_SEED = 42
```

---

## Key Documentation

| Document | Description |
|----------|-------------|
| `docs/HONEST_RESULTS.md` | Complete evaluation results |
| `results/euroc_gps_results.json` | Detection metrics |
| `results/euroc_healing_results.json` | Self-healing metrics |

---

## Version History

| Version | Date | Focus | Key Change |
|---------|------|-------|------------|
| 2.0.0 | 2026-01-01 | Synthetic evaluation | Fixed methodology |
| **3.0.0** | **2026-01-01** | **EuRoC validation** | **Real data, 97.2% AUROC** |

---

*Version 3.0.0 - 2026-01-01 (EuRoC Validated)*
