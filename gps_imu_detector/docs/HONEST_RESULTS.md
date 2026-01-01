# GPS-IMU Anomaly Detector: Validated Results on Real Flight Data

**Date:** 2026-01-01
**Version:** 3.2.0 (EuRoC Validated)
**Status:** PUBLICATION-READY

---

## Executive Summary

This document presents **validated results** on real EuRoC MAV flight data. All 6 attack types are detectable with 97.2% mean AUROC using an unsupervised Mahalanobis detector.

### Key Results

| Attack Type | AUROC @ 10x (15m) | Status |
|-------------|-------------------|--------|
| noise_injection | **100.0%** | DETECTABLE |
| drift | **99.9%** | DETECTABLE |
| bias | **99.4%** | DETECTABLE |
| coordinated | **99.4%** | DETECTABLE |
| step | **99.4%** | DETECTABLE |
| intermittent | **84.2%** | DETECTABLE |
| **Mean** | **97.2%** | — |

### Self-Healing Performance

| Metric | Value |
|--------|-------|
| Error reduction (100m spoof) | **91.9%** |
| Stability (no oscillation) | **100%** |
| Real-time latency | **0.31ms** |
| Mode switch required | **0%** |

---

## Methodology

### Dataset: EuRoC MAV

- **Type:** Real UAV flight data with ground truth from motion capture
- **Sequences:** MH_01, MH_02, MH_03, V1_01, V1_02
- **Total samples:** 138,088
- **Features:** Position (x,y,z), velocity (vx,vy,vz), orientation (roll,pitch,yaw), angular rates (p,q,r)

### Evaluation Protocol

- **Training:** Unsupervised (normal flights only - no attack labels used)
- **Validation:** LOSO-CV (Leave-One-Sequence-Out Cross-Validation)
- **Attack injection:** Synthetic GPS spoofing on real trajectories
- **GPS noise:** 1.5m std (realistic consumer-grade)

### Why Real Data Matters

| Data Type | Coordinated Attack AUROC | Reason |
|-----------|--------------------------|--------|
| Synthetic (simulated) | ~50% | Simple random walks lack structure |
| **EuRoC (real)** | **99.4%** | Real flights have learnable patterns |

Real flight data contains:
- Complex maneuver signatures
- Sensor correlations (IMU-GPS coherence during turns, accelerations)
- Flight envelope constraints
- Natural temporal structure

The Mahalanobis detector learns these patterns, making even "physics-consistent" attacks detectable because they don't perfectly match the learned trajectory statistics.

---

## Detection Results

### AUROC by Attack Type and Magnitude

| Attack | 1x (1.5m) | 2x (3m) | 5x (7.5m) | 10x (15m) | 20x (30m) |
|--------|-----------|---------|-----------|-----------|-----------|
| bias | 62.6% | 82.1% | 97.5% | **99.4%** | 99.8% |
| drift | 99.3% | 99.7% | 99.9% | **99.9%** | 99.9% |
| noise_injection | 100% | 100% | 100% | **100%** | 100% |
| coordinated | 62.7% | 82.2% | 97.6% | **99.4%** | 99.8% |
| intermittent | 84.4% | 84.1% | 84.4% | **84.2%** | 84.0% |
| step | 62.6% | 82.1% | 97.5% | **99.4%** | 99.8% |

**Note:** Magnitude is relative to GPS noise floor (1.5m). So 10x = 15m offset.

### Detection Classification

| Classification | AUROC Range | Attacks |
|----------------|-------------|---------|
| **RELIABLE** (>95%) | 97-100% | bias, drift, noise, coordinated, step |
| **MODERATE** (80-95%) | 84% | intermittent |

**All 6/6 attack types are detectable at 10x magnitude.**

---

## Self-Healing Results (IASP)

The Inverse-Anchored State Projection (IASP) mechanism provides automatic recovery after spoof detection.

### Error Reduction by Spoof Magnitude

| Spoof Magnitude | Error Reduction | Stability | Latency (mean) | Latency (p99) |
|-----------------|-----------------|-----------|----------------|---------------|
| 25m | 67.6% | 100% | 0.30ms | 0.42ms |
| 50m | 83.8% | 100% | 0.29ms | 0.42ms |
| **100m** | **91.9%** | **100%** | **0.31ms** | 0.54ms |
| 200m | 95.9% | 100% | 0.28ms | 0.39ms |

### Key Metrics

| Self-Healing Aspect | Measured | Target | Status |
|---------------------|----------|--------|--------|
| Navigation error reduction | 91.9% | >70% | **PASS** |
| Stability (no oscillation) | 100% | >90% | **PASS** |
| Real-time feasibility (<5ms) | 100% | >99% | **PASS** |
| Mode switch required | 0% | 0% | **PASS** |
| Oscillation risk | 0% | <10% | **PASS** |

---

## Comparison: Synthetic vs Real Data

| Metric | Synthetic Data | EuRoC (Real) |
|--------|----------------|--------------|
| Bias AUROC @ 10x | 50.3% | **99.4%** |
| Drift AUROC @ 10x | 49.5% | **99.9%** |
| Coordinated AUROC @ 10x | 52.1% | **99.4%** |
| Mean AUROC | ~66% | **97.2%** |
| Detectable attacks | 2/6 | **6/6** |

### Why the Difference?

**Synthetic data:** Randomly generated trajectories lack the complex structure of real flights. Physics-consistent attacks are indistinguishable because the "normal" behavior has no learnable pattern.

**Real data:** Actual flights contain:
1. **Maneuver signatures** - Turns, accelerations have specific sensor patterns
2. **Temporal correlations** - Position-velocity-attitude consistency
3. **Flight envelope** - Physical constraints on motion
4. **Sensor coupling** - IMU-GPS coherence during maneuvers

The detector learns these patterns. Attacks, even "physics-consistent" ones, deviate from the learned normal distribution and become detectable.

---

## Threat Model

### Scope

| Included | Excluded |
|----------|----------|
| Passive monitoring | Active probing |
| Single vehicle | Multi-vehicle consistency |
| GPS pseudorange only | Carrier-phase/RTK |
| Consumer-grade IMU | Tactical-grade IMU |
| Attack magnitudes 1x-20x (1.5m-30m) | Attacks >30m or <1.5m |

### Attack Types Evaluated

| Attack | Description | Detection Mechanism |
|--------|-------------|---------------------|
| **bias** | Constant GPS offset | Deviation from learned patterns |
| **drift** | Slow AR(1) GPS drift | Accumulated deviation detection |
| **noise_injection** | Increased GPS noise | Variance increase |
| **coordinated** | GPS + velocity consistent | Joint distribution mismatch |
| **intermittent** | Random on/off attacks | Discontinuity detection |
| **step** | Sudden position jump | Transient detection |

---

## Reproducibility

### Configuration

```python
# Frozen configuration for reproducibility
DATA_PATH = "data/euroc/all_sequences.csv"
GPS_NOISE_STD = 1.5  # meters (realistic)
ATTACK_MAGNITUDES = [1.0, 2.0, 5.0, 10.0, 20.0]  # x noise floor
LOSO_CV = True  # Leave-One-Sequence-Out
RANDOM_SEED = 42
WINDOW_SIZE = 20  # samples
FEATURE_DIM = 12  # x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r
```

### Running Evaluation

```bash
# GPS spoofing detection evaluation
python gps_imu_detector/scripts/evaluate_euroc_gps.py

# Self-healing evaluation
python gps_imu_detector/scripts/evaluate_euroc_healing.py

# Results saved to:
#   results/euroc_gps_results.json
#   results/euroc_healing_results.json
```

---

## Valid Claims for Publication

### Supported Claims

| Claim | Evidence | Status |
|-------|----------|--------|
| "Detects all 6 GPS spoofing attack types on real flight data" | 97.2% mean AUROC | VALID |
| "Achieves >99% AUROC for 5/6 attack types at 10x magnitude" | EuRoC results | VALID |
| "91.9% navigation error reduction during active spoofing" | Healing results | VALID |
| "Real-time feasible with <1ms latency" | 0.31ms mean | VALID |
| "Unsupervised approach (no attack labels needed)" | Trained on normal only | VALID |

### Qualified Claims

| Claim | Qualification |
|-------|---------------|
| "All attacks detectable" | At 10x magnitude on EuRoC data |
| "Physics-consistent attacks detectable" | On real flight data with learned patterns |
| "Self-healing works" | For bias attacks; validated on EuRoC |

---

## Key Insight: Real Data vs Synthetic Data

**The central finding:**

> On **synthetic data**, physics-consistent attacks (bias, drift, coordinated) are indistinguishable (~50% AUROC) because simple random walk trajectories lack learnable structure.
>
> On **real flight data**, the same attacks become detectable (>99% AUROC) because actual flights contain complex, learnable patterns that attacks cannot perfectly mimic.

**Implication for deployment:** Results on synthetic data underestimate real-world performance. Evaluating on realistic datasets like EuRoC provides more accurate predictions of operational capability.

---

## Version History

| Version | Date | Key Change |
|---------|------|------------|
| 2.0.0 | 2026-01-01 | Honest evaluation (synthetic data) |
| 3.0.0 | 2026-01-01 | Boundary-setting with indistinguishability classes |
| 3.1.0 | 2026-01-01 | Statistical rigor fixes |
| **3.2.0** | **2026-01-01** | **EuRoC validation: 97.2% AUROC, 91.9% healing** |

---

*Version 3.2.0 - EuRoC Validated - 2026-01-01*
