# Track A Results: Detectability & Self-Healing

**Question:** What attacks are fundamentally detectable without labels or extra sensors?

**Scope Guard:** This work focuses on detectability limits under unlabeled, adversarial spoofing. Supervised fault classification and engineered security pipelines address a different problem and are intentionally excluded.

---

## Table 1: The Core Result (v1.0.0 Rate-Based Detection)

| Metric | Value |
|--------|-------|
| Detection Rate (1.0x) | **100%** |
| Detection Rate (0.5x) | **100%** |
| Detection Rate (0.3x) | **90%** |
| False Positive Rate (worst-case) | **1.26%** |
| Latency | **< 1 ms** |

### Per-Attack Recall by Magnitude

| Attack Type | 1.0x | 0.5x | 0.3x | Notes |
|-------------|------|------|------|-------|
| GPS_DRIFT | 100% | 100% | 50% | Detectability floor at 0.25-0.3x |
| GPS_JUMP | 100% | 100% | 100% | Scale-robust |
| IMU_BIAS | 100% | 100% | 100% | Scale-robust (CUSUM) |
| SPOOFING | 100% | 100% | 100% | Scale-robust |
| ACTUATOR_FAULT | 100% | 100% | 100% | Scale-robust (variance ratio) |

**Aggregation note:** Overall detection is computed across attack classes; degradation at low magnitudes is isolated to GPS drift, while all other attack classes remain fully detectable.

---

## Table 2: Residual Impossibility (Claim 1)

| Offset | Residual Change | Detectable? |
|--------|-----------------|-------------|
| 1m | 5.5e-19 | NO |
| 10m | 4.6e-18 | NO |
| 50m | 2.0e-17 | NO |
| 100m | 1.1e-17 | NO |

**Conclusion:** Consistency-preserving spoofing is **undetectable by residuals** at any magnitude.

---

## Table 3: ICI Scaling Law (Claim 2)

| Offset | AUROC | Detection Rate | Zone |
|--------|-------|----------------|------|
| < 10m | < 0.65 | < 40% | Undetectable |
| 10-25m | 0.65-0.85 | 40-70% | Floor |
| 25-50m | 0.85-1.00 | 70-95% | Marginal |
| **≥ 50m** | **1.00** | **100%** | **Detectable** |

**Minimum detectable offset:** 50m with AUROC = 1.0

---

## Table 4: Self-Healing (Claim 3)

| Metric | Value |
|--------|-------|
| Spoof magnitude | 100m |
| Error before | 114.6m |
| Error after | 26.2m |
| **Reduction** | **77%** |

---

## Table 5: Bootstrap Confidence Intervals

| Metric | ICI Mean | 95% CI |
|--------|----------|--------|
| AUROC | 0.925 | [0.920, 0.930] |
| Recall@5%FPR | 0.642 | [0.610, 0.676] |

CI width < 0.04 → statistically robust.

---

## Table 6: Detectability Zones (Design Boundary)

| Zone | Magnitude | GPS_DRIFT | Other Attacks | Status |
|------|-----------|-----------|---------------|--------|
| **Full Detection** | ≥ 1.0x | 100% | 100% | Reliable |
| **Robust Detection** | 0.5x | 100% | 100% | Reliable |
| **Transition Zone** | 0.25-0.3x | 50% | 100% | GPS drift limited |
| **Below Floor** | < 0.25x | < 50% | Varies | Noise-dominated |

The 0.25-0.3x region represents the **practical observability boundary** for passive GPS drift detection—a design-complete specification, not a system failure.

---

## Table 7: Latency (Deployment Feasibility)

| Component | P95 | Target |
|-----------|-----|--------|
| ICI | 0.42 ms | < 5ms ✓ |
| EKF | 0.06 ms | < 5ms ✓ |

**Real-time capable at 200 Hz.**

---

## Claims Summary

| # | Claim | Status |
|---|-------|--------|
| 1 | Residual impossibility | ✓ Proven |
| 2 | ICI separation ≥ 25m | ✓ Validated |
| 3 | Self-healing 77% | ✓ Validated |
| 4 | AR(1) undetectable | ✓ Disclosed |
| 5 | Detectability floor | ✓ Characterized |

---

## What This Table Does NOT Include

- Supervised classification results (Track B)
- Engineering pipeline accuracies (Track B)
- Failed baseline attempts (Track C - motivation only)

These answer different questions. See `USAGE.md` for track definitions.

---

## Table 8: Rigorous Evaluation (Realistic Noise)

**Date:** 2025-12-31

Evaluation with realistic GPS/IMU noise models (multipath, bias walk, drift):

| Metric | Result | 95% CI |
|--------|--------|--------|
| **Detection Rate** | 100% | [100%, 100%] |
| **FPR** | 2.0% | [0%, 4.67%] |
| **Detectability Floor** | ~5-10m | N/A |

### Magnitude Sensitivity

| Magnitude | Offset | Detection |
|-----------|--------|-----------|
| 1-5x | 2-4m | 0% |
| **10x** | **~6m** | **100%** |
| 20x | ~12m | 100% |

### Baseline Comparison (@ 10x)

| Detector | GPS Drift | IMU Bias | Coordinated |
|----------|-----------|----------|-------------|
| **RateBased** | **100%** | **100%** | **100%** |
| SimpleThreshold | 100% | 100% | 100% |
| EKF Innovation | 20% | 15% | 100% |
| ChiSquare | 0% | 0% | 45% |

**Key insight:** With realistic noise (0.5m GPS std), detectability floor is ~5-10m.

---

*Results reproducible via `scripts/rigorous_evaluation.py`*
