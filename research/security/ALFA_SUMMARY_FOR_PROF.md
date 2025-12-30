# UAV Security Detection Research: Summary for Professor

**To:** Professor
**From:** Sreejita
**Date:** 2025-12-30
**Subject:** Honest Research Update - PINN Contradiction Resolved

## CRITICAL UPDATE: Physics Doesn't Help

See `PINN_CONTRADICTION_RESOLVED.md` for details.

| Finding | Evidence |
|---------|----------|
| Neural network detects faults | 65.7% F1 ✓ |
| Physics constraints help | w=0 > w=20 ✗ |
| "PINN-based" is accurate | Physics AUROC 0.5 ✗ |

**Recommendation:** Remove "PINN" claims from paper. Claim neural network detection only.

---

## Executive Summary

We have two research tracks for UAV sensor security:

1. **ALFA Fault Detection** - Has results but lacks reproducibility documentation
2. **GPS-IMU Anomaly Detector** - CODE FRAMEWORK ONLY, no validated results yet

**IMPORTANT:** Neither track is ready for publication without additional validation work.

---

## Track 1: ALFA Fault Detection

### Results (WITH CAVEATS)

**These numbers exist but are NOT yet publication-ready:**

| Metric | Value | Caveat |
|--------|-------|--------|
| F1 Score | 65.7% | Single experimental run |
| Precision | 83.3% | May not generalize |
| FPR | 4.5% | Threshold-dependent |
| Inference | 0.34 ms | Hardware NOT documented |
| Model Size | 0.79 MB | FP32, no quantization |

### What's Missing for Publication

- [ ] Hardware specification (CPU, RAM, OS)
- [ ] Random seeds for reproducibility
- [ ] Versioned train/test splits
- [ ] Statistical significance methodology
- [ ] Circularity verification (are sensors derived from ground truth?)
- [ ] Independent latency verification

### Comparison with Baselines

| Method | F1 Score | FPR | Status |
|--------|----------|-----|--------|
| **Our PINN** | 65.7% | **4.5%** | Needs verification |
| One-Class SVM | 96.1% | 62.9% | Baseline |
| Isolation Forest | 21.7% | 10.0% | Baseline |

**Key insight:** Our FPR is best, but claims need proper documentation.

### Paper Status
- `paper_v3_integrated.tex` exists
- Contains unverified claims that need fixing
- NOT ready for submission without reproducibility info

---

## Track 2: GPS-IMU Anomaly Detector

### HONEST STATUS

| What We Have | What We Don't Have |
|--------------|-------------------|
| ~10,000 lines of code | Trained models |
| 91 passing unit tests | Actual evaluation results |
| Evaluation scripts | Measured performance |
| Architecture design | Validated detection |

### Implementation Status

| Component | Code Exists | Validated |
|-----------|-------------|-----------|
| Feature Extractor | ✅ | ❌ |
| EKF with NIS | ✅ | ❌ |
| Physics Residuals | ✅ | ❌ |
| CNN-GRU Detector | ✅ | ❌ |
| Hybrid Scorer | ✅ | ❌ |
| Hard Negatives | ✅ | ❌ |
| Quantization | ✅ | ❌ |
| Evaluation | ✅ | ❌ |

### Validated Metrics (2025-12-30)

**CRITICAL: Only tested simple CNN-GRU baseline. Physics components NOT validated.**

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Latency (P99) | ≤5ms | 2.69ms | **PASS** |
| Model Size | <1MB | 0.03MB | **PASS** |
| Mean AUROC | ≥0.90 | 0.454 | **FAIL** |
| Recall@5%FPR | ≥95% | 1.4% | **FAIL** |

**What Was NOT Tested:**
- physics_residuals.py - ❌ NOT validated on real data
- ekf.py - ❌ NOT validated on real data
- hybrid_scorer.py - ❌ NOT validated on real data

**Key Finding:** Simple CNN-GRU = random chance. Physics components (main contribution) are UNTESTED.

### Code Deliverables

```
gps_imu_detector/
├── src/           # 17 modules - CODE EXISTS
├── scripts/       # 3 scripts - CODE EXISTS
├── configs/       # baseline.yaml - EXISTS
├── experiments/   # eval.py - EXISTS
├── ci/            # leakage_check.sh - EXISTS
├── tests/         # 91 tests - PASS
├── models/        # EMPTY
└── results/       # EMPTY
```

---

## What We CAN Honestly Claim (With Evidence)

1. We built a well-structured framework for GPS-IMU anomaly detection
2. The code follows best practices (no circular sensors, LOSO-CV)
3. 91 unit tests pass
4. **Latency:** 2.69ms P99 - MEETS target of <5ms
5. **Model size:** 0.03MB - MEETS target of <1MB
6. Evaluation infrastructure works and produces reproducible results

## What We CANNOT Claim

1. **Detection does NOT work:** AUROC 0.454 is worse than random (0.5)
2. **Recall is terrible:** 1.4% at 5% FPR (target was ≥95%)
3. The simple unsupervised CNN-GRU approach is NOT sufficient
4. NOT deployment-ready for attack detection

---

## Required Work Before Publication

### Track 1 (ALFA)
1. Document hardware used
2. Record and version random seeds
3. Create reproducible splits
4. Verify no circularity in sensors
5. Re-run with proper documentation

### Track 2 (GPS-IMU)
1. Download and prepare EuRoC dataset
2. Run training pipeline
3. Run evaluation pipeline
4. Measure actual latency
5. Document all results

---

## Recommendation

**Honest Assessment:** We have validated infrastructure, but the detection approach needs work.

**What's Good:**
- Clean architecture and code
- Proper evaluation methodology
- Latency and model size targets MET
- Reproducibility mechanisms in place

**What Needs Work:**
- Detection performance is random-level (AUROC 0.454)
- Simple unsupervised CNN-GRU is NOT sufficient
- Need physics-based or supervised approach

**Next Steps:**
1. Implement PINN-based physics residuals for unsupervised detection
2. Try supervised approach with labeled attack data
3. Implement hybrid scoring combining physics + ML

---

*Updated: 2025-12-30 with validated evaluation results.*
*Hardware: AMD64 Family 25 Model 80, Windows 11, Python 3.14.0, PyTorch 2.9.0*
*Seed: 42, Train: 3 sequences, Test: 2 sequences*
