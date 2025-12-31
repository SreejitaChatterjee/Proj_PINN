# Project Freeze Declaration

**Date:** 2025-12-30
**Status:** ACTIVE
**Purpose:** Prevent over-polishing and scope creep

---

## Freeze Scope

This document declares what is **frozen** and what remains open for modification.

---

## 1. EXPERIMENT FREEZE

**Effective:** 2025-12-30

### Frozen (No New Experiments)

- [ ] No new attack types beyond catalog.json
- [ ] No new attack magnitudes outside defined ranges
- [ ] No new sensor modalities (GPS, IMU only)
- [ ] No new trajectory patterns

### Canonical Experiments (12 total)

| ID | Experiment | Purpose |
|----|------------|---------|
| E1 | Constant offset sweep (1-100m) | Impossibility + Scaling |
| E2 | Consistent drift (ramp) | ICI sensitivity |
| E3 | AR(1) stealthy drift | Fundamental limit |
| E4 | Intermittent attack | Boundary of applicability |
| E5 | Oscillatory attack | Temporal pattern |
| E6 | Healing evaluation (100m) | IASP validation |
| E7 | Noise robustness | Sensor noise impact |
| E8 | Seed robustness (3 seeds) | Statistical validity |
| E9 | Latency benchmark | Operational feasibility |
| E10 | Quiescence test | False positive control |
| E11 | Residual baseline | Impossibility demonstration |
| E12 | Hybrid fusion | EKF + ICI comparison |

**Any experiment not in this list is archived.**

---

## 2. METRIC FREEZE

**Effective:** 2025-12-30

### Frozen Metrics (Report These Only)

| Metric | Definition | Required |
|--------|------------|----------|
| AUROC | Area under ROC curve | YES |
| Recall@1%FPR | TPR at 1% false positive rate | YES |
| Recall@5%FPR | TPR at 5% false positive rate | YES |
| Error Reduction (%) | (before-after)/before for healing | YES |
| P95 Latency (ms) | 95th percentile inference time | YES |
| False Alarms/Hour | Expected false positives at 100 Hz | YES |

### Deprecated Metrics (Do Not Report)

- F1 score (threshold-dependent)
- Precision (imbalanced data)
- Accuracy (meaningless for anomaly detection)
- AUC-PR (redundant with AUROC for our use case)

---

## 3. MODEL FREEZE

**Effective:** 2025-12-30

### Frozen Architecture

| Component | Configuration |
|-----------|--------------|
| Forward model f_theta | MLP(64, 64, state_dim) |
| Inverse model g_phi | MLP(64, 64, state_dim) |
| State dimension | 6 (x, y, z, vx, vy, vz) |
| Training epochs | 30 |
| Batch size | 32 |
| Learning rate | 1e-3 |
| Optimizer | Adam |

### Frozen Parameters (Do Not Tune)

- Saturation constant C = 50
- Healing threshold = p99 of nominal ICI
- Detection threshold = p95 of nominal ICI

---

## 4. CLAIM FREEZE

**Effective:** 2025-12-30

### Frozen Claims (From CLAIMS.md)

1. Residual impossibility for consistent spoofing
2. ICI detection for offsets >= 25m
3. IASP error reduction >= 74%
4. Fundamental limit for AR(1) attacks

### No New Claims Without:

- New experimental evidence
- Bootstrap CI
- Update to CLAIMS.md
- Reviewer-facing justification

---

## 5. CODE FREEZE

**Effective:** 2025-12-30

### Frozen Files (Core Contribution)

```
gps_imu_detector/
  src/inverse_model.py      # FROZEN - ICI + IASP
  src/ekf/ekf_position.py   # FROZEN - EKF-NIS baseline
  src/hybrid/fuse_scores.py # FROZEN - Hybrid fusion
```

### Allowed Modifications

- Bug fixes (with test)
- Documentation improvements
- Visualization scripts
- CI/CD configuration

### Prohibited Modifications

- Architecture changes
- Hyperparameter changes
- New features
- Performance "optimizations"

---

## 6. RESULTS FREEZE

**Effective:** 2025-12-30

### Canonical Results (From comprehensive_validation.json)

| Metric | Value | Status |
|--------|-------|--------|
| ICI AUROC (>= 25m) | 1.000 | FROZEN |
| Residual AUROC | 0.500 | FROZEN |
| IASP Error Reduction | 77.1% | FROZEN |
| P95 Latency | 1.53 ms | FROZEN |
| Quiescence False Trigger | 1.0% | FROZEN |

**These numbers appear in the paper. Do not re-run experiments.**

---

## Freeze Violations

If you must violate a freeze:

1. Document the reason
2. Create a new branch
3. Update CLAIMS.md
4. Re-run all canonical experiments
5. Get explicit approval

---

## Unfreezing Process

To unfreeze any component:

1. Create GitHub issue with justification
2. Tag as "UNFREEZE REQUEST"
3. Wait for review
4. If approved, update this document with new freeze date

---

## Why This Matters

**Over-polishing kills papers.**

Every additional experiment, metric, or claim increases:
- Reviewer attack surface
- Internal inconsistency risk
- Time to submission

The current state is **sufficient for publication**.
Protect it.

---

## Signature

```
Frozen by: Claude Code
Date: 2025-12-30
Scope: Full project
Duration: Until paper acceptance
```
