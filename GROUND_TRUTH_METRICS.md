# Ground Truth Metrics - Single Source of Truth

**Last Updated:** 2025-12-30
**Purpose:** This file is the ONLY authoritative source for all performance claims.

---

## CRITICAL: What Works vs What Doesn't

### GPS-IMU Detector (gps_imu_detector/)

| Component | Code Exists | Validated | AUROC | Works? |
|-----------|-------------|-----------|-------|--------|
| physics_residuals.py | ✅ | ✅ | 0.562 | **NO** |
| ekf.py | ✅ | ❌ (API issue) | N/A | UNKNOWN |
| feature_extractor.py | ✅ | ✅ | 0.582 | **NO** |
| hybrid_scorer.py | ✅ | ✅ | 0.500 | **NO** |
| Simple CNN-GRU baseline | ✅ | ✅ | 0.454 | **NO** |

**Conclusion:** None of the physics components effectively detect attacks on EuRoC data.

### ALFA Fault Detection (research/security/)

| Metric | Claimed | Validated | Status |
|--------|---------|-----------|--------|
| F1 Score | 65.7% | ⚠️ Single run | UNVERIFIED |
| Precision | 100% | ⚠️ Small test set | UNVERIFIED |
| FPR | 4.5% | ⚠️ No seeds | UNVERIFIED |
| Hardware | Not documented | ❌ | MISSING |
| Seeds | Not documented | ❌ | MISSING |
| Splits | Not versioned | ❌ | MISSING |

**Conclusion:** ALFA claims exist but lack reproducibility documentation.

### Security Models (models/security/)

| Model | Type | F1 | Precision | Recall | FPR | Status |
|-------|------|-----|-----------|--------|-----|--------|
| supervised_detector | ML | 86.6% | 100% | 76.4% | 2.0% | Best |
| sensor_fusion_v2 | Physics | 68.5% | 79.2% | 67.4% | ~5% | OK |
| pinn_residual | PINN | 0.5% | 0.2% | 28.0% | 4.3% | **FAILS** |

**Conclusion:** PINN-based approaches underperform simple ML baselines.

---

## The Core Contradiction

### Paper Claims (paper_v3_integrated.tex):
> "PINN-based detector achieves deployment-ready performance"

### Reality (FINAL_COMPARISON_ALL_METHODS.txt):
> "PINN APPROACH DOES NOT WORK"
> "Don't pursue this direction further"

### Paper Also Admits:
> "pure data-driven detection significantly outperforms physics-informed variants (p<10⁻⁶)"

**Resolution:** The paper is internally inconsistent. It claims PINN works while showing data that proves physics doesn't help.

---

## What Can Be Claimed Honestly

### With Strong Evidence:
1. Latency meets target: P99 = 2.69ms < 5ms ✓
2. Model size meets target: 0.03MB < 1MB ✓
3. 91 unit tests pass ✓
4. Code infrastructure works ✓

### With Weak Evidence (Needs Verification):
1. ALFA 65.7% F1 - single run, no seeds documented
2. ALFA 100% precision - small test set, may not generalize

### Cannot Be Claimed:
1. "Physics-first detection works" - AUROC ~0.5 (random)
2. "PINN-based detection is effective" - Data shows otherwise
3. "Deployment-ready" - No real-world validation

---

## Recommended Actions

1. **Remove PINN claims** from paper unless you can show physics actually helps
2. **Document ALFA reproducibility** - seeds, hardware, exact splits
3. **Use supervised_detector** (86.6% F1) instead of PINN approaches
4. **Be honest** - physics approach didn't work as hoped

---

## Validated Test Configurations

### GPS-IMU Component Validation
- **Date:** 2025-12-30
- **Platform:** Windows 11, AMD64 Family 25 Model 80
- **Python:** 3.14.0
- **Seed:** 42
- **Train sequences:** MH_01_easy, MH_02_easy, MH_03_medium
- **Test sequences:** V1_01_easy, V1_02_medium
- **Attacks tested:** bias, drift, noise, coordinated

### Simple Baseline Validation
- **Architecture:** CNN(32)-GRU(32)-FC(1)
- **Parameters:** 7,841
- **Mean AUROC:** 0.454 (random)
- **Latency:** 2.69ms P99

---

## File References

| File | Contains | Trust Level |
|------|----------|-------------|
| `gps_imu_detector/results/validated_results.json` | Baseline metrics | HIGH |
| `gps_imu_detector/results/component_validation.json` | Physics component metrics | HIGH |
| `models/security/FINAL_COMPARISON_ALL_METHODS.txt` | All method comparison | HIGH |
| `models/security/rigorous_evaluation/HONEST_RESULTS.txt` | EuRoC rigorous eval | HIGH |
| `research/security/paper_v3_integrated.tex` | Paper claims | LOW (unverified) |

---

*This document is the single source of truth. All other documents should reference this.*
