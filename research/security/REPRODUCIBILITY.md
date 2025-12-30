# Reproducibility Documentation

**Last Updated:** 2025-12-30
**Status:** INCOMPLETE - Missing critical information

---

## ALFA Fault Detection Results

### Claimed Metrics (paper_v3_integrated.tex)

| Metric | Value |
|--------|-------|
| F1 Score | 65.7% |
| Precision | 100% |
| Recall | 55.6% |
| FPR | 4.5% |
| Flights | 47 |

### Reproducibility Status: ❌ CANNOT REPRODUCE

| Requirement | Status | Notes |
|-------------|--------|-------|
| ALFA Data | ❌ MISSING | Not in repository |
| Random Seed | ⚠️ 42 | Used in sklearn models |
| Train/Test Split | ❌ NOT VERSIONED | No split file saved |
| Hardware Spec | ❌ NOT DOCUMENTED | - |
| Code Version | ❌ NOT TAGGED | No git tag for paper |
| Preprocessing | ⚠️ UNDOCUMENTED | scripts/security/preprocess_alfa.py referenced but not found |

### Missing Files

1. `data/attack_datasets/processed/alfa/` - Preprocessed ALFA data
2. `scripts/security/preprocess_alfa.py` - Preprocessing script
3. `configs/alfa_evaluation.yaml` - Evaluation configuration
4. `results/alfa/` - Saved results with full metadata

### To Reproduce

1. Download ALFA dataset from https://theairlab.org/alfa-dataset/
2. Preprocess with documented script (MISSING)
3. Run evaluation with documented seeds (PARTIALLY DOCUMENTED)
4. Compare to saved results (MISSING)

---

## GPS-IMU Detector Results

### Claimed Metrics

| Component | AUROC | Status |
|-----------|-------|--------|
| physics_residuals.py | 0.562 | ✅ Validated |
| feature_extractor.py | 0.582 | ✅ Validated |
| Simple CNN-GRU | 0.454 | ✅ Validated |

### Reproducibility Status: ✅ CAN REPRODUCE

| Requirement | Status | Notes |
|-------------|--------|-------|
| EuRoC Data | ✅ Present | data/euroc/all_sequences.csv |
| Random Seed | ✅ 42 | Documented |
| Train/Test Split | ✅ Documented | 3 train, 2 test sequences |
| Hardware Spec | ✅ Documented | AMD64 Family 25, Windows 11 |
| Code | ✅ Present | gps_imu_detector/validate_all_components.py |

### Reproducing

```bash
cd gps_imu_detector
python validate_all_components.py
# Results saved to results/component_validation.json
```

---

## Security Models Results

### Documented in models/security/README.md

| Model | F1 | Status |
|-------|-----|--------|
| supervised_detector | 86.6% | ✅ Best model |
| sensor_fusion_v2 | 68.5% | ✅ Second best |
| pinn_residual | 0.5% | ❌ Does not work |

### Reproducibility Status: ⚠️ PARTIAL

- Models exist but training scripts not documented
- Evaluation methodology in FINAL_COMPARISON_ALL_METHODS.txt
- Hardware not documented

---

## Critical Gaps

### 1. ALFA Results Cannot Be Reproduced
- Data not in repository
- Preprocessing not documented
- No saved splits or results

### 2. Paper Claims Contradict Evidence
- Paper: "PINN achieves deployment-ready performance"
- Evidence: PINN approach AUROC ~0.5 (random)
- Evidence: "PINN DOES NOT WORK" in FINAL_COMPARISON

### 3. 20-Seed Statistical Testing
- Paper claims "rigorous 20-seed statistical testing"
- No evidence of 20 different ALFA evaluations
- Single configuration appears to have been run

---

## Recommendations

1. **Download and include ALFA data** (or document download steps)
2. **Create versioned splits** saved to file
3. **Document all hyperparameters** in config file
4. **Run multiple seeds** and report mean±std
5. **Tag git commit** for paper version
6. **Resolve paper contradictions** - either physics helps or it doesn't

---

## Hardware Used for Validation

### GPS-IMU Validation (2025-12-30)

```
Platform: Windows-11-10.0.26200-SP0
Processor: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD
Python: 3.14.0
PyTorch: 2.9.0+cpu
Seed: 42
```

### ALFA Evaluation (Unknown)

```
Platform: NOT DOCUMENTED
Processor: NOT DOCUMENTED
Python: NOT DOCUMENTED
PyTorch: NOT DOCUMENTED
Seed: 42 (inferred from sklearn)
```

---

*This document tracks reproducibility status. Update as gaps are filled.*
