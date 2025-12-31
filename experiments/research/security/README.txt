# Residual-Based CPS Monitor

*Detectability limits of GPS spoofing under dynamics-based anomaly detection.*

---

## Novelty

> We introduce **Residual Equivalence Classes (RECs)** to characterize when GPS spoofing detection is fundamentally ill-posed. Consistent spoofing lies within the nominal REC, rendering residual-based detection impossible.

---

## Contributions

1. **Residual Equivalence Classes (RECs):** Formal lens to analyze detectability under learned dynamics
2. **Impossibility Result:** Consistent GPS spoofing lies within nominal REC → undetectable
3. **Physics Interaction:** Physics priors collapse RECs, degrading anomaly sensitivity

---

## Detection vs Identification

| Problem | Residual Detectors |
|---------|:------------------:|
| Inconsistency detection | ✅ |
| Truth identification | ❌ |

**Implication:** Spoofing defense is an *identification problem* requiring external anchors.

---

## Results

| Attack | AUROC | 95% CI | REC |
|--------|:-----:|:------:|:---:|
| drift | 0.919 | [0.89, 0.94] | Different |
| noise | 0.907 | [0.88, 0.93] | Different |
| bias | 0.866 | [0.83, 0.90] | Different |
| consistent | — | — | Same ❌ |

---

## Engineering Rigor

| Component | Status |
|-----------|:------:|
| CI leakage gate | ✅ |
| Sequence-wise splits | ✅ |
| Bootstrap CIs | ✅ |
| Cross-dataset transfer | ✅ |
| Quantization profiling | ✅ |
| Reproducibility bundle | ✅ |

---

## Reproduce

```bash
# CI check (mandatory)
python gps_imu_detector/ci/leakage_check.py --check-code

# Full evaluation
python gps_imu_detector/run_all.py --seed 42
```

See `gps_imu_detector/README.md` for details.
