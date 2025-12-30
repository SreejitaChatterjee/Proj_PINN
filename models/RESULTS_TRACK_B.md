# Track B Results: Fault Classification

**Question:** Given labels, how well can faults be classified in practice?

**Scope:** Supervised learning with labeled fault data. This is an engineering problem, not adversarial security.

---

## Table 1: PADRE Classifier (Random Forest)

| Metric | Value |
|--------|-------|
| **Accuracy** | **99.97%** |
| Precision | 99.96% |
| Recall | 100% |
| F1 | 99.98% |

### Confusion Matrix

|  | Pred Normal | Pred Faulty |
|--|-------------|-------------|
| **Actual Normal** | 402 | 2 |
| **Actual Faulty** | 0 | 5454 |

---

## Table 2: Cross-Drone Transfer

| Method | Accuracy |
|--------|----------|
| Physics-based rules | **100%** |
| ML (Bebop→Solo) | 88.9% |
| ML (Solo→Bebop) | 94.5% |

**Finding:** Physics-based rules transfer perfectly; ML requires retraining.

---

## Table 3: Model Comparison

| Model | Accuracy | Parameters |
|-------|----------|------------|
| Random Forest | 99.97% | 168 features |
| Binary CNN | 98.01% | - |
| PINN | 1.46 separation | 204,818 |

---

## What This Assumes

- Labeled training data exists
- Fault types are known a priori
- No adversarial attacker

**These assumptions do not hold in Track A.** Do not compare numbers across tracks.

---

*Results in `models/padre_*/results*.json`*
