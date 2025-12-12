# Paper Versions Summary: Publication Strategy

## IMPORTANT UPDATE (December 2024)

**After reviewing actual experimental data, several claimed results have been corrected:**

### What the Data Actually Shows:
| Architecture | Single-Step z MAE | 100-Step Position MAE | Parameters |
|-------------|-------------------|----------------------|------------|
| Baseline | 0.079m | 5.09m | 205K |
| Fourier | 0.076m | 5.09m | 302K |
| **Modular** | **0.058m** | **1.11m** | **72K** |

### Corrected Claims:
- ~~"3.5 million times worse"~~ - Fourier performs nearly identical to baseline
- ~~"51x improvement"~~ - Curriculum training gives ~25% improvement in ablation
- ~~"Expressivity-stability tradeoff"~~ - Not supported; Fourier ≈ Baseline

### Solid Findings:
1. **Modular architecture is 4.6x better** at 100-step rollout (1.11m vs 5.09m)
2. **Modular uses 65% fewer parameters** (72K vs 205K)
3. **Curriculum training provides 25% improvement** in ablation study
4. **Motor coefficients perfectly identifiable** (0% error)
5. **Inertias poorly identifiable** (50-60% error, observability limited)

---

## Publication Plan Overview

This research produces papers from the quadrotor PINN work. The core finding is that **modular architectures outperform monolithic baselines** for autoregressive stability.

---

## Paper Distribution

| Paper | Venue | Core Novelty | Status |
|-------|-------|--------------|--------|
| **Paper 1** | ACC/CDC | Stability Envelope $H_\epsilon$ (formal framework) | Needs revision |
| **Paper 2** | NeurIPS 2025 | Architecture comparison for stability | Needs revision |
| **Paper 3** | ICRA 2026 | Modular architecture design | Needs revision |
| **Paper 4** | RA-L | Practical methodology | Needs revision |
| **Paper 5** | CDC/L4DC | Prediction-Identification patterns | Draft |

---

## Detailed Paper Mapping

### Paper 1: ACC/CDC - Stability Envelope Framework

**Title:** "The Stability Envelope: A Formal Framework for Autoregressive Stability in Physics-Informed Neural Networks"

**Core Contribution:**
- Formal definition of stability envelope $H_\epsilon$
- First principled metric for evaluating learned dynamics for control
- Modular architecture achieves 4.6x better stability

**Key Equations:**
- $H_\epsilon = \max\{K : \mathbb{E}[\|\hat{\mathbf{x}}_{t+K} - \mathbf{x}_{t+K}\|] < \epsilon\}$

---

### Paper 2: Architecture Comparison

**Title:** "Physics-Informed Architecture Design for Stable Autoregressive Dynamics Prediction"

**Core Contribution:**
- Systematic comparison of monolithic, modular, and Fourier-enhanced architectures
- Finding: Modular is best, Fourier shows no difference from baseline

**Key Results (ACTUAL DATA):**
- Modular: 4.6x better 100-step stability (1.11m vs 5.09m)
- Fourier: No significant difference from baseline
- Modular uses 65% fewer parameters

---

### Paper 3: ICRA 2026 - Modular Architecture Design

**Title:** "Modular Architecture Design for Stable Physics-Informed Neural Network Dynamics Learning"

**Core Contribution:**
- Demonstration that separating translation/rotation improves stability
- Analysis of why modular design helps despite physical coupling

**Key Results:**
- Modular: 4.6x better at 100-step (1.11m vs 5.09m)
- 65% fewer parameters (72K vs 205K)

---

### Paper 4: RA-L - Practical Methodology

**Title:** "Curriculum Training for Physics-Informed Neural Networks"

**Core Contribution:**
- Training methodology for improved stability
- Ablation study of training techniques

**Key Results (ACTUAL DATA from ablation):**
- Curriculum training: 25% improvement (0.076m vs 0.101m)
- Scheduled sampling: 10% improvement
- Dropout/Energy constraints: FAILURE (hurt performance)

---

### Paper 5: CDC/L4DC - Prediction-Identification Patterns

**Title:** "Architecture-Dependent Parameter Identification in Physics-Informed Neural Networks"

**File:** `new_novelty.tex`

**Core Contribution:**
- Discovery that prediction accuracy and identification accuracy show different patterns
- Architecture-specific parameter identification results

**Key Results (FROM ACTUAL EXPERIMENTS):**
- Modular: Better mass ID (7.7% vs 40%)
- Motor coefficients: 0% error across ALL architectures
- Inertias: 50-60% error (observability limited at small angles)

---

## Honest Assessment

**Realistic Venue Targets:**
- RA-L (Robotics and Automation Letters) - Solid robotics contribution
- ICRA workshop - Preliminary results
- CDC/ACC - If theoretical analysis is strengthened

**Stretch (need more work):**
- NeurIPS/ICML - Need breakthrough finding, not incremental improvement

---

## File Locations

```
paper_versions/
├── ACC_CDC_submission.tex      # Stability Envelope H_epsilon
├── NeurIPS_2025_submission.tex # Architecture comparison
├── ICRA_2026_submission.tex    # Modular architecture design
├── RAL_submission.tex          # Practical methodology
├── new_novelty.tex             # Parameter identification
└── SUMMARY_comparison.md       # This file
```

---

## Summary

The core finding is that **modular architectures (separating translation/rotation) achieve 4.6x better autoregressive stability while using 65% fewer parameters**. This is a solid empirical result, though not a dramatic breakthrough. The "expressivity-stability tradeoff" narrative is not supported by the data.
