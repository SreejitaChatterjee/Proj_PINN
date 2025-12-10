# Paper Versions Summary: Publication Strategy

## Publication Plan Overview

This research produces **FOUR distinct papers** from the quadrotor PINN work, each with a unique focus to maximize publication output while minimizing reviewer confusion.

---

## Paper Distribution by Novelty

| Paper | Venue | Core Novelty | Status |
|-------|-------|--------------|--------|
| **Paper 1** | ACC/CDC | Stability Envelope $H_\epsilon$ (formal framework) | Ready |
| **Paper 2** | NeurIPS 2025 | Expressivity-Stability Tradeoff (main empirical result) | Ready |
| **Paper 3** | ICRA 2026 | Failure Modes Analysis (modular decoupling + Fourier drift) | Ready |
| **Paper 4** | RA-L | Curriculum Stability Training (the solution) | Ready |

---

## Detailed Paper Mapping

### Paper 1: ACC/CDC - Stability Envelope Framework

**Title:** "The Stability Envelope: A Formal Framework for Autoregressive Stability in Physics-Informed Neural Networks"

**Core Contribution:**
- Formal definition of stability envelope $H_\epsilon$
- Lipschitz stability bounds (Theorem)
- Frequency-coupling stability law (Proposition)
- First principled metric for evaluating learned dynamics for control

**Key Equations:**
- $H_\epsilon = \max\{K : \mathbb{E}[\|\hat{\mathbf{x}}_{t+K} - \mathbf{x}_{t+K}\|] < \epsilon\}$
- Stability bound based on Lipschitz constant

**What's NOT in this paper:**
- Deep failure mode analysis (saved for ICRA)
- Detailed curriculum methodology (saved for RA-L)
- Full tradeoff demonstration (saved for NeurIPS)

---

### Paper 2: NeurIPS 2025 - Expressivity-Stability Tradeoff

**Title:** "The Expressivity-Stability Tradeoff in Physics-Informed Neural Networks: Why Complex Architectures Fail at Autoregressive Dynamics Prediction"

**Core Contribution:**
- First empirical demonstration of inverse relationship between single-step accuracy and multi-step stability
- 2-6 orders of magnitude tradeoff across architectures
- Challenge to assumption that more expressive = better

**Key Results:**
- Fourier: 10x better 1-step, 3.5M x worse 100-step
- Modular: 2x better 1-step, 20x worse 100-step

**Supporting (but not main focus):**
- Stability envelope concept (brief intro)
- Failure mode overview (not detailed)
- Curriculum solution (summarized)

---

### Paper 3: ICRA 2026 - Failure Modes Analysis

**Title:** "Why Expressive Architectures Fail: Characterizing Catastrophic Instabilities in Physics-Informed Neural Networks"

**Core Contribution:**
- Deep characterization of **Modular Decoupling** failure
  - Gradient coupling coefficient $\kappa$
  - Three-phase error accumulation mechanism
- Deep characterization of **Fourier Feature Drift**
  - Sensitivity bounds: $\|\gamma(\theta+\epsilon) - \gamma(\theta)\| \propto \omega_K \epsilon$
  - Exponential feedback loop analysis

**Key Results:**
- Modular: Phase 1 (independent), Phase 2 (coupling activation), Phase 3 (catastrophic)
- Fourier: Diverges at step 60, reaches 5M+ meters by step 100

**What's NOT in this paper:**
- Stability envelope formalization (ACC/CDC)
- Full tradeoff analysis (NeurIPS)
- Detailed training methodology (RA-L)

---

### Paper 4: RA-L - Curriculum Stability Training

**Title:** "Curriculum Stability Training for Physics-Informed Neural Networks: Achieving 51x Improvement in Autoregressive Dynamics Prediction"

**Core Contribution:**
- Complete training methodology (the SOLUTION)
- Three synergistic components:
  1. Horizon Curriculum (5→10→25→50 steps)
  2. Scheduled Sampling (0%→30%)
  3. Physics-Consistent Regularization (energy + smoothness)
- Full ablation study showing each component's contribution

**Key Results:**
- 51x improvement in 100-step MAE (0.029m vs 1.49m)
- 1.1x error growth vs 17x baseline
- 18% training overhead only

**What's NOT in this paper:**
- Theoretical stability framework (ACC/CDC)
- Full failure mode analysis (ICRA)
- Broad tradeoff claims (NeurIPS)

---

## Content Saved for Future Papers

### Future Paper A: Physics-Data Conflict Bias
**Venue:** NeurIPS/ICLR (ML theory)
- Why PINNs learn wrong parameters under model mismatch
- Effective parameter absorption theory
- Closed-form bias derivation for toy systems

### Future Paper B: Observability-Limited Identification
**Venue:** ACC/CDC + RA-L
- Fisher Information limits for PINN-based SysID
- 5% inertia bound explanation
- Cramer-Rao connection

### Future Paper C: High-Excitation Paradox
**Venue:** Robotics-specific (ICRA/IROS)
- More excitation → worse ID (5%→46%)
- Trajectory design implications
- Model fidelity matching requirements

---

## Supporting Content Role (All Papers)

**Parameter Identification Results** (supporting only):
- Mass: 40% error (overestimated)
- Motor coefficients ($k_t$, $k_q$): 0% error
- Inertias ($J_{xx}$, $J_{yy}$, $J_{zz}$): 52-60% error

*These results appear in tables but are NOT the main contribution of any current paper. Parameter identification accuracy is limited due to observability constraints.*

---

## Submission Timeline Strategy

| Deadline | Venue | Paper |
|----------|-------|-------|
| May 2025 | NeurIPS 2025 | Paper 2 (Tradeoff) |
| Sep 2025 | ICRA 2026 | Paper 3 (Failure Modes) |
| Sep 2025 | ACC 2026 | Paper 1 (Stability Envelope) |
| Rolling | RA-L | Paper 4 (Curriculum Training) |

---

## Differentiation Strategy

To avoid self-plagiarism concerns:

1. **Different core claims** - Each paper has a distinct main thesis
2. **Different key figures** - Generate unique visualization for each
3. **Different theoretical depth** - ACC/CDC most formal, ICRA most practical
4. **Different baselines** - Vary comparison methods by venue
5. **Cross-reference appropriately** - Cite own work when published

---

## File Locations

```
paper_versions/
├── ACC_CDC_submission.tex      # Stability Envelope H_epsilon
├── NeurIPS_2025_submission.tex # Expressivity-Stability Tradeoff
├── ICRA_2026_submission.tex    # Failure Modes Analysis
├── RAL_submission.tex          # Curriculum Stability Training
└── SUMMARY_comparison.md       # This file
```

---

## Quick Reference: What Goes Where

| Content | ACC/CDC | NeurIPS | ICRA | RA-L |
|---------|:-------:|:-------:|:----:|:----:|
| Stability envelope $H_\epsilon$ definition | **MAIN** | Brief | Mentioned | Mentioned |
| Lipschitz stability theorem | **MAIN** | Brief | - | - |
| Expressivity-stability tradeoff | Brief | **MAIN** | Supporting | Supporting |
| 2-6 orders of magnitude claim | Table | **MAIN** | Table | Table |
| Modular decoupling analysis | Brief | Section | **MAIN** | - |
| Fourier drift analysis | Brief | Section | **MAIN** | - |
| Gradient coupling coefficient | Theory | Brief | **MAIN** | - |
| Frequency-coupling law | Theory | Brief | **MAIN** | - |
| Horizon curriculum | Brief | Brief | Mentioned | **MAIN** |
| Scheduled sampling | Brief | Brief | Mentioned | **MAIN** |
| Physics regularization | Brief | Brief | Mentioned | **MAIN** |
| Ablation study | Brief | Table | - | **MAIN** |
| Parameter ID results | Supporting | Supporting | Supporting | Supporting |

---

## Summary

This publication plan extracts **FOUR high-quality papers** from your research:

1. **ACC/CDC**: Theoretical framework (stability envelope)
2. **NeurIPS**: Big-picture phenomenon (tradeoff)
3. **ICRA**: Diagnostic analysis (failure modes)
4. **RA-L**: Practical solution (curriculum training)

Each paper stands alone with a clear, distinct contribution while collectively telling the complete story of autoregressive stability in physics-informed dynamics learning.
