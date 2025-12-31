# Track C Results: Motivation & Baselines

**Question:** Why do naïve physics or residual-based approaches fail, motivating ICI?

**Purpose:** These results support the **motivation** for Track A. They are not standalone contributions.

---

## Table 1: Physics Loss Ablation

| Weight | Mean Loss | p-value |
|--------|-----------|---------|
| **w=0** | **0.330** | - |
| w=20 | 4.502 | < 0.001 |

**Finding:** Physics loss hurts detection. Pure data-driven wins.

**Usage:** Cite in Related Work to motivate learned approach.

---

## Table 2: Residual Detector Failure

| Approach | AUROC on Consistent Spoofing |
|----------|------------------------------|
| Physics residual | 0.50 (random) |
| EKF innovation | 0.50 (random) |
| Feature-based | 0.58 |

**Finding:** All residual-based approaches fail on consistency-preserving attacks.

**Usage:** This is the "impossibility" that ICI overcomes.

---

## Table 3: Engineering Pipelines

| Pipeline | Recall | FPR | Problem |
|----------|--------|-----|---------|
| Sensor fusion v3 | 85% | 22% | High FPR |
| Enhanced detector | 74% | 0.16% | Low recall |
| Ensemble | 80% | 95% | Catastrophic FPR |

**Finding:** Engineering pipelines trade off recall vs FPR poorly.

**Usage:** Motivates principled approach (ICI) over heuristics.

---

## How to Use These Results

### In Track A Paper

**Related Work / Motivation section:**
> "Prior work on physics-based residuals achieves AUROC ≈ 0.5 on consistency-preserving attacks [cite self]. This fundamental limitation motivates our inverse-cycle approach."

**Do NOT include in Results section.**

### In Track B Paper

**Related Work section:**
> "Ablation studies show physics loss can hurt detection in adversarial settings [cite self], though supervised classification remains effective with labels."

---

## What These Results Do NOT Show

- A working detector (that's Track A)
- A deployable system (that's Track A + D)
- Supervised classification success (that's Track B)

These are **negative results** that justify why ICI was necessary.

---

*Results in `models/security/*/` and `research/security/`*
