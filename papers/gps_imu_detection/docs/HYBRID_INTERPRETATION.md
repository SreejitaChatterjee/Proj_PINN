# Hybrid Result Interpretation Guide

**Status:** Supporting Evidence (NOT a Contribution)
**Version:** 1.0.0
**Date:** 2026-01-01

---

## Executive Summary

The hybrid detector (ICI + EKF-NIS fusion) is **supporting evidence**, not a contribution. This document explains why.

---

## Key Results

| Detector | AUROC | Worst-Case R@5% | Status |
|----------|-------|-----------------|--------|
| EKF-NIS | 0.667 | 0.026 | Baseline |
| ICI | 0.972 | 0.666 | **Primary** |
| Hybrid | 0.980 | 0.676 | Supporting |

---

## Why Hybrid is NOT a Contribution

### 1. Marginal Improvement

The hybrid improves over ICI by:
- AUROC: +0.8% (0.972 → 0.980)
- Worst-case R@5%: +1.0% (0.666 → 0.676)

This is **not novel**. Simple score fusion is well-known.

### 2. ICI Defines the Boundary

The detectability boundary is defined by ICI, not the hybrid:
- ICI achieves 97.2% AUROC alone
- Hybrid cannot detect attacks that ICI cannot detect
- Hybrid cannot lower the detectability floor

### 3. EKF Helps Only in Narrow Regime

EKF-NIS provides marginal benefit only for:
- High-frequency oscillatory attacks
- Jump discontinuities (transient response)

For all other attacks, EKF contribution is negligible.

---

## Correct Framing

### DO Say

> "A lightweight hybrid fusion with classical EKF innovation tests provides marginal but consistent gains (+1% worst-case recall), validating that ICI captures the dominant detectability boundary."

### DO NOT Say

- "Novel hybrid architecture" - it's simple score averaging
- "Significant improvement" - 1% is not significant
- "Hybrid contribution" - ICI is the contribution

---

## When to Mention Hybrid

| Context | Mention? | Framing |
|---------|----------|---------|
| Abstract | NO | Focus on ICI |
| Contributions | NO | Not a contribution |
| Experiments | YES | As supporting validation |
| Discussion | YES | As future work baseline |

---

## Proper Paper Structure

### In Contributions Section

> **Contributions:**
> 1. Inverse-Cycle Instability (ICI) detector achieving 97.2% AUROC
> 2. Detectability floor characterization at 0.25-0.3x magnitude
> 3. Self-healing via IASP with 74% error reduction

**NOT:**
> 4. Hybrid ICI-EKF fusion (this is NOT a contribution)

### In Experiments Section

> We also evaluate a lightweight hybrid combining ICI with classical EKF-NIS scores. The marginal improvement (+1% worst-case recall) validates that ICI captures the dominant detection signal.

---

## Reviewer FAQ

### Q: "Why not claim hybrid as a contribution?"

**A:** The improvement is marginal (1%), and the technique (score averaging) is not novel. Claiming it as a contribution would overstate our work.

### Q: "Shouldn't you use more sophisticated fusion?"

**A:** We tested several fusion methods (weighted, learned, conditional). None significantly outperform simple averaging, confirming that ICI already captures the available signal.

### Q: "What about attention-based fusion?"

**A:** Out of scope. Our contribution is the ICI detector itself, not fusion methods. Hybrid is included only to validate that ICI defines the boundary.

---

## Consistency Checklist

Before submission, verify:

- [ ] Abstract does NOT mention hybrid
- [ ] Contributions list does NOT include hybrid
- [ ] Experiments mention hybrid as "supporting validation"
- [ ] No claims of "novel fusion" or "significant improvement"
- [ ] Hybrid table labeled as "Supporting Result"

---

*This document ensures honest framing of hybrid results as supporting evidence, not a contribution.*
