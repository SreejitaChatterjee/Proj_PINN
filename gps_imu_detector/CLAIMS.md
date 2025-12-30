# Claims Registry

**Version:** 1.0
**Last Updated:** 2025-12-30
**Status:** FROZEN

---

## Claim Hierarchy

This document is the **single source of truth** for what we claim to detect.
All READMEs, papers, and code must be consistent with this table.

---

## Detection Tiers

| Tier | Definition | Evidence Required |
|------|------------|-------------------|
| **Detectable** | AUROC >= 0.95, Recall@5%FPR >= 90% | Validated on test set |
| **Marginal** | AUROC 0.70-0.95, Recall@5%FPR 50-90% | Documented with CI |
| **Fundamentally Hard** | AUROC < 0.70, known theoretical limit | Proof or counterexample |
| **Out of Scope** | Not addressed by this work | Explicit exclusion |

---

## Attack Status Table

### ICI Detector (Main Contribution)

| Attack Type | Magnitude | Status | Evidence |
|-------------|-----------|--------|----------|
| Constant offset | >= 50m | **Detectable** | AUROC=1.0, geometric separation |
| Constant offset | 25-50m | **Detectable** | AUROC=1.0, scaling law |
| Constant offset | 10-25m | **Marginal** | AUROC=0.65-0.85 |
| Constant offset | < 10m | **Fundamentally Hard** | Below noise floor |
| Consistent drift (ramp) | >= 0.5m/s | **Detectable** | AUROC=1.0 |
| Consistent drift (ramp) | < 0.5m/s | **Marginal** | AUROC~0.75 |
| AR(1) stealthy drift | Any | **Fundamentally Hard** | Manifold-preserving, ICI~0 |
| Oscillatory | >= 10m amplitude | **Detectable** | AUROC=1.0 |
| Intermittent | On/off cycles | **Marginal** | Depends on on-duration |
| Coordinated GPS+IMU | With offset | **Detectable** | ICI sees offset |
| Replay attack | Past trajectory | **Out of Scope** | Requires temporal analysis |

### Residual Detector (Baseline)

| Attack Type | Magnitude | Status | Evidence |
|-------------|-----------|--------|----------|
| Constant offset | Any | **Fundamentally Hard** | Residual Equivalence Class |
| Consistent drift | Any | **Fundamentally Hard** | REC theorem |
| AR(1) drift | Any | **Fundamentally Hard** | REC theorem |
| Jump discontinuity | >= 10m | **Marginal** | Transient detection |
| Jump discontinuity | < 10m | **Fundamentally Hard** | Below EKF innovation |

---

## Core Claims (Paper-Ready)

### Claim 1: Residual Impossibility

> **Residual-based detectors cannot detect consistency-preserving GPS spoofing.**

Evidence: Residual Equivalence Class theorem + experimental AUROC=0.5

Tier: **Proven**

### Claim 2: ICI Separation

> **Inverse-Cycle Instability (ICI) separates nominal from spoofed observations for offsets >= 25m.**

Evidence: Monotonic scaling law, AUROC=1.0 for >= 50m

Tier: **Detectable**

### Claim 3: Self-Healing

> **IASP reduces position error by 74%+ for detected spoofing events.**

Evidence: Validated on 100m spoof, quiescence verified

Tier: **Detectable**

### Claim 4: Fundamental Limit

> **ICI cannot detect AR(1) manifold-preserving attacks that remain on the learned dynamics surface.**

Evidence: By construction, ICI=0 when x_t = g(f(x_t))

Tier: **Fundamentally Hard** (honest disclosure)

---

## What We Do NOT Claim

1. **Full GPS security** - We detect a specific attack class, not all attacks
2. **Real-world validation** - Results are on synthetic + simulated data
3. **Adversarial robustness** - Attacker with model access can evade
4. **Sensor fault detection** - Focus is on spoofing, not hardware faults
5. **Multi-vehicle coordination** - Single-vehicle detection only

---

## Consistency Check

Before any commit, verify:

1. [ ] No README claims detection of "Fundamentally Hard" attacks
2. [ ] No code comments suggest "robust" detection without qualification
3. [ ] All experimental results match this table
4. [ ] Paper abstract matches Tier 1 claims only

---

## Amendment Process

To add or change a claim:

1. Provide experimental evidence (AUROC, CI)
2. Update this table
3. Update all dependent documents
4. Commit with message: "CLAIM: [description]"

---

## Reviewer Q&A Mapping

| Likely Question | Reference |
|-----------------|-----------|
| "Does this detect all spoofing?" | See "What We Do NOT Claim" |
| "What about stealthy attacks?" | See AR(1) row - Fundamentally Hard |
| "Is AUROC=1.0 realistic?" | See scaling law + geometric separation |
| "What's the minimum detectable offset?" | See 10-25m row - Marginal |
