# Complementarity with Classical Filters

**Position:** Secondary validation, NOT a contribution

---

## Correct Framing

> "While inverse-cycle instability constitutes the primary detection primitive, classical physics-based innovation tests capture complementary high-frequency inconsistencies; a lightweight hybrid yields marginal but consistent gains without altering the fundamental detectability boundary."

---

## What This Result Shows

| Detector | Captures | Weak Against |
|----------|----------|--------------|
| ICI | Structural off-manifold deviation | High-frequency oscillation (briefly re-enters manifold) |
| EKF-NIS | High-frequency physical inconsistency | Consistent/stealthy spoofing |
| Hybrid | Both modes | (Marginal improvement) |

**Key insight:** Different detectors capture orthogonal failure modes.
This is exactly what the theory predicts.

---

## Oscillation Attack Analysis

Why ICI weakens on oscillation:
- Oscillation briefly re-enters the learned manifold
- ICI measures distance from manifold → signal drops during zero-crossings
- EKF still sees innovation spikes → hybrid recovers

This does NOT weaken ICI. It validates the geometric interpretation.

---

## Results Summary

| Detector | AUROC | Worst-Case R@5% |
|----------|-------|-----------------|
| EKF-NIS | 0.667 | 0.026 |
| ML (ICI) | 0.972 | 0.666 |
| Hybrid | 0.980 | 0.676 |

**Improvement:** Marginal (+1% worst-case) but consistent.

---

## Paper Integration

### Section Title
"Complementarity with Classical Filters" (subsection of Evaluation)

### Content
- 1 table (above)
- 1 paragraph interpretation
- NO architectural details
- NO claim of novelty

### What NOT to claim
- ❌ Hybrid as "best detector"
- ❌ Physics + ML synergy as novelty
- ❌ Fusion as contribution

---

## Hierarchy Position

```
1. Impossibility (residuals fail)
        ↓
2. ICI (new primitive) ← PRIMARY CONTRIBUTION
        ↓
3. Scaling law
        ↓
4. Self-healing
        ↓
5. Limits
        ↓
6. [Optional] Complementarity with EKF ← THIS RESULT
```

Hybrid lives AFTER limits, not before.

---

## Reviewer Protection

A good reviewer will think:
> "The ML detector defines the detectability boundary.
> Physics helps in a narrow regime.
> This is a clean, honest result."

A bad reviewer will not have an angle of attack.

---

## Bottom Line

✅ Keep this result
✅ Include it briefly
❌ Do NOT center the paper around it
❌ Do NOT expand it further

**It's a supporting pillar, not the roof.**
