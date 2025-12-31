# 5. Limits: What ICI Cannot Detect

**Hierarchy Position:** Honest disclosure of fundamental limits

**Prerequisite:** Understand all previous folders first

## Fundamental Limits

ICI is not a universal spoofing detector. It has fundamental blind spots.

### Limit 1: Small Offsets (< 10m)

| Offset | ICI AUROC | Status |
|--------|-----------|--------|
| 1m | 0.52 | Fundamentally Hard |
| 5m | 0.52 | Fundamentally Hard |
| 10m | 0.66 | Marginal |

**Reason:** Small offsets are indistinguishable from sensor noise.

### Limit 2: AR(1) Manifold-Preserving Attacks

An attacker who knows the dynamics can craft attacks that stay ON the learned manifold:

```python
# AR(1) drift that preserves consistency
x_spoofed[t] = x_true[t] + drift[t]
drift[t] = 0.99 * drift[t-1] + noise
```

| Attack | ICI AUROC | Reason |
|--------|-----------|--------|
| AR(1) drift | ~0.5 | Stays on manifold |

**This is a theorem, not a bug.** If g(f(x)) ≈ x, then ICI ≈ 0.

### Limit 3: Attacker with Model Access

If the attacker has access to f_θ and g_φ, they can:
1. Compute the manifold projection
2. Craft attacks that minimize ICI
3. Evade detection

**Mitigation:** Model confidentiality, frequent retraining.

## Experiments in This Folder

- `ar1_attack.py` - Demonstrate AR(1) evasion
- `minimum_detectable_offset.py` - Find detection floor
- `adversarial_evasion.py` - White-box attack analysis

## Why We Disclose This

1. **Honesty:** Reviewers will ask. Pre-empt the question.
2. **Scope:** Define what we claim vs. what we don't.
3. **Future work:** These limits are research opportunities.

## Comparison to Residual Limits

| Limit | Residual | ICI |
|-------|----------|-----|
| Small offsets | Fundamentally Hard | Fundamentally Hard |
| Large offsets | Fundamentally Hard | **Detectable** |
| AR(1) drift | Fundamentally Hard | Fundamentally Hard |
| Consistent ramp | Fundamentally Hard | **Detectable** |

**Key difference:** ICI expands the detectable attack space, but does not cover all attacks.

## Reviewer Q&A

**Q: So ICI can be evaded?**
A: Yes, by sophisticated attackers with model knowledge. This is true of ALL learned detectors.

**Q: What's the value if limits exist?**
A: ICI detects attacks that residuals fundamentally cannot. That's the contribution.

**Q: Should we add more defenses?**
A: Out of scope. This paper establishes ICI. Defense-in-depth is future work.
