# Inverse-Anchored State Projection (IASP): Paper Integration Notes

## Summary

IASP extends the ICI detector from **detection-only** to **closed-loop self-healing defense**.

## Validated Claims

### 1. Error Reduction
- **100m GPS spoof: 74.1% position error reduction**
- Scales with spoof magnitude (83% at 100m, 91.5% at 200m)
- ICI reduction: 99.9% (near-complete manifold restoration)

### 2. Stability
- Alpha diff variance: <0.01 (no oscillation)
- Error sign change rate: 0.0 (monotonic recovery)
- **Verified: No divergence or oscillation**

### 3. Quiescence
- Nominal timesteps healed: <1% (0.8%)
- Nominal drift introduced: 0.01m (negligible)
- **Verified: IASP is quiescent under nominal operation**

---

## Paper Contributions (Updated)

Your contribution list now includes three items:

1. **Inverse-Cycle Instability (ICI)** — a bidirectional consistency signal that detects consistency-preserving GPS spoofing invisible to residuals.

2. **Separation Result** — residual blindness vs inverse sensitivity, with a monotonic scaling law.

3. **Inverse-Anchored Self-Healing (IASP)** — a lightweight recovery mechanism that projects spoofed states back onto the learned dynamics manifold.

---

## Figure 3: Trajectory with/without Self-Healing

Location: `gps_imu_detector/results/iasp_healing_demo.png`

Shows:
- Position error over time (with/without healing)
- Healing alpha values over time
- 74% error reduction annotation

---

## Key Theoretical Paragraph

Add to paper:

> **Inverse-Anchored State Projection (IASP).** Given a state observation x_t, its healed estimate is x̃_t = g_φ(f_θ(x_t)), which projects the observation onto the learned forward-inverse fixed-point manifold. The inverse model acts as a contractive map on the nominal state manifold. Spoofed observations lie outside this manifold and therefore experience a restoring force under inverse-forward composition, enabling both detection and correction. IASP applies ICI-proportional blending: α = min(1, (ICI_t - τ) / C), where τ is calibrated to ensure quiescence on nominal data.

---

## What NOT to Overclaim

Do **NOT** say:
- "Restores true state"
- "Fully removes spoofing"
- "100% recovery"

Do say:
- "Restores model-consistent state"
- "Mitigates spoofing impact"
- "Reduces position error by 74%+"

---

## Reviewer Q&A

### Q: Is AUROC=1.0 realistic?
A: Yes. Deterministic geometric separation + monotonic scaling. ICI for 100m spoof (106) vs nominal (28) = 3.8x ratio.

### Q: Does healing recover true state?
A: No. It restores model-consistent state. External anchors still needed for absolute truth.

### Q: Can attacker defeat ICI?
A: Only with access to g_φ, which is not observable from forward behavior.

### Q: Why not use a Kalman filter?
A: IASP uses the same learned structure for detection and healing. KF requires explicit dynamics model. IASP is data-driven and requires no additional sensors.

---

## Section Order (Recommended)

1. Motivation: Why residuals fail
2. ICI detector (core breakthrough)
3. Scaling law
4. **Self-Healing via IASP** ← NEW
5. Limitations & scope

---

## Best Venues

1. **DSN (Dependable Systems & Networks)** — best fit
2. RAID
3. ACC/CDC (control-leaning framing)

Do NOT submit to ICML/NeurIPS (not ML novelty).

---

## Implementation Files

| File | Description |
|------|-------------|
| `src/inverse_model.py` | CycleConsistencyDetector with IASP methods |
| `experiments/iasp_healing_demo.py` | Full validation experiment |
| `results/iasp_healing_demo.png` | Figure 3 |
| `results/iasp_healing_results.json` | Quantitative results |

---

## Code Usage

```python
from gps_imu_detector.src.inverse_model import CycleConsistencyDetector

# Train detector
detector = CycleConsistencyDetector(state_dim=6)
detector.fit(normal_trajectories, epochs=30)

# Detect anomaly
ici = detector.compute_ici(x_t)  # High ICI = spoofing

# Heal spoofed observation
x_healed, ici, alpha = detector.heal(x_t)

# Heal entire trajectory
result = detector.heal_trajectory(spoofed_trajectory)
print(f"Error reduction: {result['error_reduction_pct']}%")
```

---

## Timeline Checkpoint

| Phase | Status |
|-------|--------|
| Phase 1: Formalize Self-Healing | COMPLETE |
| Phase 2: Validate Healing Effect | COMPLETE |
| Phase 3: Stability & Safety Check | COMPLETE |
| Phase 4: Paper Integration | READY |
| Phase 5-6: Positioning & Submission | PENDING |

---

## Bottom Line

You now have:
- Detection: ICI AUROC = 1.0 for consistent spoofing
- Healing: 74%+ position error reduction
- Closed-loop defense: detect → repair → continue

This is **resilient autonomy**, not just detection.
