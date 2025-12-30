# PINN Contradiction - Resolution

**Date:** 2025-12-30
**Issue:** Paper claims PINN works, but evidence shows it doesn't

---

## The Contradiction

### Paper Says (paper_v3_integrated.tex):
> "We present a Physics-Informed Neural Network (PINN) based detector that achieves deployment-ready performance"

### Evidence Says (FINAL_COMPARISON_ALL_METHODS.txt):
> "PINN APPROACH DOES NOT WORK"
> "Learned mappings ≠ physics constraint checking"
> "Don't pursue this direction further"

### Paper Also Admits:
> "pure data-driven detection significantly outperforms physics-informed variants (p<10⁻⁶, effect size 13.6×)"

---

## Resolution

### What Actually Happened

1. **ALFA fault detection works** - The detector achieves 65.7% F1
2. **But physics constraints don't help** - w=0 (no physics) outperforms w=20 (physics)
3. **The "PINN" is really just a neural network** - trained on flight data
4. **Physics loss hurts performance** - empirically verified

### Why Physics Doesn't Help for Fault Detection

1. **Fault dynamics violate physics** - When an engine fails, the UAV no longer follows normal physics
2. **Self-consistency doesn't detect bias** - A biased sensor is self-consistent
3. **Learned mappings ≠ constraint checking** - The network learns to predict, not verify physics

### What Should Be Claimed

| Claim | Valid? | Reason |
|-------|--------|--------|
| "Neural network detects faults" | ✅ Yes | 65.7% F1 achieved |
| "PINN-based detection" | ❌ No | Physics constraints don't help |
| "Physics-informed" | ⚠️ Misleading | Physics loss hurts performance |
| "Deployment-ready" | ⚠️ Needs work | Missing reproducibility |

---

## Paper Recommendations

### Option 1: Honest Revision
- Remove "PINN" from title/claims
- Present as "neural network-based fault detection"
- Acknowledge physics doesn't help for this task
- Emphasize the 65.7% F1 with 4.5% FPR as the contribution

### Option 2: Investigate Further
- Test if physics helps in low-data regime
- Try different physics formulations
- Compare on different fault types
- May find cases where physics does help

### Option 3: Different Framing
- "We tried physics-informed approaches but..."
- "Counter-intuitively, pure data-driven works better"
- Present as a negative result with positive detection performance

---

## Summary

**The detector works. The physics doesn't help.**

The paper should be revised to:
1. Not claim PINN/physics is the key contribution
2. Present honest comparison of physics vs no-physics
3. Claim the neural network detection performance
4. Add reproducibility documentation

---

## Evidence Summary

| Source | PINN AUROC | Data-only AUROC | Winner |
|--------|------------|-----------------|--------|
| GPS-IMU (physics_residuals) | 0.562 | 0.582 (features) | Data |
| GPS-IMU (CNN-GRU) | 0.454 | N/A | Random |
| ALFA (w=20 vs w=0) | Lower | Higher | Data |
| FINAL_COMPARISON | "DOES NOT WORK" | Better | Data |

**Consistent finding: Physics constraints don't improve detection.**

---

*This document resolves the PINN contradiction by acknowledging the evidence.*
