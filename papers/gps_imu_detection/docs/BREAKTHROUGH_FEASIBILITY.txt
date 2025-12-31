# Breakthrough Detection Methods: Feasibility Analysis

**The Core Technical Gap:**
Residual-based detection tests `x_{t+1} ≈ f_θ(x_t)`. A consistent spoofer ensures `x̃_{t+1} ≈ f_θ(x̃_t)`. Single-step prediction is informationally insufficient.

**The Lever:** Test properties of the dynamics model itself, not its residuals.

---

## Feasibility Matrix

| Option | Method | Novelty | Implementation Effort | Expected AUROC Gain | Risk | Recommendation |
|--------|--------|---------|----------------------|---------------------|------|----------------|
| **1** | Model Inversion Instability (MIIT) | ★★★★★ | Medium | +0.2-0.3 | Low | **✓ PICK THIS** |
| **2** | Jacobian Spectrum Fingerprinting | ★★★★☆ | Low | +0.1-0.2 | Medium | ✓ Secondary |
| **3** | Multi-Step Reachability Violation | ★★★★★ | High | +0.3-0.4 | High | Future work |

---

## Option 1: Model Inversion Instability Test (MIIT)

### Core Idea
Instead of asking "Does the next state match the model?", ask "Is this trajectory invertible under the learned dynamics?"

### Technical Mechanism
```
Forward:  x_t  →[f_θ]→  x_{t+1}
Inverse:  x_{t+1}  →[g_φ]→  x̂_t
Score:    ||x_t - g_φ(f_θ(x_t))||
```

### Why It Works
- Nominal trajectories lie on a stable forward-inverse manifold
- Spoofed trajectories (even consistent ones) do NOT
- GPS spoofing fabricates observations, not true latent state transitions
- Inverse dynamics becomes ill-conditioned for spoofed data

### Feasibility Assessment

| Criterion | Score | Notes |
|-----------|-------|-------|
| Uses existing PINN? | ✓ | Forward model already trained |
| New model needed? | Yes | Inverse g_φ (~same architecture) |
| Training data? | ✓ | Same dataset |
| Compute cost? | Low | One additional forward pass |
| Reviewer appeal? | High | Novel detection primitive |
| Attacker difficulty? | High | Cannot spoof model geometry |

### Implementation Steps
1. Define inverse model `g_φ` (mirror of forward model)
2. Train on same data: `g_φ(x_{t+1}) → x_t`
3. Compute cycle consistency: `||x_t - g_φ(f_θ(x_t))||`
4. Use as anomaly score (higher = more anomalous)

### Expected Outcome
- Residual AUROC ≈ 0.5 for consistent spoofing
- **Inverse-cycle AUROC ≥ 0.7** for consistent spoofing
- This is a **new detection primitive**

---

## Option 2: Jacobian Spectrum Fingerprinting

### Core Idea
Consistent spoofing preserves state values but cannot preserve local system sensitivity.

### Technical Mechanism
```
J_t = ∂f_θ/∂x_t   (Jacobian at timestep t)

Track:
- Eigenvalues of J_t
- Condition number κ(J_t)
- Spectral entropy H(λ)
```

### Why It Works
- Nominal flight: slowly varying Jacobian spectrum
- Spoofed trajectories: force model into off-manifold regions
- Jacobian becomes unstable BEFORE residuals change
- This is **pre-residual detection**

### Feasibility Assessment

| Criterion | Score | Notes |
|-----------|-------|-------|
| Uses existing PINN? | ✓ | Just compute gradients |
| New model needed? | No | Use autograd |
| Training data? | ✓ | Same dataset |
| Compute cost? | Medium | Jacobian computation per step |
| Reviewer appeal? | High | Exploits differentiability |
| Attacker difficulty? | Very High | Cannot spoof model sensitivity |

### Implementation Steps
1. Compute Jacobian: `J = torch.autograd.functional.jacobian(f_θ, x_t)`
2. Extract eigenvalues: `λ = torch.linalg.eigvals(J)`
3. Compute features: condition number, spectral entropy, max eigenvalue
4. Train classifier on Jacobian features

### Expected Outcome
- Residuals: identical for consistent spoofing
- **Jacobian spectrum: separable**
- Detection before residual deviation

---

## Option 3: Multi-Step Reachability Violation

### Core Idea
Check if trajectory lies within the reachable tube under bounded controls.

### Technical Mechanism
```
Instead of: x_{t+1} ∈ N(f_θ(x_t))
Check:      {x_{t:t+H}} ∈ R_H(x_t)

Where R_H is the reachable set under bounded controls.
```

### Why It Works
- Spoofer can fake one trajectory
- Cannot satisfy ALL reachable alternatives
- Control-consistent futures must be bounded

### Feasibility Assessment

| Criterion | Score | Notes |
|-----------|-------|-------|
| Uses existing PINN? | Partial | Need control bounds |
| New model needed? | Yes | Reachability computation |
| Training data? | Need control inputs | May not have |
| Compute cost? | High | Set propagation |
| Reviewer appeal? | Very High | Bridges control + learning + security |
| Attacker difficulty? | Extreme | Requires solving optimal control |

### Implementation Steps
1. Estimate control bounds from data
2. Implement zonotope/interval propagation
3. Check trajectory membership in reachable tube
4. Flag violations as anomalies

### Expected Outcome
- Strongest theoretical guarantee
- Highest implementation complexity
- **Future work** candidate

---

## Recommendation

### Primary: Option 1 (MIIT)

**Why:**
- Minimal additional training (inverse model)
- Clear mathematical signal
- Easy to explain to reviewers
- Hard for attackers to defeat
- Directly leverages existing PINN infrastructure

**Success Criterion:**
```
Residual AUROC ≈ 0.5 (consistent spoofing)
Inverse-cycle AUROC ≥ 0.7 (consistent spoofing)
```

If achieved, this represents a **new class of spoofing detector**.

### Secondary: Option 2 (Jacobian)

**Why:**
- Zero additional training
- Pure computation on existing model
- Can be added quickly as complementary signal
- Novel in CPS security literature

### Future Work: Option 3 (Reachability)

**Why:**
- Strongest theoretical foundation
- Requires more infrastructure
- Natural extension for control-theoretic venues (ACC/CDC)

---

## One-Sentence Technical Novelty

> We introduce inverse-cycle instability as a new detection signal, showing that while consistency-preserving GPS spoofing defeats residual-based detectors, it induces structural inconsistency in the learned dynamics that can be detected without external sensors.

---

## Implementation Priority

| Week | Task |
|------|------|
| 1 | Implement inverse model g_φ |
| 1 | Train on existing data |
| 2 | Compute cycle consistency scores |
| 2 | Evaluate on consistent spoofing |
| 3 | Add Jacobian features (Option 2) |
| 3 | Combine signals, report results |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Inverse model doesn't converge | Use same architecture as forward, verified training |
| Cycle consistency uninformative | Fall back to Jacobian (Option 2) |
| Attacker adapts | Theoretical argument: cannot spoof both forward AND inverse |

---

*This analysis prioritizes achievable novelty over theoretical perfection.*
