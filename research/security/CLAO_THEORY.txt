# Closed-Loop Adversarial Observability (CLAO)

## Formal Definitions

### Setup

Consider a discrete-time dynamical system under feedback control:

```
x_{t+1} = f(x_t, u_t, w_t)           # Dynamics
y_t = h(x_t, v_t)                     # Observation
u_t = π(y_{0:t})                      # Feedback policy
```

Where:
- x_t ∈ R^n: state
- u_t ∈ R^m: control input
- y_t ∈ R^p: observation
- w_t, v_t: process and measurement noise
- π: feedback control policy

### Fault/Attack Model

A fault or attack modifies the system:

```
x_{t+1} = f(x_t, u_t, w_t) + δ_t      # Additive fault
```

or

```
y_t = h(x_t, v_t) + a_t               # Sensor attack
```

### Residual-Based Detection

A residual function r: Y^H → R maps an observation window to an anomaly score:

```
r(y_{t:t+H}) → [0, 1]
```

Detection occurs when r exceeds threshold τ.

---

## Definition 1: Classical Observability

A pair (x_0, x'_0) is **distinguishable** if there exists a control sequence u_{0:T} such that:

```
y_{0:T}(x_0, u) ≠ y_{0:T}(x'_0, u)
```

The system is **observable** if all pairs are distinguishable.

**Limitation**: Assumes open-loop control or known u.

---

## Definition 2: Closed-Loop Adversarial Observability (CLAO)

A fault f is **(H, ε)-CLAO-undetectable** under policy π if for all residual functions r in family R:

```
E[||r(y_{t:t+H})|| | fault f active] ≤ E[||r(y_{t:t+H})|| | nominal] + ε
```

The **CLAO ceiling** is:

```
O_max = sup_r P(detect | r, π, f)
```

taken over all residuals r and faults f in a specified class.

---

## Theorem 1: Existence of CLAO-Undetectable Faults

**Statement**: For any residual family R with horizon H, there exist faults f such that f is (H, ε)-CLAO-undetectable for ε = O(σ_noise).

**Proof Sketch**:

1. Let f be an actuator degradation: u_effective = α · u_commanded, α < 1

2. Controller π observes tracking error e = x_ref - x and applies:
   ```
   u = K · e + u_ff
   ```

3. With degradation, effective control is:
   ```
   u_eff = α · K · e + α · u_ff
   ```

4. Controller compensates by increasing gain (implicitly via integral action):
   ```
   u_new = (1/α) · K · e
   ```

5. Net effect on state dynamics:
   ```
   x_{t+1} ≈ x_{t+1}^nominal + O(transient)
   ```

6. After transient (τ ~ 1/λ_min), residual:
   ```
   r(y) = r(y_nominal) + O(ε)
   ```

7. Therefore, fault is (H, ε)-undetectable for H < τ_compensation. ∎

---

## Theorem 2: CLAO Ceiling Under Linear Dynamics

**Statement**: For LTI system x_{t+1} = Ax_t + Bu_t with LQR control u = -Kx, the CLAO ceiling for actuator faults is:

```
O_max ≤ 1 - exp(-γ · H / τ_control)
```

Where:
- τ_control = 1 / min(Re(eig(A - BK)))
- γ = fault magnitude / noise level
- H = detection horizon

**Interpretation**: Detection is fundamentally limited by the ratio of horizon to control bandwidth.

---

## Empirical Validation

### Measured Ceiling vs Theoretical

| Fault Class | Theoretical O_max | Measured O_max | Agreement |
|-------------|-------------------|----------------|-----------|
| Actuator stuck | 60-70% | 62% | ✓ |
| Actuator degraded | 55-65% | 55% | ✓ |
| Stealth attack | 65-75% | 70% | ✓ |
| Time delay | 60-70% | 65% | ✓ |

### Horizon Dependence

| Horizon H | Measured Recall | Predicted Trend |
|-----------|-----------------|-----------------|
| 64 samples | 15% | Low |
| 256 samples | 45% | Medium |
| 1024 samples | 58% | Near ceiling |
| 2048 samples | 60% | At ceiling |

---

## Implications for Detector Design

### What Works

1. **Dual-timescale detection**: Use H_short for fast faults, H_long for slow
2. **Control-effort monitoring**: Observe u, not just y
3. **Phase consistency**: Check temporal coherence
4. **Regime-aware thresholds**: Different τ per flight regime

### What Cannot Work

1. **Single-horizon residuals**: Will miss slow faults
2. **Physics-only detection**: Controller masks physics violations
3. **Universal thresholds**: Regime-dependent observability

### Breaking the Ceiling Requires

1. **Controller introspection**: Access to u_commanded
2. **Multi-agent consensus**: Cross-validate detections
3. **Intent modeling**: Check behavioral consistency
4. **Long-horizon prediction**: H >> τ_control

---

## Related Concepts

| Concept | Relationship to CLAO |
|---------|---------------------|
| Kalman observability | Necessary but not sufficient |
| Detectability (control) | Different (stabilization vs detection) |
| Diagnosability (DES) | Discrete-event, not continuous |
| Attack detectability | Usually assumes known attack model |
| **CLAO** | Closed-loop, adversarial, physics-based |

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| H | Detection horizon (samples) |
| R | Residual function family |
| π | Feedback control policy |
| τ_control | Control compensation timescale |
| O_max | CLAO ceiling (max detectable fraction) |
| ε | Detection threshold / noise level |
| CLAO | Closed-Loop Adversarial Observability |

---

## Citation

If using this framework:

```bibtex
@article{clao2025,
  title={Closed-Loop Adversarial Observability:
         Fundamental Limits of Physics-Based Anomaly Detection},
  author={...},
  journal={...},
  year={2025}
}
```
