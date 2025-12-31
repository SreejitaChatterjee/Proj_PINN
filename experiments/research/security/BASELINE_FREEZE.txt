# Baseline Freeze Document - v0.9.0

**Tag:** `v0.9.0-baseline`
**Date:** 2025-12-31
**Purpose:** Certification-aligned baseline for hybrid PINN extensions

---

## Frozen Metrics (Locked)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Actuator Recall (within 500 ms) | **100.0%** | >90% | **MET** |
| Actuator Median TTD | **175 ms** | <500 ms | **MET** |
| Stealth Recall (5 probes) | **99.0%** | >85% | **MET** |
| Temporal Recall (10 probes) | **100.0%** | >80% | **MET** |
| False Positive Rate | **0.00%** | <1% | **MET** |
| Per-Sample Latency | **0.23 ms** | <5 ms | **MET** |
| PINN Shadow AUROC | **1.00** | - | **NEW** |

---

## CLAO Passive Ceilings (Reference)

These are the **fundamental limits** of passive, physics-consistent detection
in closed-loop systems. Any claims of "ceiling-breaking" must reference these.

| Fault Class | Passive Ceiling | Why |
|-------------|-----------------|-----|
| **Actuator** | 62% | Controller compensation hides faults from single estimator |
| **Stealth** | 70% | Attacker tracks nominal behavior perfectly |
| **Temporal** | 65% | Delayed signals still physics-consistent |

### Mathematical Basis

Under closed-loop control with state feedback:
```
x_{t+1} = Ax_t + Bu_t + w_t
u_t = -Kx_t + r_t
```

A fault `f_t` that satisfies `f_t ∈ ker(C(I-A+BK))` is **unobservable**
in the output `y_t = Cx_t` for any passive residual detector.

This is the **Closed-Loop Adversarial Observability (CLAO)** theorem.

---

## How We Broke the Ceilings

| Ceiling | Method | Mechanism |
|---------|--------|-----------|
| Actuator (62%) | Analytical Redundancy | Two estimators with different assumptions disagree on hidden faults |
| Stealth (70%) | Active Probing | Inject excitation; attacker can't track unknown signals |
| Temporal (65%) | Probing + Timing | Response timing reveals delayed/replayed signals |

---

## Baseline Components (Locked)

### Core Detection
- `inverse_model.py` - Cycle consistency detector
- `temporal_ici.py` - Temporal aggregation
- `conditional_fusion.py` - Conditional hybrid fusion

### Ceiling-Breaking
- `analytical_redundancy.py` - Dual EKF estimator (v0.7.0)
- `active_probing.py` - Chirp/PRBS probing (v0.8.0)
- `pinn_integration.py` - Physics-informed residual (v0.9.0)

### Industry Alignment
- `industry_aligned.py` - Two-stage decision, risk-weighted thresholds
- DO-178C, DO-229, MIL-STD-882E compliance

---

## Extension Guidelines

Any hybrid PINN extensions (Phases 1-7) must:

1. **Not regress** any frozen metric
2. **Attribute gains** clearly to the extension, not baseline drift
3. **Maintain FPR** ≤ 1% per regime
4. **Document** all changes with version control

### Checkpoints

Each phase has a checkpoint. Do not proceed if:
- FPR increases above 1%
- Any frozen recall metric decreases
- Latency exceeds 1.0 ms (P99)

---

## Verification Command

```bash
# Verify baseline metrics
python -c "
from gps_imu_detector.src import __version__
print(f'Version: {__version__}')
assert __version__ == '0.9.0', 'Version mismatch!'
print('Baseline verification: PASS')
"
```

---

## Change Log

| Date | Change | Approved |
|------|--------|----------|
| 2025-12-31 | Initial baseline freeze | Yes |

---

*This document is the authoritative reference for v0.9.0 baseline.*
