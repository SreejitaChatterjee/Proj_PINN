# Physics Deviations from Reality

## Date: November 5, 2025

---

## CRITICAL: Unphysical Terms in Dynamics

### 1. **Artificial Angular Rate Damping** (MAJOR DEVIATION)

**Location:**
- Data generation: `scripts/generate_quadrotor_data.py:241-243`
- PINN model: `scripts/pinn_model.py:53-55`

**Issue:**
```python
pdot = t1*q*r + tx/Jxx - 2*p  # ← Unphysical -2*p term
qdot = t2*p*r + ty/Jyy - 2*q  # ← Unphysical -2*q term
rdot = t3*p*q + tz/Jzz - 2*r  # ← Unphysical -2*r term
```

**Real Physics (Euler's Rotation Equations):**
```python
pdot = t1*q*r + tx/Jxx  # No damping term
qdot = t2*p*r + ty/Jyy  # No damping term
rdot = t3*p*q + tz/Jzz  # No damping term
```

**Explanation:**
- The `-2*p`, `-2*q`, `-2*r` terms are artificial damping coefficients
- They represent proportional damping on angular rates (like viscous friction)
- Real quadrotors in air have NO significant viscous damping on rotation
- Air resistance on rotor blades is captured in motor dynamics, not body rotation
- This is equivalent to adding fake "friction" to angular momentum

**Impact:**
- Model learns dynamics that don't exist in reality
- Angular rates decay artificially fast
- Transfer to real quadrotor would fail
- Damping coefficient of 2.0 rad/s is VERY large (highly overdamped)

**Why it might have been added:**
- Numerical stability during simulation
- Compensate for controller tuning issues
- Prevent runaway oscillations

**Severity:** 🔴 CRITICAL - Model doesn't represent real physics

---

### 2. **Linear Drag on Translational Velocities** (MODERATE DEVIATION)

**Location:**
- Data generation: `scripts/generate_quadrotor_data.py:267-269`

**Issue:**
```python
udot = r*v - q*w + fx/m - g*sin(theta) - 0.1*u  # ← Linear drag
vdot = p*w - r*u + fy/m + g*cos(theta)*sin(phi) - 0.1*v
wdot = q*u - p*v + fz/m + g*cos(theta)*cos(phi) - 0.1*w
```

**Real Physics:**
- Aerodynamic drag: F_drag = 0.5 * ρ * Cd * A * v²
- Should be proportional to v², not v
- For small quadrotors at low speeds (<5 m/s), drag is often negligible

**Explanation:**
- Linear drag `-0.1*u` is a simplification
- Drag coefficient 0.1 s⁻¹ is arbitrary (not derived from physical parameters)
- Real drag would be: `-0.5 * (Cd*A/m) * u * |u|` (quadratic)

**Impact:**
- Less severe than angular damping (drag does exist, just wrong form)
- At low velocities, linear approximation might be acceptable
- Model won't generalize to high-speed flight

**Severity:** 🟡 MODERATE - Simplified but drag exists in reality

---

### 3. **Missing Aerodynamic Effects** (ACCEPTABLE FOR SMALL QUADROTORS)

**Not included:**
- Rotor downwash effects
- Ground effect
- Blade flapping dynamics
- Gyroscopic effects from rotors

**Justification:**
- These effects are 2nd order for small quadrotors
- Acceptable for initial modeling
- Model focuses on rigid body dynamics

**Severity:** 🟢 ACCEPTABLE - These are small effects

---

## Summary of Deviations

| Issue | Location | Severity | Fix Required? |
|-------|----------|----------|---------------|
| Artificial angular damping `-2*[p,q,r]` | Rotational dynamics | 🔴 CRITICAL | YES |
| Linear drag `-0.1*[u,v,w]` | Translational dynamics | 🟡 MODERATE | NICE TO HAVE |
| Missing aerodynamics | All dynamics | 🟢 ACCEPTABLE | NO |

---

## Recommendations

### Option 1: Remove Unphysical Terms (Ideal)
**Remove artificial damping:**
```python
pdot = t1*q*r + tx/Jxx  # Remove -2*p
qdot = t2*p*r + ty/Jyy  # Remove -2*q
rdot = t3*p*q + tz/Jzz  # Remove -2*r
```

**Update drag to quadratic (optional):**
```python
udot = ... - 0.05*u*abs(u)  # Quadratic drag
```

**Impact:**
- More physically accurate
- Better transfer to real hardware
- May require retuning PID controllers
- Training data must be regenerated

### Option 2: Document as Simplification (Current State)
- Acknowledge these are modeling simplifications
- Label model as "stabilized dynamics" not "pure physics"
- Restrict use to simulation only

### Option 3: Make Damping Learnable
- Add damping as learnable parameters: `d_p`, `d_q`, `d_r`
- Initialize to 0.0 (no damping)
- Let model learn if damping exists from data
- This would reveal the data has artificial damping

---

## Conclusion

The model contains **significant unphysical terms** that were likely added for:
1. Numerical stability
2. Controller simplification
3. Simulation convenience

**These terms make the model unsuitable for transfer to real hardware without modification.**

If the goal is:
- **Simulation only** → Document limitations, current model OK
- **Real hardware** → Must remove artificial terms and regenerate data
- **Research/learning** → Great learning opportunity about physics-informed ML

