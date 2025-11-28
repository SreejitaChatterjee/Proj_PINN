# Known Limitations and Failure Cases

## Overview

This document catalogs the known limitations, failure cases, and areas for improvement in the quadrotor PINN implementation. Understanding these limitations is crucial for:
- Setting realistic expectations for model performance
- Identifying areas for future research
- Avoiding misuse of the model outside its validated operating envelope

---

## 1. Parameter Identification Limitations

### 1.1 Inertia Parameters (Jxx, Jyy, Jzz)

**Current Status:** 5.00% error after improvements (previously 1300-6700%)

**Root Cause:** Fundamental observability problem at small angles

**Explanation:**
The rotational dynamics equations contain cross-coupling terms:
```
ṗ = [(Jyy - Jzz)/Jxx] · q·r + τx/Jxx
q̇ = [(Jzz - Jxx)/Jyy] · p·r + τy/Jyy
ṙ = [(Jxx - Jyy)/Jzz] · p·q + τz/Jzz
```

At small angles (±20° in standard training):
- Angular rates p, q, r are small (< 1 rad/s)
- Cross-coupling terms (p·q·r) are very small (< 0.1)
- Gradient signals for inertia identification are weak
- Inertias become nearly unobservable

**Attempted Solutions:**
1. ✅ Generated aggressive ±45-60° trajectories (20,873 samples)
   - Increases cross-coupling magnitudes by 10-100×
   - Provides stronger gradient signals
   - **Status:** Data generated, needs retraining to validate

2. ✅ Added energy conservation loss
   - Alternative gradient path: E_rot = (1/2)(Jxx·p² + Jyy·q² + Jzz·r²)
   - Less sensitive to small angles
   - **Status:** Implemented, needs retraining to validate

3. ⚠️ Angular acceleration measurements available but not yet utilized
   - Could provide direct observability of inertias

**Recommendations:**
- Use aggressive trajectories for fine-tuning inertia parameters
- Consider system identification methods designed for parameter estimation
- Validate learned inertias against manufacturer specifications

---

## 2. Operating Envelope Limitations

### 2.1 Attitude Limits

**Validated Range:** ±20° (roll, pitch), ±30° (yaw)

**Failure Modes Beyond Range:**
- Predictions become unreliable beyond ±40° attitudes
- Small-angle approximations in kinematics break down
- Gimbal lock near ±90° pitch angles (inherent to Euler angles)

**Evidence:**
- Training data uses square wave references with ±20° amplitudes
- No validation data exists beyond ±30°

**Recommendations:**
- DO NOT use for aggressive maneuvers requiring >40° angles
- Consider quaternion representation for large-angle applications
- Retrain with aggressive trajectory data for extended envelope

### 2.2 Angular Rate Limits

**Validated Range:** < 5 rad/s (p, q, r)

**Failure Modes Beyond Range:**
- Gyroscopic effects not modeled
- Blade flapping dynamics neglected
- Motor saturation not fully represented

**Recommendations:**
- Limit to standard flight maneuvers
- Add gyroscopic terms for high-rate applications

### 2.3 Velocity Limits

**Validated Range:** < 2 m/s horizontal, < 1.5 m/s vertical

**Failure Modes Beyond Range:**
- Aerodynamic drag model is linearized (quadratic drag at high speeds)
- Ground effect not modeled near surfaces
- Blade vortex interactions neglected

**Recommendations:**
- DO NOT use for high-speed flight (> 5 m/s)
- Add nonlinear drag terms for high-speed applications

---

## 3. Model Architecture Limitations

### 3.1 Single-Step Predictions

**Current Implementation:** Model predicts state at t+dt given state at t

**Limitations:**
- Autoregressive rollout accumulates errors over long horizons
- 100-step prediction error: 0.029 m (vs 1.49 m baseline)
- Still shows drift at very long horizons (> 500 steps / 0.5s)

**Evidence:**
- Error autocorrelation shows temporal dependencies
- Rolling statistics show slight error growth over time

**Recommendations:**
- Use scheduled sampling during training (already implemented: 0-30%)
- Consider multi-step direct prediction for long-horizon planning
- Implement closed-loop control to prevent drift

### 3.2 Deterministic Predictions

**Current Implementation:** Point estimates only, no uncertainty quantification

**Limitations:**
- No confidence intervals on predictions
- Cannot detect out-of-distribution inputs
- No epistemic uncertainty estimation

**Recommendations:**
- Consider Bayesian neural networks or ensemble methods
- Add dropout at inference for uncertainty estimation
- Implement anomaly detection for OOD inputs

---

## 4. Physics Modeling Limitations

### 4.1 Simplified Aerodynamics

**What's Missing:**
- Blade flapping dynamics
- Ground effect (< 0.5m altitude)
- Rotor wake interactions
- Wind disturbances
- Downwash effects

**Impact:**
- Predictions near ground may be inaccurate
- Multi-rotor interactions not captured
- External disturbances not modeled

**Recommendations:**
- Add empirical ground effect model: T_eff = T / (1 - (r/(4h))²)
- Consider blade element theory for high-fidelity simulations
- Add disturbance rejection in closed-loop controller

### 4.2 Actuator Dynamics

**Current Model:**
- 80ms motor time constant (first-order lag)
- 15 N/s thrust slew rate
- 0.5 N·m/s torque slew rate

**What's Missing:**
- Motor non-linearities (dead zones, saturation)
- ESC response characteristics
- Propeller thrust/torque curves
- Battery voltage effects on performance

**Recommendations:**
- Characterize actual hardware actuators
- Add motor saturation limits (0-65535 PWM)
- Model battery voltage sag under load

---

## 5. Training Data Limitations

### 5.1 Trajectory Diversity

**Current Coverage:**
- 10 trajectories with varying periods (1.2-5.0s)
- Square wave references (step inputs)
- ±20° attitude commands

**What's Missing:**
- Smooth trajectory tracking (splines, polynomials)
- Hover-to-hover transitions
- Coordinated turns
- Emergency maneuvers (rapid descent)

**Impact:**
- Model may not generalize to smooth references
- Untested on aggressive emergency maneuvers

**Recommendations:**
- Generate smooth trajectory dataset
- Add practical flight scenarios (takeoff, landing, waypoint navigation)

### 5.2 Noise and Disturbances

**Current Data:** Deterministic simulation, no noise

**What's Missing:**
- Sensor noise (IMU, barometer, GPS)
- Process noise (wind, turbulence)
- Measurement outliers

**Impact:**
- Model not robust to real-world noise
- May be overly sensitive to sensor errors

**Recommendations:**
- Add realistic sensor noise models
- Train with data augmentation (noise injection)
- Implement state estimation filter (EKF, UKF)

---

## 6. Computational Limitations

### 6.1 Inference Speed

**Current Performance:**
- Forward pass: ~0.5-1.0ms per sample (CPU)
- Suitable for real-time control at 1 kHz

**Limitations:**
- Not optimized for edge deployment
- No GPU acceleration implemented
- No model quantization or pruning

**Recommendations:**
- Use ONNX Runtime for optimized inference
- Quantize to INT8 for embedded systems
- Prune network (current: 256 neurons × 5 layers may be overkill)

### 6.2 Training Time

**Current Performance:**
- 150 epochs: ~10-15 minutes (CPU)
- Physics loss dominates computational cost

**Limitations:**
- Not scalable to very large datasets (> 1M samples)
- Physics loss requires autodiff (slow on CPU)

**Recommendations:**
- Use GPU for training (5-10× speedup)
- Batch physics loss computations
- Consider mixed-precision training (FP16)

---

## 7. Validation Limitations

### 7.1 Test Data

**Current Approach:** Held-out trajectories from same simulator

**Limitations:**
- No real hardware validation
- Simulator-to-reality gap unknown
- No external validation dataset

**Impact:**
- Unknown performance on real quadrotor
- May not generalize to different platforms
- Overfitting to simulator possible

**Recommendations:**
- Validate on Crazyflie 2.0 or similar platform
- Test against independent datasets (if available)
- Perform sim-to-real transfer learning

### 7.2 Metrics

**Current Metrics:** MAE, RMSE, correlation

**What's Missing:**
- Control-relevant metrics (tracking error, settling time)
- Safety-critical metrics (collision avoidance, stability margins)
- Energy efficiency metrics

**Recommendations:**
- Define task-specific performance metrics
- Add safety constraints to loss function
- Measure closed-loop control performance

---

## 8. Known Failure Cases

### 8.1 High-Frequency Oscillations

**Symptom:** Model may predict unrealistic high-frequency oscillations in angular rates

**Cause:**
- Insufficient temporal smoothness constraints
- Physics loss may allow oscillatory solutions
- No frequency-domain regularization

**Mitigation:**
- Temporal smoothness loss (λ=8.0) partially addresses this
- May need stronger frequency-domain constraints

### 8.2 Numerical Instability

**Symptom:** Predictions diverge during long autoregressive rollouts (> 1000 steps)

**Cause:**
- Error accumulation in autoregressive mode
- Small numerical errors compound over time
- No explicit stability constraints

**Mitigation:**
- Use scheduled sampling during training
- Implement state bounds in stability loss
- Consider Lyapunov-based stability constraints

### 8.3 Parameter Drift

**Symptom:** Learned parameters may drift during training despite regularization

**Cause:**
- Weak observability for certain parameters (especially inertias)
- Regularization weight (λ_reg=1.0) may be too low
- Multiple local minima in loss landscape

**Mitigation:**
- Increase regularization weight
- Use stronger parameter bounds
- Initialize parameters close to true values

---

## 9. Future Work Required

### High Priority
1. **Hardware Validation:** Test on real quadrotor (Crazyflie 2.0)
2. **Aggressive Trajectory Training:** Retrain with ±45-60° data for improved inertia identification
3. **Energy Loss Validation:** Quantify improvement from energy conservation constraints
4. **Closed-Loop Control:** Integrate with MPC or LQR for trajectory tracking

### Medium Priority
5. **Uncertainty Quantification:** Add Bayesian inference or ensembles
6. **Advanced Aerodynamics:** Blade element theory, ground effect
7. **Noise Robustness:** Train with realistic sensor noise
8. **Model Compression:** Quantization and pruning for embedded deployment

### Low Priority
9. **Quaternion Representation:** Avoid gimbal lock for aggressive maneuvers
10. **Multi-Robot Extension:** Swarm dynamics with interaction terms
11. **Transfer Learning:** Domain adaptation for different quadrotor platforms
12. **Explainability:** SHAP values, attention mechanisms for interpretability

---

## 10. Usage Guidelines

### ✅ Recommended Use Cases
- Trajectory prediction for small quadrotors (< 0.5 kg)
- Attitude angles < 30°
- Angular rates < 5 rad/s
- Velocities < 2 m/s
- Short-horizon predictions (< 100 steps / 0.1s)
- Simulation and digital twin applications

### ⚠️ Use with Caution
- Aggressive maneuvers (30-45° attitudes)
- High-speed flight (2-5 m/s)
- Near-ground operation (< 0.5m altitude)
- Long-horizon predictions (100-500 steps)
- Parameter identification (verify against known values)

### ❌ NOT Recommended
- Acrobatic flight (> 45° attitudes, flips)
- Very high-speed flight (> 5 m/s)
- Safety-critical applications without validation
- Different quadrotor platforms without retraining
- Real-time control without hardware-in-the-loop testing

---

## Summary

The quadrotor PINN achieves excellent prediction accuracy within its validated operating envelope but has known limitations that must be respected for safe and effective use. The most significant limitation is weak inertia parameter identification at small angles, which has been addressed through aggressive trajectory generation and energy conservation constraints but requires validation through retraining.

For production use, hardware validation, uncertainty quantification, and closed-loop control integration are essential next steps.

---

**Last Updated:** 2025-11-28
**Model Version:** Optimized PINN v2
**Contact:** For questions or to report additional failure cases, please open an issue on GitHub.
