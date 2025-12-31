# Mathematical Vulnerability Analysis: Attack Detection System

## Executive Summary

Four parallel analyses from Control Theory, Information Theory, Adversarial ML, and Signal Processing perspectives reveal fundamental mathematical vulnerabilities in the detection architecture.

**Critical Finding**: An adversary with system knowledge can craft attacks achieving **32+ meters** of position deviation while remaining undetected.

---

## 1. Unified Vulnerability Map

| Attack Vector | Source | Undetectable Magnitude | Time to Execute |
|---------------|--------|------------------------|-----------------|
| Slow GPS drift | Signal Processing | 32m at 0.5Hz | 2 seconds |
| Bias injection | Control Theory | 1670m | 10 seconds |
| Noise-floor hiding | Information Theory | 0.28m/sample | Cumulative |
| Threshold gaming | Adversarial ML | 0.49m jumps | Instant |
| Coordinated GPS+IMU | Control Theory | Unlimited | Continuous |
| EKF nullspace | Control Theory | Velocity/attitude | Continuous |

---

## 2. Control Theory Vulnerabilities

### 2.1 Observability Blind Spots

**EKF State Structure**:
```
x = [δp(3), δv(3), δθ(3), δba(3), δbg(3)]  # 15 states
```

**Measurement Matrix** (position only):
```
H = [I₃ | 0₃ₓ₁₂]  →  Only observes δp
```

**Unobservable Subspace**:
```
Null(H) = {velocity, attitude, biases}
```

**Attack**: Inject errors in velocity/attitude/bias → NIS = 0

### 2.2 Optimal Minimum-Jerk Attack

For a position deviation of A meters over T seconds:
```
δp(t) = A × [10(t/T)³ - 15(t/T)⁴ + 6(t/T)⁵]

Max jerk constraint: A = 1.67 × T³ meters
```

**For T = 10 seconds**: A = **1670 meters** undetectable!

### 2.3 Coordinated Attack Equations

To maintain kinematic consistency:
```
d(δp_gps)/dt = δv_gps ± 0.1 m/s
d(δv_gps)/dt = δa_imu ± 0.5 m/s²
```

**Attack Design**:
```python
δv(t) = d(δp)/dt      # Velocity matches position derivative
δa(t) = d²(δp)/dt²    # Acceleration matches velocity derivative
```

---

## 3. Information Theory Bounds

### 3.1 Minimum Detectable Attack (NIS-based)

```
a_min = √(χ²_threshold) × σ_sensor
```

| Confidence | χ² Threshold | Min Attack (σ=0.1m) |
|------------|--------------|---------------------|
| 95% | 7.815 | **0.28m** |
| 99% | 11.345 | **0.34m** |

**Implication**: Any attack < 0.28m is fundamentally undetectable.

### 3.2 Channel Capacity for Hidden Attacks

```
C = ½ log₂(1 + SNR) bits/sample
  = ½ log₂(1 + (0.28)²/(0.1)²)
  = 1.57 bits/sample
```

Over 1000 samples: **1570 bits** of attack information hidden.

### 3.3 Cross-Sensor Byzantine Bound

With n=4 redundant sensors (PADRE):
- **1 compromised**: Cannot influence median
- **2 compromised**: Partial control
- **3 compromised**: Full consensus control

**Minimum for consensus flip**: 2 sensors

### 3.4 Cramér-Rao Lower Bound

```
Var(â) ≥ σ²_sensor = 0.01 m²
```

No estimator can do better than sensor noise variance.

---

## 4. Adversarial ML Vulnerabilities

### 4.1 Threshold Gaming

**Fixed Thresholds** (from attack_classifier.py):
```python
jump_threshold = 0.5m        # Attack: 0.49m jumps
drift_threshold = 0.01 m/s   # Attack: 0.0099 m/s
mag_diff_threshold = 0.2 rad # Attack: 0.19 rad
```

**Attack**: Operate at threshold - ε

### 4.2 Score Normalization Poisoning

```python
# Online stats update
mean_new = mean + (value - mean) / n
std_new = √((std² × (n-2) + δ × δ') / (n-1))
```

**Attack**:
1. Inject elevated values during calibration
2. Inflates mean and std
3. Future attacks appear "normal"

**Effect**: After poisoning, threshold effectively becomes:
```
τ_effective = τ_nominal × σ_inflated / σ_true
```

### 4.3 Ensemble Weakness

```python
min_detectors_agree = 2  # Only need to evade 3 of 5 detectors!
```

**Attack Strategy**: Design attacks orthogonal to ≥3 detectors.

### 4.4 EMA Smoothing Exploitation

```python
α = 0.3  # Smoothing factor
ema = α × score + (1-α) × ema_prev
```

**Attack Pattern**: Attack every 7 samples (EMA decay interval):
```
Interval = ceil(ln(0.1) / ln(0.7)) = 7 samples
```

---

## 5. Signal Processing Vulnerabilities

### 5.1 Jerk Filter Transfer Function

```
H_jerk(z) = (1 - z⁻¹)³ / dt³
```

**Frequency Response**:
| Frequency | Relative Gain | Detectability |
|-----------|---------------|---------------|
| 0 Hz (DC) | 0 | **Invisible** |
| 0.5 Hz | 0.003 | **Nearly invisible** |
| 10 Hz | 0.029 | Weak |
| 100 Hz | 1.0 | Full |

### 5.2 Maximum Undetectable Amplitude

For sinusoidal attack at frequency f:
```
A_max = 100 / (2πf)³  meters
```

| Frequency | Max Amplitude |
|-----------|---------------|
| 0.1 Hz | 4025 m |
| 0.5 Hz | **32 m** |
| 1.0 Hz | 4.0 m |
| 2.0 Hz | 0.5 m |

### 5.3 Boxcar Integration Nulls

Window size = 10 samples at dt=0.005s → 50ms window

**Null frequencies**: f = k × 20 Hz (k = 1,2,3...)

Attack at 20 Hz integrates to **exactly zero** over each window.

### 5.4 Aliasing Attack

Nyquist = 100 Hz. Attack at 198 Hz aliases to 2 Hz.
- Injected: 198 Hz, 50mm amplitude
- Appears as: 2 Hz drift (significant for navigation)
- Jerk check: Sees 2 Hz (passes threshold)

---

## 6. Optimal Stealth Attack Recipe

### 6.1 Combined Attack Design

```python
def optimal_stealth_attack(t, dt=0.005):
    """
    Exploit all identified vulnerabilities simultaneously.
    """
    attack = np.zeros((len(t), 3))

    # 1. Slow drift (0.5 Hz) - 32m amplitude allowed
    f_drift = 0.5
    A_drift = 30.0  # meters (under 32m limit)
    attack[:, 0] += A_drift * np.sin(2*np.pi*f_drift*t)

    # 2. Coordinated velocity to maintain kinematic consistency
    vel_attack = A_drift * 2*np.pi*f_drift * np.cos(2*np.pi*f_drift*t)

    # 3. Bias injection in EKF unobservable subspace
    bias_attack = 0.01 * t  # Slow drift in bias (invisible to NIS)

    # 4. Intermittent jumps (every 7 samples, 0.49m each)
    for i in range(0, len(t), 7):
        attack[i, :] += 0.48  # Just under 0.5m threshold

    return attack, vel_attack
```

### 6.2 Attack Effectiveness

| Component | Evasion Method | Contribution |
|-----------|----------------|--------------|
| 30m drift | Filter nullspace | Primary |
| Coord. velocity | Kinematic consistency | Enables drift |
| 0.48m jumps | Threshold gaming | 68m over 1000 samples |
| Bias injection | EKF nullspace | Unbounded |

**Total deviation over 10 seconds**: >100 meters

---

## 7. Stackelberg Game Equilibrium

**Defender Strategy** (Leader - commits first):
- Weights from 5-point grid search
- Fixed thresholds: [0.3, 0.5, 0.7, 0.9]
- Publicly known parameters

**Attacker Strategy** (Follower - best response):
```
θ_A* = argmax ||δ||  subject to S_fused(δ) < 0.3
```

**Equilibrium Outcome**:
- Attacker achieves significant deviation (10-100+ meters)
- Defender achieves ~35-40% detection rate on optimal attacks

---

## 8. Vulnerability Severity Matrix

| Vulnerability | Severity | Exploitability | Fix Complexity |
|---------------|----------|----------------|----------------|
| Low-freq nullspace | **CRITICAL** | Easy | High (fundamental) |
| EKF observability gaps | **CRITICAL** | Medium | Medium |
| Threshold gaming | HIGH | Easy | Low (randomize) |
| Normalization poisoning | HIGH | Medium | Medium |
| Ensemble min_agree=2 | HIGH | Easy | Low (change to 1) |
| NIS noise floor | MEDIUM | Easy | Fundamental limit |
| EMA smoothing lag | MEDIUM | Easy | Low (reduce α) |
| Boxcar nulls | LOW | Hard | Low (vary window) |

---

## 9. Recommended Defenses

### 9.1 Immediate Fixes

```python
# 1. Lower agreement threshold
min_detectors_agree = 1  # Not 2

# 2. Randomize thresholds at runtime
threshold = base_threshold + np.random.uniform(-0.05, 0.05)

# 3. Add cumulative drift monitoring
cumulative_drift += innovation
if cumulative_drift > long_term_threshold:
    flag_anomaly()

# 4. Multi-rate jerk checking
for dt in [0.005, 0.02, 0.1]:
    check_jerk(pos, dt)
```

### 9.2 Architectural Changes

1. **Cross-state EKF updates**: Add velocity to position measurement
2. **Robust statistics**: Use median instead of mean in normalizer
3. **Adaptive thresholds**: Learn from recent clean data
4. **Spectral monitoring**: Track PSD changes, not just instantaneous
5. **CUSUM/SPRT**: Replace instantaneous NIS with sequential tests

### 9.3 Fundamental Limits (Cannot Fix)

- Cramér-Rao bound on estimation
- Shannon capacity in noise floor
- Trade-off: sensitivity vs false positive rate

---

## 10. Conclusion

The detection system has **mathematically exploitable vulnerabilities** at multiple levels:

1. **Filter design**: Triple differentiation creates low-frequency nullspace
2. **Statistical**: Fixed thresholds enable gaming
3. **Information-theoretic**: Noise floor provides attack channel
4. **Control-theoretic**: EKF has unobservable subspaces

An adversary with system knowledge can craft attacks achieving **>100 meters** of position deviation while remaining below all detection thresholds. The most critical vulnerability is the **low-frequency nullspace** in the jerk checker, allowing 32m amplitude attacks at 0.5 Hz.

**Overall Security Assessment**: The system provides defense against naive attacks but is vulnerable to sophisticated adversaries with system knowledge.
