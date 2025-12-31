# Principled Improvements to GPS-IMU Detector

**Version:** 1.0
**Date:** 2025-12-30
**Status:** Implemented

---

## Scope Guard

> These improvements respect the **detectability floor**. They improve performance in the marginal regime (10-25m) through variance reduction and smarter fusion, but they do NOT:
> - Violate FPR constraints
> - Claim to detect fundamentally undetectable attacks
> - Change the core ICI detection primitive

---

## What Was Locked (NOT Changed)

| Component | Why Locked |
|-----------|------------|
| Residual impossibility | Proven theorem - cannot be "fixed" |
| Detectability floor (~25m) | Fundamental limit at 1% FPR |
| Worst-case recall ~65-70% | Expected under constraints |
| ICI primitive | Core innovation - working correctly |

---

## What Was Improved

### 1. Temporal ICI Aggregation

**File:** `src/temporal_ici.py`

**Problem:** Single-sample ICI has variance from nominal dynamics, making marginal attacks hard to separate.

**Solution:** Aggregate ICI over sliding window:
```
ICI_agg(t) = mean(ICI[t-k:t])
```

**Methods:**
- **Window Mean:** Simple variance reduction (~sqrt(k) factor)
- **EWMA:** Exponentially weighted for smooth tracking
- **CUSUM:** Cumulative sum for drift detection

**Expected Improvement:**
- Worst-case recall: 67% → ~75% (at 5% FPR)
- Variance reduced by ~sqrt(20) for window_size=20

**NOT a new detection primitive** - same ICI with reduced measurement noise.

---

### 2. Conditional Hybrid Fusion

**File:** `src/conditional_fusion.py`

**Problem:** Naive fusion `S = w_e * EKF + w_m * ICI` dilutes ICI when EKF sees nothing (consistent spoofing).

**Solution:** Conditional fusion based on innovation spectrum:
```
S(t) = ICI(t) + EKF(t) * I(high_freq_innovation)
```

where `I(.)` is 1 when EKF innovation has high-frequency content.

**How it works:**
1. Analyze EKF innovation spectrum in sliding window
2. If high-frequency content > threshold → include EKF
3. Otherwise → use ICI only (no dilution)

**When EKF is active:**
- Jump attacks (high-frequency spike)
- Oscillatory attacks (periodic component)

**When EKF is inactive:**
- Consistent spoofing (EKF sees nothing)
- Normal operation

**Expected Improvement:**
- +5-10% recall on consistent spoofing (no dilution)
- Same performance on jump attacks

---

### 3. IASP v2: Improved Self-Healing

**File:** `src/iasp_v2.py`

**Problem:** Single-step IASP projection may not fully converge to manifold.

**Solutions:**

#### 3.1 Multi-Step Iteration
```python
for k in range(n_iterations):
    x_{k+1} = g_phi(f_theta(x_k))
    if converged: break
```

Iterate projection until ICI stops decreasing.

#### 3.2 Confidence-Weighted Healing
```python
confidence = sigmoid((ICI - threshold) / scale)
alpha = confidence * max_alpha
```

Smooth, proportional blending based on detection confidence.

#### 3.3 Gradual Projection
```python
correction = clip(x_projected - x, -max_step, max_step)
x_healed = x + alpha * correction
```

Limit step size to prevent discontinuities.

#### 3.4 Rate-Limited Alpha
```python
alpha_t = alpha_{t-1} + clip(alpha_new - alpha_{t-1}, -rate, rate)
```

Prevent oscillation by limiting alpha changes.

**Expected Improvement:**
- Error reduction: 77% → 85-90%
- Stability: Improved (no oscillation)
- Quiescence: Preserved (p99 threshold)

---

## Configuration

### Temporal ICI
```python
from gps_imu_detector.src.temporal_ici import TemporalICIConfig

config = TemporalICIConfig(
    window_size=20,          # 100ms at 200Hz
    ewma_alpha=0.15,         # Smoothing factor
    cusum_threshold=5.0,     # Drift threshold
    cusum_slack=0.5,         # Allowable drift
)
```

### Conditional Fusion
```python
from gps_imu_detector.src.conditional_fusion import ConditionalFusionConfig

config = ConditionalFusionConfig(
    fs=200.0,                # Sample rate
    freq_cutoff=5.0,         # High-freq boundary (Hz)
    highfreq_threshold=0.3,  # Power fraction to activate EKF
    w_ici=0.7,               # ICI weight
    w_ekf=0.3,               # EKF weight (when active)
)
```

### IASP v2
```python
from gps_imu_detector.src.iasp_v2 import IASPv2Config

config = IASPv2Config(
    n_iterations=3,          # Projection iterations
    confidence_mode='sigmoid',
    sigmoid_scale=0.1,
    max_step_size=10.0,      # Max correction (m)
    momentum=0.9,            # Smooth healing
    max_alpha=0.95,          # Max blending
    alpha_rate_limit=0.1,    # Prevent oscillation
)
```

---

## Usage Examples

### Temporal Aggregation
```python
from gps_imu_detector.src.temporal_ici import TemporalICIAggregator

agg = TemporalICIAggregator()
agg.calibrate(nominal_ici_scores)

# Streaming
for ici_t in ici_stream:
    result = agg.update(ici_t)
    if result['alarms']['window']:
        raise_alarm()

# Batch
agg_scores = agg.score_trajectory(ici_scores, mode='window')
```

### Conditional Fusion
```python
from gps_imu_detector.src.conditional_fusion import ConditionalHybridFusion

fusion = ConditionalHybridFusion()
fusion.calibrate(nominal_ici, nominal_ekf_innovation)

for t in range(T):
    result = fusion.detect(ici_t, ekf_innovation_t)
    if result['alarm']:
        raise_alarm()
```

### IASP v2 Healing
```python
from gps_imu_detector.src.iasp_v2 import IASPv2Healer

healer = IASPv2Healer(detector, config)
healer.calibrate(nominal_trajectories)

result = healer.heal_trajectory(spoofed_trajectory)
healed = result['healed_trajectory']
```

---

## Test Coverage

```bash
pytest gps_imu_detector/tests/test_improvements.py -v
```

Tests verify:
- Temporal aggregation reduces variance
- Conditional fusion outperforms naive fusion
- IASP v2 improves error reduction
- **Detectability floor is respected**

---

## Important Notes

1. **These are NOT magic fixes.** They improve performance in the marginal regime through principled statistical techniques.

2. **The detectability floor remains.** Attacks below ~25m will still be marginal at 1% FPR.

3. **The core ICI primitive is unchanged.** These are post-processing improvements, not new detection methods.

4. **Quiescence is preserved.** All improvements have <1% false positive rate on nominal data when properly calibrated.

---

## Summary

| Improvement | Technique | Expected Gain |
|------------|-----------|---------------|
| Temporal ICI | Sliding window | Recall 67% → 75% |
| Conditional Fusion | Spectrum-gated EKF | +5-10% on consistent |
| IASP v2 | Multi-step + confidence | Error reduction 77% → 85-90% |

**Total expected improvement:**
- Worst-case recall: 67% → ~75%
- Healing error reduction: 77% → ~85-90%

**Without violating:**
- FPR constraints
- Quiescence
- Impossibility results
