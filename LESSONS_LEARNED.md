# PINN Optimization: Lessons Learned

## Executive Summary

We attempted to optimize a baseline PINN for quadrotor dynamics through three iterations:
1. **Phase 0:** Fourier features + full optimizations → **Catastrophic failure** (5M m error)
2. **Phase 1:** Removed Fourier, kept optimizations → **Significant improvement but still unstable** (177 m error)
3. **Baseline:** Simple 5-layer MLP with 250 epochs → **Best autoregressive performance** (1.49 m error)

**Key Finding:** Architectural optimizations that improve single-step accuracy can **destroy autoregressive stability**. For control applications, the baseline PINN remains superior.

---

## Complete Performance Comparison

### Autoregressive Rollout (100 steps, 0.1s) - CRITICAL METRIC

| State | Baseline | Vanilla Optimized | Fourier Optimized | Winner |
|-------|----------|-------------------|-------------------|--------|
| **z (m)** | **1.49** | 177.46 | 5,199,034 | ✅ **Baseline** |
| **φ (rad)** | **0.018** | 0.240 | 8,596 | ✅ **Baseline** |
| **θ (rad)** | **0.003** | 0.047 | 1,747 | ✅ **Baseline** |
| **ψ (rad)** | **0.032** | 0.503 | 6,108 | ✅ **Baseline** |
| **p (rad/s)** | **0.067** | 4.303 | 11,932 | ✅ **Baseline** |
| **q (rad/s)** | **0.167** | 1.289 | 7,006 | ✅ **Baseline** |
| **r (rad/s)** | **0.084** | 0.173 | 1,864 | ✅ **Baseline** |
| **vz (m/s)** | **1.55** | 251.97 | 5,552,459 | ✅ **Baseline** |

**Result:** Baseline wins **ALL states** by 1-100x for vanilla, 1000-3500x for Fourier.

### Single-Step Accuracy (Teacher-Forced)

| State | Baseline | Vanilla Optimized | Winner |
|-------|----------|-------------------|--------|
| **z (m)** | 0.0872 | **0.0088** ✅ | Optimized (10x better) |
| **φ (rad)** | 0.0008 | **0.0001** ✅ | Optimized (6x better) |
| **θ (rad)** | 0.0005 | **0.0001** ✅ | Optimized (6x better) |
| **p (rad/s)** | 0.0029 | **0.0014** ✅ | Optimized (2x better) |
| **vz (m/s)** | 0.0454 | **0.0092** ✅ | Optimized (5x better) |

**Result:** Optimized wins single-step accuracy by 2-10x.

### Model Characteristics

| Metric | Baseline | Vanilla Optimized | Change |
|--------|----------|-------------------|--------|
| **Parameters** | ~100,000 | 35,470 | **-65%** ✅ |
| **Architecture** | 5-layer MLP, tanh | Residual + Modular, Swish | More complex |
| **Training Epochs** | 250 | 100 | -60% time ✅ |
| **Single-step MAE** | Good | **Excellent** ✅ | 2-10x better |
| **100-step MAE** | **1.49 m** ✅ | 177 m | **119x worse** ❌ |
| **Real-world usable?** | **Yes** ✅ | **No** ❌ | **Critical** |

---

## What Went Wrong: Three Failure Modes

### Failure Mode 1: Fourier Features (Phase 0)

**Implementation:**
- Encoded periodic states as `[x, sin(πx), cos(πx), sin(2πx), cos(2πx)]`
- Increased input dimension 12 → 48
- Goal: Capture periodicity of angles/rates

**Expected Benefit:**
- Better representation of periodic dynamics
- Reduced hidden layer size needed

**Actual Result:**
- ❌ **Catastrophic divergence** at t=0.06s
- VZ → 5.5 million m/s
- Z → 5.2 million m

**Root Cause:**
```
Fourier features extrapolate poorly outside training distribution:
- Training: states normalized to ~[-1, 1]
- Autoregressive: small drift → x = 1.05
- Fourier: sin(2π × 1.05) completely different from sin(2π × 1.0)
- Feedback loop: bad prediction → worse Fourier features → catastrophic
```

**Lesson:** Fourier features are excellent for interpolation but **dangerous for autoregressive prediction** due to poor extrapolation.

### Failure Mode 2: Modular Architecture (Both Phases)

**Implementation:**
- Separate TranslationalModule (z, vz) and RotationalModule (φ, θ, ψ, p, q, r)
- Each module: independent 64-neuron network
- Goal: Reduce parameter interference

**Expected Benefit:**
- Specialized learning for translation vs rotation
- Faster convergence (~30% based on literature)

**Actual Result:**
- ✅ Better single-step predictions
- ❌ **Modules decouple during autoregressive rollout**
- Translational and rotational dynamics lose coordination
- Errors accumulate independently, then interact catastrophically

**Root Cause:**
```
Quadrotor dynamics are fundamentally coupled:
- Thrust depends on angles: F_z = -T·cos(θ)·cos(φ)
- Angles affect acceleration
- Modular design breaks this coupling during rollout
- Baseline's single network maintains implicit coupling
```

**Lesson:** Domain knowledge suggests modularity, but **dynamics coupling requires monolithic architecture** for autoregressive stability.

### Failure Mode 3: Training Duration Mismatch

**Implementation:**
- Baseline: 250 epochs, simple architecture
- Optimized: 100 epochs, complex architecture (residual + modular + multi-step)

**Expected Benefit:**
- Complex architecture converges faster
- 60% time savings

**Actual Result:**
- ❌ Complex architecture needed **MORE** epochs, not fewer
- 100 epochs insufficient for stability
- Multi-step rollout loss added overhead but not enough epochs to converge

**Root Cause:**
```
Complexity tax:
- Residual layers: more parameters to learn
- Modular design: coordination between modules
- Multi-step rollout: 5x compute per rollout epoch
- Energy loss: additional constraint to satisfy

Simple baseline learned faster with fewer constraints.
```

**Lesson:** **Complex architectures need more training, not less**. Don't cut epochs when adding complexity.

---

## Why Single-Step ≠ Multi-Step Performance

### The Paradox

| Metric | Baseline | Optimized | Paradox |
|--------|----------|-----------|---------|
| Single-step MAE | 0.087 m | **0.009 m** (10x better) | Optimized wins |
| 100-step MAE | **1.49 m** | 177 m (119x worse) | **Baseline wins** |

**How can the model be better at 1-step but worse at 100-step?**

### The Answer: Distribution Shift + Error Accumulation

**Training Distribution:**
- Input: (x_t, u_t) from real data
- Output: x_{t+1} from real data
- Model learns: f(x_real) → x_real

**Autoregressive Inference Distribution:**
- Input: (x̂_t, u_t) from model's predictions
- Output: x̂_{t+1}
- Model encounters: f(x̂_predicted) → but trained on f(x_real)

**Error Accumulation:**
```
Step 1:  ε₁ = 0.009 m   (small, optimized model excellent)
Step 2:  x̂₂ = x₂ + ε₁   (slightly wrong input)
         ε₂ = 0.009 + δ₂ (error from wrong input)
Step 3:  x̂₃ = x₃ + ε₁ + ε₂ + δ₃
...
Step 100: ε₁₀₀ = Σ(ε_i + δ_i) = 177 m  (catastrophic)
```

**Why Baseline is More Robust:**
- Simpler architecture → fewer failure modes
- 250 epochs → extensive scheduled sampling
- Scheduled sampling exposed model to its own errors during training
- Learned to be robust to distribution shift

**Why Optimized Failed:**
- Modular architecture → coordination breaks under distribution shift
- Only 100 epochs → insufficient scheduled sampling
- 5-step rollout training << 100-step test
- Never learned to handle long-term compounding

---

## What Worked: Positive Findings

Despite the autoregressive failure, several optimizations succeeded:

### ✅ 1. Residual Connections + Swish Activation

**Impact:** Better gradient flow, smoother training
- Validation loss converged 2x faster
- No vanishing gradient issues
- Single-step accuracy improved 2-10x

**Recommendation:** Keep for any PINN, even without other optimizations.

### ✅ 2. Adaptive Physics Loss Weighting

**Impact:** Prevented early training instability
```python
λ_physics(epoch) = λ_max · (1 - exp(-k · epoch))
```
- Early: Low physics weight → fast data fitting
- Late: High physics weight → enforce constraints
- Smoother convergence than fixed weights

**Recommendation:** Use for all multi-objective PINN training.

### ✅ 3. Energy-Based Constraints (When Properly Weighted)

**Impact:** Improved parameter identification (when λ_energy = 0.05, not 5.0)
- Global energy conservation check
- Helps learn inertia tensors
- Must be soft constraint to avoid destabilization

**Recommendation:** Use with very low weight (0.01-0.05) as regularizer.

### ✅ 4. Multi-Step Rollout Loss

**Impact:** Reduced rollout error from 197,963 → 29,004 (6.8x)
- Model learned to predict its own errors
- Critical for autoregressive applications
- **But 5 steps << 100 steps needed**

**Recommendation:** Use rollout loss with K ≥ test horizon / 2. For 100-step test, need ≥50-step rollout training.

### ✅ 5. Gradient Clipping

**Impact:** Prevented exploding gradients during multi-step rollout
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Recommendation:** Essential for any multi-step training.

---

## The Right Way to Optimize PINNs for Autoregressive Prediction

Based on our failures, here's what actually works:

### Principle 1: Prioritize Autoregressive Stability Over Single-Step Accuracy

**Don't optimize:**
- Single-step teacher-forced error
- Model complexity / parameter count

**Do optimize:**
- Multi-step autoregressive error
- Long-term prediction stability
- Distribution shift robustness

### Principle 2: Simple Architectures with Long Training

**Instead of:** Complex architecture (modular + residual + Fourier) + 100 epochs

**Use:** Simple architecture (single MLP) + 250-500 epochs with heavy scheduled sampling

**Why:** Scheduled sampling is the ONLY proven technique for autoregressive stability.

### Principle 3: Match Training Horizon to Test Horizon

**If testing 100-step rollout:**
- Train with ≥50-step rollout loss (not 5-step)
- Use scheduled sampling from epoch 0
- Increase horizon gradually: 5 → 10 → 25 → 50 steps

### Principle 4: Keep Dynamics Coupled

**Don't:** Separate modules for coupled dynamics
**Do:** Single network that learns coupling implicitly

For quadrotors: z, vz, φ, θ, ψ, p, q, r are fundamentally coupled through thrust equation. Separating them breaks physics.

### Principle 5: More Complexity = More Training

**Rule of thumb:**
- Baseline architecture: N epochs
- +Residual connections: 1.2× N epochs
- +Modular design: 1.5× N epochs
- +Multi-step rollout: 2× N epochs
- +Fourier features: Don't use for autoregressive

**Our mistake:** Added 3 complexity factors but trained with 0.4× epochs.

---

## Recommended Improvements to Baseline (Conservative)

If we want to improve baseline WITHOUT breaking autoregressive stability:

### Option A: Longer Training with Scheduled Sampling

```python
# Current baseline
epochs = 250
scheduled_sampling = 0.3 * (epoch / epochs)  # 0% → 30%

# Improved baseline
epochs = 400
scheduled_sampling = 0.7 * (epoch / epochs)  # 0% → 70%
```

**Expected:** 10-20% autoregressive improvement, no risk.

### Option B: Add Residual Connections Only

Keep everything else the same, just add skip connections:
```python
class ResidualMLP(nn.Module):
    def forward(self, x):
        return x + self.net(x)  # Residual
```

**Expected:** 2x faster convergence, maintain stability.

### Option C: Multi-Step Rollout Loss (Conservative)

```python
# Add 10-step rollout loss every 5 epochs
if epoch % 5 == 0:
    rollout_loss = multistep_rollout(data, num_steps=10)
    loss += 0.5 * rollout_loss
```

**Expected:** 20-30% autoregressive improvement.

### Option D: Hybrid: A + B + C

- 400 epochs
- Residual connections
- Heavy scheduled sampling (70%)
- 10-step rollout loss

**Expected:** 30-50% autoregressive improvement, low risk.

---

## Final Recommendations

### For This Project (Quadrotor PINN)

**Use baseline PINN as the production model:**
- z MAE: 1.49 m (acceptable for MPC)
- Proven stable over 100-step rollout
- Simple architecture → easy to debug
- Well-trained (250 epochs)

**Document optimized versions as research:**
- "Explored architectural optimizations (Fourier, modular, residual)"
- "Found single-step accuracy vs autoregressive stability trade-off"
- "Established that simple architectures with long training outperform complex architectures for control"

### For Future PINN Projects

**Autoregressive applications (control, forecasting):**
1. Start simple (single MLP)
2. Train long (300-500 epochs)
3. Heavy scheduled sampling (50-70%)
4. Multi-step rollout loss (K = test horizon / 2)
5. Only add complexity if autoregressive metrics improve

**Single-step applications (system ID, parameter estimation):**
1. Optimize single-step accuracy
2. Use complex architectures (Fourier, modular, residual)
3. Shorter training OK
4. Don't worry about autoregressive stability

---

## Metrics Summary Table

| Model | Params | Epochs | Single-Step z MAE | 100-Step z MAE | Usable? |
|-------|--------|--------|-------------------|----------------|---------|
| **Baseline** ✅ | ~100K | 250 | 0.087 m | **1.49 m** | ✅ **Yes** |
| Vanilla Optimized | 35K | 100 | **0.009 m** ✅ | 177 m ❌ | ❌ No |
| Fourier Optimized | 37K | 100 | 0.041 m | 5.2M m ❌ | ❌ No |

**Winner: Baseline PINN**

---

## Key Takeaways

1. **Simpler is better** for autoregressive prediction
2. **Single-step accuracy ≠ multi-step stability**
3. **Fourier features are dangerous** for autoregressive rollout
4. **Modular architectures break coupling** in coupled dynamical systems
5. **Complex models need MORE training**, not less
6. **Scheduled sampling is essential** for autoregressive applications
7. **Match training horizon to test horizon**
8. **65% parameter reduction means nothing** if the model doesn't work

**Bottom line:** We successfully optimized the wrong objective. For control applications, baseline PINN with simple architecture and long training remains the gold standard.
