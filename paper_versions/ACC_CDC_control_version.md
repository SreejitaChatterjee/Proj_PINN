# ACC/CDC Version: Control Theory Framing

## Target Venues
- **ACC** (American Control Conference)
- **CDC** (IEEE Conference on Decision and Control)
- **Page limit:** 6 pages

---

## Proposed Title Options

**Option A (System ID focus):**
> Simultaneous State Prediction and Parameter Identification for Quadrotor Dynamics Using Physics-Informed Neural Networks

**Option B (Observability focus):**
> Observability-Limited Parameter Identification in Physics-Informed Neural Networks: A Quadrotor Case Study

**Option C (Stability focus):**
> Stability Analysis of Autoregressive Physics-Informed Neural Networks for Nonlinear System Identification

---

## Abstract (200 words)

Physics-Informed Neural Networks (PINNs) enable simultaneous dynamics learning and parameter identification by embedding governing equations into neural network training. We present a systematic study of PINN-based system identification for 6-DOF quadrotor dynamics, addressing two critical challenges: autoregressive prediction stability and parameter observability limits. Our analysis reveals that architectural modifications improving single-step prediction accuracy can catastrophically destabilize multi-step rollouts—a finding with direct implications for model predictive control applications. We identify modular architecture decoupling and Fourier feature extrapolation as primary failure mechanisms, and develop a curriculum-based training methodology achieving 51× improvement in 100-step prediction stability. For parameter identification, we characterize observability limits using Fisher Information Matrix analysis: mass and motor coefficients achieve 0% identification error due to strong gradient signals from translational dynamics, while inertia parameters saturate at 5% error due to weak cross-coupling excitation at small angles (±20°). Experiments with aggressive maneuvers (±45-60°) intended to improve inertia observability instead degraded identification due to simulator-model mismatch—demonstrating that increased excitation requires matched model fidelity. Our results establish practical bounds for PINN-based system identification and provide validated guidelines for deploying learned dynamics in closed-loop control.

---

## Contributions (Control Theory Framing)

1. **Autoregressive Stability Analysis:**
   - Characterized stability properties of PINNs under recursive prediction
   - Identified failure mechanisms: gradient decoupling, distribution shift
   - Quantified: 100-3,500,000× degradation from unstable architectures

2. **Parameter Observability Analysis:**
   - Fisher Information Matrix analysis for 6 physical parameters
   - Established theoretical limits on inertia identification at small angles
   - Experimental validation of Cramér-Rao bounds

3. **Stable Training Methodology:**
   - Curriculum learning over increasing prediction horizons
   - 51× improvement in 100-step stability
   - Maintains parameter identification accuracy

4. **Model Mismatch Characterization:**
   - Documented failure mode when excitation exceeds model fidelity
   - PINN learns "effective" parameters compensating for missing physics
   - Practical guidance on training envelope selection

---

## Paper Structure (6 pages)

### I. Introduction (0.5 pages)
- Motivation: Learning-based system identification for MPC
- Challenge 1: Multi-step prediction stability
- Challenge 2: Parameter observability limits
- Contribution summary

### II. Problem Formulation (1 page)

#### A. Quadrotor Dynamics
State vector: $\mathbf{x} = [x, y, z, \phi, \theta, \psi, p, q, r, v_x, v_y, v_z]^T \in \mathbb{R}^{12}$

Control input: $\mathbf{u} = [T, \tau_x, \tau_y, \tau_z]^T \in \mathbb{R}^4$

Dynamics:
$$\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}; \boldsymbol{\theta})$$

where $\boldsymbol{\theta} = [m, J_{xx}, J_{yy}, J_{zz}, k_t, k_q]^T$ are unknown parameters.

#### B. PINN Formulation
Neural network: $\hat{\mathbf{x}}_{t+1} = g_\phi(\mathbf{x}_t, \mathbf{u}_t)$

Physics loss:
$$\mathcal{L}_{physics} = \left\| \frac{\hat{\mathbf{x}}_{t+1} - \mathbf{x}_t}{\Delta t} - f(\mathbf{x}_t, \mathbf{u}_t; \hat{\boldsymbol{\theta}}) \right\|^2$$

where $\hat{\boldsymbol{\theta}}$ are learnable parameters.

#### C. Autoregressive Rollout
$$\hat{\mathbf{x}}_{t+k} = g_\phi^{(k)}(\mathbf{x}_t, \mathbf{u}_{t:t+k-1})$$

Stability criterion: $\|\hat{\mathbf{x}}_{t+k} - \mathbf{x}_{t+k}\| \leq \epsilon(k)$ for bounded error growth.

#### D. Observability Analysis
Fisher Information Matrix element:
$$\mathcal{I}_{ij} = \mathbb{E}\left[\frac{\partial \log p(\mathbf{y}|\boldsymbol{\theta})}{\partial \theta_i} \frac{\partial \log p(\mathbf{y}|\boldsymbol{\theta})}{\partial \theta_j}\right]$$

Cramér-Rao bound: $\text{Var}(\hat{\theta}_i) \geq [\mathcal{I}^{-1}]_{ii}$

### III. Stability Analysis (1.5 pages)

#### A. Failure Mode I: Gradient Decoupling
Modular architecture separates:
- Translational module: predicts $z, v_z$
- Rotational module: predicts $\phi, \theta, \psi, p, q, r$

**Problem:** Breaks physical coupling:
$$\ddot{z} = -\frac{T \cos\theta \cos\phi}{m} + g - D(v_z)$$

During rollout, errors in $\phi, \theta$ cause thrust projection errors → $z$ divergence.

**Quantified:** 30m drift at 100 steps (vs 1.49m baseline)

#### B. Failure Mode II: Feature Extrapolation
Fourier encoding: $\gamma(x) = [\sin(\omega_k x), \cos(\omega_k x)]_{k=1}^K$

**Problem:** Small state perturbation → large feature space shift:
$$\|\gamma(x + \epsilon) - \gamma(x)\| = O(\omega_K \epsilon)$$

For high frequencies, small errors amplify exponentially in rollout.

**Quantified:** >100m drift at 100 steps

#### C. Proposed Solution: Curriculum Learning
Progressive horizon extension: 5 → 10 → 25 → 50 steps

**Rationale:** Learn short-term error correction before extending to longer horizons.

Combined with scheduled sampling (expose model to own predictions) and physics regularization.

**Result:** 51× improvement (0.029m vs 1.49m at 100 steps)

### IV. Parameter Observability Analysis (1 page)

#### A. Sensitivity Analysis
For roll dynamics:
$$\dot{p} = \frac{\tau_x}{J_{xx}} + \frac{(J_{yy} - J_{zz})}{J_{xx}} q \cdot r$$

Parameter sensitivity:
$$\frac{\partial \dot{p}}{\partial J_{xx}} = -\frac{\tau_x}{J_{xx}^2} + \frac{(J_{yy} - J_{zz})}{J_{xx}^2} q \cdot r$$

At small angles ($\pm 20°$): $|q|, |r| < 0.5$ rad/s → cross-coupling term negligible.

Compare to vertical dynamics:
$$\ddot{z} = -\frac{T}{m} + g$$
$$\frac{\partial \ddot{z}}{\partial m} = \frac{T}{m^2}$$

Mass has direct, strong gradient signal; inertias have weak signals at small angles.

#### B. Experimental Validation
**Table I: Parameter Identification Results**

| Parameter | True Value | Learned | Error |
|-----------|------------|---------|-------|
| m | 0.068 kg | 0.0680 kg | 0.00% |
| $k_t$ | 0.01 | 0.0100 | 0.00% |
| $k_q$ | 7.83e-4 | 7.83e-4 | 0.00% |
| $J_{xx}$ | 6.86e-5 | 7.21e-5 | 5.00% |
| $J_{yy}$ | 9.20e-5 | 9.66e-5 | 5.00% |
| $J_{zz}$ | 1.37e-4 | 1.43e-4 | 5.00% |

Results match observability predictions: strong-observable parameters → 0% error; weak-observable → 5% limit.

#### C. Model Mismatch Effect
Aggressive trajectories (±45-60°) generated to improve inertia observability.

**Expected:** Stronger cross-coupling → better inertia gradients → lower error

**Observed:** Inertia errors INCREASED (5% → 46%)

**Cause:** Simulator uses linearized drag; real aerodynamics nonlinear at large angles.
PINN learns "effective" inertias that compensate for missing physics.

**Implication:** Excitation must match model fidelity.

### V. Experimental Results (1 page)

#### A. Training Configuration
- 10 trajectories, 49,382 samples
- 80/20 train/test split (time-based)
- 250 epochs, curriculum schedule
- Physics weight: 20.0, Energy weight: 5.0

#### B. Stability Comparison
**Table II: 100-Step Autoregressive Performance**

| Architecture | z MAE (m) | $\phi$ MAE (rad) | Status |
|--------------|-----------|------------------|--------|
| Baseline | 1.49 | 0.018 | Reference |
| Modular | 30.0 | 0.24 | Failed |
| Fourier | 5.2M | 8596 | Catastrophic |
| **Proposed** | **0.029** | **0.001** | **51× better** |

#### C. Multi-Horizon Evaluation
Error growth: 0.026m → 0.029m (1.1×) from 1 to 100 steps
Baseline growth: 0.087m → 1.49m (17×)
Proposed method: 15× more stable

### VI. Conclusion (0.25 pages)
- Autoregressive stability ≠ single-step accuracy
- Parameter observability limits identification precision
- Model mismatch can degrade identification despite improved excitation
- Curriculum learning essential for stable multi-step prediction

---

## Key Equations to Emphasize

1. **Euler equations** (show physical coupling)
2. **Fisher Information** (connects to observability theory)
3. **Cramér-Rao bound** (theoretical limit on estimation)
4. **Sensitivity expressions** (explains mass vs inertia difference)
5. **Curriculum schedule** (practical contribution)

---

## Control Theory Angle Differentiators

### What Makes This a Control Paper:
- System identification framework (not just function approximation)
- Observability analysis using Fisher information
- Stability analysis of recursive prediction
- Connection to MPC requirements
- Physical parameter interpretability

### Language to Use:
- "System identification" not "learning"
- "Observability" not "learnability"
- "Stability" not "robustness"
- "Parameter estimation" not "parameter learning"
- "Closed-loop" and "MPC" context

### Comparisons to Include:
- Classical system ID (least squares, EKF) as context
- Mention Kalman filter limitations that motivate learning
- Connect to adaptive control literature

---

## What ACC/CDC Reviewers Want

1. **Theoretical grounding:** Fisher information analysis ✓
2. **Stability analysis:** Autoregressive error characterization ✓
3. **Physical interpretation:** All parameters have meaning ✓
4. **Practical relevance:** MPC application context ✓

### Potential Concerns:
- "Why not just use EKF for parameter estimation?"
  - Answer: PINN learns nonlinear dynamics + parameters jointly; EKF requires known structure
- "Is the stability analysis formal?"
  - Answer: Empirical characterization; formal Lyapunov analysis is future work
- "How does this compare to adaptive control?"
  - Answer: PINN is offline learning; adaptive is online. Different use cases.

---

## Additional Work for This Version

### Must-Have:
1. **Cleaner Fisher information derivation** (0.5 page math)
2. **Comparison to least-squares baseline** (shows PINN advantage)
3. **Tighter connection to MPC** (motivates why stability matters)

### Nice-to-Have:
1. **Formal Lyapunov-style stability bound**
2. **Comparison to EKF-based identification**
3. **Computational complexity analysis**
