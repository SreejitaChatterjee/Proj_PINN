# RA-L / T-RO Version: Robotics Journal Framing

## Target Venues
- **RA-L** (IEEE Robotics and Automation Letters) - with ICRA/IROS presentation option
- **T-RO** (IEEE Transactions on Robotics) - longer, more comprehensive
- **Page limit:** 6-8 pages (RA-L), 12-16 pages (T-RO)

---

## Proposed Title

> Physics-Informed Neural Networks for Quadrotor Dynamics: Achieving Stable Long-Horizon Prediction Through Curriculum Learning and System Identification

**Alternative:**
> Stable Autoregressive Dynamics Learning for Quadrotor Control: A Physics-Informed Approach with Simultaneous Parameter Identification

---

## Abstract (200 words for RA-L)

Physics-Informed Neural Networks (PINNs) offer a principled approach to learning robot dynamics by embedding physical laws into neural networks. However, deploying learned dynamics for model predictive control requires stable multi-step predictions—a capability we show is fundamentally distinct from single-step accuracy. Through systematic experiments on 6-DOF quadrotor dynamics, we demonstrate that common architectural improvements (Fourier features, modular design) improve single-step accuracy by 2–10× but destabilize 100-step rollouts by 100–3,500,000×. We identify gradient decoupling and feature-space extrapolation as primary failure mechanisms. Our proposed training methodology—combining curriculum learning over progressively longer horizons, scheduled sampling, and energy conservation constraints—achieves 51× improvement in 100-step prediction (0.029m vs 1.49m MAE) while simultaneously identifying physical parameters: 0% error for mass and motor coefficients, 5% for inertias. We further characterize observability limits through Fisher information analysis and document a critical negative result: aggressive training maneuvers intended to improve inertia identification instead caused hallucinated dynamics due to simulator-model mismatch. Our results establish practical guidelines for deploying PINNs in safety-critical robot control and provide a validated methodology for simultaneous dynamics learning and system identification.

---

## Contributions (Journal Framing - More Comprehensive)

1. **Systematic Characterization of PINN Stability:**
   - First comprehensive study of autoregressive stability in physics-informed dynamics learning
   - Quantified failure modes: modular decoupling, Fourier extrapolation
   - Established that single-step metrics mislead for control applications

2. **Complete Training Methodology:**
   - 10-component training pipeline with ablation studies
   - Curriculum learning (5→50 steps), scheduled sampling (0→30%)
   - Energy conservation and temporal smoothness losses
   - 51× improvement with detailed analysis

3. **Simultaneous System Identification:**
   - Joint dynamics learning + parameter identification framework
   - Characterized observability limits through Fisher information
   - 0% error on mass/motor coefficients, 5% on inertias

4. **Observability and Simulation Fidelity Analysis:**
   - Theoretical analysis of inertia parameter observability
   - Empirical validation of Cramér-Rao bounds
   - Negative result: aggressive maneuvers + model mismatch → hallucinated dynamics

5. **Practical Deployment Guidelines:**
   - When to trust simulation-based training
   - Architecture selection criteria for control
   - Evaluation protocols for learned dynamics

---

## Paper Structure (8 pages RA-L / 14 pages T-RO)

### I. Introduction (1 page)
- Motivation: Learning dynamics for quadrotor control
- Gap: Single-step vs multi-step evaluation
- Contributions (5 points above)

### II. Related Work (1 page)
- **A. Physics-Informed Neural Networks**
  - Original PINN formulation (Raissi et al.)
  - Applications to dynamical systems
  - Gap: Limited study of autoregressive stability

- **B. Quadrotor Dynamics and System Identification**
  - Classical system ID approaches
  - Learning-based methods
  - Hybrid physics-learning approaches

- **C. Model-Based Control and Learned Dynamics**
  - MPC with learned models
  - Distribution shift in model-based RL
  - Importance of multi-step accuracy

### III. Problem Formulation (1 page)

#### A. Quadrotor Dynamics
- Full 12-state model: position, attitude, angular rates, velocities
- Newton-Euler equations with Euler angles
- Body-to-inertial frame transformations

#### B. PINN Architecture
- Network structure: 5 layers, 256 neurons, 204,818 parameters
- 6 learnable physics parameters: m, Jxx, Jyy, Jzz, kt, kq
- Physics loss embedding Newton-Euler equations

#### C. Autoregressive Rollout
- Definition: $\hat{x}_{t+k} = f_\theta^{(k)}(x_t, u_{t:t+k})$
- Stability metric: 100-step MAE
- Distribution shift: $p_{train}(x) \neq p_{rollout}(\hat{x})$

#### D. Observability Analysis
- Fisher Information Matrix for parameter identification
- Cramér-Rao bound on parameter estimation variance
- Inertia observability at small vs large angles

### IV. Failure Mode Analysis (1.5 pages)

#### A. Experimental Setup
- Four architecture variants tested
- Same physics constraints, different network structures
- Consistent training configuration (except architecture)

#### B. Modular Architecture Failure
- Design: Separate translational and rotational modules
- Problem: Breaks $\ddot{z} = -T\cos\theta\cos\phi/m + g$ coupling
- Mechanism: Independent error accumulation → catastrophic interaction
- Result: 30m drift at 100 steps (vs 1.49m baseline)

#### C. Fourier Feature Failure
- Design: Periodic encoding of angular states
- Problem: Extrapolation outside training distribution
- Mechanism: Small state error → large feature jump → feedback explosion
- Result: 100m+ drift at 100 steps

#### D. Training Horizon Mismatch
- 5-step training vs 100-step testing
- Model never sees compounding errors during training
- Divergence onset: t ≈ 0.06–0.08s consistently

### V. Proposed Methodology (1.5 pages)

#### A. Curriculum Learning
- Progressive horizon: 5 → 10 → 25 → 50 steps
- Epoch schedule: 50 epochs per stage
- Rationale: Learn short-term error correction first

#### B. Scheduled Sampling
- Replace ground truth with predictions: 0% → 30%
- Exposes model to its own error distribution
- Bridges train-test distribution gap

#### C. Physics-Consistent Regularization
- **Energy Conservation Loss:**
  $$\mathcal{L}_{energy} = \left(\frac{dE}{dt} - P_{in} + P_{drag}\right)^2$$

- **Temporal Smoothness Loss:**
  $$\mathcal{L}_{smooth} = \sum_i \text{ReLU}\left(\left|\frac{dx_i}{dt}\right| - v_{max,i}\right)^2$$

- **Stability Bounding Loss:**
  $$\mathcal{L}_{stab} = \sum_i \text{ReLU}(|x_i| - x_{max,i})^2$$

#### D. Complete Training Pipeline
- AdamW optimizer with cosine annealing (epochs 0-230)
- L-BFGS fine-tuning (epochs 230-250)
- Loss weights: data=1, physics=20, temporal=2, energy=5, stability=0.05

#### E. Ablation Study
- Progressive component addition
- Curriculum alone: 45% improvement
- + Scheduled sampling: 70% total
- + Dropout: 92% total
- + Energy conservation: 98% total (51× improvement)

### VI. Experiments (2 pages)

#### A. Data Generation
- 10 diverse trajectories, 49,382 samples
- Square wave references with varied periods (1.2–5.0s)
- Realistic motor dynamics (80ms time constant, slew rate limits)
- 80/20 train/test split (time-based, not random)

#### B. Autoregressive Stability Results
- **Table I:** All architectures compared
- Optimized v2: 0.029m at 100 steps (vs 1.49m baseline)
- Error growth: 1.1× (vs 17× baseline)

#### C. Multi-Horizon Evaluation
- **Table II:** Performance at 1, 10, 50, 100 steps
- Minimal degradation: 0.026m → 0.029m (1.1×)
- Demonstrates true learned dynamics, not memorization

#### D. Parameter Identification Results
- **Table III:** All 6 parameters
- Mass, kt, kq: 0.00% error
- Jxx, Jyy, Jzz: 5.00% error (observability limit)

#### E. Observability Validation
- Fisher information analysis explains 5% inertia limit
- Mass has 100× stronger gradient signal than inertia
- Experimental results match theoretical predictions

#### F. Aggressive Trajectory Experiment (Negative Result)
- Generated ±45–60° maneuvers for stronger inertia gradients
- **Result:** Inertia errors INCREASED (5% → 46%)
- **Cause:** Simulator model mismatch at large angles
- PINN learned hallucinated dynamics to fit invalid data
- **Lesson:** Aggressive excitation requires matched fidelity

### VII. Discussion (0.5 pages)

#### A. Practical Guidelines
- Always evaluate on autoregressive rollout, not single-step
- Prefer monolithic architectures for coupled dynamics
- Avoid Fourier features for autoregressive applications
- Match training horizon to deployment horizon
- Verify simulator fidelity before aggressive excitation

#### B. Limitations
- Simulation-only validation
- Single quadrotor platform
- Fixed physics structure (known equations)
- No real-time implementation

### VIII. Conclusion (0.25 pages)
- Summary of contributions
- Key finding: Training methodology > architectural complexity
- Future work: Real hardware validation, model predictive control integration

---

## Key Figures and Tables

### Figures (6-7 total):
1. **System Overview** (architecture + data flow)
2. **Failure Mode Comparison** (divergence curves, all architectures)
3. **Failure Mechanism Illustrations** (modular coupling, Fourier extrapolation)
4. **Training Methodology Diagram** (curriculum + losses)
5. **Ablation Study** (progressive improvement)
6. **Parameter Identification** (bar chart with bounds)
7. **Aggressive Trajectory Results** (negative result illustration)

### Tables (4-5 total):
1. Architecture comparison (single-step vs 100-step)
2. Multi-horizon evaluation (1, 10, 50, 100 steps)
3. Parameter identification results (6 parameters)
4. Trajectory specifications (10 training trajectories)
5. Training configuration (hyperparameters, loss weights)

---

## RA-L Specific Notes

### Presentation Option
- If accepted with ICRA/IROS presentation
- 6-page limit strict
- Prioritize: Sections I, III, IV, V, VI.A-D

### Supplementary Material
- Full trajectory specifications → supplement
- Detailed ablation results → supplement
- Additional visualization → video attachment

---

## T-RO Extended Version (If Pursuing)

### Additional Sections:
- Detailed Newton-Euler derivation
- Complete Fisher information analysis
- Extended ablation studies
- Computational cost analysis
- Comparison with classical system ID
- Real-time implementation considerations

### Extended Experiments:
- More trajectory variations
- Noise robustness analysis
- Sensitivity to hyperparameters
- Comparison with other PINN formulations

---

## Strengthening Actions

### Before Submission:
1. **Energy conservation figure** - Show energy drift comparison
2. **Trajectory diversity visualization** - All 10 trajectories
3. **Computational cost table** - Training/inference time

### For Revision:
1. **Real Crazyflie data** - Even small validation
2. **MPC integration demo** - Shows practical utility
3. **Noise robustness study** - Sensor noise handling
