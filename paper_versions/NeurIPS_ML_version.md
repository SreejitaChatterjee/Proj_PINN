# NeurIPS/ICLR Version: Machine Learning Theory Framing

## Target Venues
- **NeurIPS 2025** (Conference on Neural Information Processing Systems)
- **ICLR 2026** (International Conference on Learning Representations)
- **Page limit:** 9 pages + unlimited appendix

---

## Proposed Title Options

**Option A (Phenomenon-focused):**
> The Expressivity-Stability Tradeoff: Why Complex Neural Architectures Fail at Autoregressive Dynamics Prediction

**Option B (Theory-flavored):**
> On the Instability of Expressive Physics-Informed Networks Under Autoregressive Rollout

**Option C (Provocative):**
> Architectural Complexity is the Enemy of Temporal Stability in Learned Dynamics Models

---

## Abstract (300 words)

Physics-Informed Neural Networks (PINNs) embed physical constraints into neural networks to learn dynamics from data. While prior work evaluates PINNs on single-step prediction accuracy, practical deployment—in model predictive control, simulation, or forecasting—requires stable multi-step autoregressive rollout where model predictions feed back as inputs. We demonstrate a fundamental expressivity-stability tradeoff: architectural modifications that improve single-step accuracy by 2–10× can catastrophically destabilize autoregressive rollouts by 10²–10⁶×, causing prediction errors to explode from centimeters to kilometers within 100 steps.

Through systematic experiments on a 12-dimensional nonlinear dynamical system (6-DOF quadrotor), we identify two primary failure mechanisms. First, modular architectures that decouple physical subsystems for independent learning break implicit dynamic coupling, causing coordinated errors to accumulate independently then interact catastrophically. Second, Fourier feature embeddings—highly effective for stationary function approximation—suffer severe distribution shift during rollout: small state perturbations map to large feature-space discontinuities, triggering exponential error growth.

We propose a stability-oriented training framework combining curriculum learning over increasing rollout horizons, scheduled sampling to expose models to their own error distributions, and physics-based regularizers enforcing energy conservation and temporal smoothness. This approach achieves 51× improvement in 100-step prediction stability while maintaining parameter identification accuracy, demonstrating that training methodology—not architectural expressivity—determines autoregressive stability.

Our results challenge the prevailing assumption that more expressive architectures yield better learned dynamics models. We establish that standard single-step evaluation metrics are fundamentally misleading for autoregressive applications and propose rollout-based evaluation protocols. These findings have broad implications for neural dynamics learning, neural ODEs, model-based reinforcement learning, and any domain requiring stable long-horizon prediction from learned models.

---

## Contributions (ML Theory Framing)

1. **Identified Expressivity-Stability Tradeoff:**
   - First systematic demonstration that architectural expressivity inversely correlates with autoregressive stability in physics-informed learning
   - Quantified: 2–10× single-step improvement → 100–3,500,000× rollout degradation

2. **Characterized Two Failure Mechanisms:**
   - **Modular decoupling:** Gradient isolation breaks dynamic coupling in autoregressive chains
   - **Fourier extrapolation:** High-frequency features amplify distribution shift exponentially

3. **Proposed Stability-Oriented Training Framework:**
   - Curriculum learning (progressive horizon extension)
   - Scheduled sampling (exposure to own error distribution)
   - Physics-consistent regularization (energy, smoothness)
   - Demonstrated 51× improvement with principled ablations

4. **Established Evaluation Methodology:**
   - Showed single-step metrics are inadequate for autoregressive applications
   - Proposed multi-horizon evaluation protocol for learned dynamics

---

## Paper Structure (9 pages)

### 1. Introduction (1 page)
- **Opening:** "We show that architectural choices validated by single-step benchmarks can introduce instabilities invisible to standard evaluation yet fatal in deployment"
- **Problem:** Gap between supervised learning (i.i.d. samples) and autoregressive deployment (distribution shift)
- **Contribution:** Expressivity-stability tradeoff + training methodology

### 2. Related Work (0.75 pages)
- Physics-informed neural networks
- Neural ODEs and continuous dynamics learning
- Model-based RL (distribution shift in learned models)
- Scheduled sampling, DAgger (exposure bias in sequence models)
- Fourier features / positional encodings

### 3. Problem Setting (0.75 pages)
- General formulation: $x_{t+1} = f_\theta(x_t, u_t)$
- Autoregressive rollout: $\hat{x}_{t+k} = f_\theta^{(k)}(\hat{x}_t, u_{t:t+k})$
- Distribution shift: $p_{train}(x) \neq p_{rollout}(\hat{x})$
- Physics-informed constraint: $\mathcal{L}_{physics} = \|f_\theta - f_{physics}\|^2$
- Experimental system: 12D quadrotor (representative nonlinear dynamics)

### 4. The Expressivity-Stability Tradeoff (2 pages)

#### 4.1 Experimental Setup
- Four architectures: Baseline, Modular, Fourier, Optimized
- Same physics constraints, different network structures
- Evaluation: 1-step (teacher-forced) vs 100-step (autoregressive)

#### 4.2 Main Result
- **Table 1:** Single-step vs 100-step accuracy for all architectures
- Key finding: Inverse correlation between single-step and multi-step performance

#### 4.3 Failure Mode I: Modular Architecture Decoupling
- **Mechanism:** Separate translation/rotation modules → broken $\ddot{z} = f(\phi, \theta, T)$ coupling
- **Analysis:** Error accumulates independently in each module, then interacts
- **Figure:** Error trajectory decomposition showing decoupled growth

#### 4.4 Failure Mode II: Fourier Feature Extrapolation
- **Mechanism:** $\text{Fourier}(x + \epsilon) \neq \text{Fourier}(x) + O(\epsilon)$ for high frequencies
- **Analysis:** Small state drift → large feature-space jump → catastrophic feedback
- **Figure:** Feature space visualization showing extrapolation discontinuity

### 5. Stability-Oriented Training (1.5 pages)

#### 5.1 Curriculum Learning Over Rollout Horizon
- Progressive schedule: 5 → 10 → 25 → 50 steps
- **Intuition:** Learn short-term error correction before extending horizon
- **Ablation:** Direct 50-step training fails

#### 5.2 Scheduled Sampling
- Replace ground truth with model predictions: 0% → 30%
- **Intuition:** Expose model to its own error distribution during training
- Bridges train-test distribution gap

#### 5.3 Physics-Consistent Regularization
- Energy conservation: $\frac{dE}{dt} = P_{in} - P_{drag}$
- Temporal smoothness: $\|\frac{d\hat{x}}{dt} - v_{max}\|^+$
- Prevents unphysical predictions in rollout

#### 5.4 Complete Training Pipeline
- **Algorithm 1:** Full training procedure
- Loss weighting schedule

### 6. Experiments (2 pages)

#### 6.1 Main Results
- **Table 2:** Complete comparison across all metrics
- 51× improvement in 100-step MAE
- 15× reduction in error growth rate (1.1× vs 17× over 100 steps)

#### 6.2 Ablation Study
- **Figure:** Progressive addition of components
- Each component contributes; full combination is synergistic

#### 6.3 Parameter Identification
- Simultaneous dynamics learning + system identification
- 0% error on motor coefficients ($k_t$, $k_q$); 40% error on mass; 52-60% error on inertias

#### 6.4 Analysis: Why Training Methodology > Architecture
- Dropout regularization effect
- Monolithic coupling preservation
- Error distribution exposure

### 7. Discussion and Broader Impact (1 page)

#### 7.1 Implications for Neural Dynamics Learning
- Single-step benchmarks are insufficient
- Propose rollout-based evaluation standard

#### 7.2 Connection to Model-Based RL
- Distribution shift in learned models
- Implications for MBPO, Dreamer, etc.

#### 7.3 Limitations
- Single dynamical system (quadrotor)
- Simulation only (no real-world validation)
- Fixed physics structure (known equations)

#### 7.4 Future Directions
- Theoretical analysis: When does curriculum guarantee stability?
- Extension to unknown physics (pure data-driven)
- Real-world validation

### 8. Conclusion (0.25 pages)

---

## Key Figures (5-6 total)

1. **Expressivity-Stability Tradeoff** (Fig 1, teaser)
   - Left: Single-step accuracy (complex wins)
   - Right: 100-step accuracy (simple wins)
   - Arrow showing inverse relationship

2. **Failure Mode Diagrams** (Fig 2, two-panel)
   - (a) Modular coupling break illustration
   - (b) Fourier feature extrapolation in feature space

3. **Error Growth Curves** (Fig 3)
   - All architectures on log scale
   - Divergence points marked

4. **Training Methodology Overview** (Fig 4)
   - Curriculum schedule
   - Scheduled sampling curve
   - Loss component diagram

5. **Ablation Study** (Fig 5)
   - Progressive component addition
   - Shows synergistic effect

6. **Evaluation Protocol Comparison** (Fig 6, if space)
   - Teacher-forced vs autoregressive
   - Shows misleading nature of single-step

---

## What NeurIPS/ICLR Reviewers Will Want

1. **Theoretical grounding:**
   - Add section on error propagation theory
   - Lipschitz bounds on autoregressive error growth
   - Connection to stability theory in dynamical systems

2. **Broader experimental validation:**
   - At minimum: 1-2 additional dynamical systems
   - Cart-pole, pendulum, Lorenz attractor
   - Show same pattern holds

3. **Comparison to neural ODEs:**
   - How does NODE handle this?
   - Is the problem specific to discrete-time PINNs?

4. **Formal characterization:**
   - Define "stability envelope" mathematically
   - Conditions for bounded error growth

---

## Additional Work Needed for This Version

### Must-Have (before submission):
1. **Second dynamical system** (cart-pole or Lorenz)
   - 1-2 weeks work
   - Shows generalization beyond quadrotor

2. **Neural ODE baseline**
   - Does continuous-time formulation help?
   - Important for ML audience

3. **Error propagation analysis**
   - Even informal Lipschitz-style argument
   - Connects to theory

### Nice-to-Have:
4. **Lyapunov-inspired stability loss**
   - Would strengthen theoretical contribution

5. **Transformer baseline**
   - Sequence models on dynamics

---

## Positioning Statement (for cover letter)

This paper reveals a fundamental but overlooked phenomenon in neural dynamics learning: architectural modifications that improve single-step accuracy can catastrophically destabilize autoregressive rollouts. We systematically characterize two failure mechanisms—modular decoupling and Fourier extrapolation—and develop a training methodology that resolves them. Our findings challenge standard evaluation practices and have direct implications for neural ODEs, model-based RL, and any domain deploying learned dynamics models. The work bridges physics-informed learning, dynamical systems, and practical ML deployment.
