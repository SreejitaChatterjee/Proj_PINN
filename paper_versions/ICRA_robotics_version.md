# ICRA/IROS Version: Robotics-Focused Framing

## Target Venues
- **ICRA 2026** (IEEE International Conference on Robotics and Automation)
- **IROS 2026** (IEEE/RSJ International Conference on Intelligent Robots and Systems)
- **Page limit:** 6-8 pages

---

## Proposed Title Options

**Option A (Problem-focused):**
> Why Your Learned Dynamics Model Fails at Control: Autoregressive Stability in Physics-Informed Neural Networks for Quadrotor Systems

**Option B (Solution-focused):**
> Stable Long-Horizon Prediction for Quadrotor Control: A Curriculum-Based Physics-Informed Neural Network Approach

**Option C (Negative result emphasis):**
> When Better Accuracy Means Worse Control: The Autoregressive Stability Paradox in Learned Quadrotor Dynamics

---

## Abstract (250 words)

Physics-Informed Neural Networks (PINNs) promise accurate dynamics models for robot control by embedding physical laws into learning. However, we reveal a critical gap: models achieving excellent single-step accuracy can catastrophically fail during closed-loop control due to compounding prediction errors. Through systematic experiments on 6-DOF quadrotor dynamics, we demonstrate that architectural sophistication—including Fourier feature embeddings and modular physics-network separation—degrades long-horizon stability by 100–3,500,000× despite improving single-step metrics by 2–10×. We identify two failure mechanisms: (1) modular architectures break dynamic coupling essential for stable recursive prediction, and (2) Fourier features suffer catastrophic extrapolation when states drift outside training distributions. To address these challenges, we develop a stability-oriented training methodology combining curriculum learning (5→50 step horizons), scheduled sampling (0→30%), and energy conservation constraints. Our approach achieves 51× improvement in 100-step prediction accuracy (0.029m vs 1.49m MAE) while simultaneously identifying physical parameters with 0% error for mass/motor coefficients and 5% for inertias. Critically, experiments with aggressive maneuvers (±45–60°) reveal fundamental simulation-to-reality limitations: the PINN learned hallucinated dynamics outside the training envelope, degrading identification despite improved theoretical observability. These results establish that autoregressive stability—not single-step accuracy—is the critical metric for deploying learned dynamics in safety-critical robot control, and provide practical guidelines for PINN-based system identification.

---

## Contributions (Robotics Framing)

1. **Failure Mode Identification for Robot Control:**
   - First systematic characterization of how architectural choices in learned dynamics models affect closed-loop control stability
   - Quantified: Modular PINN → 30m drift, Fourier PINN → 100m+ drift, vs 1.49m baseline at 100 steps

2. **Practical Training Methodology:**
   - 10-component training pipeline achieving 51× stability improvement
   - Directly applicable to quadrotor MPC and trajectory tracking
   - Complete implementation provided for reproducibility

3. **System Identification Results:**
   - Demonstrated simultaneous dynamics learning + parameter identification
   - 0% error on mass, thrust/torque coefficients
   - 5% error on inertia tensor (characterized as observability limit)

4. **Negative Result with Practical Implications:**
   - Aggressive training maneuvers can DEGRADE identification due to sim-model mismatch
   - Provides guidance on when simulation-based training is trustworthy

---

## Paper Structure (6 pages)

### I. Introduction (0.75 pages)
- **Hook:** "A learned dynamics model that achieves 0.009m single-step error can diverge to 177m in 100 steps—rendering it useless for MPC"
- **Gap:** Prior PINN work evaluates single-step; control needs multi-step stability
- **Contribution summary:** 4 bullets above

### II. Related Work (0.5 pages)
- PINNs for robotics (sparse, your niche)
- Quadrotor system identification (classical vs learning)
- Model-based RL / learned dynamics (DAgger, MBPO connection)
- Sim-to-real transfer (connect to your hallucination finding)

### III. Problem Formulation (0.5 pages)
- 12-state quadrotor dynamics
- PINN architecture with 6 learnable parameters
- Autoregressive rollout definition
- Stability metric: 100-step MAE

### IV. Failure Mode Analysis (1.25 pages)
- **A. Modular Architecture Failure**
  - Breaking translation-rotation coupling
  - Error accumulation mechanism
  - Figure: Divergence curves for all architectures

- **B. Fourier Feature Catastrophe**
  - Out-of-distribution extrapolation
  - Frequency space explosion diagram

- **C. Training Horizon Mismatch**
  - 5-step training vs 100-step testing gap

### V. Proposed Methodology (1.0 pages)
- Curriculum learning schedule (5→10→25→50)
- Scheduled sampling (0→30%)
- Energy conservation loss
- Temporal smoothness constraints
- Table: 10 components and their purposes

### VI. Experiments (1.5 pages)
- **A. Autoregressive Stability Comparison**
  - Table: Baseline vs Optimized v2 vs failed variants
  - Figure: Error growth curves

- **B. Parameter Identification**
  - Table: All 6 parameters with errors
  - Fisher information connection

- **C. Aggressive Trajectory Negative Result**
  - Figure: Parameter drift with aggressive data
  - Explanation of hallucination mechanism

### VII. Conclusion (0.5 pages)
- Key takeaway: Evaluate on autoregressive rollout, not single-step
- Practical deployment guidelines
- Limitations and future work (real hardware validation)

---

## Key Figures (4-5 total)

1. **Autoregressive Stability Comparison** (Fig 1, full width)
   - 4 curves: Baseline, Modular, Fourier, Optimized v2
   - X: timesteps, Y: position error (log scale)

2. **Failure Mechanism Diagrams** (Fig 2, two-panel)
   - Left: Modular coupling break
   - Right: Fourier extrapolation explosion

3. **Training Methodology Overview** (Fig 3)
   - Curriculum schedule + loss components

4. **Parameter Identification Results** (Fig 4)
   - Bar chart with error bars
   - Highlight observability limit

5. **Aggressive Trajectory Negative Result** (Fig 5, if space)
   - Before/after parameter values
   - Hallucination illustration

---

## Reviewers Will Ask

1. **"Why not just use longer training horizons from the start?"**
   - Answer: Curriculum is necessary; direct 50-step training diverges (ablation in supplement)

2. **"How does this compare to pure black-box models (LSTM, Transformer)?"**
   - Answer: Add 1 baseline comparison (LSTM diverges faster, no parameter ID)

3. **"Real hardware validation?"**
   - Answer: Future work; simulation study establishes methodology first

4. **"Generalization to other robots?"**
   - Answer: Principles are general; quadrotor is representative 6-DOF system

---

## Strengthening Actions Before Submission

1. **Add LSTM/MLP baseline** (1-2 days work)
   - Shows PINN advantage for stability

2. **Ablation study figure** (have data, need plot)
   - Shows each component's contribution

3. **Cleaner failure mode diagrams** (1 day)
   - Currently text-heavy; need visual explanation

4. **Code release preparation** (ongoing)
   - GitHub repo with training scripts
