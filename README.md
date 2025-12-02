# Physics-Informed Neural Networks for Quadrotor Dynamics

A comprehensive implementation of Physics-Informed Neural Networks (PINNs) for learning and predicting quadrotor dynamics with physics constraints and parameter identification.

## 🎯 Project Overview

This project implements a PINN that:
- **Learns 12-state quadrotor dynamics** (position, orientation, angular rates, velocities)
- **Identifies 6 physical parameters** (mass, inertias, motor coefficients)
- **Enforces physics constraints** through custom loss functions
- **Achieves 51× improvement** over baseline at 100-step prediction horizon
- **Maintains <5% parameter error** for all learnable parameters

### Key Features

✅ **Complete 6-DOF quadrotor dynamics modeling**
✅ **Physics-informed loss with temporal smoothness constraints**
✅ **Realistic actuator models** (motor dynamics, slew rate limits)
✅ **10 diverse training trajectories** with square wave references
✅ **Comprehensive visualization suite** (176 plots generated)
✅ **Extensive ablation studies and performance analysis**

## 📊 Results Summary

| Metric | Baseline | Optimized PINN v2 | Improvement |
|--------|----------|-------------------|-------------|
| **100-step horizon error** | 1.49 m | 0.029 m | **51× better** |
| **Altitude MAE** | 0.440 m | 0.022 m | **95% reduction** |
| **Angular rate MAE** | 0.36-1.31 rad/s | 0.002-0.004 rad/s | **99.5-99.7% reduction** |
| **Mass identification** | - | **0.07% error** | Near-perfect |
| **Motor coefficients (kt, kq)** | - | **0.01%, 0.00% error** | Near-perfect |
| **Inertias (Jxx, Jyy, Jzz)** | 1300-6700% error | **5.00% error** | Observability-limited |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Matplotlib
- Scikit-learn, Joblib

### Installation

```bash
# Clone the repository
git clone https://github.com/SreejitaChatterjee/Proj_PINN.git
cd Proj_PINN

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Generate Training Data

```bash
# Generate 10 diverse quadrotor trajectories with realistic physics
python scripts/generate_quadrotor_data.py
```

**Output:** `data/quadrotor_training_data.csv` (49,382 samples)

**Features:**
- 10 trajectories with varying square wave references (1.2s to 5.0s periods)
- Motor dynamics (80ms time constant)
- Slew rate limits (15 N/s thrust, 0.5 N·m/s torque)
- Reference filtering (250ms time constant)

#### 2. Train the PINN Model

```bash
# Train with physics-informed loss and parameter identification
python scripts/train.py
```

**Output:**
- `models/quadrotor_pinn.pth` - Trained model weights
- `models/scalers.pkl` - Input/output scalers
- Training curves and convergence plots

**Training Configuration:**
- 150 epochs, Adam optimizer (lr=0.0005)
- Batch size: 64
- Physics loss weight: 20.0
- Regularization weight: 1.0

#### 3. Evaluate Model Performance

```bash
# Evaluate on held-out test trajectories
python scripts/evaluate.py
```

**Output:** Comprehensive metrics (MAE, RMSE, correlation) for all 12 states

#### 4. Generate Visualizations

```bash
# Generate all 16 time-series comparison plots
python scripts/generate_all_18_plots.py

# Generate individual trajectory plots (160 plots: 10 trajectories × 16 variables)
python scripts/generate_individual_trajectory_plots.py

# Generate comparative plots (16 plots with all 10 trajectories as subplots)
python scripts/generate_comparative_trajectory_plots.py
```

**Output directories:**
- `results/detailed/` - Individual variable time-series
- `results/individual_trajectories/` - Per-trajectory analysis
- `results/comparative_trajectories/` - Side-by-side comparisons

## 📁 Project Structure

```
Proj_PINN/
├── data/                          # Training/test data
│   ├── quadrotor_training_data.csv
│   └── aggressive_test_trajectories.pkl
├── models/                        # Trained models and scalers
│   ├── quadrotor_pinn.pth
│   └── scalers.pkl
├── scripts/                       # Main Python scripts
│   ├── generate_quadrotor_data.py      # Data generation
│   ├── train.py                        # PINN training
│   ├── evaluate.py                     # Model evaluation
│   ├── pinn_model.py                   # PINN architecture
│   ├── plot_utils.py                   # Plotting utilities
│   ├── generate_all_18_plots.py        # Generate all plots
│   ├── generate_individual_trajectory_plots.py
│   └── generate_comparative_trajectory_plots.py
├── results/                       # Visualization outputs
│   ├── detailed/                  # Time-series analysis
│   ├── individual_trajectories/   # Per-trajectory plots
│   └── comparative_trajectories/  # Comparative analysis
├── reports/                       # Documentation
│   ├── quadrotor_pinn_report.pdf  # Complete technical report (78 pages)
│   └── quadrotor_pinn_report.tex  # LaTeX source
└── README.md                      # This file
```

## 🔬 Technical Details

### PINN Architecture

```
Input (16 features) → [256 neurons, 5 layers, Tanh activation, Dropout 0.3]
                    → Output (12 states + 6 parameters)
```

**Input features:**
- Current states: x, y, z, φ, θ, ψ, p, q, r, vx, vy, vz (12)
- Control inputs: thrust, τx, τy, τz (4)

**Output predictions:**
- Next-step states: x, y, z, φ, θ, ψ, p, q, r, vx, vy, vz (12)
- Learned parameters: m, Jxx, Jyy, Jzz, kt, kq (6)

### Physics-Informed Loss Function

```
L_total = L_data + λ_physics·L_physics + λ_reg·L_reg + λ_temporal·L_temporal
```

**Components:**
1. **Data Loss (MSE):** Prediction accuracy on training samples
2. **Physics Loss:** Violation of quadrotor dynamics equations
   - Rotational dynamics (Euler equations)
   - Translational dynamics (Newton's laws)
   - Position kinematics (body-to-inertial transformation)
3. **Regularization Loss:** Parameter constraints (bounded ranges)
4. **Temporal Smoothness:** Derivative consistency constraints

**Loss weights:** λ_physics = 20.0, λ_reg = 1.0, λ_temporal = 8.0

### Quadrotor Dynamics

The PINN enforces these physics equations:

**Rotational Dynamics (Euler equations):**
```
ṗ = (τx + (Jyy - Jzz)qr) / Jxx
q̇ = (τy + (Jzz - Jxx)pr) / Jyy
ṙ = (τz + (Jxx - Jyy)pq) / Jzz
```

**Translational Dynamics (Newton's laws in body frame):**
```
v̇x = (rv_y - qv_z) - g·sin(θ)
v̇y = (pv_z - rv_x) + g·sin(φ)cos(θ)
v̇z = (qv_x - pv_y) + g·cos(φ)cos(θ) - T/m
```

**Position Kinematics (body to inertial frame):**
```
ẋ = R(φ,θ,ψ) · [vx, vy, vz]ᵀ
```

## 📈 Trajectory Differentiation

The 10 training trajectories use diverse square wave references:

| Trajectory | Description | Period Range | Amplitude Characteristics |
|------------|-------------|--------------|--------------------------|
| 0 | Standard square wave | 2.0-3.0s | ±10° roll, ±5° pitch/yaw, -5 to -3m alt |
| 1 | Fast aggressive | 1.5-2.5s | ±15° roll, ±8° pitch, ±10° yaw |
| 2 | Slow gentle | 3.0-4.0s | ±5° roll, ±3° pitch, ±5° yaw |
| 3 | Asymmetric | 2.0-2.5s | -12° to +8° roll, ±12° yaw |
| 4 | High amplitude | 1.8-2.2s | ±20° roll, ±10° pitch |
| 5 | Medium frequency | 2.5-3.5s | ±8° roll, ±4° pitch |
| 6 | Large asymmetric | 3.0-4.0s | -6° to +12° roll, -8° to +16° yaw |
| 7 | Very fast high amplitude | 1.2-1.8s | ±18° roll, ±12° pitch, ±15° yaw |
| 8 | Very slow | 4.0-5.0s | ±6° roll, ±4° pitch |
| 9 | Mixed frequency | 2.2-3.0s | ±10° roll, ±8° pitch, ±14° yaw |

This diversity ensures comprehensive coverage of the quadrotor's operational envelope.

## 🛠️ Advanced Features

### Curriculum Learning

Progressive multi-step rollout training:
- Epochs 0-50: 5-step predictions
- Epochs 50-100: 10-step predictions
- Epochs 100-150: 25-step predictions
- Epochs 150-230: 50-step predictions
- Epochs 230-250: L-BFGS fine-tuning

### Scheduled Sampling

Gradually increases teacher forcing ratio:
- Starts at 0% (full teacher forcing)
- Increases to 30% by end of training
- Improves autoregressive stability

### Realistic Actuator Dynamics

**Motor Time Constant:** 80ms first-order lag
```
T_actual(t+dt) = T_actual(t) + (dt/(τ+dt))·(T_cmd - T_actual)
```

**Slew Rate Limits:**
- Thrust: 15 N/s maximum
- Torques: 0.5 N·m/s maximum

**Reference Filtering:** 250ms time constant for smooth setpoint transitions

## 📖 Documentation

### Complete Technical Report

See [`reports/quadrotor_pinn_report.pdf`](reports/quadrotor_pinn_report.pdf) (78 pages) for:
- Detailed methodology
- Complete physics derivations
- Ablation studies
- Optimization progression
- Comprehensive results analysis
- Future work directions

### Key Sections

1. **Project Overview** - Architecture and objectives
2. **Implementation Process** - 25-step development pipeline
3. **Complete Results** - All performance metrics
4. **State Analysis** - All 19 variables analyzed
5. **Visual Results** - 16 comparative trajectory plots
6. **Baseline Model** - Initial implementation
7. **Optimization** - Advanced techniques explored
8. **Autoregressive Stability** - Long-horizon prediction analysis
9. **Optimized PINN v2** - Final solution achieving 51× improvement
10. **Conclusion** - Summary and future work

## 🎓 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@software{chatterjee2024quadrotor_pinn,
  author = {Chatterjee, Sreejita},
  title = {Physics-Informed Neural Networks for Quadrotor Dynamics},
  year = {2024},
  url = {https://github.com/SreejitaChatterjee/Proj_PINN}
}
```

## 📝 License

This project is available for academic and research purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

**Author:** Sreejita Chatterjee
**Repository:** [https://github.com/SreejitaChatterjee/Proj_PINN](https://github.com/SreejitaChatterjee/Proj_PINN)

## 🙏 Acknowledgments

This project implements Physics-Informed Neural Networks (PINNs) for quadrotor system identification and prediction, combining:
- Deep learning (PyTorch)
- Classical control theory
- Nonlinear dynamics
- Multi-objective optimization

## 🚧 Current Limitations & Future Work

### Remaining Limitations

1. **Inertia Observability:** Jxx/Jyy/Jzz achieve 5% error (reduced from 1300-6700% baseline), but further improvement limited by fundamental observability at small angles
2. **Simplified Aerodynamics:** Only linear drag modeled (no blade flapping, ground effect)
3. **Limited Operating Envelope:** Training focuses on ±20° attitudes

### Planned Improvements

1. **Aggressive Trajectories:** Add ±45-60° maneuvers for better inertia identification
2. **Angular Accelerations:** Include ṗ, q̇, ṙ measurements for stronger observability
3. **Energy Constraints:** Alternative physics formulations for parameter identification
4. **Real Hardware Validation:** Test on Crazyflie 2.0 platform
5. **Advanced Aerodynamics:** Blade flapping, ground effect, rotor wash
6. **Real-time Control:** Online state estimation and MPC integration

## 📊 Performance Metrics

### State Prediction Accuracy

| State | MAE | RMSE | Correlation |
|-------|-----|------|-------------|
| **X Position** | 0.023 m | 0.064 m | 0.999 |
| **Y Position** | 0.031 m | 0.098 m | 0.998 |
| **Altitude (Z)** | 0.070 m | 0.165 m | 0.999 |
| **Roll (φ)** | 0.0008 rad (0.045°) | 0.0012 rad | 0.999 |
| **Pitch (θ)** | 0.0005 rad (0.028°) | 0.0008 rad | 0.999 |
| **Yaw (ψ)** | 0.0009 rad (0.052°) | 0.0015 rad | 0.999 |
| **Roll Rate (p)** | 0.0034 rad/s | 0.0054 rad/s | 0.999 |
| **Pitch Rate (q)** | 0.0014 rad/s | 0.0024 rad/s | 0.999 |
| **Yaw Rate (r)** | 0.0029 rad/s | 0.0044 rad/s | 0.999 |
| **X Velocity** | 0.008 m/s | 0.018 m/s | 0.997 |
| **Y Velocity** | 0.012 m/s | 0.027 m/s | 0.990 |
| **Z Velocity** | 0.040 m/s | 0.074 m/s | 0.999 |

### Parameter Identification

| Parameter | True Value | Learned Value | Absolute Error |
|-----------|------------|---------------|----------------|
| **Mass (m)** | 0.068 kg | 0.068 kg | **0.07%** |
| **Thrust coeff (kt)** | 0.01 | 0.01 | **0.01%** |
| **Torque coeff (kq)** | 7.83×10⁻⁴ | 7.83×10⁻⁴ | **0.00%** |
| **Jxx** | 6.86×10⁻⁵ | 7.21×10⁻⁵ | **5.00%** |
| **Jyy** | 9.20×10⁻⁵ | 9.66×10⁻⁵ | **5.00%** |
| **Jzz** | 1.37×10⁻⁴ | 1.43×10⁻⁴ | **5.00%** |

---

**Built with ❤️ using Physics-Informed Neural Networks**
