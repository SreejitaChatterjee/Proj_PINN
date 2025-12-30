# PINN Dynamics

**Learn dynamics models from data with optional physics constraints.**

A production-ready framework for physics-informed neural networks applied to dynamical systems.

## Features

- **Easy System Definition**: Define any dynamical system in <20 lines
- **Real Data Training**: Train on sensor data without control inputs
- **Multi-Step Prediction**: Autoregressive rollout with uncertainty quantification
- **Deployment Ready**: Export to ONNX/TorchScript for embedded systems

## Installation

```bash
pip install -e .
```

Or install dependencies only:
```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run demo (< 30 seconds)
python demo.py --real    # Real EuRoC flight data (recommended)
python demo.py           # Simulated data
```

**Output:**
```
PINN Demo: Quadrotor Dynamics Prediction
[1/3] Loading model... 272,908 parameters
[2/3] Loading test data... EuRoC MAV (real flight data)
[3/3] Running 100-step rollout...

RESULTS
  99-step rollout errors:
    Position MAE:  3.45 cm
    Attitude MAE:  2.76 deg
```

## Usage

### Load a Pre-trained Model

```python
from pinn_dynamics import QuadrotorPINN, Predictor

# Load model
model = QuadrotorPINN()
model.load_state_dict(torch.load('models/quadrotor_pinn_diverse.pth'))

# Create predictor
predictor = Predictor(model, scaler_X, scaler_y)

# Single-step prediction
next_state = predictor.predict(state, control)

# Multi-step rollout
trajectory = predictor.rollout(initial_state, controls, steps=100)

# With uncertainty quantification
result = predictor.rollout_with_uncertainty(initial_state, controls, n_samples=50)
print(f"Mean: {result.mean}, Std: {result.std}")
```

### Define a Custom System

```python
from pinn_dynamics import DynamicsPINN

class MyRobot(DynamicsPINN):
    def __init__(self):
        super().__init__(
            state_dim=6,
            control_dim=2,
            learnable_params={'mass': 1.0, 'inertia': 0.1}
        )

    def physics_loss(self, inputs, outputs, dt=0.01):
        # Your physics equations here
        x, v = inputs[:, :3], inputs[:, 3:6]
        force = inputs[:, 6:8]

        accel = force / self.params['mass']
        v_next_pred = v + accel * dt

        return ((outputs[:, 3:6] - v_next_pred)**2).mean()
```

### Train a Model

```python
from pinn_dynamics import QuadrotorPINN, Trainer
from pinn_dynamics.data import prepare_data

# Load data
train_loader, val_loader, scaler_X, scaler_y = prepare_data('data/training.csv')

# Create model and trainer
model = QuadrotorPINN()
trainer = Trainer(model, device='cuda', lr=0.001)

# Train with physics constraints
trainer.fit(
    train_loader, val_loader,
    epochs=100,
    weights={'physics': 10.0, 'temporal': 5.0}
)

# Save
torch.save(model.state_dict(), 'my_model.pth')
```

### Export for Deployment

```python
from pinn_dynamics.inference import export_onnx, export_torchscript

# ONNX (cross-platform)
export_onnx(model, 'model.onnx')

# TorchScript (C++ inference)
export_torchscript(model, 'model.pt')
```

## Built-in Systems

| System | States | Controls | Description |
|--------|--------|----------|-------------|
| `QuadrotorPINN` | 12 | 4 | 6-DOF quadrotor with learnable mass/inertia |
| `PendulumPINN` | 2 | 1 | Simple pendulum |
| `CartPolePINN` | 4 | 1 | Inverted pendulum on cart |

## UAV Fault Detection (Security Extension)

### GPS-IMU Detector (VALIDATED)

**Detection:** Mean AUROC 0.845 [0.82, 0.87] across five attack types with real-time latency (P95 = 2.73ms).

**Theoretical Implication:** Consistent GPS spoofing lies within the nominal Residual Equivalence Class (REC), making residual-based detection ill-posed without external trust anchors.

**Counter-Intuitive:** Physics priors degrade detection; w=0 outperforms w=20.

**Engineering Rigor:** CI leakage gates, bootstrap CIs, cross-dataset transfer, quantization profiling.

```bash
# CI check (mandatory before evaluation)
python gps_imu_detector/ci/leakage_check.py --check-code

# Full evaluation
python gps_imu_detector/run_all.py --seed 42
```

See `gps_imu_detector/README.md` for details.

### Quick Example
```python
from pinn_dynamics.security import AnomalyDetector
from pinn_dynamics import QuadrotorPINN, Predictor
import torch

# Load trained fault detector
model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)
model.load_state_dict(torch.load('models/security/detector_w0_seed0.pth'))

# Create predictor and detector
predictor = Predictor(model, scaler_X, scaler_y, device='cpu')
detector = AnomalyDetector(predictor, threshold=0.1707, use_physics=False)

# Real-time detection loop
for t in range(len(flight_data) - 1):
    state = flight_data[t, :12]
    control = flight_data[t, 12:16]
    next_state_measured = flight_data[t+1, :12]

    result = detector.detect(state, control, next_state_measured)

    if result.is_anomaly:
        print(f"[{t}] FAULT DETECTED!")
        print(f"  Anomaly Score: {result.score:.3f}")
        print(f"  Uncertainty: {result.uncertainty:.3f}")
        # Trigger emergency protocol...
```

### Counter-Intuitive Finding
Pure data-driven learning (w=0) **significantly outperforms** physics-informed variants (w=20) for fault detection (p<10^-6, effect size 13.6×).

**Why?** Fault dynamics violate Newton-Euler assumptions. Physics constraints penalize learning fault behavior, destroying the anomaly detection signal.

**Lesson:** Domain knowledge can hurt when detecting violations of those constraints.

### Dataset
Evaluated on **CMU ALFA** (Advanced Large-scale Flight Archive):
- 47 real UAV flights from Carnegie Mellon
- 5 fault categories: engine failures, rudder stuck, aileron stuck, elevator stuck, unknown
- Zero synthetic data - all real flight tests

### Key Files
- **GPS-IMU Detector**: `gps_imu_detector/run_all.py` - One-command reproduction
- **CI Leakage Gate**: `gps_imu_detector/ci/leakage_check.py` - Mandatory pre-evaluation check
- **Statistical Rigor**: `gps_imu_detector/src/statistical_rigor.py` - Bootstrap CIs
- **Reproducibility**: `gps_imu_detector/docs/REPRODUCIBILITY.md` - Full guide
- **ALFA Scripts**: `scripts/security/feature_alfa_detector.py` - Best ALFA detector
- **Examples**: `examples/uav_fault_detection.py` - Working example

### Quick Start
```bash
# GPS-IMU detector (validated, recommended)
python gps_imu_detector/ci/leakage_check.py --check-code  # CI gate
python gps_imu_detector/run_all.py --seed 42              # Full eval

# ALFA dataset (experimental)
python scripts/security/feature_alfa_detector.py --seeds 5
```

### Documentation
- `gps_imu_detector/README.md` - GPS-IMU detector (validated results)
- `research/security/README.md` - Security research summary

## Project Structure

```
pinn_dynamics/           # Framework package
├── systems/             # System implementations
│   ├── base.py          # DynamicsPINN base class
│   ├── quadrotor.py     # QuadrotorPINN
│   ├── pendulum.py      # PendulumPINN
│   └── cartpole.py      # CartPolePINN
├── training/            # Training infrastructure
│   ├── trainer.py       # Trainer class
│   └── losses.py        # Loss functions
├── inference/           # Prediction & export
│   ├── predictor.py     # High-level predictor
│   └── export.py        # ONNX/TorchScript export
├── data/                # Data loading
│   ├── loaders.py       # Generic CSV loader
│   └── euroc.py         # EuRoC MAV dataset
└── utils/               # Utilities
    └── config.py        # Configuration management

examples/                # Usage examples
├── quickstart.py        # Load and predict
├── custom_system.py     # Define your own system
├── train_model.py       # Train on your data
├── train_real_data.py   # Train on EuRoC
└── export_model.py      # Export for deployment

research/                # Research artifacts
├── paper/               # Publication sources
├── ablation/            # Ablation studies
└── weight_sweep/        # Physics weight experiments
```

## Real Data: EuRoC MAV Dataset

Train on real flight data from [ETH Zurich](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets):

```python
from pinn_dynamics.data import load_euroc

# Downloads ~1GB automatically
data = load_euroc('MH_01_easy')
```

Available sequences: `MH_01_easy`, `MH_02_easy`, `MH_03_medium`, `V1_01_easy`, etc.

## API Server

```bash
uvicorn scripts.api:app --port 8000
```

Endpoints:
- `GET /health` - Health check
- `POST /predict/{model}` - Single-step prediction
- `POST /rollout/{model}` - Multi-step rollout

## Docker

```bash
docker-compose up api       # Production API
docker-compose up dev       # Development environment
docker-compose run test     # Run tests
```

## Research Observation

In our experiments with specific hyperparameters (w=20, lr=1e-3), we observed w=0 outperforming w=20 for rollout stability. However, this does **not** imply physics loss is harmful in general---the interaction between physics constraints, optimization dynamics, and hyperparameters requires further investigation.

The paper introduces the **stability envelope** $H_\epsilon$, a formal metric for evaluating autoregressive stability.

See `paper_versions/` for the full analysis.

## Requirements

- Python 3.9+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn
- Optional: FastAPI (API), ONNX (export)

## Examples

See the `examples/` directory:
- `quickstart.py` - Load and predict in 10 lines
- `custom_system.py` - Define a double pendulum
- `train_model.py` - Train on your own data
- `train_real_data.py` - Train on EuRoC dataset
- `export_model.py` - Export for deployment

## License

MIT
