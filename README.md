# PINN Framework for Dynamics Learning

A framework for learning dynamical systems with physics-informed neural networks.

## Quick Start

```bash
# Install (development mode)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt

# Run demo (< 30 seconds)
python demo.py              # Simulated data
python demo.py --real       # Real EuRoC flight data
```

**Output:**
```
PINN Demo: Quadrotor Dynamics Prediction
[1/3] Loading model... 204,818 parameters
[2/3] Loading test data... Trajectory 2: 100 timesteps
[3/3] Running 100-step rollout...

RESULTS
  100-step rollout errors:
    Position MAE:  5.83 m
    Attitude MAE:  0.05 rad (3.10 deg)
```

## What This Does

Learns dynamics models from data:
- **Input**: State + control at time t
- **Output**: State at time t+1
- **Physics**: Optional constraint loss enforcing known equations

Currently implements **6-DOF quadrotor dynamics** as the reference system.

## Research Finding

Physics loss doesn't improve autoregressive stability. In rigorous experiments (20 seeds, 100 epochs):

| Configuration | 100-Step Error | Variance |
|--------------|----------------|----------|
| No physics loss | 1.57m | 1.06m |
| Physics weight=20 | 2.75m | 1.57m |

**Training regime and architecture matter more than physics constraints.**

See `paper_versions/ACC_CDC_submission.tex` for full analysis.

## Project Structure

```
Proj_PINN/
├── demo.py                 # Quick demo (start here)
├── scripts/
│   ├── pinn_model.py       # PINN architecture
│   ├── train.py            # Training loop
│   └── evaluate.py         # Evaluation metrics
├── models/                 # Trained weights
├── data/                   # Training/test data
└── paper_versions/         # Research paper
```

## Real Data: EuRoC MAV Dataset

Train on real quadrotor flight data from the [EuRoC MAV dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets):

```bash
# Download and preprocess (MH_01_easy, ~1GB)
python scripts/load_euroc.py --sequence MH_01_easy

# Available sequences: MH_01_easy, MH_02_easy, MH_03_medium, V1_01_easy, V1_02_medium
```

This provides real IMU + ground truth data for dynamics learning.

## Training Your Own Model

```bash
# Generate simulated data
python scripts/generate_quadrotor_data.py

# Train
python scripts/train_with_diverse_data.py

# Evaluate
python scripts/evaluate_diverse_model.py
```

## API

### Base Class

```python
from scripts.pinn_base import DynamicsPINN

class MySystemPINN(DynamicsPINN):
    def __init__(self):
        super().__init__(
            state_dim=4,
            control_dim=1,
            hidden_size=128,
            learnable_params={'mass': 1.0, 'length': 0.5}
        )

    def physics_loss(self, inputs, outputs, dt=0.01):
        # Implement your system's governing equations
        ...
```

### Built-in Systems

```python
from scripts.pinn_base import PendulumPINN, CartPolePINN
from scripts.pinn_model import QuadrotorPINN

# Simple pendulum (2 states, 1 control)
pendulum = PendulumPINN()

# Cart-pole (4 states, 1 control)
cartpole = CartPolePINN()

# Quadrotor (12 states, 4 controls)
quadrotor = QuadrotorPINN()

# Model summary
print(pendulum.summary())
```

### Training Loop

```python
model = QuadrotorPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    pred = model(inputs)

    # Supervised loss
    data_loss = F.mse_loss(pred, targets)

    # Physics loss (optional)
    phys_loss = model.physics_loss(inputs, pred, dt=0.001)

    loss = data_loss + 0.1 * phys_loss
    loss.backward()
    optimizer.step()
```

### Rollout

```python
# Autoregressive prediction
trajectory = model.rollout(initial_state, controls)  # [n_steps, state_dim]
```

## API Server

```bash
# Start server
uvicorn scripts.api:app --reload --port 8000

# Endpoints
GET  /health              # Health check
GET  /info/{model}        # Model info
POST /predict/{model}     # Single-step prediction
POST /rollout/{model}     # Multi-step rollout
```

Example request:
```bash
curl -X POST http://localhost:8000/predict/quadrotor \
  -H "Content-Type: application/json" \
  -d '{"state": [0,0,-1,0,0,0,0,0,0,0,0,0], "control": [0.67,0,0,0]}'
```

## Docker

```bash
# Development
docker-compose up dev

# Production API
docker-compose up api

# Run tests
docker-compose run test
```

## ONNX Export

```bash
# Export model to ONNX
python scripts/export.py --model quadrotor --output models/quadrotor.onnx

# Verify export
python scripts/export.py --model quadrotor --verify
```

## Experiment Tracking

```python
from scripts.tracking import ExperimentTracker

# Local logging
tracker = ExperimentTracker(backend="local")

# Weights & Biases
tracker = ExperimentTracker(backend="wandb", project="pinn-dynamics")

# MLflow
tracker = ExperimentTracker(backend="mlflow")

tracker.log_params({"lr": 0.001})
tracker.log_metrics({"loss": 0.5}, step=1)
tracker.finish()
```

## Configuration

```bash
# Train with config file
python scripts/train.py --config configs/quadrotor.yaml
```

See `configs/` for example configurations.

## Testing

```bash
pytest tests/ -v
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn
- FastAPI, Uvicorn (for API)
- See `requirements.txt` for full list

## Citation

```bibtex
@software{pinn_dynamics_2024,
  title = {PINN Framework for Dynamics Learning},
  year = {2024},
  url = {https://github.com/[username]/Proj_PINN}
}
```

## License

MIT
