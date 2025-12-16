# PINN Dynamics Framework

## What This Is
A **production-ready framework** for learning dynamics models from data with optional physics constraints.

## Core Features
1. **Easy System Definition** - Define any dynamical system in <20 lines
2. **Real Data Training** - Train on sensor data without control inputs
3. **Multi-Step Prediction** - Autoregressive rollout with uncertainty quantification
4. **Deployment Ready** - Export to ONNX/TorchScript for embedded systems

## Key Research Finding
Physics loss doesn't improve (and may hurt) autoregressive rollout stability.
Training regime and architecture matter more than physics constraints.
See `research/paper/` for the full analysis.

## Package Structure
```
pinn_dynamics/           # Main framework package
├── systems/             # DynamicsPINN base + implementations
├── training/            # Trainer class and losses
├── inference/           # Predictor, ONNX/TorchScript export
├── data/                # Data loaders (CSV, EuRoC)
└── utils/               # Config management

scripts/                 # Legacy scripts (still functional)
examples/                # Usage examples
research/                # Research artifacts (paper, ablation, etc.)
```

## Usage
```bash
# Quick demo
python demo.py --real    # Real EuRoC data
python demo.py           # Simulated data

# Install as package
pip install -e .

# Then use
from pinn_dynamics import QuadrotorPINN, Trainer, Predictor
```

## Datasets
- **EuRoC MAV** (real): 138K samples from ETH Zurich MAV sequences
- **Simulated**: 100 diverse trajectories for ablation studies

## Built-in Systems
| System | States | Controls |
|--------|--------|----------|
| QuadrotorPINN | 12 | 4 |
| PendulumPINN | 2 | 1 |
| CartPolePINN | 4 | 1 |
