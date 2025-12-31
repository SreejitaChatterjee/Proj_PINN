# PINN Dynamics Framework

## What This Is
A **production-ready framework** for learning dynamics models from data with optional physics constraints.

## Core Features
1. **Easy System Definition** - Define any dynamical system in <20 lines
2. **Real Data Training** - Train on sensor data without control inputs
3. **Multi-Step Prediction** - Autoregressive rollout with uncertainty quantification
4. **Deployment Ready** - Export to ONNX/TorchScript for embedded systems

## Key Research Observation
In our experiments (w=20, lr=1e-3, 20 seeds), w=0 achieved 1.74±1.03m vs w=20 achieved 2.72±1.54m (p=0.024, d=0.75).

**Important:** This does NOT imply physics loss is harmful in general. The result may be specific to:
- Our hyperparameter choices (w=20, learning rate, batch size)
- Our physics loss formulation
- Our architecture

The higher variance with physics loss (1.54m vs 1.03m) suggests it may require more careful tuning.
Future work should explore adaptive weighting, alternative formulations, and low-data regimes.

See `paper_versions/ACC_CDC_submission.tex` for the stability envelope framework.

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
