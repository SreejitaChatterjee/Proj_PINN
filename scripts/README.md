# Quadrotor PINN - Streamlined Codebase

## Core Files (4 files, ~400 lines total)

### `pinn_model.py` (91 lines)
Unified PINN model with physics losses and parameter identification.

### `train.py` (105 lines)
Complete training pipeline with data loading, optimization, and loss tracking.

### `evaluate.py` (75 lines)
Model evaluation with error metrics and rollout predictions.

### `plot_utils.py` (110 lines)
All visualization utilities for training curves, state comparisons, and parameter convergence.

## Quick Start

```python
# Training
python train.py

# Evaluation
python evaluate.py

# Custom training
from train import Trainer, prepare_data
from pinn_model import QuadrotorPINN

train_loader, val_loader = prepare_data('data/quadrotor_training_data.csv')
model = QuadrotorPINN()
trainer = Trainer(model)
trainer.train(train_loader, val_loader, epochs=150)
```

## Model Architecture
- Input: 15 states (z, angles, rates, controls, angular accelerations)
- Output: 8 next states
- Parameters: 6 learnable (m, Jxx, Jyy, Jzz, kt, kq) + 1 fixed (g)
- Physics: Euler rotational dynamics + vertical dynamics

## Loss Components
1. **Data loss**: MSE between predicted and true states
2. **Physics loss**: Residuals from physical equations
3. **Regularization loss**: Pull parameters toward true values

## Legacy Files (Can be deleted)
All other `.py` files in this directory are redundant and can be removed.
