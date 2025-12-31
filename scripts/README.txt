# Quadrotor PINN - Scripts

## Core Files

### Model
- `pinn_model.py` - PINN architecture for simulated data (16 inputs: states + controls)
- `pinn_base.py` - Base class for custom dynamics systems

### Training
- `train_euroc.py` - **Train on real EuRoC MAV data** (recommended)
- `train.py` - Train on simulated data
- `load_euroc.py` - Download and preprocess EuRoC dataset

### Data Generation
- `generate_diverse_training_data.py` - Generate simulated trajectories

## Quick Start

```bash
# Train on real EuRoC data (recommended)
python train_euroc.py

# Or train on simulated data
python generate_diverse_training_data.py
python train_with_diverse_data.py

# Run demo
python ../demo.py --real   # EuRoC model
python ../demo.py          # Simulated model
```

## Model Architectures

### EuRoC Model (real data)
- Input: 15 features (12 states + 3 IMU accelerations)
- Output: 12 next states
- Data: 138K samples from 5 ETH Zurich MAV sequences
- Performance: 11cm position MAE on 100-step rollout

### Simulated Model
- Input: 16 features (12 states + 4 controls)
- Output: 12 next states
- Data: 500K samples from numerical simulation
- Physics: Learnable mass and inertia parameters

## Loss Components
1. **Data loss**: MSE between predicted and true states
2. **Kinematic loss**: Position derivatives match velocities
3. **Smoothness loss**: Penalize unrealistic state jumps
4. **Physics loss**: (Simulated only) Euler equations residuals
