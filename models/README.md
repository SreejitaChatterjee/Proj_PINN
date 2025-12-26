# Trained Models

This directory contains pre-trained PINN models for dynamics prediction and fault detection.

---

## Fault Detection Models (`security/`)

**Location:** `models/security/`

### Best Detector (w=0, seed 0)
**File:** `detector_w0_seed0.pth`
- **Type:** Pure data-driven (physics weight w=0)
- **Architecture:** 5 layers × 256 units, tanh, dropout 0.1
- **Parameters:** 204,818 (0.79 MB)
- **Val Loss:** 0.3301
- **Performance:**
  - F1: 65.7%
  - FPR: 4.5%
  - Precision: 100% (on ALFA dataset)
  - Inference: 0.34 ms (CPU)
- **Usage:**
  ```python
  from pinn_dynamics import QuadrotorPINN
  model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)
  model.load_state_dict(torch.load('models/security/detector_w0_seed0.pth'))
  ```

### Multi-Seed Ensemble (w=0, seeds 0-19)
**Files:** `detector_w0_seed0.pth` through `detector_w0_seed19.pth`
- **Count:** 20 models
- **Mean Val Loss:** 0.330 ± 0.007
- **Use Case:** Ensemble prediction or variance estimation
- **Usage:**
  ```python
  # Load all 20 models for ensemble
  models = []
  for seed in range(20):
      model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)
      model.load_state_dict(torch.load(f'models/security/detector_w0_seed{seed}.pth'))
      models.append(model)
  ```

### Physics-Informed Detector (w=20, seed 0)
**File:** `detector_w20_seed0.pth`
- **Type:** Physics-informed (physics weight w=20)
- **Val Loss:** 4.502 ± 0.147
- **Performance:** Significantly worse than w=0 (p<10^-6)
- **Use Case:** Comparison/ablation study
- **Finding:** Physics constraints hurt fault detection because faults violate Newton-Euler assumptions

---

## Dynamics Prediction Models (Legacy)

### QuadrotorPINN Diverse
**File:** `quadrotor_pinn_diverse.pth`
- **Type:** Dynamics prediction (NOT fault detection)
- **Training:** 100 diverse trajectories
- **Parameters:** 272,908
- **Use Case:** Multi-step trajectory prediction
- **Usage:**
  ```python
  from pinn_dynamics import QuadrotorPINN, Predictor
  model = QuadrotorPINN()
  model.load_state_dict(torch.load('models/quadrotor_pinn_diverse.pth'))
  predictor = Predictor(model, scaler_X, scaler_y)
  trajectory = predictor.rollout(initial_state, controls, steps=100)
  ```

---

## Model Specifications

| Model | Task | Params | Size | Val Loss | Inference |
|-------|------|--------|------|----------|-----------|
| `detector_w0_seed0.pth` | Fault Detection | 204,818 | 0.79 MB | 0.330 | 0.34 ms |
| `detector_w20_seed0.pth` | Fault Detection (w/ physics) | 204,818 | 0.79 MB | 4.502 | 0.34 ms |
| `quadrotor_pinn_diverse.pth` | Dynamics Prediction | 272,908 | 1.04 MB | --- | 0.45 ms |

---

## Quick Start

### Fault Detection
```python
from pinn_dynamics import QuadrotorPINN, Predictor
from pinn_dynamics.security import AnomalyDetector

# Load detector
model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)
model.load_state_dict(torch.load('models/security/detector_w0_seed0.pth'))

# Create detector
predictor = Predictor(model, scaler_X, scaler_y)
detector = AnomalyDetector(predictor, threshold=0.1707, use_physics=False)

# Detect faults
result = detector.detect(state, control, next_state_measured)
if result.is_anomaly:
    print(f"FAULT! Score={result.score:.3f}")
```

### Dynamics Prediction
```python
from pinn_dynamics import QuadrotorPINN, Predictor

# Load model
model = QuadrotorPINN()
model.load_state_dict(torch.load('models/quadrotor_pinn_diverse.pth'))

# Create predictor
predictor = Predictor(model, scaler_X, scaler_y)

# Predict trajectory
states_pred = predictor.rollout(initial_state, controls, steps=100)
```

---

## Training New Models

### Fault Detector
```bash
python scripts/security/train_detector.py \
    --physics_weight 0 \
    --num_seeds 20 \
    --epochs 500 \
    --hidden_size 256 \
    --num_layers 5 \
    --dropout 0.1
```

### Dynamics Model
```bash
python scripts/train_pinn.py \
    --data data/my_trajectories.csv \
    --epochs 100 \
    --physics_weight 10.0
```

---

## Model Archive

Old/experimental models not in active use:

- `archive/detector_initial.pth` - First attempt (low performance)
- `archive/detector_lstm.pth` - LSTM variant (future work)

---

## References

### Fault Detection Models
- **Paper:** "Low-False-Alarm UAV Fault Detection via PINNs" (ACSAC 2025 submission)
- **Dataset:** CMU ALFA (47 real UAV flights)
- **Results:** See `research/security/`

### Dynamics Models
- **Paper:** See `research/paper/`
- **Dataset:** EuRoC MAV (ETH Zurich)

---

## License

MIT - See LICENSE in root directory
