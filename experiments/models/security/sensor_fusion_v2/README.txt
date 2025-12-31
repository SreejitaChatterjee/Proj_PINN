# Sensor Fusion Detector v2 (RECOMMENDED)

## Overview
Physics-first hybrid detector. Pure physics checks handle 67% of attacks; minimal MLP only for hard cases.

## Architecture
- **Type**: No PINN, physics-first with minimal learning
- **Components**:
  1. Pure physics detector (3 thresholds):
     - Position-velocity consistency
     - Attitude-rate consistency
     - Kinematic integration consistency
  2. Residual learner (small MLP, only for physics-passed samples)
- **Decision**: Physics first; learning fills gaps

## Performance
| Metric | Value |
|--------|-------|
| **Precision** | 79.2% |
| **Recall** | 67.4% |
| **F1 Score** | 68.5% |
| **FPR** | ~5% |

## Best For
| Attack | Recall | F1 |
|--------|--------|-----|
| GPS gradual drift | 100% | 99.2% |
| GPS oscillating | 100% | 98.9% |
| IMU gradual drift | 96.4% | 97.6% |
| Actuator degraded | 92.8% | 95.9% |
| Adaptive attack | 97.5% | 98.4% |
| Slow ramp | 100% | 99.8% |

## Files
- `detector_v2.py` - Detector implementation
- `run_v2.py` - Evaluation script
- `results.json` - Per-attack results

## Usage
```python
from sensor_fusion_detector.detector_v2 import HybridDetector

detector = HybridDetector()
detector.train_residual(normal_data)  # Optional
results = detector.detect(sensor_data)
```

## Why v2 is Recommended
- **Best F1 score** (68.5%) across all models
- **Balanced** precision (79.2%) and recall (67.4%)
- **Simple** - mostly physics, minimal learning
- **Robust** - physics checks don't overfit
