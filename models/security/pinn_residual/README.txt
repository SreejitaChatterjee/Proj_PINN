# PINN Residual Ensemble Detector

## Overview
Hybrid detector combining PINN prediction residuals with temporal consistency checks using dual-trigger logic.

## Architecture
- **Type**: PINN-based ensemble
- **Components**:
  1. Residual detector (PINN prediction errors)
  2. Temporal detector (consistency over time)
- **Decision**: OR logic (either trigger = attack)

## Performance
| Metric | Value |
|--------|-------|
| **Precision** | 0.23% |
| **Recall** | 28.0% |
| **F1 Score** | 0.46% |
| **FPR** | 4.3% |

## Best For (by category recall)
- **IMU attacks**: 48.0% average recall
- **Actuator attacks**: 45.1% average recall
- **GPS attacks**: 30.4% average recall

## Specific Attack Performance
| Attack | Recall |
|--------|--------|
| Accel saturation | 100% |
| GPS oscillating | 95.3% |
| Thrust manipulation | 94.1% |
| Gyro saturation | 92.3% |
| GPS multipath | 92.3% |

## Files
- `pinn_residual_detector.py` - Detector implementation
- `pinn_residual_evaluation.json` - Evaluation results

## Usage
```python
from pinn_dynamics.security.pinn_residual_detector import PINNResidualDetector

detector = PINNResidualDetector(pinn_model, threshold_percentile=99.0)
results = detector.detect(sensor_data)
```

## When to Use
- Need both residual and temporal detection
- Moderate FPR tolerance (4.3%)
- Physical consistency validation
