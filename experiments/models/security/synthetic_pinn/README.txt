# Synthetic PINN Detector

## Overview
Neural network trained on synthetic attack data using PINN (Physics-Informed Neural Network) residuals for anomaly detection.

## Architecture
- **Type**: PINN-based supervised detector
- **Input**: 12-state quadrotor data + 4 control inputs
- **Method**: Learns to distinguish normal vs attack patterns from PINN prediction residuals

## Performance
| Metric | Value |
|--------|-------|
| **Precision** | 93.8% |
| **Recall** | 18.7% |
| **F1 Score** | 31.2% |
| **FPR** | 0.28% |

## Best For
- **Control hijack**: 99.97% recall, 98.9% F1
- **Thrust manipulation**: 99.4% recall, 98.9% F1
- **Gyro saturation**: 100% recall, 98.6% F1
- **Accel saturation**: 99.5% recall, 97.8% F1

## Files
- `pinn_synthetic_detector.pth` - Trained model weights
- `scalers_synthetic.pkl` - Input/output scalers
- `synthetic_detector_config.json` - Model configuration

## Usage
```python
from pinn_dynamics.security.supervised_detector import SupervisedDetector

detector = SupervisedDetector.load("models/security/synthetic_pinn")
predictions = detector.detect(sensor_data)
```

## When to Use
- Low false positive rate is critical (0.28% FPR)
- Detecting actuator/control attacks specifically
- High precision applications
