# Sequence PINN Detector

## Overview
Temporal sequence-based PINN detector that analyzes 20-step windows to capture attack patterns over time.

## Architecture
- **Type**: PINN-based temporal detector
- **Input**: 20-step sequences of 12-state + 4 control
- **Method**: PINN prediction errors aggregated over temporal windows

## Performance
| Metric | Value |
|--------|-------|
| **Precision** | 0.3% |
| **Recall** | 44.5% |
| **F1 Score** | 0.6% |
| **FPR** | 5.0% |

## Best For (by category recall)
- **IMU attacks**: 83.6% average recall
- **Stealth attacks**: 51.4% average recall
- **GPS attacks**: 40.4% average recall

## Specific Attack Performance
| Attack | Recall |
|--------|--------|
| IMU constant bias | 100% |
| IMU noise injection | 100% |
| Intermittent attack | 100% |
| GPS oscillating | 100% |
| Gyro saturation | 100% |

## Files
- `sequence_pinn_detector.pth` - Trained model weights
- `scalers_sequence.pkl` - Input/output scalers
- `sequence_detector_config.json` - Model configuration

## Usage
```python
from pinn_dynamics.security.supervised_detector import SequencePINNDetector

detector = SequencePINNDetector.load("models/security/sequence_pinn")
predictions = detector.detect(sensor_data, sequence_length=20)
```

## When to Use
- Temporal pattern detection needed
- IMU attack detection priority
- Intermittent/periodic attack detection
