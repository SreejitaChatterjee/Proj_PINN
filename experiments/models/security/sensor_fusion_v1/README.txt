# Sensor Fusion Detector v1

## Overview
CNN + BiLSTM autoencoder trained on normal data. Uses reconstruction error as anomaly score.

## Architecture
- **Type**: No PINN, learned reconstruction
- **Components**:
  1. Physics consistency layer (20 features, no learning)
  2. Multi-scale temporal CNN (kernels 3, 5, 7)
  3. 2-layer Bidirectional LSTM
  4. Reconstruction decoder
- **Decision**: Physics OR learned predictions

## Performance
| Metric | Value |
|--------|-------|
| **Precision** | 92.8% |
| **Recall** | 29.8% |
| **F1 Score** | 45.1% |
| **FPR** | 0.51% |

## Best For
| Attack | F1 Score |
|--------|----------|
| GPS oscillating | 98.7% |
| GPS multipath | 98.9% |
| IMU constant bias | 97.4% |
| Gyro saturation | 97.7% |
| Thrust manipulation | 98.4% |

## Files
- `detector.py` - Detector implementation
- `run_experiment.py` - Training/evaluation script
- `evaluation_results.json` - Per-attack results

## Usage
```python
from sensor_fusion_detector.detector import AttackDetector, train_detector

model = AttackDetector()
train_detector(model, train_data, val_data)
predictions = model(sensor_data)['predictions']
```

## When to Use
- Minimum false positives critical (0.51% FPR)
- High precision applications (92.8%)
- Willing to miss some attacks for reliability
