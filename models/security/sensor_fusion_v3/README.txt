# Sensor Fusion Detector v3

## Overview
7 specialized detectors with NO learning. Each detector targets specific attack signatures.

## Architecture
- **Type**: No PINN, no learning, pure physics ensemble
- **Components** (7 detectors):
  1. **Physics**: pos-vel + kinematic consistency
  2. **Gyro Saturation**: rate magnitude > 3 rad/s
  3. **Control Hijack**: thrust deviation > 3N
  4. **Time Delay**: adaptive attitude-rate threshold
  5. **Resonance**: FFT peak detection
  6. **Drift (CUSUM)**: cumulative sum on residuals
  7. **Intermittent**: 5-sigma state jumps
- **Decision**: Weighted combination, any trigger = attack

## Performance
| Metric | Value |
|--------|-------|
| **Precision** | 22.2% |
| **Recall** | 85.3% |
| **F1 Score** | 33.8% |
| **FPR** | ~20% |

## Best For (highest recall)
| Attack | Recall |
|--------|--------|
| GPS (all types) | 100% |
| IMU (all types) | 99-100% |
| Control hijack | 98.8% |
| Replay attack | 100% |
| Intermittent | 100% |
| Resonance | 66.3% |
| Time delay | 40.9% |
| Stealthy coordinated | 35.9% |

## Files
- `detector_v3.py` - Detector implementation
- `run_v3.py` - Evaluation script
- `results.json` - Per-attack results

## Usage
```python
from sensor_fusion_detector.detector_v3 import ComprehensiveDetector

detector = ComprehensiveDetector()
detector.calibrate(clean_data)  # Set thresholds
results = detector.detect(sensor_data)
```

## When to Use
- Maximum recall priority (85.3%)
- Security research / forensics
- Acceptable false positive rate (~20%)
- Detecting stealthy/temporal attacks
