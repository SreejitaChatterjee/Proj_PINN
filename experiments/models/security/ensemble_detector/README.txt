# Ensemble Detector

## Overview
Multi-detector voting ensemble combining cross-sensor validation, statistical analysis, similarity matching, and PINN predictions.

## Architecture
- **Type**: PINN-based multi-detector ensemble
- **Components**:
  1. Cross-sensor consistency detector
  2. Statistical anomaly detector
  3. Similarity-based detector
  4. PINN residual detector
  5. Sequence PINN detector
- **Decision**: Voting threshold (default 0.5)

## Performance
| Metric | Value |
|--------|-------|
| **Precision** | 0.03% |
| **Recall** | 80.3% |
| **F1 Score** | 0.06% |
| **FPR** | 94.7% |

## Best For (by category recall)
- **Mag/Baro attacks**: 100% recall
- **Actuator attacks**: 100% recall
- **Temporal attacks**: 100% recall
- **GPS attacks**: 88.1% average recall

## Specific Attack Performance
| Attack | Recall |
|--------|--------|
| 20+ attack types | 100% |
| Stealthy coordinated | 68.1% |
| Adaptive attack | 42.5% |

## Files
- `ensemble_detector.py` - Detector implementation
- `ensemble_evaluation_results.json` - Evaluation results

## Usage
```python
from pinn_dynamics.security.ensemble_detector import EnsembleDetector

detector = EnsembleDetector(voting_threshold=0.5)
results = detector.detect(sensor_data)
```

## When to Use
- Maximum recall is priority (catch everything)
- False positives are acceptable
- Security research / analysis
- NOT for production (94.7% FPR)
