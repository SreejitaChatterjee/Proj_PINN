# Spoofing Attack Detection Models

**STATUS: EXPERIMENTAL - Results on Synthetic Data Only**

**WARNING:** All metrics below are from synthetic attack simulations, NOT validated
on real-world attacks. Use with caution. See research/security/UAV_FAULT_DETECTION.md
for validated results on real CMU ALFA flight data (AUROC 0.575).

This directory contains experimental models for UAV sensor spoofing attack detection.

## Model Overview

| Model | Type | Precision | Recall | F1 | FPR | Best For |
|-------|------|-----------|--------|-----|-----|----------|
| [supervised_detector](./supervised_detector/) | ML | 100% | **76.4%** | **86.6%** | 2.0% | **BEST OVERALL** |
| [synthetic_pinn](./synthetic_pinn/) | PINN | 93.8% | 18.7% | 31.2% | 0.28% | Control/actuator attacks |
| [sequence_pinn](./sequence_pinn/) | PINN | 0.3% | 44.5% | 0.6% | 5.0% | Temporal patterns |
| [pinn_residual](./pinn_residual/) | PINN | 0.2% | 28.0% | 0.5% | 4.3% | Physics violations |
| [ensemble_detector](./ensemble_detector/) | PINN | 0.03% | 80.3% | 0.06% | 94.7% | Catch everything |
| [sensor_fusion_v1](./sensor_fusion_v1/) | No PINN | 92.8% | 29.8% | 45.1% | 0.51% | Minimum false alarms |
| [sensor_fusion_v2](./sensor_fusion_v2/) | No PINN | 79.2% | 67.4% | 68.5% | ~5% | Balanced detection |
| [sensor_fusion_v3](./sensor_fusion_v3/) | No PINN | 22.2% | 85.3% | 33.8% | ~20% | Maximum recall |

## Quick Comparison

### By Approach

| Approach | Models | Key Insight |
|----------|--------|-------------|
| **Supervised ML** | supervised_detector | Random Forest on statistical features - best F1 (86.6%) |
| **PINN-based** | synthetic_pinn, sequence_pinn, pinn_residual, ensemble | Uses physics-informed neural network residuals |
| **Physics-only** | sensor_fusion_v2, sensor_fusion_v3 | Pure physics consistency checks, no/minimal learning |
| **Learned** | sensor_fusion_v1 | CNN+LSTM autoencoder on physics features |

### By Use Case

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Production deployment** | supervised_detector | Best F1 (86.6%), 100% precision, 2% FPR |
| **Low false alarms** | sensor_fusion_v1 | 92.8% precision, 0.51% FPR |
| **Catch everything** | sensor_fusion_v3 | 85.3% recall |
| **Control hijack detection** | synthetic_pinn | 99.97% recall on control attacks |
| **Security research** | ensemble_detector | 80%+ recall, analyze all attacks |

## Per-Attack Performance (Recall %)

| Attack | Synth | Seq | Resid | Ens | v1 | v2 | v3 |
|--------|-------|-----|-------|-----|-----|-----|-----|
| gps_gradual_drift | 35.9 | 61.5 | 11.5 | 100 | 31.9 | **100** | **100** |
| gps_sudden_jump | 0.5 | 23.5 | 0 | 100 | 0.6 | 84.5 | **100** |
| gps_oscillating | 0.3 | **100** | 95.3 | 62.8 | **100** | **100** | **100** |
| imu_constant_bias | 67.4 | **100** | 52.9 | 38.2 | 98.1 | 99.7 | **100** |
| imu_gradual_drift | 65.1 | 91.3 | 55.1 | 39.1 | 88.5 | 96.4 | **99.0** |
| gyro_saturation | **100** | **100** | 92.3 | **100** | **100** | 0.6 | **100** |
| accel_saturation | 99.5 | **100** | **100** | **100** | **99.9** | **100** | **100** |
| barometer_spoofing | 71.7 | 92.3 | 42.3 | **100** | 0.9 | **100** | **100** |
| control_hijack | **100** | 80.8 | 73.1 | **100** | 0.8 | 0.7 | 98.8 |
| thrust_manipulation | 99.4 | 97.1 | 94.1 | **100** | **99.7** | **100** | **100** |
| stealthy_coordinated | 0.4 | 0 | 1.4 | **68.1** | 0.5 | 0.9 | 35.9 |
| replay_attack | 0 | 0 | 7.0 | **100** | 1.5 | 20.0 | **100** |
| adaptive_attack | 0.4 | 18.4 | 0 | 42.5 | 0.5 | **97.5** | **100** |

## Directory Structure

```
models/security/
├── README.md                    # This file
├── supervised_detector/         # Random Forest classifier (BEST)
│   ├── classifier.pkl
│   └── evaluation_results.json
├── synthetic_pinn/              # PINN-based supervised detector
│   ├── README.md
│   ├── pinn_synthetic_detector.pth
│   ├── scalers_synthetic.pkl
│   └── synthetic_detector_config.json
├── sequence_pinn/               # Temporal PINN detector
│   ├── README.md
│   ├── sequence_pinn_detector.pth
│   ├── scalers_sequence.pkl
│   └── sequence_detector_config.json
├── pinn_residual/               # PINN residual ensemble
│   ├── README.md
│   ├── pinn_residual_detector.py
│   └── pinn_residual_evaluation.json
├── ensemble_detector/           # Multi-detector voting
│   ├── README.md
│   ├── ensemble_detector.py
│   └── ensemble_evaluation_results.json
├── sensor_fusion_v1/            # CNN+LSTM autoencoder
│   ├── README.md
│   ├── detector.py
│   ├── run_experiment.py
│   └── evaluation_results.json
├── sensor_fusion_v2/            # Physics-first hybrid (RECOMMENDED)
│   ├── README.md
│   ├── detector_v2.py
│   ├── run_v2.py
│   └── results.json
└── sensor_fusion_v3/            # 7-detector ensemble
    ├── README.md
    ├── detector_v3.py
    ├── run_v3.py
    └── results.json
```

## Key Findings

1. **Supervised ML wins**: Random Forest classifier achieves best F1 (86.6%) with 100% precision
2. **PINN vs No-PINN**: No-PINN models (v2, v3) generally outperform PINN-based models
3. **Exception**: Synthetic PINN excels at control/actuator attacks (99%+ recall)
4. **Category performance**: IMU attacks (95.1%) > GPS (86.3%) > Coordinated (76.7%) > Stealth (58.0%)
5. **Hardest attacks**: time_delay (2.7%), control_hijack (5.4%), gps_meaconing (10.6%)

## Recommended Deployment

**Primary**: Use `supervised_detector` for best overall performance:
- 76.4% recall across all 30 attack types
- 100% precision (zero false positives on attacks)
- 2.0% FPR on clean data
- Category performance: GPS 86.3%, IMU 95.1%, Actuator 62.8%, Stealth 58.0%

```python
# Load and use supervised detector
from pinn_dynamics.security.supervised_detector import SupervisedAttackClassifier
detector = SupervisedAttackClassifier.load("models/security/supervised_detector/classifier.pkl")
predictions, probs = detector.predict_batch(sensor_data)
```
