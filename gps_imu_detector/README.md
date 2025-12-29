# GPS-IMU Anomaly Detector

A physics-first, multi-scale unsupervised fusion detector for GPS-IMU anomaly detection.
Optimized for real-time inference at 200 Hz on commodity CPUs.

## Novelty Claim

Physics-first multi-scale unsupervised fusion combining:
- PINN residuals (physics violations)
- EKF integrity proxies (NIS)
- Multi-scale statistical features

Yields superior worst-case recall and cross-domain robustness for GPS-IMU anomaly detection.

## Directory Structure

```
gps_imu_detector/
├── src/                    # Core implementation
│   ├── feature_extractor.py    # Streaming multi-scale features
│   ├── physics_residuals.py    # Analytic + PINN residuals
│   ├── ekf.py                  # Simple EKF with NIS
│   ├── model.py                # 1D CNN + GRU detector
│   ├── hybrid_scorer.py        # Weighted fusion scoring
│   ├── attack_generator.py     # Hard negative generator
│   └── data_loader.py          # Sequence-aware data loading
├── data/                   # Dataset storage
├── experiments/            # Experiment configs and logs
├── results/                # Evaluation results
├── models/                 # Trained model artifacts
├── docs/                   # Documentation
│   ├── EVALUATION_PROTOCOL.md
│   └── REPRODUCIBILITY_CHECKLIST.md
└── tests/                  # Unit and integration tests
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run sanity check
python -m pytest tests/ -v

# Train detector
python src/train.py --config config.yaml

# Evaluate
python src/evaluate.py --model models/detector.pth
```

## Key Design Principles

1. **No Circular Sensors**: Never derive sensor from ground truth
2. **Sequence-Wise Splits**: LOSO-CV to prevent temporal leakage
3. **Train-Only Preprocessing**: Scalers fit on training normal data only
4. **Domain-Knowledge Thresholds**: Contamination from priors, not tuned on attacks
5. **CPU-First**: All components optimized for <5ms per timestep

## Target Metrics

- Latency: ≤5ms per timestep on 4-core CPU
- Recall@5%FPR: ≥95% on validated attack suite
- Cross-dataset drop: ≤10% absolute after domain adaptation

## Datasets

- **Primary**: EuRoC MAV (IMU + MoCap as pseudo-GPS)
- **Transfer**: PX4 SITL, Blackbird
- **Attack Catalog**: Bias, drift, ramp, coordinated, intermittent
