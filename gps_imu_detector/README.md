# GPS-IMU Anomaly Detector

A physics-first, multi-scale unsupervised fusion detector for GPS-IMU anomaly detection.
Optimized for real-time inference at 200 Hz on commodity CPUs.

## Project Status: Phases 0-5 Complete

| Phase | Description | Status | Lines |
|-------|-------------|--------|-------|
| 0 | Setup & Governance | Complete | ~200 |
| 1-2 | Core Pipeline | Complete | ~3,375 |
| 3 | Hardening & Robustness | Complete | ~800 |
| 4 | Quantization & Optimization | Complete | ~600 |
| 5 | Rigorous Evaluation | Complete | ~999 |
| **Total** | | **Complete** | **~5,974** |

## Novelty Claim

Physics-first multi-scale unsupervised fusion combining:
- PINN residuals (physics violations)
- EKF integrity proxies (NIS)
- Multi-scale statistical features
- Hard negative mining for robustness
- Adversarial training for worst-case performance

Yields superior worst-case recall and cross-domain robustness for GPS-IMU anomaly detection.

## Directory Structure

```
gps_imu_detector/
├── src/                        # Core implementation (~5,000 lines)
│   ├── data_loader.py              # Sequence-aware LOSO-CV splits
│   ├── feature_extractor.py        # Streaming O(1) multi-scale features
│   ├── physics_residuals.py        # Analytic + lightweight PINN residuals
│   ├── ekf.py                      # 15-state EKF with NIS integrity proxy
│   ├── model.py                    # 1D CNN + GRU detector (<100K params)
│   ├── hybrid_scorer.py            # Calibrated weighted fusion
│   ├── train.py                    # LOSO-CV training pipeline
│   ├── hard_negatives.py           # Stealth attack generation
│   ├── attribution.py              # Multi-task attack classification
│   ├── transfer.py                 # Cross-dataset transfer evaluation
│   ├── hardened_training.py        # Curriculum learning + hard negatives
│   ├── quantization.py             # INT8/ONNX/TorchScript export
│   ├── inference.py                # Real-time streaming pipeline
│   └── evaluate.py                 # Rigorous LOSO-CV evaluation
├── data/                       # Dataset storage
├── experiments/                # Experiment configs and logs
├── results/                    # Evaluation results (JSON)
├── models/                     # Trained model artifacts
├── docs/                       # Documentation
│   ├── EVALUATION_PROTOCOL.md      # Strict evaluation rules
│   └── REPRODUCIBILITY_CHECKLIST.md
└── tests/                      # Unit and integration tests (~600 lines)
    ├── test_pipeline.py            # Phase 0-2 tests (15 tests)
    ├── test_hardening.py           # Phase 3 tests (19 tests)
    ├── test_optimization.py        # Phase 4 tests (11 tests)
    └── test_evaluation.py          # Phase 5 tests (12 tests)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests (57 tests)
python -m pytest tests/ -v

# Train detector with LOSO-CV
python src/train.py --config config.yaml --data /path/to/data

# Run rigorous evaluation
python src/evaluate.py --data /path/to/data --output ./results

# Export for deployment
python src/inference.py --export --format onnx
```

## Key Design Principles

1. **No Circular Sensors**: Never derive sensor from ground truth
2. **Sequence-Wise Splits**: LOSO-CV to prevent temporal leakage
3. **Train-Only Preprocessing**: Scalers fit on training normal data only
4. **Domain-Knowledge Thresholds**: Contamination from priors, not tuned on attacks
5. **CPU-First**: All components optimized for <5ms per timestep
6. **Hardening**: Iterative hard negative mining for worst-case robustness

## Architecture

### Core Detection Pipeline
```
Raw Sensors (200 Hz)
    │
    ├─→ Feature Extractor (O(1) streaming)
    │       └─→ Multi-scale stats [5, 10, 25] windows
    │
    ├─→ Physics Checker
    │       ├─→ PVA consistency residuals
    │       ├─→ Jerk bounds (50 m/s³)
    │       └─→ Energy conservation
    │
    ├─→ EKF Integrity
    │       └─→ NIS (Normalized Innovation Squared)
    │
    └─→ CNN-GRU Detector
            └─→ <100K params, <5ms inference
                    │
                    ▼
            Hybrid Scorer (calibrated fusion)
                    │
                    ▼
            Anomaly Score [0, 1]
```

### Hardening Pipeline (Phase 3)
```
Trained Detector
    │
    ▼
Hard Negative Generator
    ├─→ AR(1) slow drift
    ├─→ Coordinated GPS+IMU
    ├─→ Intermittent on/off
    ├─→ Below-threshold ramp
    └─→ Adversarial (PGD)
            │
            ▼
    Retrain on Hard Negatives
            │
            ▼
    Hardened Detector
```

## Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Latency | ≤5ms | Per timestep on 4-core CPU |
| Recall@5%FPR | ≥95% | On validated attack suite |
| Worst-case Recall | ≥80% | Across all attack types |
| Cross-dataset Drop | ≤10% | After domain adaptation |
| Model Size | <1MB | For embedded deployment |

## Attack Catalog

| Attack Type | Description | Detection Difficulty |
|-------------|-------------|---------------------|
| Bias | Constant offset | Easy |
| Drift | AR(1) slow ramp | Medium |
| Noise | Increased variance | Easy |
| Coordinated | Multi-sensor consistent | Hard |
| Intermittent | On/off timing | Hard |
| Ramp | Below-threshold linear | Very Hard |
| Adversarial | PGD perturbation | Very Hard |

## Datasets

- **Primary**: EuRoC MAV (IMU + MoCap as pseudo-GPS)
- **Transfer**: PX4 SITL, Blackbird
- **Real Faults**: ALFA (CMU AirLab, 47 flights)

## Implementation Details

### Phase 1-2: Core Components
- `StreamingFeatureExtractor`: O(1) per-timestep with Numba optimization
- `AnalyticPhysicsChecker`: Jerk, PVA, energy, attitude residuals
- `SimpleEKF`: 15-state with NIS computation
- `CNNGRUDetector`: 32-64 channels, 64 GRU hidden
- `HybridScorer`: Grid-search calibrated weights

### Phase 3: Hardening
- `HardNegativeGenerator`: 7 stealth attack variants
- `AdversarialAttackGenerator`: PGD with epsilon sweep
- `DomainRandomizer`: Noise, jitter, motion regime augmentation
- `TransferEvaluator`: MMD domain shift, CORAL alignment
- `AttributionHead`: Attack type + sensor classification

### Phase 4: Optimization
- `ModelQuantizer`: INT8 dynamic/static quantization
- `ONNXExporter`: Cross-platform deployment
- `LatencyBenchmark`: Warmup + P99 latency
- `StreamingInference`: Maintains GRU state

### Phase 5: Evaluation
- `RigorousEvaluator`: Full LOSO-CV with bootstrapping
- Per-attack recall at 1%, 5%, 10% FPR
- Ablation studies (model size variants)
- Latency benchmarks vs 5ms target

## Test Coverage

```
tests/test_pipeline.py      - 15 tests (Phase 0-2)
tests/test_hardening.py     - 19 tests (Phase 3)
tests/test_optimization.py  - 11 tests (Phase 4, 2 skipped for Python 3.14)
tests/test_evaluation.py    - 12 tests (Phase 5)
─────────────────────────────────────────────────
Total                       - 57 tests
```

## References

1. EuRoC MAV Dataset: Burri et al., IJRR 2016
2. ALFA Dataset: Keipour et al., IJRR 2021
3. Physics-Informed Neural Networks: Raissi et al., JCP 2019
4. Adversarial Robustness: Madry et al., ICLR 2018
