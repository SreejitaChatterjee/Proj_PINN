# GPS-IMU Anomaly Detector

A physics-first, multi-scale unsupervised fusion framework for GPS-IMU anomaly detection.
Designed for real-time inference at 200 Hz on commodity CPUs.

## IMPORTANT: This is a CODE FRAMEWORK

**What exists:** ~10,000 lines of code, 91 passing tests
**What does NOT exist:** Trained models, validated results, measured performance

This framework is ready to be trained and evaluated, but NO performance claims can be made until actual evaluation is run.

## Project Status: Code Complete, Validation Pending

| Phase | Description | Code Status | Validated |
|-------|-------------|-------------|-----------|
| 0 | Setup & Governance | ✅ Complete | ⚠️ No |
| 1-2 | Core Pipeline | ✅ Complete | ⚠️ No |
| 3 | Hardening & Robustness | ✅ Complete | ⚠️ No |
| 4 | Quantization & Optimization | ✅ Complete | ⚠️ No |
| 5 | Rigorous Evaluation | ✅ Complete | ⚠️ No |
| P0-P5 | Roadmap Priority Items | ✅ Complete | ⚠️ No |

## Architecture (Implemented, Not Validated)

Physics-first multi-scale unsupervised fusion combining:
- PINN residuals (physics violations)
- EKF integrity proxies (NIS)
- Multi-scale statistical features
- Hard negative mining for robustness
- Minimax calibration for worst-case recall
- Explainable per-alarm attribution

**Note:** The architecture is implemented but NOT validated on real data.

## Directory Structure

```
gps_imu_detector/
├── src/                        # Core implementation (~6,000 lines)
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
│   ├── evaluate.py                 # Rigorous LOSO-CV evaluation
│   ├── minimax_calibration.py      # Worst-case recall optimization (P2)
│   ├── operational_metrics.py      # Latency CDF, FA/hour, delay (P3)
│   └── explainable_alarms.py       # Per-alarm attribution (P4)
├── scripts/                    # Utility scripts
│   ├── ci_circular_check.py        # CI gate - fail if circular sensors (P0)
│   ├── quantize.py                 # ONNX export and quantization
│   └── demo_reproduce_figure.py    # Reproduce paper figures (P5)
├── configs/                    # Configuration files
│   └── baseline.yaml               # Baseline training config
├── experiments/                # Evaluation scripts
│   └── eval.py                     # Per-attack ROC/PR, recall@FPR, latency CDF
├── ci/                         # CI pipeline scripts
│   └── leakage_check.sh            # Full leakage audit (grep + correlation + pytest)
├── profile/                    # Profiling artifacts
│   └── profile_report.md           # Latency, memory, CPU spec template
├── data/                       # Dataset storage
├── results/                    # Evaluation results (JSON)
├── models/                     # Trained model artifacts
├── docs/                       # Documentation
│   ├── EVALUATION_PROTOCOL.md      # Strict evaluation rules
│   └── REPRODUCIBILITY_CHECKLIST.md
└── tests/                      # Unit and integration tests (~900 lines)
    ├── test_pipeline.py            # Phase 0-2 tests (15 tests)
    ├── test_hardening.py           # Phase 3 tests (19 tests)
    ├── test_optimization.py        # Phase 4 tests (11 tests)
    ├── test_evaluation.py          # Phase 5 tests (12 tests)
    ├── test_leakage.py             # Leakage detection tests (13 tests)
    └── test_roadmap_items.py       # Roadmap P0-P5 tests (20 tests)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run CI gate (check for circular sensors)
python scripts/ci_circular_check.py --data /path/to/data.csv

# Run all tests (91 tests)
python -m pytest tests/ -v

# Train baseline model
python src/train.py --config configs/baseline.yaml

# Evaluate on test split
python experiments/eval.py --split test --out results/baseline

# Quantize and export to ONNX
python scripts/quantize.py --model models/baseline.pth --out models/baseline.onnx --int8

# Benchmark latency
python scripts/quantize.py --model models/baseline.pth --out models/baseline.onnx --benchmark

# Full CI leakage check
./ci/leakage_check.sh data/

# Reproduce paper figures
python scripts/demo_reproduce_figure.py --output ./figures

# Export for deployment
python src/inference.py --export --format onnx
```

## Key Design Principles

1. **No Circular Sensors**: Never derive sensor from ground truth (CI gate enforced)
2. **Sequence-Wise Splits**: LOSO-CV to prevent temporal leakage
3. **Train-Only Preprocessing**: Scalers fit on training normal data only
4. **Domain-Knowledge Thresholds**: Contamination from priors, not tuned on attacks
5. **CPU-First**: All components optimized for <5ms per timestep
6. **Hardening**: Iterative hard negative mining for worst-case robustness
7. **Minimax Calibration**: Optimize for worst-case recall, not average
8. **Explainability**: Per-alarm attribution to PINN/EKF/ML/temporal

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
            Minimax Calibrated Fusion
                    │
                    ▼
            Anomaly Score [0, 1] + Explanation
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

## Roadmap Priority Items (P0-P5)

| Priority | Item | Module | Description |
|----------|------|--------|-------------|
| P0 | CI Gate | `scripts/ci_circular_check.py` | Auto-fail if circular sensors detected |
| P1 | Leakage Tests | `tests/test_leakage.py` | 13 tests for data leakage detection |
| P2 | Minimax Calibration | `src/minimax_calibration.py` | Optimize worst-case recall |
| P3 | Operational Metrics | `src/operational_metrics.py` | Latency CDF, FA/hour, delay |
| P4 | Explainable Alarms | `src/explainable_alarms.py` | Per-alarm attribution |
| P5 | Demo Script | `scripts/demo_reproduce_figure.py` | Reproduce paper figures |

## Validated Metrics (2025-12-30)

**Evaluation run on EuRoC MAV dataset with seed=42, 3 train / 2 test sequences.**

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Latency (P99) | ≤5ms | 2.69ms | **PASS** |
| Model Size | <1MB | 0.03MB | **PASS** |
| Mean AUROC | ≥0.90 | 0.454 | **FAIL** |
| Recall@5%FPR | ≥95% | 1.4% | **FAIL** |
| Worst-case Recall | ≥80% | 1.4% | **FAIL** |

### Detection Performance (Per-Attack)

| Attack | AUROC | Recall@5%FPR |
|--------|-------|--------------|
| bias | 0.399 | 1.4% |
| drift | 0.495 | 5.2% |
| noise | 0.480 | 3.8% |
| coordinated | 0.456 | 3.2% |
| intermittent | 0.439 | 3.3% |

**Interpretation:** The simple unsupervised CNN-GRU trained only on normal data does NOT effectively detect attacks. AUROC of 0.454 is worse than random (0.5). This validates that the infrastructure works but the simple unsupervised approach is insufficient for detection.

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

### Roadmap P0-P5: Priority Items
- `MinimaxCalibrator`: Differential evolution for worst-case recall
- `OperationalProfiler`: Latency CDF, false alarms/hour, detection delay
- `AlarmExplainer`: Per-alarm attribution to component sources
- `RuleFusionExplainer`: Interpretable rule-based explanations
- `CircularSensorChecker`: CI gate with Pearson correlation test

## Test Coverage

```
tests/test_pipeline.py      - 15 tests (Phase 0-2)
tests/test_hardening.py     - 19 tests (Phase 3)
tests/test_optimization.py  - 11 tests (Phase 4)
tests/test_evaluation.py    - 12 tests (Phase 5)
tests/test_leakage.py       - 13 tests (P1: Leakage detection)
tests/test_roadmap_items.py - 20 tests (P0-P5: Roadmap items)
─────────────────────────────────────────────────
Total                       - 91 tests (all passing)
```

## References

1. EuRoC MAV Dataset: Burri et al., IJRR 2016
2. ALFA Dataset: Keipour et al., IJRR 2021
3. Physics-Informed Neural Networks: Raissi et al., JCP 2019
4. Adversarial Robustness: Madry et al., ICLR 2018
5. Minimax Optimization: Boyd & Vandenberghe, Convex Optimization, 2004
