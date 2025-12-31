# PINN vs Raw Statistics: When to Use What

## Executive Summary

For the PADRE motor fault detection dataset, **raw statistical features achieve 99.7% accuracy** while PINN-based approaches achieve only ~31%. This document explains why and when each approach is appropriate.

## Experimental Results

### Model Comparison on PADRE Dataset (10,050 samples)

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| Random Forest (raw stats) | **99.73%** | **99.86%** | <1s |
| Random Forest (motor ID) | **99.20%** | - | <1s |
| MLP (raw stats) | 99.64% | 99.81% | ~5s |
| Sensor-PINN | 31.05% | 42.06% | ~20s |
| State-PINN (converted) | ~16% F1 | - | ~5min |

### Why PINN Failed on This Task

1. **Signal Already Clear**: Raw sensor statistics (mean, std, range) contain obvious fault signatures
2. **Physics Conversion Destroys Signal**: Converting IMU data to state vectors introduces:
   - Position drift (±2000m from integration)
   - Velocity instability (±27 m/s)
   - Angular rate clamping (76% at limits)
3. **Wrong Tool for Classification**: PINNs predict dynamics, not classify faults
4. **Class Imbalance**: 93% faulty samples means even random guessing achieves high accuracy

### Root Cause Analysis

```
Raw PADRE Data → [mean, std, range per sensor] → RF Classifier → 99.7% accuracy
                           ↓
Raw PADRE Data → [Physics Conversion] → Drift/Noise → PINN → 31% accuracy
```

The physics conversion step amplifies noise more than signal because:
- Complementary filter accumulates attitude drift
- Velocity solver is numerically unstable
- Position integration has no ground truth anchor

## When PINN Adds Value

### PINN is BETTER when:

| Scenario | Why PINN Helps |
|----------|----------------|
| **Low data (<100 samples)** | Physics constraints regularize, prevent overfitting |
| **Extrapolation** | Physics enables generalization to unseen conditions |
| **State estimation** | Only way to get position/velocity from IMU |
| **Multi-step prediction** | Physics ensures trajectory consistency |
| **Interpretability** | Residuals show which physics law is violated |

### Raw Stats are BETTER when:

| Scenario | Why Raw Stats Win |
|----------|-------------------|
| **Abundant data (>1000 samples)** | Statistical patterns dominate |
| **Classification tasks** | Direct mapping input→label |
| **Real-time requirements** | Simple features, fast inference |
| **Clear signal in raw data** | No need to transform |

## Low-Data Regime Analysis

| Training Samples | RF Accuracy | Notes |
|------------------|-------------|-------|
| 10 | 93.17% | Class imbalance helps |
| 20 | 92.84% | Slight overfitting |
| 50 | 93.00% | Stable |
| 100 | 95.69% | Improving |
| 200 | 96.38% | Good |
| 500 | 97.65% | Very good |
| 1000 | 98.21% | Excellent |
| 7035 | 99.73% | Near perfect |

Note: Even with 10 samples, RF achieves 93% due to class imbalance (93% faulty).

## Classification Approaches Compared

| Approach | Best Accuracy | Worst Class | Notes |
|----------|---------------|-------------|-------|
| Single-label (6 classes) | 99.3% | Motor A: 95.6% | Confused by multi-fault cases |
| **Multi-label (4 binary)** | **99.2% avg** | **Motor D: 98.5%** | **Recommended** |

### Why Multi-Label is Better

The dataset contains files where multiple motors fail simultaneously:
- `1002.csv`: Motor A + D faulty
- `1022.csv`: Motor A + C + D faulty
- `1122.csv`: All motors faulty

Single-label classification forces these into a "Multiple" class, which shares features with single-motor faults. Multi-label asks 4 independent questions: "Is Motor X faulty?"

## Recommended Architecture for PADRE

```
┌─────────────────────────────────────────────────────────┐
│                 PADRE Fault Detection                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Raw IMU Data (24 channels)                             │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐                                       │
│  │ Window (256) │                                       │
│  └──────────────┘                                       │
│         │                                                │
│         ▼                                                │
│  ┌──────────────────────────────────────┐               │
│  │ Statistical Features (72 features)   │               │
│  │ - Mean per channel (24)              │               │
│  │ - Std per channel (24)               │               │
│  │ - Range per channel (24)             │               │
│  └──────────────────────────────────────┘               │
│         │                                                │
│         ├────────────────┬──────────────┐               │
│         ▼                ▼              ▼               │
│  ┌────────────┐   ┌────────────┐  ┌────────────┐       │
│  │ RF Binary  │   │ RF Motor   │  │ CNN/LSTM   │       │
│  │ (99.7%)    │   │ ID (99.2%) │  │ (optional) │       │
│  └────────────┘   └────────────┘  └────────────┘       │
│         │                │                              │
│         ▼                ▼                              │
│  ┌────────────┐   ┌────────────┐                       │
│  │ Fault?     │   │ Which      │                       │
│  │ Yes/No     │   │ Motor?     │                       │
│  └────────────┘   └────────────┘                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Top Discriminative Features

From Random Forest feature importance:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | B_aZ_std | 0.1134 |
| 2 | B_aZ_range | 0.0738 |
| 3 | C_gX_std | 0.0725 |
| 4 | B_gX_range | 0.0673 |
| 5 | C_gX_range | 0.0611 |
| 6 | B_gZ_std | 0.0561 |
| 7 | B_gX_std | 0.0549 |
| 8 | D_gZ_range | 0.0373 |
| 9 | D_aZ_std | 0.0233 |
| 10 | B_gZ_range | 0.0225 |

Key insight: **Vibration patterns (std, range) on motors B, C, D** are most discriminative.

## Files Created

- `pinn_dynamics/systems/sensor_pinn.py` - Sensor-PINN for raw IMU (not recommended for this task)
- `scripts/train_padre_classifier.py` - Raw stats classifier (recommended)
- `models/padre_classifier/` - Saved models

## Conclusion

**For PADRE motor fault detection**: Use Random Forest on raw statistical features.

**Save PINN for**:
- State estimation tasks
- Low-data scenarios
- Physics-constrained prediction
- Interpretable anomaly detection

---
*Analysis conducted: December 2024*
*Dataset: PADRE (Parrot Bebop 2, 20 files, 10K+ windows)*
