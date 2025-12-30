# Evaluation Protocol (Reviewer Contract)

This document specifies the **non-negotiable evaluation rules** for the GPS-IMU Anomaly Detector.
Any results reported MUST follow these rules. Violation invalidates all metrics.

---

## 1. Data Splits

### Rule: Sequence-wise splits by flight ID ONLY

- **NO** random shuffling of samples across flights
- **NO** sample from flight X appearing in both train and test
- All samples from a given flight belong to exactly ONE split

### Split Definition

```
Train:  Flights for model training and scaler fitting
Val:    Flights for threshold calibration and early stopping
Test:   Flights for final metric reporting (NEVER seen during training)
```

### Verification

```bash
python ci/leakage_check.py --splits configs/splits.json
```

---

## 2. Scaler Discipline

### Rule: Scalers fit on training NORMAL data ONLY

- `scaler.fit()` called ONLY on train split
- `scaler.transform()` used for val/test
- **NEVER** `scaler.fit_transform()` on combined data

### Rationale

If scaler sees attack data or test data, statistics leak future information.

---

## 3. Threshold Calibration

### Rule: Thresholds calibrated on CLEAN VALIDATION data

- Threshold set to achieve target FPR (e.g., 1% or 5%) on **clean val data**
- **NEVER** calibrate on test data
- **NEVER** calibrate on attack data

### Protocol

1. Run detector on val (normal only)
2. Compute score distribution
3. Set threshold at (100 - target_FPR) percentile
4. Apply fixed threshold to test

---

## 4. Banned Sensors (Circular Features)

The following sensors/features are **BANNED** because they create circular dependencies:

| Sensor | Reason |
|--------|--------|
| `baro_alt` / `barometric_altitude` | Fused with GPS in flight controller |
| `mag_heading` / `magnetic_heading` | Fused with IMU in flight controller |
| `derived_*` | Explicitly derived from target |
| `fused_*` | Multi-sensor fusion output |
| `filtered_*` | Kalman-filtered state |
| `ekf_*` | EKF state estimates |
| `*_ground_truth*` | Ground truth labels |

### Allowed Features

| Feature | Source | Safe? |
|---------|--------|-------|
| `gps_x, gps_y, gps_z` | Raw GPS receiver | YES |
| `gps_vx, gps_vy, gps_vz` | Raw GPS velocity | YES |
| `imu_ax, imu_ay, imu_az` | Raw accelerometer | YES |
| `imu_wx, imu_wy, imu_wz` | Raw gyroscope | YES |
| `roll, pitch, yaw` | IMU-only AHRS | YES (if documented) |

### Verification

```bash
python ci/leakage_check.py --data data/features.csv --correlation-threshold 0.9
```

---

## 5. Decision Rule

### Rule: N consecutive anomalous samples required for alarm

- Single-sample anomalies are often noise
- Require N consecutive samples above threshold
- N is fixed BEFORE test evaluation

### Recommended Values

| Application | N | Rationale |
|-------------|---|-----------|
| High sensitivity | 3 | Fast detection, more false alarms |
| Balanced | 5 | Default recommendation |
| High specificity | 10 | Fewer false alarms, slower detection |

---

## 6. Attack Injection

### Rule: Attacks injected at TEST TIME ONLY

- Training data is 100% normal/clean
- Attacks applied to test sequences only
- Attack parameters documented in `attacks/catalog.json`

### Attack Catalog Structure

```json
{
  "attack_type": {
    "parameters": {...},
    "injection_time": "random | fixed",
    "duration": "seconds",
    "seed": 42
  }
}
```

---

## 7. Metrics Reported

### Primary Metrics (MUST report)

| Metric | Definition |
|--------|------------|
| AUROC | Area under ROC curve |
| Recall@1%FPR | True positive rate at 1% false positive rate |
| Recall@5%FPR | True positive rate at 5% false positive rate |
| F1 | Harmonic mean of precision and recall |
| Detection Latency | Time from attack start to first alarm (ms) |

### Per-Attack Breakdown (MUST report)

- Worst-case attack performance
- Best-case attack performance
- Attack-specific recall values

### Statistical Confidence (MUST report)

- 95% confidence intervals via bootstrap (>=500 resamples)
- Cross-validation variance (if applicable)

---

## 8. Reproducibility Requirements

### Required Artifacts

- [ ] `configs/splits.json` - Train/val/test flight IDs
- [ ] `attacks/catalog.json` - Attack definitions with seeds
- [ ] `results/metrics.json` - All reported metrics
- [ ] Random seeds documented in all scripts

### One-Command Reproduction

```bash
./run_hybrid_eval.sh
```

Must reproduce headline metrics within +/- 2%.

---

## 9. Negative Results to Report

### Failure Modes (MUST disclose)

- Attacks where detector fails (Recall < 50%)
- Minimum detectable offset/magnitude
- False alarm rate under distribution shift

---

## Checklist Before Submission

- [ ] `ci/leakage_check.py --check-code` passes
- [ ] `ci/leakage_check.py --splits configs/splits.json` passes
- [ ] No banned sensors in feature list
- [ ] Threshold calibrated on val (not test)
- [ ] N-consecutive rule documented
- [ ] Bootstrap CIs reported
- [ ] Worst-case attack disclosed
- [ ] One-command reproduction tested

---

## Violation = Invalid Results

If ANY rule above is violated, reported metrics are **INVALID**.

This is not optional. This is the minimum bar for credible evaluation.
