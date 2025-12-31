# Reproducibility Guide

**One-command reproduction for GPS-IMU Detector evaluation.**

---

## Quick Start

```bash
# Full reproduction (generates all artifacts)
python gps_imu_detector/run_all.py --seed 42

# Run CI leakage check first
python gps_imu_detector/ci/leakage_check.py --check-code --root .
```

---

## What Gets Produced

| Artifact | Path | Description |
|----------|------|-------------|
| Per-attack CSV | `results/per_attack_results.csv` | AUROC, AUPR, recall per attack |
| Full JSON | `results/full_results.json` | Complete results with CIs |
| Latency profile | `results/latency_profile.json` | P50/P95/P99 latency |
| ONNX model | `results/detector.onnx` | Deployable model |
| Report | `results/EVALUATION_REPORT.md` | Human-readable summary |

---

## Seeds & Splits

| Component | Seed | Documentation |
|-----------|------|---------------|
| Main RNG | 42 | `--seed` argument |
| Attack injection | 42 | Same as main |
| Bootstrap CI | 42 | Fixed in code |
| Splits | N/A | `configs/splits.json` |

---

## Hardware Requirements

- **CPU**: Any modern x86_64
- **RAM**: 4 GB minimum
- **GPU**: Not required (CPU inference)
- **Time**: ~2 minutes total

---

## Dependency Versions

```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
onnx>=1.12.0 (optional)
onnxruntime>=1.10.0 (optional)
```

---

## Verification Checklist

```bash
# 1. Check leakage gates pass
python gps_imu_detector/ci/leakage_check.py --check-code

# 2. Run full pipeline
python gps_imu_detector/run_all.py --seed 42

# 3. Verify key metrics
cat results/full_results.json | jq '.mean_auroc'
# Expected: ~0.845

# 4. Verify latency
cat results/latency_profile.json | jq '.p95_ms'
# Expected: <5.0 ms

# 5. Verify ONNX export
ls -la results/detector.onnx
```

---

## Expected Results

| Metric | Value | Tolerance |
|--------|-------|-----------|
| Mean AUROC | 0.845 | ±0.02 |
| Worst AUROC | 0.666 | ±0.05 |
| P95 latency | <5.0 ms | Hard limit |

---

## Troubleshooting

### ONNX export fails
```bash
# Run without ONNX
python gps_imu_detector/run_all.py --skip-onnx
```

### Leakage check fails
```bash
# Review violations
python gps_imu_detector/ci/leakage_check.py --check-code 2>&1 | head -50
```

### Different AUROC values
- Check seed is 42
- Verify PyTorch version >=1.9.0
- Different platforms may have minor floating-point differences

---

## CI Integration

```yaml
# .github/workflows/reproduce.yml
name: Reproduce Results
on: [push, pull_request]

jobs:
  reproduce:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python gps_imu_detector/ci/leakage_check.py --check-code
      - run: python gps_imu_detector/run_all.py --seed 42
      - run: |
          AUROC=$(cat results/full_results.json | python -c "import json,sys; print(json.load(sys.stdin)['mean_auroc'])")
          if (( $(echo "$AUROC < 0.8" | bc -l) )); then
            echo "AUROC too low: $AUROC"
            exit 1
          fi
```

---

## Contact

For reproduction issues, please file a GitHub issue with:
1. Full command run
2. Python/PyTorch versions
3. Error message or unexpected output
