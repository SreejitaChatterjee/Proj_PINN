# Profiling Report: GPS-IMU Anomaly Detector

**Date:** 2025-12-30
**Commit:** See git log
**Author:** Validated automatically

---

## 1. Hardware Specification

| Component | Specification |
|-----------|---------------|
| CPU | AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD |
| OS | Windows-11-10.0.26200-SP0 |
| Python | 3.14.0 |
| PyTorch | 2.9.0+cpu |
| ONNX Runtime | Not tested |

---

## 2. Model Specification

| Property | Value |
|----------|-------|
| Architecture | CNN(32)-GRU(32)-FC(1) |
| Conv Channels | 32 |
| GRU Hidden | 32 |
| Total Parameters | 7,841 |
| Model Size (FP32) | 0.030 MB |
| Model Size (INT8) | Not tested |

---

## 3. Latency Benchmarks

### 3.1 PyTorch (FP32)

| Metric | Value |
|--------|-------|
| Mean | 1.752 ms |
| P50 | 1.689 ms |
| P95 | 2.122 ms |
| P99 | 2.687 ms |
| Samples | 1000 |
| Warmup | 100 |

### 3.2 ONNX Runtime (FP32)

| Metric | Value |
|--------|-------|
| Mean | Not tested |
| P50 | Not tested |
| P99 | Not tested |
| Threads | 1 (single-thread) |

### 3.3 ONNX Runtime (INT8 Quantized)

| Metric | Value |
|--------|-------|
| Mean | Not tested |
| P50 | Not tested |
| P99 | Not tested |
| Speedup vs FP32 | Not tested |

### 3.4 Full Pipeline (Feature Extraction + Inference)

| Component | Latency (ms) | % of Total |
|-----------|--------------|------------|
| Feature Extraction | Not tested | - |
| Physics Residuals | Not tested | - |
| EKF Update | Not tested | - |
| CNN-GRU Inference | 1.752 | 100% |
| Hybrid Scoring | Not tested | - |
| **Total** | ~1.8 ms | - |

---

## 4. Memory Footprint

| Metric | Value |
|--------|-------|
| Peak CPU Memory | Not measured |
| Model Memory (FP32) | 0.030 MB |
| Model Memory (INT8) | Not tested |
| Feature Buffer | Not measured |
| Total Working Set | Not measured |

---

## 5. Detection Performance (VALIDATED - POOR RESULTS)

**IMPORTANT: The simple unsupervised detector does NOT effectively detect attacks.**

### 5.1 Per-Attack Results

| Attack | AUROC | Recall@1%FPR | Recall@5%FPR | Recall@10%FPR |
|--------|-------|--------------|--------------|---------------|
| bias | 0.399 | 0.000 | 0.014 | 0.039 |
| drift | 0.495 | 0.009 | 0.052 | 0.101 |
| noise | 0.480 | 0.004 | 0.038 | 0.084 |
| coordinated | 0.456 | 0.002 | 0.032 | 0.066 |
| intermittent | 0.439 | 0.006 | 0.033 | 0.068 |

### 5.2 Overall

| Metric | Value |
|--------|-------|
| Mean AUROC | 0.454 |
| Worst-case Recall@5%FPR | 0.014 (bias) |

**Interpretation:** AUROC of 0.454 is WORSE than random (0.5). The simple unsupervised approach trained only on normal data does NOT learn to distinguish attacks.

---

## 6. Target Compliance

| Target | Requirement | Measured | Status |
|--------|-------------|----------|--------|
| Latency | ≤5 ms/timestep | 2.687 ms (P99) | **PASS** |
| Model Size | <1 MB | 0.030 MB | **PASS** |
| Recall@5%FPR | ≥95% | 1.4% | **FAIL** |
| Mean AUROC | >0.90 | 0.454 | **FAIL** |

---

## 7. Latency CDF

```
Percentile | Latency (ms)
-----------|-------------
P50        | 1.689
P95        | 2.122
P99        | 2.687
```

---

## 8. Experimental Configuration

| Setting | Value |
|---------|-------|
| Random Seed | 42 |
| Epochs | 10 |
| Sequence Length | 25 |
| Batch Size | 256 |
| Train Sequences | MH_01_easy, MH_02_easy, MH_03_medium |
| Test Sequences | V1_01_easy, V1_02_medium |
| Train Samples | 92,675 |
| Test Samples | 45,413 |

---

## 9. Conclusion

**Latency and model size targets are MET.**

**Detection performance is NOT acceptable:**
- The simple unsupervised CNN-GRU trained only on normal data essentially performs at random chance
- AUROC of 0.454 indicates no discrimination between normal and attack data
- This is expected for unsupervised learning without attack examples during training

**Next steps for improved detection:**
1. Use supervised learning with labeled attack data
2. Implement physics residuals (PINN-based) for unsupervised anomaly detection
3. Use reconstruction-based approaches (autoencoder)
4. Implement hybrid scoring with physics consistency checks

---

*Report generated automatically from validation run*
*Last updated: 2025-12-30*
