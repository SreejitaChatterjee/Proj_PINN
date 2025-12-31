# GPS-IMU Detector Evaluation Report

Generated: 2025-12-30T13:05:43.326563
Seed: 42

## Overall Performance
- Mean AUROC: 0.845
- Worst AUROC: 0.666
- Worst Attack: intermittent

## Per-Attack Results
| Attack | AUROC | AUPR | Recall@1%FPR | Recall@5%FPR | Recall@10%FPR |
|--------|-------|------|--------------|--------------|---------------|
| bias | 0.866 | 0.934 | 0.377 | 0.606 | 0.726 |
| drift | 0.919 | 0.966 | 0.691 | 0.807 | 0.847 |
| noise | 0.907 | 0.959 | 0.638 | 0.764 | 0.819 |
| coordinated | 0.869 | 0.933 | 0.322 | 0.570 | 0.709 |
| intermittent | 0.666 | 0.825 | 0.191 | 0.307 | 0.377 |

## Latency Profile
- Mean: 3.13 ms
- P50: 2.25 ms
- P95: 3.92 ms
- P99: 12.27 ms

## Target Compliance
- Latency < 5ms: PASS