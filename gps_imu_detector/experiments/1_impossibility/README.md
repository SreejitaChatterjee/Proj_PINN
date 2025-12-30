# 1. Impossibility: Why Residuals Fail

**Hierarchy Position:** Foundation of the contribution

**Key Insight:**
Residual-based detectors achieve AUROC = 0.5 on consistency-preserving GPS spoofing.
This is not a failure of implementation—it is a fundamental impossibility.

## The Residual Equivalence Class (REC) Theorem

For any spoofed observation x' that preserves dynamics consistency:
```
||f(x') - x'_{t+1}|| = ||f(x) - x_{t+1}||
```

The forward model cannot distinguish x from x' based on residuals alone.

## Evidence

| Metric | Residual | Expected (Random) |
|--------|----------|-------------------|
| AUROC | 0.500 | 0.500 |
| Recall@5%FPR | 0.050 | 0.050 |

**Conclusion:** Residual = Random Guessing

## Experiments in This Folder

- `run_impossibility.py` - Demonstrates AUROC=0.5 for residual detector
- `residual_vs_ici.py` - Side-by-side comparison

## What This Proves

1. Traditional residual-based detection is fundamentally blind to stealthy spoofing
2. A new detection primitive is required
3. This motivates ICI (see `2_ici_detector/`)

## Reviewer Q&A

**Q: Maybe the residual detector just needs tuning?**
A: No. The REC theorem proves geometric equivalence. No tuning can break it.

**Q: What about ensemble residuals?**
A: All residual variants suffer from REC. See baselines/ for evidence.
