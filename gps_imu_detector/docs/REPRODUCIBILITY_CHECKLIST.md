# Reproducibility Checklist

Use this checklist before submitting results or claiming metrics.

## Environment Setup

- [ ] Python version documented (3.10+)
- [ ] requirements.txt with exact pinned versions
- [ ] CUDA/CPU configuration documented
- [ ] Random seeds set and documented
- [ ] Hardware specs recorded (CPU model, RAM, cores)

## Data Preparation

- [ ] Dataset sources documented with download links
- [ ] Data preprocessing scripts provided
- [ ] Sequence split indices saved to file
- [ ] Attack catalog with parameters saved
- [ ] Seed files for attack generation provided

## Model Training

- [ ] config.yaml with all hyperparameters
- [ ] Training logs saved (loss curves, metrics per epoch)
- [ ] Model checkpoints saved (.pth files)
- [ ] Scaler/preprocessor objects saved (.pkl files)
- [ ] Early stopping criteria documented

## Evaluation

- [ ] Evaluation script provided
- [ ] Results match evaluation protocol
- [ ] Per-fold results saved (not just aggregates)
- [ ] Per-attack breakdown provided
- [ ] Confidence intervals computed

## Ablation Studies

- [ ] Each ablation has separate config
- [ ] Statistical significance tests run
- [ ] P-values reported
- [ ] Effect sizes calculated

## Cross-Dataset Transfer

- [ ] Source dataset clearly identified
- [ ] Target dataset(s) clearly identified
- [ ] Zero-shot results reported
- [ ] Adaptation procedure documented (if used)

## Artifact Bundle

Required files for reproduction:
```
artifacts/
├── environment/
│   ├── requirements.txt
│   └── environment.yml (conda)
├── data/
│   ├── splits/
│   │   └── sequence_splits.json
│   └── attacks/
│       ├── attack_catalog.json
│       └── seeds.json
├── models/
│   ├── detector.pth
│   ├── scaler.pkl
│   └── config.yaml
├── results/
│   ├── metrics.json
│   ├── per_fold_results.csv
│   ├── per_attack_results.csv
│   └── ablation_results.csv
└── scripts/
    ├── train.py
    ├── evaluate.py
    └── reproduce_all.sh
```

## Quick Verification Steps

1. Clone repo on fresh machine
2. Install requirements: `pip install -r requirements.txt`
3. Run smoke test: `pytest tests/ -v`
4. Run full evaluation: `python evaluate.py --reproduce`
5. Compare metrics to reported values (within tolerance)

## Tolerance Thresholds

Due to floating-point non-determinism:
- Recall: ±0.5%
- FPR: ±0.1%
- Latency: ±10%

If results differ by more, investigate:
- Different hardware (CPU vs GPU)
- Different library versions
- Missing seed settings

## Sign-Off

Before claiming results:

Researcher: ___________________ Date: ___________

- [ ] I have verified all checklist items
- [ ] I have run reproduction on a clean environment
- [ ] All artifacts are saved and documented
