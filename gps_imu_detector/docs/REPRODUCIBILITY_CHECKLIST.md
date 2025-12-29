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

## Evaluation (Phase 5)

- [ ] Evaluation script provided
- [ ] Results match evaluation protocol
- [ ] Per-fold results saved (not just aggregates)
- [ ] Per-attack breakdown provided
- [ ] Confidence intervals computed (bootstrap N=100+)
- [ ] Detection delay metrics included

## Ablation Studies

- [ ] Each ablation has separate config
- [ ] Statistical significance tests run
- [ ] P-values reported
- [ ] Effect sizes calculated

## Cross-Dataset Transfer (Phase 3)

- [ ] Source dataset clearly identified
- [ ] Target dataset(s) clearly identified
- [ ] Zero-shot results reported
- [ ] MMD domain shift computed
- [ ] Adaptation procedure documented (if used)

## Hardening Evaluation (Phase 3)

- [ ] Hard negative attacks tested
- [ ] Adversarial robustness (PGD) evaluated
- [ ] Stealth attack recall reported
- [ ] Domain randomization applied

## Latency Benchmarks (Phase 4)

- [ ] Warmup iterations documented
- [ ] Mean latency reported
- [ ] P99 latency reported
- [ ] FP32 vs INT8 comparison
- [ ] Target hardware documented

## Quantization (Phase 4)

- [ ] Model size before/after quantization
- [ ] Accuracy drop from quantization
- [ ] ONNX export tested
- [ ] TorchScript export tested

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
│       ├── hard_negatives.json    # Phase 3
│       └── seeds.json
├── models/
│   ├── detector.pth
│   ├── detector_int8.pth          # Phase 4
│   ├── detector.onnx              # Phase 4
│   ├── scaler.pkl
│   └── config.yaml
├── results/
│   ├── metrics.json
│   ├── per_fold_results.json
│   ├── per_attack_results.json
│   ├── ablation_results.json      # Phase 5
│   ├── transfer_results.json      # Phase 3
│   └── latency_results.json       # Phase 4
└── scripts/
    ├── train.py
    ├── evaluate.py
    ├── hard_negatives.py          # Phase 3
    ├── quantization.py            # Phase 4
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
- Recall: +/-0.5%
- FPR: +/-0.1%
- Latency: +/-10%
- AUROC: +/-0.01

If results differ by more, investigate:
- Different hardware (CPU vs GPU)
- Different library versions
- Missing seed settings

## Phase-Specific Checklists

### Phase 1-2 (Core Pipeline)
- [ ] All 15 core tests passing
- [ ] Feature extractor produces valid output
- [ ] Physics residuals computed correctly
- [ ] EKF NIS values reasonable
- [ ] Hybrid scorer calibrated

### Phase 3 (Hardening)
- [ ] All 19 hardening tests passing
- [ ] Hard negatives generated successfully
- [ ] Adversarial attacks evaluated
- [ ] Transfer evaluation complete
- [ ] Domain shift quantified

### Phase 4 (Optimization)
- [ ] All 11 optimization tests passing (2 may skip for Python 3.14)
- [ ] INT8 quantization successful
- [ ] ONNX export valid
- [ ] Latency meets <5ms target
- [ ] Model size <1MB

### Phase 5 (Evaluation)
- [ ] All 12 evaluation tests passing
- [ ] LOSO-CV complete
- [ ] Ablation studies done
- [ ] Bootstrap CIs computed
- [ ] Results JSON exported

## Sign-Off

Before claiming results:

Researcher: ___________________ Date: ___________

- [ ] I have verified all checklist items
- [ ] I have run reproduction on a clean environment
- [ ] All 57 tests passing
- [ ] All artifacts are saved and documented
