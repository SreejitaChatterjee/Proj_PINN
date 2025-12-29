# Evaluation Protocol

This document defines the strict evaluation rules for scientifically valid results.
**All experiments MUST follow this protocol.**

## 1. Data Splitting Rules

### 1.1 Sequence-Wise Splits Only
- **NEVER** use random index-based splits
- Use Leave-One-Sequence-Out Cross-Validation (LOSO-CV)
- Each sequence is a complete flight/trajectory
- Train on N-1 sequences, test on 1 sequence
- Repeat for all sequences, report mean ± std

### 1.2 Temporal Ordering
- Within each sequence, maintain temporal order
- No shuffling of timesteps within a sequence
- This prevents temporal leakage

## 2. Preprocessing Rules

### 2.1 Scaler Fitting
- **FIT scalers ONLY on training normal data**
- Transform validation/test data using training-fitted scalers
- Never refit scalers on test data

### 2.2 Feature Extraction
- Compute features independently per sequence
- No cross-sequence feature computation
- Window features must not cross sequence boundaries

## 3. Threshold and Hyperparameter Rules

### 3.1 Contamination Setting
- Set from domain knowledge or clean validation data
- Acceptable values: 0.01, 0.03, 0.05 (based on expected anomaly rate)
- **NEVER tune contamination on attacked data**

### 3.2 Detection Thresholds
- Set thresholds to achieve target FPR on clean validation
- Report metrics at fixed FPR: 1%, 5%, 10%
- **NEVER tune thresholds using attack data**

### 3.3 Hyperparameter Tuning
- Use nested CV: outer loop for evaluation, inner loop for tuning
- Hyperparameters tuned only on training/validation splits
- Test set never influences hyperparameters

## 4. Sensor Independence Rules

### 4.1 Circular Sensor Prohibition
- **NEVER derive sensor readings from ground truth**
- Prohibited: barometer = f(ground_truth_altitude)
- Prohibited: magnetometer = f(ground_truth_heading)
- If proxies needed for demo, mark as `demo_only` and exclude from metrics

### 4.2 Independence Verification
- Run correlation test between all sensor pairs
- If correlation > 0.9 between derived and source sensor, reject
- Document all sensor sources and derivations

### 4.3 Ground Truth Usage
- Ground truth used ONLY for:
  - Creating attack labels (for evaluation)
  - Computing physics residuals (if explicitly modeling known dynamics)
- Ground truth NEVER used as input feature to detector

## 5. Attack Generation Rules

### 5.1 Reproducibility
- All attacks generated with fixed seeds
- Document attack parameters in catalog
- Save seed files for exact reproduction

### 5.2 Attack Catalog
Required attack types:
1. **Bias**: Constant offset on single sensor
2. **Drift**: Slow ramp over time (AR(1) with high coefficient)
3. **Noise**: Increased variance
4. **Coordinated**: Multiple sensors attacked consistently
5. **Intermittent**: On/off attacks with random timing

### 5.3 Magnitude Ranges
- Test multiple magnitudes: [0.1, 0.25, 0.5, 1.0, 2.0, 4.0] × baseline std
- Report per-magnitude breakdown

## 6. Metric Reporting Rules

### 6.1 Required Metrics
- **Recall@1%FPR**: Detection rate at 1% false alarm
- **Recall@5%FPR**: Detection rate at 5% false alarm
- **AUROC**: Area under ROC curve
- **AUPRC**: Area under Precision-Recall curve
- **Worst-case recall**: Minimum recall across all attack types

### 6.2 Confidence Intervals
- Report mean ± std across CV folds
- Use bootstrap (N=1000) for single-dataset confidence intervals
- Report 95% confidence intervals

### 6.3 Per-Attack Breakdown
- Report metrics for each attack type separately
- Report metrics for each magnitude separately
- Identify worst-performing attack type

## 7. Cross-Dataset Transfer Rules

### 7.1 Transfer Protocol
- Train ONLY on source dataset (e.g., EuRoC)
- Test on completely different target dataset (e.g., PX4, Blackbird)
- No fine-tuning on target data for "zero-shot" evaluation
- Report transfer drop: (source_metric - target_metric)

### 7.2 Domain Adaptation
- If using domain adaptation, clearly separate from zero-shot
- Report both zero-shot and adapted metrics
- Document adaptation procedure

## 8. Ablation Rules

### 8.1 Required Ablations
1. Remove PINN residuals → measure impact
2. Remove EKF NIS → measure impact
3. Remove ML detector → measure impact
4. Remove multi-scale features → measure impact
5. Single window vs multi-scale → measure impact

### 8.2 Statistical Significance
- Use paired t-test or Wilcoxon signed-rank test
- Report p-values for ablation comparisons
- Claim significance only if p < 0.05

## 9. Reproducibility Requirements

### 9.1 Required Artifacts
- Exact random seeds
- Data split indices
- Attack generator scripts with parameters
- Trained model checkpoints
- Evaluation scripts

### 9.2 Documentation
- README with setup instructions
- requirements.txt with pinned versions
- config.yaml with all hyperparameters

## 10. Prohibited Practices

❌ Random index splits within sequences
❌ Fitting scalers on test data
❌ Tuning thresholds on attack data
❌ Deriving sensors from ground truth
❌ Reporting only best-case metrics
❌ Hiding negative results or failed attacks
❌ Cherry-picking sequences or attack types
❌ Using future information in features

## Checklist Before Reporting Results

- [ ] Used sequence-wise splits (LOSO-CV)
- [ ] Scalers fit on training normal data only
- [ ] Thresholds set from clean validation or domain prior
- [ ] No circular sensors in evaluation
- [ ] Reported recall at fixed FPR (1%, 5%)
- [ ] Reported worst-case per-attack recall
- [ ] Reported cross-dataset transfer metrics
- [ ] Provided confidence intervals
- [ ] Ran required ablations
- [ ] Artifacts saved for reproduction
