# PINN Model Anomaly Report

## Date: November 4, 2025

## Critical Anomalies Identified

### 1. **Data Scaling Mismatch** (CRITICAL)

**Issue**: Training uses StandardScaler but evaluation doesn't

**Details**:
- Training (`scripts/train.py:87-88`): Data is normalized using StandardScaler
  - Inputs scaled to mean=0, std=1
  - Outputs scaled to mean=0, std=1
- Evaluation (`scripts/evaluate.py:35-37`): Raw unscaled data fed directly to model
  - Model expects normalized inputs
  - Receives raw values with different scales
  
**Impact**: 
- Prediction errors: 143-1767% relative error on angular states
- Model cannot make accurate predictions on raw data
- Evaluation metrics are meaningless

**Solution Required**: 
- Save scaler during training (pickle/joblib)
- Load and apply scaler in evaluation
- OR remove scaling from training entirely

---

### 2. **MAPE Calculation Issue** (HIGH)

**Issue**: Mean Absolute Percentage Error uses inappropriate denominator

**Details**:
- Formula: `MAPE = mean(|pred - actual| / (actual + 1e-8)) * 100`
- Data contains near-zero values (initial conditions)
- Division by near-zero → astronomical percentages (18M%)

**Impact**:
- MAPE metrics are meaningless (millions of %)
- Cannot compare model performance using MAPE

**Solution**:
- Use MAPE only when |actual| > threshold
- Use normalized RMSE or MAE instead
- Report relative error: MAE / mean(|actual|)

---

### 3. **Evaluation on Initial Conditions** (MEDIUM)

**Issue**: Evaluation includes transient initial conditions

**Details**:
- First 1000 samples have near-zero states
- Model trained on full trajectories expects flight dynamics
- Comparing against near-zero actuals inflates errors

**Solution**:
- Skip initial transient period in evaluation
- Evaluate on steady-state flight regions
- Or evaluate separately for different flight phases

---

## Recommendations

1. **Immediate**: Fix data scaling mismatch in evaluation
2. **Short-term**: Implement proper error metrics (skip MAPE)
3. **Long-term**: Separate evaluation for different flight regimes

## Model Parameter Status

Despite evaluation issues, learned parameters are good:
- kt, kq: 0% error (perfect)
- m: 0.2% error
- Jxx, Jyy, Jzz: 15% error

The model architecture is sound; evaluation methodology needs fixing.
