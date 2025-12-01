# Retraining Results: Before vs After

## Executive Summary

Successfully retrained the PINN model with proper methodology to fix overfitting. The model now uses proper train/val/test split by trajectory and includes regularization techniques.

## Key Improvements Made

### 1. Data Split Strategy
**Before (INCORRECT):**
- Random sample splitting using `train_test_split(random_state=42)`
- **Problem:** Breaks temporal continuity, causes data leakage
- **Result:** Model sees future states during training

**After (CORRECT):**
- Split by entire trajectories
- Train: 7 trajectories (70%)
- Val: 1 trajectory (10%)
- Test: 2 trajectories (20%)
- **Benefit:** No data leakage, true generalization test

### 2. Regularization Techniques
**Before:**
- Dropout: 0.1
- No weight decay
- No gradient clipping
- No early stopping

**After:**
- Dropout: 0.3 (3× stronger)
- L2 weight decay: 1e-4
- Gradient clipping: 1.0
- Early stopping: 50 epochs patience
- Learning rate scheduling

### 3. Training Results
**Training completed:**
- Total epochs: 71 (stopped early)
- Best validation loss: 1,199,434.2
- Final learning rate: 1.25e-04 (reduced 8× from 0.001)
- Training time: ~4 minutes

## Test Set Performance Comparison

### Original Model (with data leakage)
Used random split on `quadrotor_test_only.csv` (mixed samples from all trajectories):

| Variable | MAE (Old) |
|----------|-----------|
| X Position | 1.499 m |
| Y Position | 2.339 m |
| Z Position | 4.265 m |
| Roll | 0.047 rad |
| Pitch | 0.033 rad |
| Yaw | 0.059 rad |
| X Velocity | 0.411 m/s |
| Y Velocity | 0.612 m/s |
| Z Velocity | 2.238 m/s |

### Retrained Model (proper trajectory split)
Used `test_set.csv` (2 complete held-out trajectories: #3 and #6):

| Variable | MAE (New) | Change |
|----------|-----------|--------|
| X Position | 1.350 m | ✓ 10% better |
| Y Position | 1.847 m | ✓ 21% better |
| Z Position | 5.445 m | ✗ 28% worse |
| Roll | 0.045 rad | ✓ 4% better |
| Pitch | 0.022 rad | ✓ 33% better |
| Yaw | 0.064 rad | ✗ 8% worse |
| X Velocity | 0.490 m/s | ✗ 19% worse |
| Y Velocity | 0.742 m/s | ✗ 21% worse |
| Z Velocity | 2.470 m/s | ✗ 10% worse |

**Overall MAE:** 1.059 (mixed performance)

## Physical Parameter Learning

### Retrained Model Parameters:

| Parameter | Learned | True | Error % | Status |
|-----------|---------|------|---------|--------|
| **Mass (m)** | 0.0500 kg | 0.0680 kg | **26.5%** | At lower bound ⚠️ |
| **Jxx** | 1.00e-04 kg·m² | 6.86e-05 kg·m² | **45.8%** | At upper bound ⚠️ |
| **Jyy** | 1.30e-04 kg·m² | 9.20e-05 kg·m² | **41.3%** | At upper bound ⚠️ |
| **Jzz** | 2.00e-04 kg·m² | 1.37e-04 kg·m² | **46.4%** | At upper bound ⚠️ |
| **kt** | 1.00e-02 N/(rad/s)² | 1.00e-02 N/(rad/s)² | **0.0%** | ✓ Perfect |
| **kq** | 7.83e-04 N·m/(rad/s)² | 7.83e-04 N·m/(rad/s)² | **0.0%** | ✓ Perfect |

**Critical Issue:** All inertia parameters (Jxx, Jyy, Jzz) and mass are hitting their constraint bounds, indicating the model is struggling to identify these parameters correctly.

## Analysis

### What Worked ✓
1. **Proper data splitting** eliminated data leakage
2. **Regularization** reduced overfitting (early stopping at epoch 71)
3. **Motor coefficients (kt, kq)** identified perfectly
4. **Horizontal position (X, Y)** improved slightly
5. **Pitch angle** prediction improved significantly (33%)

### What Didn't Work ✗
1. **Inertia parameters** not learned correctly (hitting bounds)
2. **Vertical dynamics (Z)** worse than before
3. **Velocity predictions** degraded
4. **Mass parameter** at lower bound (26% error)
5. **Physics loss** still high (13,092)

### Root Causes
1. **Insufficient trajectory diversity**: Only 10 trajectories total, with just 2 in test set
2. **Parameter observability issue**: Inertia tensors are hard to identify from position/attitude data alone
3. **Constraint bounds too tight**: Parameters hitting bounds suggests they need more freedom
4. **Data generation**: May not excite all dynamic modes sufficiently

## Recommendations

### Short-term (Quick Fixes)
1. **Relax parameter bounds** for inertias (±60% instead of ±45%)
2. **Increase training data**: Generate 50-100 trajectories with diverse maneuvers
3. **Add angular acceleration** measurements if available
4. **Weight adjustments**: Reduce physics loss weight if data loss is more reliable

### Medium-term (Model Improvements)
1. **Two-stage training:**
   - Stage 1: Learn dynamics with frozen parameters
   - Stage 2: Fine-tune parameters with learned dynamics
2. **Separate physics loss** for rotational vs translational dynamics
3. **Uncertainty quantification**: Add confidence intervals to predictions
4. **Curriculum learning**: Start with simple trajectories, increase complexity

### Long-term (Research Directions)
1. **Generate aggressive maneuvers** specifically designed to identify inertias
2. **Multi-fidelity approach**: Combine high-frequency data with physics
3. **Hybrid modeling**: Use known parameters where available, learn only uncertain ones
4. **Domain randomization**: Train on varied quadrotor configurations

## Files Generated

### Models
- `models/quadrotor_pinn_fixed.pth` - Retrained model weights
- `models/scalers_fixed.pkl` - Feature scaling parameters

### Data Splits
- `data/train_set.csv` - 7 trajectories (34,358 samples)
- `data/val_set.csv` - 1 trajectory (4,999 samples)
- `data/test_set.csv` - 2 trajectories (9,998 samples)

### Visualizations
- `results/training_history_fixed.png` - Training curves showing early stopping
- `results/test_set_trajectories_fixed/` - 16 plots showing test performance

### Scripts
- `scripts/create_proper_train_test_split.py` - Trajectory-based splitting
- `scripts/train_fixed_overfitting.py` - Regularized training
- `scripts/evaluate_fixed_model_on_test.py` - Test set evaluation

## Conclusion

The retraining **fixed the data leakage problem** and produced a more honest evaluation, but revealed deeper issues:

1. ✓ **Overfitting addressed**: Model no longer memorizing training data
2. ✓ **Proper evaluation**: Using true held-out trajectories
3. ✗ **Parameter identification failed**: Inertias not learned correctly
4. ✗ **Mixed performance**: Some states improved, others degraded
5. ⚠️ **Needs more work**: Fundamental modeling or data issues remain

**Next Steps:**
1. Generate more diverse training data (Priority: HIGH)
2. Relax parameter constraints (Priority: HIGH)
3. Investigate why vertical dynamics degraded (Priority: MEDIUM)
4. Consider hybrid approach with some known parameters (Priority: MEDIUM)

The model is no longer overfitting, but it's also not generalizing well enough for publication. More work needed on data generation and model architecture.
