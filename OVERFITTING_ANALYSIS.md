# OVERFITTING ANALYSIS - CRITICAL ISSUE FOUND

## Summary
The model shows **severe overfitting**. The trajectory plots in your paper show **training data performance**, NOT test set performance as claimed.

## Evidence

### 1. What Your Paper Claims (Caption on page ~18-33)
> "X Position vs Time - All 10 Trajectories (Overall MAE: **0.023 m**, RMSE: **0.064 m**)"
> Caption states: "PINN v2 Performance on **Held-Out Test Set**"

### 2. Actual Performance on TRUE Test Set

**TEST SET (20% held-out, never seen during training):**
```
Variable                    Training MAE  →  Test MAE    Degradation
─────────────────────────────────────────────────────────────────────
X Position                   0.023 m      →  1.499 m      65× WORSE
Y Position                   ~0.03 m      →  2.339 m      78× WORSE
Z Position                   ~0.05 m      →  4.265 m      85× WORSE
Roll Angle                   ~0.001 rad   →  0.047 rad    47× WORSE
Pitch Angle                  ~0.001 rad   →  0.033 rad    33× WORSE
Yaw Angle                    ~0.002 rad   →  0.059 rad    30× WORSE
X Velocity                   ~0.01 m/s    →  0.411 m/s    41× WORSE
Y Velocity                   ~0.02 m/s    →  0.612 m/s    31× WORSE
Z Velocity                   ~0.05 m/s    →  2.238 m/s    45× WORSE
```

## Root Cause

### Problem 1: Random Splitting of Time Series Data
The training script uses `sklearn.train_test_split(test_size=0.2, random_state=42)` which:
- Randomly shuffles samples from ALL trajectories
- Breaks temporal continuity
- Allows the model to "see" future states during training
- Results in data leakage

### Problem 2: Plotting Training Data
The plotting script `generate_comparative_trajectory_plots.py`:
- Loads from `quadrotor_training_data.csv` (ALL data, including training set)
- Shows perfect overlaps because model memorized the training data
- Paper incorrectly labels these as "held-out test set" results

## Why You Only See One Line

You asked: "why is there just one line?"

**Answer:** Because on the **training data**, predictions perfectly overlap ground truth (memorization/overfitting). The two lines are:
- Blue solid: Ground truth
- Red dashed: Predictions (invisible because perfectly overlapping)

On the **actual test set**, you would see clear separation between the lines.

## Impact on Paper Claims

Your paper claims:
1. ❌ "Test set: Last 20% of data (9,873 timesteps) - completely unseen during training"
2. ❌ "The model demonstrates exceptional stability with only 1.1× error growth"
3. ❌ "51× improvement over baseline at 100-step horizon"
4. ❌ "<10% accuracy degradation on unseen trajectories"

**Reality:**
1. ✅ Test set exists but plots show TRAINING data
2. ❌ Actual test degradation is 30-85× worse, not 1.1×
3. ❓ Baseline comparison needs verification
4. ❌ Actual degradation is 3000-8500%, not <10%

## Recommended Actions

### Immediate (Fix Paper)
1. **Replace all trajectory plots** with test set versions (already generated in `results/test_set_trajectories/`)
2. **Update all performance metrics** to report test set numbers
3. **Add honest discussion** of the overfitting problem
4. **Remove or correct claims** about generalization

### Short-term (Fix Model)
1. **Use proper time series splitting:**
   - Reserve entire trajectories for testing (not random samples)
   - Use chronological split or k-fold cross-validation respecting trajectory boundaries

2. **Add regularization:**
   - Increase dropout (currently 0.1)
   - Add L2 regularization
   - Reduce model capacity if needed

3. **Early stopping:**
   - Monitor validation loss during training
   - Stop before overfitting occurs

### Long-term (Improve Methodology)
1. **Generate more diverse training data:**
   - More trajectory types
   - Different initial conditions
   - Noise injection

2. **Domain randomization:**
   - Vary physical parameters during training
   - Test on out-of-distribution scenarios

3. **Explicit evaluation of extrapolation:**
   - Test on longer time horizons than trained
   - Test on more aggressive maneuvers

## Files Created

1. **Data Splits:**
   - `data/quadrotor_train_only.csv` (39,484 samples)
   - `data/quadrotor_test_only.csv` (9,871 samples)

2. **Test Set Plots:**
   - `results/test_set_trajectories/01_x_test_trajectories.png` (and 15 more)

3. **Scripts:**
   - `scripts/split_train_test_data.py`
   - `scripts/generate_test_trajectory_plots.py`

## Conclusion

This is a **severe case of overfitting** that invalidates most claims in the paper. The model has essentially **memorized the training data** but **does not generalize** to held-out test data.

The good news: You discovered this before publication. The fix requires:
1. Retraining with proper cross-validation
2. Honest reporting of actual test performance
3. Addressing the overfitting through regularization

**Current Status:** ⚠️ Paper claims are NOT supported by actual test set performance
