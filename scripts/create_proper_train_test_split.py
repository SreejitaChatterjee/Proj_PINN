"""Create proper time-series train/validation/test split by reserving entire trajectories"""

from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "quadrotor_training_data.csv"
TRAIN_OUTPUT = PROJECT_ROOT / "data" / "train_set.csv"
VAL_OUTPUT = PROJECT_ROOT / "data" / "val_set.csv"
TEST_OUTPUT = PROJECT_ROOT / "data" / "test_set.csv"

# Split configuration (by trajectory, not by samples)
# Train: 70%, Val: 15%, Test: 15%
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def main():
    print("=" * 80)
    print("CREATING PROPER TIME-SERIES TRAIN/VAL/TEST SPLIT")
    print("(Splitting by ENTIRE TRAJECTORIES to prevent data leakage)")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")

    # Get trajectory information
    trajectory_ids = sorted(df["trajectory_id"].unique())
    n_trajectories = len(trajectory_ids)
    print(f"Total trajectories: {n_trajectories}")

    # Calculate number of trajectories for each split
    n_train = int(n_trajectories * TRAIN_RATIO)
    n_val = int(n_trajectories * VAL_RATIO)
    n_test = n_trajectories - n_train - n_val

    print(f"\nSplit configuration:")
    print(f"  Train: {n_train} trajectories ({n_train/n_trajectories*100:.1f}%)")
    print(f"  Val:   {n_val} trajectories ({n_val/n_trajectories*100:.1f}%)")
    print(f"  Test:  {n_test} trajectories ({n_test/n_trajectories*100:.1f}%)")

    # Shuffle trajectories with fixed seed for reproducibility
    np.random.seed(42)
    shuffled_ids = np.random.permutation(trajectory_ids)

    # Split trajectory IDs
    train_traj_ids = shuffled_ids[:n_train]
    val_traj_ids = shuffled_ids[n_train : n_train + n_val]
    test_traj_ids = shuffled_ids[n_train + n_val :]

    print(f"\nTrajectory assignments:")
    print(f"  Train IDs: {sorted(train_traj_ids.tolist())}")
    print(f"  Val IDs:   {sorted(val_traj_ids.tolist())}")
    print(f"  Test IDs:  {sorted(test_traj_ids.tolist())}")

    # Create splits
    df_train = df[df["trajectory_id"].isin(train_traj_ids)].copy()
    df_val = df[df["trajectory_id"].isin(val_traj_ids)].copy()
    df_test = df[df["trajectory_id"].isin(test_traj_ids)].copy()

    # Display statistics
    print(f"\n{'='*80}")
    print("SPLIT STATISTICS")
    print(f"{'='*80}")
    print(f"Train set:")
    print(f"  Samples: {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  Trajectories: {df_train['trajectory_id'].nunique()}")
    print(f"  Avg samples per trajectory: {len(df_train)/n_train:.0f}")

    print(f"\nValidation set:")
    print(f"  Samples: {len(df_val):,} ({len(df_val)/len(df)*100:.1f}%)")
    print(f"  Trajectories: {df_val['trajectory_id'].nunique()}")
    print(f"  Avg samples per trajectory: {len(df_val)/n_val:.0f}")

    print(f"\nTest set:")
    print(f"  Samples: {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")
    print(f"  Trajectories: {df_test['trajectory_id'].nunique()}")
    print(f"  Avg samples per trajectory: {len(df_test)/n_test:.0f}")

    # Save splits
    print(f"\n{'='*80}")
    print("SAVING SPLITS")
    print(f"{'='*80}")

    print(f"Saving train set to: {TRAIN_OUTPUT}")
    df_train.to_csv(TRAIN_OUTPUT, index=False)

    print(f"Saving validation set to: {VAL_OUTPUT}")
    df_val.to_csv(VAL_OUTPUT, index=False)

    print(f"Saving test set to: {TEST_OUTPUT}")
    df_test.to_csv(TEST_OUTPUT, index=False)

    print(f"\n{'='*80}")
    print("SUCCESS!")
    print(f"{'='*80}")
    print("Key differences from previous split:")
    print("  ✓ Entire trajectories kept together (no temporal leakage)")
    print("  ✓ Train/val/test are completely independent")
    print("  ✓ 3-way split for proper validation during training")
    print("  ✓ Each set has diverse trajectory coverage")
    print(f"\nUse these files for retraining to get honest generalization metrics.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
