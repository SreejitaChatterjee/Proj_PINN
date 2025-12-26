"""Create proper time-series train/validation/test split for diverse dataset"""

from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "quadrotor_training_data_diverse.csv"
TRAIN_OUTPUT = PROJECT_ROOT / "data" / "train_set_diverse.csv"
VAL_OUTPUT = PROJECT_ROOT / "data" / "val_set_diverse.csv"
TEST_OUTPUT = PROJECT_ROOT / "data" / "test_set_diverse.csv"

# Split configuration (by trajectory, not by samples)
# Train: 70%, Val: 15%, Test: 15%
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def main():
    print("=" * 80)
    print("CREATING TRAIN/VAL/TEST SPLIT FOR DIVERSE DATASET")
    print("(Splitting by ENTIRE TRAJECTORIES)")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df):,}")

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

    print(f"\nFirst/Last trajectory IDs in each split:")
    print(f"  Train: {train_traj_ids[0]} ... {train_traj_ids[-1]}")
    print(f"  Val:   {val_traj_ids[0]} ... {val_traj_ids[-1]}")
    print(f"  Test:  {test_traj_ids[0]} ... {test_traj_ids[-1]}")

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
    file_size_mb = TRAIN_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    print(f"\nSaving validation set to: {VAL_OUTPUT}")
    df_val.to_csv(VAL_OUTPUT, index=False)
    file_size_mb = VAL_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    print(f"\nSaving test set to: {TEST_OUTPUT}")
    df_test.to_csv(TEST_OUTPUT, index=False)
    file_size_mb = TEST_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    print(f"\n{'='*80}")
    print("SUCCESS!")
    print(f"{'='*80}")
    print("Key improvements over previous split:")
    print("  1. 10x more training data (7 -> 70 trajectories)")
    print("  2. 15x more validation data (1 -> 15 trajectories)")
    print("  3. 7.5x more test data (2 -> 15 trajectories)")
    print("  4. Much better diversity in all splits")
    print("  5. Proper trajectory-based splitting (no leakage)")
    print(f"\nNext: Retrain model with this diverse dataset")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
