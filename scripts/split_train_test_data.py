"""Split data into train and test sets using the same split as training"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'quadrotor_training_data.csv'
TRAIN_OUTPUT = PROJECT_ROOT / 'data' / 'quadrotor_train_only.csv'
TEST_OUTPUT = PROJECT_ROOT / 'data' / 'quadrotor_test_only.csv'

def main():
    print("="*80)
    print("SPLITTING DATA INTO TRAIN AND TEST SETS")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")

    # Get trajectory information
    trajectory_ids = sorted(df['trajectory_id'].unique())
    print(f"Total trajectories: {len(trajectory_ids)}")

    # Prepare data in same format as training script
    state_cols = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    input_features = state_cols + ['thrust', 'torque_x', 'torque_y', 'torque_z']

    # Build sample indices matching training script logic
    sample_indices = []
    for traj_id in trajectory_ids:
        df_traj = df[df['trajectory_id'] == traj_id]
        traj_start_idx = df_traj.index[0]
        # Each trajectory contributes (len-1) samples
        for i in range(len(df_traj) - 1):
            sample_indices.append(traj_start_idx + i)

    print(f"Total training samples (excluding last timestep per trajectory): {len(sample_indices)}")

    # Split indices using SAME random_state as training
    train_idx, test_idx = train_test_split(
        sample_indices,
        test_size=0.2,
        random_state=42  # MUST match training script
    )

    print(f"\nSplit sizes:")
    print(f"  Train indices: {len(train_idx)}")
    print(f"  Test indices: {len(test_idx)}")
    print(f"  Split ratio: {len(train_idx)/(len(train_idx)+len(test_idx)):.1%} train")

    # Create train and test dataframes
    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()

    # Save to CSV
    print(f"\nSaving train data to: {TRAIN_OUTPUT}")
    df_train.to_csv(TRAIN_OUTPUT, index=False)
    print(f"  Train samples: {len(df_train)}")
    print(f"  Train trajectories: {df_train['trajectory_id'].nunique()}")

    print(f"\nSaving test data to: {TEST_OUTPUT}")
    df_test.to_csv(TEST_OUTPUT, index=False)
    print(f"  Test samples: {len(df_test)}")
    print(f"  Test trajectories: {df_test['trajectory_id'].nunique()}")

    # Statistics
    print(f"\n{'='*80}")
    print("SPLIT STATISTICS")
    print(f"{'='*80}")
    print(f"Original data: {len(df)} samples")
    print(f"Train set: {len(df_train)} samples ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Test set: {len(df_test)} samples ({len(df_test)/len(df)*100:.1f}%)")
    print(f"\nNote: This split matches the train_test_split used during training (random_state=42)")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
