#!/usr/bin/env python3
"""
Combine standard training data with aggressive trajectories for improved inertia identification.
"""

from pathlib import Path

import pandas as pd


def main():
    PROJECT_ROOT = Path(__file__).parent.parent

    # Load standard training data
    df_standard = pd.read_csv(PROJECT_ROOT / "data" / "quadrotor_training_data.csv")
    print(
        f"Standard training data: {len(df_standard)} samples, {df_standard['trajectory_id'].nunique()} trajectories"
    )

    # Load aggressive trajectories
    df_aggressive = pd.read_csv(PROJECT_ROOT / "data" / "aggressive_inertia_trajectories.csv")
    print(
        f"Aggressive trajectory data: {len(df_aggressive)} samples, {df_aggressive['trajectory_id'].nunique()} trajectories"
    )

    # Combine datasets
    df_combined = pd.concat([df_standard, df_aggressive], ignore_index=True)
    print(
        f"\nCombined data: {len(df_combined)} samples, {df_combined['trajectory_id'].nunique()} trajectories"
    )

    # Save combined dataset
    output_path = PROJECT_ROOT / "data" / "combined_training_data.csv"
    df_combined.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")
    print(f"Trajectory IDs: {sorted(df_combined['trajectory_id'].unique())}")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING DATA SUMMARY")
    print("=" * 80)
    print(f"Standard trajectories (0-9):  {len(df_standard):,} samples")
    print(f"Aggressive trajectories (100-104): {len(df_aggressive):,} samples")
    print(f"Total: {len(df_combined):,} samples")
    print("=" * 80)


if __name__ == "__main__":
    main()
