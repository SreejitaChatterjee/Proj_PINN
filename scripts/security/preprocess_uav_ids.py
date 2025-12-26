"""
Preprocess UAV-IDS dataset (Kaggle) for PINN anomaly detection.

UAV-IDS Dataset: Real DJI Phantom flights with network + sensor attacks.
Source: https://www.kaggle.com/datasets/monirul101/uav-ids-dataset

Converts to QuadrotorPINN format:
    State (12): [x, y, z, phi, theta, psi, p, q, r, vx, vy, vz]
    Control (4): [thrust, torque_x, torque_y, torque_z]
    Label (1): [0=normal, 1=attack]
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_uav_ids_file(file_path: Path) -> pd.DataFrame:
    """
    Load UAV-IDS CSV file.

    Expected columns (DJI Phantom telemetry):
    - timestamp, lat, lon, alt
    - roll, pitch, yaw (degrees)
    - vx, vy, vz (m/s)
    - ax, ay, az (accelerometer)
    - gx, gy, gz (gyro)
    - attack_type (string: 'Normal', 'GPS_Spoofing', 'DoS', etc.)
    """
    df = pd.read_csv(file_path)
    return df


def convert_to_pinn_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert UAV-IDS telemetry to PINN 12-state format.

    Args:
        df: Raw UAV-IDS dataframe

    Returns:
        DataFrame with PINN-compatible states
    """
    # Create output dataframe
    pinn_df = pd.DataFrame()

    # Timestamp
    pinn_df["timestamp"] = df["timestamp"]

    # Position (convert GPS to local ENU frame)
    lat0, lon0, alt0 = df["lat"].iloc[0], df["lon"].iloc[0], df["alt"].iloc[0]
    R_earth = 6371000  # meters

    pinn_df["x"] = (df["lon"] - lon0) * np.cos(np.deg2rad(lat0)) * R_earth
    pinn_df["y"] = (df["lat"] - lat0) * R_earth
    pinn_df["z"] = df["alt"] - alt0

    # Attitude (convert degrees to radians)
    pinn_df["phi"] = np.deg2rad(df["roll"])
    pinn_df["theta"] = np.deg2rad(df["pitch"])
    pinn_df["psi"] = np.deg2rad(df["yaw"])

    # Angular rates (rad/s)
    pinn_df["p"] = np.deg2rad(df["gx"])  # Gyro x → roll rate
    pinn_df["q"] = np.deg2rad(df["gy"])  # Gyro y → pitch rate
    pinn_df["r"] = np.deg2rad(df["gz"])  # Gyro z → yaw rate

    # Velocity (body frame, m/s)
    pinn_df["vx"] = df["vx"]
    pinn_df["vy"] = df["vy"]
    pinn_df["vz"] = df["vz"]

    # Labels (0=normal, 1=attack)
    if "attack_type" in df.columns:
        pinn_df["label"] = (df["attack_type"] != "Normal").astype(int)
        pinn_df["attack_type"] = df["attack_type"]
    else:
        pinn_df["label"] = 0  # Assume all normal if no label

    # Estimate controls from state changes
    pinn_df = estimate_controls(pinn_df)

    return pinn_df


def estimate_controls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate control inputs from state trajectory.

    Since UAV-IDS doesn't provide motor commands, estimate from dynamics.
    """
    dt = df["timestamp"].diff().fillna(0.01)

    # Physical parameters (DJI Phantom 3)
    m = 1.28  # kg (mass)
    g = 9.81  # m/s^2
    Jxx, Jyy, Jzz = 0.0123, 0.0123, 0.0224  # kg*m^2 (inertia)

    # Estimate thrust from vertical acceleration + gravity
    dvz = df["vz"].diff() / dt
    df["thrust"] = m * (dvz + g)

    # Estimate torques from angular accelerations
    df["torque_x"] = Jxx * df["p"].diff() / dt
    df["torque_y"] = Jyy * df["q"].diff() / dt
    df["torque_z"] = Jzz * df["r"].diff() / dt

    # Fill NaNs (first row)
    df[["thrust", "torque_x", "torque_y", "torque_z"]] = df[
        ["thrust", "torque_x", "torque_y", "torque_z"]
    ].fillna(0)

    # Clip unrealistic values
    df["thrust"] = df["thrust"].clip(0, 20)  # 0-20N
    df["torque_x"] = df["torque_x"].clip(-1, 1)  # ±1 N*m
    df["torque_y"] = df["torque_y"].clip(-1, 1)
    df["torque_z"] = df["torque_z"].clip(-0.5, 0.5)

    return df


def split_by_attack_type(df: pd.DataFrame, output_dir: Path):
    """
    Split dataset by attack type and save separately.

    Args:
        df: Preprocessed dataframe
        output_dir: Output directory
    """
    if "attack_type" not in df.columns:
        # Save all as single file
        output_file = output_dir / "all_data.csv"
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        return

    attack_types = df["attack_type"].unique()

    for attack_type in attack_types:
        subset = df[df["attack_type"] == attack_type]

        # Create filename
        filename = f"{attack_type.lower().replace(' ', '_')}.csv"
        output_file = output_dir / filename

        # Save
        subset.to_csv(output_file, index=False)

        # Create metadata
        meta = {
            "attack_type": attack_type,
            "n_samples": len(subset),
            "attack_ratio": subset["label"].mean(),
            "duration_seconds": subset["timestamp"].max() - subset["timestamp"].min(),
        }

        meta_file = output_dir / f"{attack_type.lower().replace(' ', '_')}_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"  Saved: {output_file} ({len(subset)} samples, {meta['attack_ratio']*100:.1f}% attacks)"
        )


def create_train_val_test_split(
    output_dir: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    """
    Create train/val/test splits from processed data.

    Strategy:
    - Train: Normal data only (60%)
    - Val: Normal + some attacks for threshold tuning (20%)
    - Test: Remaining attacks for final evaluation (20%)
    """
    # Load all normal data
    normal_file = output_dir / "normal.csv"
    if not normal_file.exists():
        print("  Warning: No normal.csv found, skipping split")
        return

    normal_df = pd.read_csv(normal_file)

    # Split normal data
    n_total = len(normal_df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_df = normal_df.iloc[:n_train]
    val_normal_df = normal_df.iloc[n_train : n_train + n_val]
    test_normal_df = normal_df.iloc[n_train + n_val :]

    # Load attack data
    attack_files = list(output_dir.glob("*.csv"))
    attack_files = [f for f in attack_files if f.stem != "normal"]

    val_attack_dfs = []
    test_attack_dfs = []

    for attack_file in attack_files:
        attack_df = pd.read_csv(attack_file)

        # Split attacks 50/50 between val and test
        n_attack = len(attack_df)
        n_val_attack = n_attack // 2

        val_attack_dfs.append(attack_df.iloc[:n_val_attack])
        test_attack_dfs.append(attack_df.iloc[n_val_attack:])

    # Combine
    val_df = pd.concat([val_normal_df] + val_attack_dfs, ignore_index=True)
    test_df = pd.concat([test_normal_df] + test_attack_dfs, ignore_index=True)

    # Shuffle
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"\n  Train: {len(train_df)} samples (100% normal)")
    print(f"  Val: {len(val_df)} samples ({val_df['label'].mean()*100:.1f}% attacks)")
    print(f"  Test: {len(test_df)} samples ({test_df['label'].mean()*100:.1f}% attacks)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess UAV-IDS dataset")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to UAV-IDS directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pinn_dynamics/data/attack_datasets/processed/uav_ids",
        help="Output directory",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UAV-IDS Dataset Preprocessing")
    print("=" * 60)

    # Find all CSV files
    csv_files = list(input_dir.glob("*.csv"))

    if len(csv_files) == 0:
        print("No CSV files found in input directory!")
        print(f"Looking in: {input_dir.absolute()}")
        return

    print(f"Found {len(csv_files)} CSV files")

    # Process each file
    all_data = []
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        df = load_uav_ids_file(csv_file)
        pinn_df = convert_to_pinn_format(df)
        all_data.append(pinn_df)

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal samples: {len(combined_df)}")

    # Split by attack type
    print("\nSplitting by attack type...")
    split_by_attack_type(combined_df, output_dir)

    # Create train/val/test splits
    print("\nCreating train/val/test splits...")
    create_train_val_test_split(output_dir)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Output: {output_dir.absolute()}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train PINN: python demo.py --data processed/uav_ids/train.csv")
    print("2. Evaluate: python scripts/security/evaluate_detector.py")


if __name__ == "__main__":
    main()
