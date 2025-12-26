"""
Preprocess attack datasets to match QuadrotorPINN state format.

Converts raw attack data to standardized format:
    State (12): [x, y, z, phi, theta, psi, p, q, r, vx, vy, vz]
    Control (4): [thrust, torque_x, torque_y, torque_z]
    Label (1): [0=normal, 1=attack]

Output: CSV files compatible with existing PINN data loaders
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def extract_state_from_imu_gps(
    gps_data: pd.DataFrame,
    imu_data: pd.DataFrame,
    mag_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convert GPS + IMU + Magnetometer to 12-state representation.

    Args:
        gps_data: DataFrame with [timestamp, lat, lon, alt, vx, vy, vz]
        imu_data: DataFrame with [timestamp, ax, ay, az, gx, gy, gz]
        mag_data: Optional magnetometer [timestamp, mx, my, mz]

    Returns:
        DataFrame with [timestamp, x, y, z, phi, theta, psi, p, q, r, vx, vy, vz]
    """
    # Synchronize timestamps (nearest neighbor)
    merged = pd.merge_asof(
        gps_data.sort_values("timestamp"),
        imu_data.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=0.05,  # 50ms tolerance
    )

    # Convert GPS (lat/lon) to local frame (meters)
    # Simple ENU conversion (good enough for local flights)
    lat0, lon0 = merged["lat"].iloc[0], merged["lon"].iloc[0]
    R_earth = 6371000  # meters

    merged["x"] = (merged["lon"] - lon0) * np.cos(np.deg2rad(lat0)) * R_earth
    merged["y"] = (merged["lat"] - lat0) * R_earth
    merged["z"] = merged["alt"] - merged["alt"].iloc[0]

    # Angular rates from gyro (already body frame)
    merged["p"] = merged["gx"]
    merged["q"] = merged["gy"]
    merged["r"] = merged["gz"]

    # Estimate attitude from accelerometer (static assumption)
    # phi (roll) from ay, theta (pitch) from ax
    g = 9.81
    merged["phi"] = np.arctan2(merged["ay"], merged["az"])
    merged["theta"] = np.arctan2(-merged["ax"], np.sqrt(merged["ay"] ** 2 + merged["az"] ** 2))

    # Yaw from magnetometer (if available)
    if mag_data is not None:
        mag_merged = pd.merge_asof(
            merged,
            mag_data.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=0.05,
        )
        # Simple 2D magnetometer heading
        mag_merged["psi"] = np.arctan2(mag_merged["my"], mag_merged["mx"])
        merged = mag_merged
    else:
        # Integrate gyro yaw (drift alert!)
        dt = merged["timestamp"].diff().fillna(0)
        merged["psi"] = (merged["r"] * dt).cumsum()

    # Velocity from GPS (already in NED/ENU)
    # Note: May need body-to-inertial rotation
    # For now, assume GPS velocity is in body frame (common for UAV datasets)
    merged["vx"] = merged.get("vx", 0)
    merged["vy"] = merged.get("vy", 0)
    merged["vz"] = merged.get("vz", 0)

    # Select final state vector
    state_columns = [
        "timestamp",
        "x",
        "y",
        "z",
        "phi",
        "theta",
        "psi",
        "p",
        "q",
        "r",
        "vx",
        "vy",
        "vz",
    ]
    return merged[state_columns]


def add_attack_labels(
    state_df: pd.DataFrame,
    attack_intervals: list[Tuple[float, float]],
) -> pd.DataFrame:
    """
    Add binary attack labels based on time intervals.

    Args:
        state_df: State trajectory
        attack_intervals: List of (start_time, end_time) tuples

    Returns:
        DataFrame with added 'label' column (0=normal, 1=attack)
    """
    state_df["label"] = 0  # Default: normal

    for start, end in attack_intervals:
        mask = (state_df["timestamp"] >= start) & (state_df["timestamp"] <= end)
        state_df.loc[mask, "label"] = 1

    return state_df


def synthesize_controls(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate control inputs from state trajectory.

    For real data without logged controls, estimate from physics.
    This is an approximation - ideally dataset includes controls.

    Args:
        state_df: State trajectory

    Returns:
        DataFrame with added control columns
    """
    # Simple finite difference for derivatives
    dt = state_df["timestamp"].diff().fillna(0.01)

    # Estimate thrust from vertical acceleration
    dvz_dt = state_df["vz"].diff() / dt
    m_est = 0.068  # kg (typical quadrotor)
    g = 9.81
    state_df["thrust"] = m_est * (dvz_dt + g)

    # Estimate torques from angular accelerations
    J_est = {"xx": 6.86e-5, "yy": 9.2e-5, "zz": 1.366e-4}
    state_df["torque_x"] = J_est["xx"] * state_df["p"].diff() / dt
    state_df["torque_y"] = J_est["yy"] * state_df["q"].diff() / dt
    state_df["torque_z"] = J_est["zz"] * state_df["r"].diff() / dt

    # Fill NaNs (first row)
    state_df[["thrust", "torque_x", "torque_y", "torque_z"]] = state_df[
        ["thrust", "torque_x", "torque_y", "torque_z"]
    ].fillna(0)

    return state_df


def process_drone_fusion_dataset(input_dir: Path, output_dir: Path):
    """Process IEEE Drone Sensor Fusion Dataset."""
    print("Processing Drone Fusion Dataset...")

    # Expected structure (adjust based on actual dataset)
    scenarios = list(input_dir.glob("scenario_*"))

    for scenario in scenarios:
        print(f"  Processing {scenario.name}...")

        # Load sensor data (adjust column names based on actual format)
        gps_file = scenario / "gps.csv"
        imu_file = scenario / "imu.csv"
        mag_file = scenario / "magnetometer.csv"

        if not gps_file.exists() or not imu_file.exists():
            print(f"    ⚠️  Missing sensor files, skipping...")
            continue

        gps_data = pd.read_csv(gps_file)
        imu_data = pd.read_csv(imu_file)
        mag_data = pd.read_csv(mag_file) if mag_file.exists() else None

        # Extract state
        state_df = extract_state_from_imu_gps(gps_data, imu_data, mag_data)

        # Load attack metadata (if provided)
        attack_meta = scenario / "attacks.json"
        if attack_meta.exists():
            with open(attack_meta) as f:
                attack_info = json.load(f)
                attack_intervals = attack_info.get("intervals", [])
        else:
            # Assume second half is attack (common in datasets)
            total_time = state_df["timestamp"].max()
            attack_intervals = [(total_time / 2, total_time)]

        # Add labels
        state_df = add_attack_labels(state_df, attack_intervals)

        # Synthesize controls
        state_df = synthesize_controls(state_df)

        # Save
        output_file = output_dir / f"{scenario.name}.csv"
        state_df.to_csv(output_file, index=False)
        print(f"    ✓ Saved to {output_file}")

        # Save metadata
        meta = {
            "scenario": scenario.name,
            "attack_type": (
                attack_info.get("type", "unknown") if attack_meta.exists() else "unknown"
            ),
            "attack_intervals": attack_intervals,
            "n_samples": len(state_df),
            "attack_ratio": state_df["label"].mean(),
        }
        with open(output_dir / f"{scenario.name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Preprocess attack datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["drone_fusion", "cyber_physical", "alfa"],
        required=True,
        help="Dataset to preprocess",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with raw data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pinn_dynamics/data/attack_datasets/processed",
        help="Output directory",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Preprocessing {args.dataset}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    if args.dataset == "drone_fusion":
        process_drone_fusion_dataset(input_dir, output_dir)
    elif args.dataset == "cyber_physical":
        # TODO: Implement based on dataset structure
        print("⚠️  Cyber-Physical dataset preprocessing not yet implemented")
    elif args.dataset == "alfa":
        # TODO: Implement ROS bag processing
        print("⚠️  ALFA dataset requires ROS bag processing (TODO)")

    print("\n✓ Preprocessing complete!")
    print(f"Processed data saved to: {output_dir}")


if __name__ == "__main__":
    main()
