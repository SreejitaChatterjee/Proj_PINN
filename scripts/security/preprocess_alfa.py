"""
Preprocess CMU ALFA Dataset for PINN Anomaly Detection.

Converts ALFA fixed-wing UAV data to 12-state quadrotor-equivalent format.
Note: ALFA uses fixed-wing aircraft, but we extract analogous states.

State (12): [x, y, z, phi, theta, psi, p, q, r, vx, vy, vz]
Control (4): [throttle, aileron, elevator, rudder] (analogous to thrust/torques)
Label: [0=normal, 1=fault] + fault_type

Usage:
    python scripts/security/preprocess_alfa.py \\
        --input data/attack_datasets/12707963/processed/ \\
        --output data/attack_datasets/processed/alfa/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def quaternion_to_euler(qx, qy, qz, qw):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        qx, qy, qz, qw: Quaternion components

    Returns:
        (roll, pitch, yaw) in radians
    """
    # Using scipy for robust conversion
    r = Rotation.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = r.as_euler("xyz", degrees=False)
    return roll, pitch, yaw


def load_flight_data(flight_dir: Path) -> pd.DataFrame:
    """
    Load and merge all CSV files for a single flight.

    Args:
        flight_dir: Directory containing CSV files for one flight

    Returns:
        Merged dataframe with 12-state representation
    """
    flight_name = flight_dir.name

    # Load key CSV files
    pose_file = flight_dir / f"{flight_name}-mavros-local_position-pose.csv"
    imu_file = flight_dir / f"{flight_name}-mavros-imu-data.csv"
    vel_file = flight_dir / f"{flight_name}-mavros-local_position-velocity.csv"
    rc_file = flight_dir / f"{flight_name}-mavros-rc-out.csv"

    # Check if files exist
    if not pose_file.exists():
        print(f"  Warning: Missing pose file for {flight_name}")
        return None

    # Load pose (position + orientation)
    pose_df = pd.read_csv(pose_file)
    pose_df = pose_df.rename(columns=lambda x: x.replace("field.", ""))

    # Extract position
    pose_df["x"] = pose_df["pose.position.x"]
    pose_df["y"] = pose_df["pose.position.y"]
    pose_df["z"] = pose_df["pose.position.z"]

    # Convert quaternion to Euler angles
    euler_angles = np.array(
        [
            quaternion_to_euler(
                row["pose.orientation.x"],
                row["pose.orientation.y"],
                row["pose.orientation.z"],
                row["pose.orientation.w"],
            )
            for _, row in pose_df.iterrows()
        ]
    )

    pose_df["phi"] = euler_angles[:, 0]  # roll
    pose_df["theta"] = euler_angles[:, 1]  # pitch
    pose_df["psi"] = euler_angles[:, 2]  # yaw

    # Timestamp for merging
    pose_df["timestamp"] = pose_df["%time"] / 1e9  # Convert nanoseconds to seconds

    # Load IMU (angular rates)
    if imu_file.exists():
        imu_df = pd.read_csv(imu_file)
        imu_df = imu_df.rename(columns=lambda x: x.replace("field.", ""))
        imu_df["p"] = imu_df["angular_velocity.x"]  # roll rate
        imu_df["q"] = imu_df["angular_velocity.y"]  # pitch rate
        imu_df["r"] = imu_df["angular_velocity.z"]  # yaw rate
        imu_df["timestamp"] = imu_df["%time"] / 1e9

        # Merge with pose (nearest timestamp)
        pose_df = pd.merge_asof(
            pose_df.sort_values("timestamp"),
            imu_df[["timestamp", "p", "q", "r"]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=0.1,  # 100ms tolerance
        )
    else:
        pose_df["p"] = 0
        pose_df["q"] = 0
        pose_df["r"] = 0

    # Load velocity
    if vel_file.exists():
        vel_df = pd.read_csv(vel_file)
        vel_df = vel_df.rename(columns=lambda x: x.replace("field.", ""))
        vel_df["vx"] = vel_df["twist.linear.x"]
        vel_df["vy"] = vel_df["twist.linear.y"]
        vel_df["vz"] = vel_df["twist.linear.z"]
        vel_df["timestamp"] = vel_df["%time"] / 1e9

        pose_df = pd.merge_asof(
            pose_df.sort_values("timestamp"),
            vel_df[["timestamp", "vx", "vy", "vz"]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=0.1,
        )
    else:
        pose_df["vx"] = 0
        pose_df["vy"] = 0
        pose_df["vz"] = 0

    # Load RC outputs (controls)
    if rc_file.exists():
        rc_df = pd.read_csv(rc_file)
        rc_df = rc_df.rename(columns=lambda x: x.replace("field.", ""))

        # Fixed-wing controls (4 channels typical)
        # Map to quadrotor-like controls
        if "channels0" in rc_df.columns:
            rc_df["thrust"] = rc_df["channels2"]  # Throttle → thrust
            rc_df["torque_x"] = rc_df["channels0"]  # Aileron → roll torque
            rc_df["torque_y"] = rc_df["channels1"]  # Elevator → pitch torque
            rc_df["torque_z"] = rc_df["channels3"]  # Rudder → yaw torque

            rc_df["timestamp"] = rc_df["%time"] / 1e9

            pose_df = pd.merge_asof(
                pose_df.sort_values("timestamp"),
                rc_df[["timestamp", "thrust", "torque_x", "torque_y", "torque_z"]].sort_values(
                    "timestamp"
                ),
                on="timestamp",
                direction="nearest",
                tolerance=0.1,
            )
        else:
            pose_df["thrust"] = 0
            pose_df["torque_x"] = 0
            pose_df["torque_y"] = 0
            pose_df["torque_z"] = 0
    else:
        pose_df["thrust"] = 0
        pose_df["torque_x"] = 0
        pose_df["torque_y"] = 0
        pose_df["torque_z"] = 0

    # Select final columns
    state_cols = [
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
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    final_df = pose_df[state_cols + control_cols].copy()

    # Remove NaNs
    final_df = final_df.dropna()

    return final_df


def extract_fault_label(flight_name: str) -> tuple:
    """
    Extract fault type from flight directory name.

    Args:
        flight_name: e.g., "carbonZ_2018-07-18-15-53-31_1_engine_failure"

    Returns:
        (label, fault_type) where label=0 for normal, 1 for fault
    """
    if "no_failure" in flight_name:
        return 0, "Normal"
    elif "engine_failure" in flight_name:
        return 1, "Engine_Failure"
    elif "elevator" in flight_name.lower():
        return 1, "Elevator_Stuck"
    elif "aileron" in flight_name.lower():
        return 1, "Aileron_Stuck"
    elif "rudder" in flight_name.lower():
        return 1, "Rudder_Stuck"
    else:
        # Unknown, mark as fault
        return 1, "Unknown_Fault"


def process_all_flights(input_dir: Path, output_dir: Path):
    """
    Process all 47 flights in ALFA dataset.

    Args:
        input_dir: Directory with processed ALFA data
        output_dir: Output directory for preprocessed CSVs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all flight directories
    flight_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    print(f"Found {len(flight_dirs)} flight sequences")

    results = []
    for i, flight_dir in enumerate(flight_dirs, 1):
        print(f"\n[{i}/{len(flight_dirs)}] Processing {flight_dir.name}...")

        # Load and merge CSVs
        df = load_flight_data(flight_dir)

        if df is None or len(df) == 0:
            print(f"  Skipped (no data)")
            continue

        # Add fault label
        label, fault_type = extract_fault_label(flight_dir.name)
        df["label"] = label
        df["fault_type"] = fault_type

        # Save to output
        output_file = output_dir / f"{flight_dir.name}.csv"
        df.to_csv(output_file, index=False)

        # Metadata
        meta = {
            "flight_name": flight_dir.name,
            "fault_type": fault_type,
            "n_samples": len(df),
            "duration_seconds": float(df["timestamp"].max() - df["timestamp"].min()),
            "fault_ratio": float(df["label"].mean()),
        }

        meta_file = output_dir / f"{flight_dir.name}_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        results.append(meta)
        print(
            f"  [OK] Saved {len(df)} samples ({meta['duration_seconds']:.1f}s, fault={fault_type})"
        )

    # Create summary
    summary = {
        "total_flights": len(results),
        "normal_flights": sum(1 for r in results if r["fault_type"] == "Normal"),
        "fault_flights": sum(1 for r in results if r["fault_type"] != "Normal"),
        "fault_types": list(set(r["fault_type"] for r in results)),
        "total_samples": sum(r["n_samples"] for r in results),
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total flights: {summary['total_flights']}")
    print(f"Normal flights: {summary['normal_flights']}")
    print(f"Fault flights: {summary['fault_flights']}")
    print(f"Fault types: {', '.join(summary['fault_types'])}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"\nOutput: {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ALFA dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/attack_datasets/12707963/processed",
        help="Path to ALFA processed directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/attack_datasets/processed/alfa",
        help="Output directory",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    print("=" * 60)
    print("ALFA Dataset Preprocessing")
    print("=" * 60)

    process_all_flights(input_dir, output_dir)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Train PINN on normal flights")
    print("2. Evaluate on fault scenarios")
    print("=" * 60)


if __name__ == "__main__":
    main()
