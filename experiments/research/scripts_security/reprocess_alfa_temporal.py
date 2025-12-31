"""
Reprocess ALFA data with CORRECT per-sample temporal labels.

The original preprocessing labeled ALL samples in a fault flight as "fault",
but in reality the fault occurs mid-flight. This script uses the failure_status
files to determine the exact fault onset time and labels samples correctly.

Usage:
    python scripts/security/reprocess_alfa_temporal.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def quaternion_to_euler(qx, qy, qz, qw):
    """Convert quaternion to Euler angles."""
    r = Rotation.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = r.as_euler("xyz", degrees=False)
    return roll, pitch, yaw


def get_fault_onset_time(flight_dir: Path, flight_name: str) -> float:
    """
    Get the exact timestamp when the fault first occurs.

    Returns:
        Fault onset time in seconds, or None if no fault status file.
    """
    # Try different failure status file patterns
    patterns = [
        f"{flight_name}-failure_status-engines.csv",
        f"{flight_name}-failure_status-surfaces.csv",
        f"{flight_name}-failure_status.csv",
    ]

    for pattern in patterns:
        status_file = flight_dir / pattern
        if status_file.exists():
            df = pd.read_csv(status_file)
            # Find first row where fault is active (data=1 or similar)
            if "field.data" in df.columns:
                fault_rows = df[df["field.data"] == 1]
            elif "data" in df.columns:
                fault_rows = df[df["data"] == 1]
            else:
                continue

            if len(fault_rows) > 0:
                # Convert nanoseconds to seconds
                onset_ns = fault_rows["%time"].iloc[0]
                return onset_ns / 1e9

    return None


def load_flight_with_temporal_labels(flight_dir: Path) -> pd.DataFrame:
    """
    Load flight data and add CORRECT per-sample labels.

    For fault flights:
      - Samples before fault onset: label=0
      - Samples after fault onset: label=1

    For normal flights:
      - All samples: label=0
    """
    flight_name = flight_dir.name

    # Determine flight type from name
    if "no_failure" in flight_name:
        is_fault_flight = False
        fault_type = "Normal"
    elif "engine_failure" in flight_name:
        is_fault_flight = True
        fault_type = "Engine_Failure"
    elif "elevator" in flight_name.lower():
        is_fault_flight = True
        fault_type = "Elevator_Stuck"
    elif "aileron" in flight_name.lower():
        is_fault_flight = True
        fault_type = "Aileron_Stuck"
    elif "rudder" in flight_name.lower():
        is_fault_flight = True
        fault_type = "Rudder_Stuck"
    else:
        is_fault_flight = True
        fault_type = "Unknown_Fault"

    # Load pose data
    pose_file = flight_dir / f"{flight_name}-mavros-local_position-pose.csv"
    if not pose_file.exists():
        print(f"  Warning: Missing pose file for {flight_name}")
        return None

    pose_df = pd.read_csv(pose_file)
    pose_df = pose_df.rename(columns=lambda x: x.replace("field.", ""))

    # Extract position
    pose_df["x"] = pose_df["pose.position.x"]
    pose_df["y"] = pose_df["pose.position.y"]
    pose_df["z"] = pose_df["pose.position.z"]

    # Convert quaternion to Euler
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
    pose_df["phi"] = euler_angles[:, 0]
    pose_df["theta"] = euler_angles[:, 1]
    pose_df["psi"] = euler_angles[:, 2]

    # Timestamp
    pose_df["timestamp"] = pose_df["%time"] / 1e9

    # Load IMU
    imu_file = flight_dir / f"{flight_name}-mavros-imu-data.csv"
    if imu_file.exists():
        imu_df = pd.read_csv(imu_file)
        imu_df = imu_df.rename(columns=lambda x: x.replace("field.", ""))
        imu_df["p"] = imu_df["angular_velocity.x"]
        imu_df["q"] = imu_df["angular_velocity.y"]
        imu_df["r"] = imu_df["angular_velocity.z"]
        imu_df["timestamp"] = imu_df["%time"] / 1e9
        pose_df = pd.merge_asof(
            pose_df.sort_values("timestamp"),
            imu_df[["timestamp", "p", "q", "r"]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=0.1,
        )
    else:
        pose_df["p"] = 0
        pose_df["q"] = 0
        pose_df["r"] = 0

    # Load velocity
    vel_file = flight_dir / f"{flight_name}-mavros-local_position-velocity.csv"
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

    # Load controls
    rc_file = flight_dir / f"{flight_name}-mavros-rc-out.csv"
    if rc_file.exists():
        rc_df = pd.read_csv(rc_file)
        rc_df = rc_df.rename(columns=lambda x: x.replace("field.", ""))
        if "channels0" in rc_df.columns:
            rc_df["thrust"] = rc_df["channels2"]
            rc_df["torque_x"] = rc_df["channels0"]
            rc_df["torque_y"] = rc_df["channels1"]
            rc_df["torque_z"] = rc_df["channels3"]
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

    # Select columns
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
    final_df = final_df.dropna()

    if len(final_df) == 0:
        return None

    # NOW: Add CORRECT temporal labels
    final_df["fault_type"] = fault_type

    if not is_fault_flight:
        # Normal flight: all samples are normal
        final_df["label"] = 0
        final_df["fault_onset"] = np.nan
        n_pre_fault = len(final_df)
        n_post_fault = 0
    else:
        # Fault flight: find onset time and label accordingly
        fault_onset = get_fault_onset_time(flight_dir, flight_name)

        if fault_onset is not None:
            final_df["label"] = (final_df["timestamp"] >= fault_onset).astype(int)
            final_df["fault_onset"] = fault_onset
            n_pre_fault = (final_df["label"] == 0).sum()
            n_post_fault = (final_df["label"] == 1).sum()
        else:
            # No fault status file - mark all as fault (conservative)
            final_df["label"] = 1
            final_df["fault_onset"] = final_df["timestamp"].iloc[0]
            n_pre_fault = 0
            n_post_fault = len(final_df)

    return final_df, n_pre_fault, n_post_fault, fault_type


def main():
    input_dir = Path("data/alfa/processed/processed")
    output_dir = Path("data/alfa/temporal")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ALFA TEMPORAL REPROCESSING")
    print("=" * 70)
    print("\nThis corrects the labeling error where all samples in fault flights")
    print("were labeled as 'fault'. Now we use failure_status to find the exact")
    print("fault onset time and label samples correctly.\n")

    flight_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(flight_dirs)} flights\n")

    stats = {
        "total_flights": 0,
        "normal_flights": 0,
        "fault_flights": 0,
        "total_samples": 0,
        "normal_samples": 0,
        "fault_samples": 0,
        "pre_fault_samples": 0,  # Normal samples within fault flights
        "fault_types": {},
    }

    for i, flight_dir in enumerate(flight_dirs, 1):
        result = load_flight_with_temporal_labels(flight_dir)

        if result is None:
            print(f"[{i}/{len(flight_dirs)}] {flight_dir.name}: SKIPPED (no data)")
            continue

        df, n_pre_fault, n_post_fault, fault_type = result

        # Save
        output_file = output_dir / f"{flight_dir.name}.csv"
        df.to_csv(output_file, index=False)

        # Update stats
        stats["total_flights"] += 1
        stats["total_samples"] += len(df)

        if fault_type == "Normal":
            stats["normal_flights"] += 1
            stats["normal_samples"] += len(df)
        else:
            stats["fault_flights"] += 1
            stats["normal_samples"] += n_pre_fault  # Pre-fault samples are normal!
            stats["pre_fault_samples"] += n_pre_fault
            stats["fault_samples"] += n_post_fault

        if fault_type not in stats["fault_types"]:
            stats["fault_types"][fault_type] = 0
        stats["fault_types"][fault_type] += 1

        print(f"[{i}/{len(flight_dirs)}] {flight_dir.name}")
        print(f"    Type: {fault_type}")
        print(f"    Pre-fault (label=0): {n_pre_fault}")
        print(f"    Post-fault (label=1): {n_post_fault}")

    # Save stats (convert numpy types to Python types)
    stats["timestamp"] = datetime.now().isoformat()
    stats = {k: (int(v) if isinstance(v, (np.integer, np.int64)) else v) for k, v in stats.items()}
    stats_file = output_dir / "processing_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total flights: {stats['total_flights']}")
    print(f"  Normal flights: {stats['normal_flights']}")
    print(f"  Fault flights: {stats['fault_flights']}")
    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"  Normal samples (label=0): {stats['normal_samples']}")
    print(f"    From normal flights: {stats['normal_samples'] - stats['pre_fault_samples']}")
    print(f"    From pre-fault periods: {stats['pre_fault_samples']}")
    print(f"  Fault samples (label=1): {stats['fault_samples']}")
    print(f"\nFault types: {stats['fault_types']}")
    print(f"\nOutput: {output_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
