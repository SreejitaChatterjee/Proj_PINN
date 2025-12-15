#!/usr/bin/env python3
"""
EuRoC MAV Dataset Loader

Downloads and preprocesses EuRoC MAV data for dynamics learning.
Source: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

Data format:
- IMU (imu0/data.csv): timestamp, wx, wy, wz, ax, ay, az
- Ground truth (state_groundtruth_estimate0/data.csv):
  timestamp, px, py, pz, qw, qx, qy, qz, vx, vy, vz, bwx, bwy, bwz, bax, bay, baz
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation

# EuRoC sequences (easiest to hardest)
SEQUENCES = {
    'MH_01_easy': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip',
    'MH_02_easy': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip',
    'MH_03_medium': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip',
    'V1_01_easy': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip',
    'V1_02_medium': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip',
}


def download_sequence(sequence_name='MH_01_easy', data_dir='data/euroc'):
    """Download a EuRoC sequence if not already present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    seq_dir = data_dir / sequence_name
    if seq_dir.exists():
        print(f"Sequence {sequence_name} already exists at {seq_dir}")
        return seq_dir

    url = SEQUENCES.get(sequence_name)
    if not url:
        raise ValueError(f"Unknown sequence: {sequence_name}. Available: {list(SEQUENCES.keys())}")

    zip_path = data_dir / f"{sequence_name}.zip"

    print(f"Downloading {sequence_name}...")
    print(f"  URL: {url}")
    print(f"  This may take a few minutes (~1GB)")

    urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    zip_path.unlink()  # Remove zip after extraction
    print(f"Done. Data at {seq_dir}")
    return seq_dir


def load_imu(seq_dir):
    """Load IMU data: angular velocity and linear acceleration."""
    seq_dir = Path(seq_dir)
    # Handle both structures: seq_dir/mav0/imu0 or seq_dir/imu0
    imu_path = seq_dir / 'mav0' / 'imu0' / 'data.csv'
    if not imu_path.exists():
        imu_path = seq_dir / 'imu0' / 'data.csv'
    if not imu_path.exists():
        # Check parent for mav0
        imu_path = seq_dir.parent / 'mav0' / 'imu0' / 'data.csv'

    df = pd.read_csv(imu_path, header=0, names=[
        'timestamp', 'wx', 'wy', 'wz', 'ax', 'ay', 'az'
    ])
    df['timestamp'] = df['timestamp'] / 1e9  # ns to seconds
    return df


def load_ground_truth(seq_dir):
    """Load ground truth state: position, orientation, velocity."""
    seq_dir = Path(seq_dir)
    # Handle both structures
    gt_path = seq_dir / 'mav0' / 'state_groundtruth_estimate0' / 'data.csv'
    if not gt_path.exists():
        gt_path = seq_dir / 'state_groundtruth_estimate0' / 'data.csv'
    if not gt_path.exists():
        gt_path = seq_dir.parent / 'mav0' / 'state_groundtruth_estimate0' / 'data.csv'

    df = pd.read_csv(gt_path, header=0, names=[
        'timestamp', 'px', 'py', 'pz',
        'qw', 'qx', 'qy', 'qz',
        'vx', 'vy', 'vz',
        'bwx', 'bwy', 'bwz',  # gyro bias
        'bax', 'bay', 'baz'   # accel bias
    ])
    df['timestamp'] = df['timestamp'] / 1e9  # ns to seconds
    return df


def quat_to_euler(qw, qx, qy, qz):
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    r = Rotation.from_quat([qx, qy, qz, qw])  # scipy uses xyzw order
    return r.as_euler('xyz')  # roll, pitch, yaw


def prepare_dynamics_data(seq_dir, dt=0.005):
    """
    Prepare data for dynamics learning.

    Returns DataFrame with:
    - State: x, y, z, roll, pitch, yaw, p, q, r, vx, vy, vz (12 states)
    - Pseudo-controls: ax, ay, az (from IMU, after bias correction)
    - Next state (for supervised learning)

    Args:
        seq_dir: Path to EuRoC sequence
        dt: Desired timestep (will resample data)
    """
    print(f"Loading data from {seq_dir}...")

    imu = load_imu(seq_dir)
    gt = load_ground_truth(seq_dir)

    print(f"  IMU samples: {len(imu)}")
    print(f"  GT samples: {len(gt)}")

    # Convert quaternion to Euler angles
    eulers = np.array([quat_to_euler(row.qw, row.qx, row.qy, row.qz)
                       for _, row in gt.iterrows()])
    gt['roll'] = eulers[:, 0]
    gt['pitch'] = eulers[:, 1]
    gt['yaw'] = eulers[:, 2]

    # Merge IMU and GT on nearest timestamp
    # First, create a common time grid
    t_start = max(imu['timestamp'].min(), gt['timestamp'].min())
    t_end = min(imu['timestamp'].max(), gt['timestamp'].max())

    # Resample to fixed dt
    t_grid = np.arange(t_start, t_end, dt)

    # Interpolate GT to grid
    from scipy.interpolate import interp1d

    gt_cols = ['px', 'py', 'pz', 'roll', 'pitch', 'yaw', 'vx', 'vy', 'vz']
    gt_interp = {}
    for col in gt_cols:
        f = interp1d(gt['timestamp'], gt[col], kind='linear', fill_value='extrapolate')
        gt_interp[col] = f(t_grid)

    # Interpolate IMU to grid
    imu_cols = ['wx', 'wy', 'wz', 'ax', 'ay', 'az']
    imu_interp = {}
    for col in imu_cols:
        f = interp1d(imu['timestamp'], imu[col], kind='linear', fill_value='extrapolate')
        imu_interp[col] = f(t_grid)

    # Build output dataframe
    data = pd.DataFrame({
        'timestamp': t_grid,
        'x': gt_interp['px'],
        'y': gt_interp['py'],
        'z': gt_interp['pz'],
        'roll': gt_interp['roll'],
        'pitch': gt_interp['pitch'],
        'yaw': gt_interp['yaw'],
        'p': imu_interp['wx'],  # angular rates from gyro
        'q': imu_interp['wy'],
        'r': imu_interp['wz'],
        'vx': gt_interp['vx'],
        'vy': gt_interp['vy'],
        'vz': gt_interp['vz'],
        # IMU as pseudo-controls
        'ax': imu_interp['ax'],
        'ay': imu_interp['ay'],
        'az': imu_interp['az'],
    })

    print(f"  Resampled to {len(data)} samples at {dt*1000:.1f}ms")
    print(f"  Duration: {t_end - t_start:.1f}s")

    return data


def create_training_pairs(data, window=1):
    """
    Create input-output pairs for dynamics learning.

    Input: state_t + control_t (15 features)
    Output: state_{t+1} (12 features)
    """
    state_cols = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    control_cols = ['ax', 'ay', 'az']

    X = []
    y = []

    for i in range(len(data) - window):
        # Current state + control
        state = data[state_cols].iloc[i].values
        control = data[control_cols].iloc[i].values
        inp = np.concatenate([state, control])

        # Next state
        next_state = data[state_cols].iloc[i + window].values

        X.append(inp)
        y.append(next_state)

    return np.array(X), np.array(y)


def main():
    """Download MH_01_easy and prepare for training."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', default='MH_01_easy', choices=list(SEQUENCES.keys()))
    parser.add_argument('--output', default='data/euroc_processed.csv')
    args = parser.parse_args()

    # Download
    seq_dir = download_sequence(args.sequence)

    # Process
    data = prepare_dynamics_data(seq_dir, dt=0.005)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")

    # Print stats
    print("\nData statistics:")
    print(data.describe())


if __name__ == '__main__':
    main()
