"""
Generate realistic synthetic attacks by injecting faults into clean EuRoC data.

Attack types (based on published literature):
1. GPS Spoofing - Gradual position drift (Tippenhauer CCS'11)
2. IMU Injection - Constant accelerometer bias (Shoukry CHES'13)
3. Sensor Dropout - Random data loss (jamming)
4. Replay Attack - Repeat old trajectory

Usage:
    python scripts/security/generate_synthetic_attacks.py \\
        --input data/euroc_mav/ \\
        --output data/attack_datasets/synthetic/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json


class SyntheticAttackGenerator:
    """Generate physics-based attack scenarios from clean data."""

    def __init__(self, clean_data: pd.DataFrame, seed: int = 42):
        """
        Args:
            clean_data: Clean flight trajectory with columns:
                [timestamp, x, y, z, phi, theta, psi, p, q, r, vx, vy, vz,
                 thrust, torque_x, torque_y, torque_z]
            seed: Random seed for reproducibility
        """
        self.clean_data = clean_data.copy()
        self.rng = np.random.RandomState(seed)

    def gps_spoofing_gradual_drift(
        self,
        drift_magnitude: float = 10.0,
        drift_duration: float = 30.0,
        attack_start_ratio: float = 0.3,
    ) -> pd.DataFrame:
        """
        GPS Spoofing: Gradual position drift.

        Simulates stealthy GPS spoofing attack where attacker gradually
        shifts reported position over time.

        Reference: Tippenhauer et al. "On the Requirements for Successful
        GPS Spoofing Attacks" (CCS 2011)

        Args:
            drift_magnitude: Total drift distance (meters)
            drift_duration: Duration of drift (seconds)
            attack_start_ratio: When attack starts (0.0-1.0)

        Returns:
            DataFrame with GPS spoofing attack injected
        """
        data = self.clean_data.copy()
        data['label'] = 0
        data['attack_type'] = 'Normal'

        # Find attack start/end indices
        n_total = len(data)
        attack_start_idx = int(n_total * attack_start_ratio)
        dt = data['timestamp'].diff().mean()
        attack_duration_samples = int(drift_duration / dt)
        attack_end_idx = min(attack_start_idx + attack_duration_samples, n_total)

        # Create smooth drift trajectory
        n_attack = attack_end_idx - attack_start_idx
        drift_x = np.linspace(0, drift_magnitude, n_attack)
        drift_y = np.linspace(0, drift_magnitude * 0.5, n_attack)  # Drift in x and y
        drift_z = np.linspace(0, drift_magnitude * 0.2, n_attack)  # Small z drift

        # Inject drift
        data.loc[attack_start_idx:attack_end_idx-1, 'x'] += drift_x
        data.loc[attack_start_idx:attack_end_idx-1, 'y'] += drift_y
        data.loc[attack_start_idx:attack_end_idx-1, 'z'] += drift_z

        # Mark as attack
        data.loc[attack_start_idx:attack_end_idx-1, 'label'] = 1
        data.loc[attack_start_idx:attack_end_idx-1, 'attack_type'] = 'GPS_Spoofing'

        return data

    def imu_bias_injection(
        self,
        accel_bias: float = 0.5,
        gyro_bias: float = 0.1,
        attack_start_ratio: float = 0.3,
        attack_duration: float = 20.0,
    ) -> pd.DataFrame:
        """
        IMU Injection: Constant bias added to accelerometer/gyro.

        Simulates sensor spoofing where attacker injects false accelerations.
        This violates Newton's laws (F=ma) and is detectable by physics.

        Reference: Shoukry et al. "Non-invasive Spoofing Attacks for
        Anti-lock Braking Systems" (CHES 2013)

        Args:
            accel_bias: Accelerometer bias (m/s²)
            gyro_bias: Gyro bias (rad/s)
            attack_start_ratio: When attack starts
            attack_duration: Duration (seconds)

        Returns:
            DataFrame with IMU bias attack
        """
        data = self.clean_data.copy()
        data['label'] = 0
        data['attack_type'] = 'Normal'

        # Attack window
        n_total = len(data)
        attack_start_idx = int(n_total * attack_start_ratio)
        dt = data['timestamp'].diff().mean()
        attack_duration_samples = int(attack_duration / dt)
        attack_end_idx = min(attack_start_idx + attack_duration_samples, n_total)

        # Inject constant bias
        # Note: We're affecting velocity change (from accelerometer bias)
        n_attack = attack_end_idx - attack_start_idx
        velocity_change = accel_bias * dt * np.arange(n_attack)

        data.loc[attack_start_idx:attack_end_idx-1, 'vx'] += velocity_change
        data.loc[attack_start_idx:attack_end_idx-1, 'p'] += gyro_bias
        data.loc[attack_start_idx:attack_end_idx-1, 'q'] += gyro_bias * 0.7

        # Mark as attack
        data.loc[attack_start_idx:attack_end_idx-1, 'label'] = 1
        data.loc[attack_start_idx:attack_end_idx-1, 'attack_type'] = 'IMU_Injection'

        return data

    def sensor_dropout(
        self,
        dropout_probability: float = 0.3,
        attack_start_ratio: float = 0.4,
        attack_duration: float = 15.0,
    ) -> pd.DataFrame:
        """
        Sensor Dropout: Random data loss (jamming attack).

        Simulates GPS/IMU jamming where sensor data is intermittently lost.
        Detector must handle missing data gracefully.

        Args:
            dropout_probability: Fraction of samples lost during attack
            attack_start_ratio: When attack starts
            attack_duration: Duration (seconds)

        Returns:
            DataFrame with sensor dropout attack
        """
        data = self.clean_data.copy()
        data['label'] = 0
        data['attack_type'] = 'Normal'

        # Attack window
        n_total = len(data)
        attack_start_idx = int(n_total * attack_start_ratio)
        dt = data['timestamp'].diff().mean()
        attack_duration_samples = int(attack_duration / dt)
        attack_end_idx = min(attack_start_idx + attack_duration_samples, n_total)

        # Random dropout mask
        n_attack = attack_end_idx - attack_start_idx
        dropout_mask = self.rng.rand(n_attack) < dropout_probability

        # Set GPS position/velocity to NaN (sensor failure)
        data.loc[attack_start_idx:attack_end_idx-1, 'x'] = np.where(
            dropout_mask, np.nan, data.loc[attack_start_idx:attack_end_idx-1, 'x']
        )
        data.loc[attack_start_idx:attack_end_idx-1, 'y'] = np.where(
            dropout_mask, np.nan, data.loc[attack_start_idx:attack_end_idx-1, 'y']
        )
        data.loc[attack_start_idx:attack_end_idx-1, 'z'] = np.where(
            dropout_mask, np.nan, data.loc[attack_start_idx:attack_end_idx-1, 'z']
        )

        # Mark as attack
        data.loc[attack_start_idx:attack_end_idx-1, 'label'] = 1
        data.loc[attack_start_idx:attack_end_idx-1, 'attack_type'] = 'Sensor_Dropout'

        return data

    def replay_attack(
        self,
        replay_window: int = 100,
        attack_start_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """
        Replay Attack: Repeat old sensor trajectory.

        Attacker replays previously recorded sensor data. This creates
        temporal inconsistency detectable by physics prediction.

        Args:
            replay_window: Number of samples to replay
            attack_start_ratio: When attack starts

        Returns:
            DataFrame with replay attack
        """
        data = self.clean_data.copy()
        data['label'] = 0
        data['attack_type'] = 'Normal'

        # Attack window
        n_total = len(data)
        attack_start_idx = int(n_total * attack_start_ratio)

        # Ensure we have enough history to replay
        if attack_start_idx < replay_window * 2:
            attack_start_idx = replay_window * 2

        attack_end_idx = min(attack_start_idx + replay_window, n_total)

        # Copy old trajectory
        replay_source_start = attack_start_idx - replay_window - 100
        replay_source_end = replay_source_start + (attack_end_idx - attack_start_idx)

        # Replay position/attitude/velocity (GPS + IMU)
        state_cols = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz']
        for col in state_cols:
            data.loc[attack_start_idx:attack_end_idx-1, col] = \
                data.loc[replay_source_start:replay_source_end-1, col].values

        # Mark as attack
        data.loc[attack_start_idx:attack_end_idx-1, 'label'] = 1
        data.loc[attack_start_idx:attack_end_idx-1, 'attack_type'] = 'Replay_Attack'

        return data

    def generate_all_attacks(self) -> dict:
        """Generate all attack types."""
        return {
            'gps_spoofing': self.gps_spoofing_gradual_drift(),
            'imu_injection': self.imu_bias_injection(),
            'sensor_dropout': self.sensor_dropout(),
            'replay_attack': self.replay_attack(),
            'clean': self.clean_data.copy(),
        }


def load_euroc_data(euroc_path: Path) -> pd.DataFrame:
    """
    Load preprocessed EuRoC data.

    Expects CSV with columns: [timestamp, x, y, z, phi, theta, psi,
                                p, q, r, vx, vy, vz, thrust, ...]
    """
    # Try to find preprocessed CSV
    csv_files = list(euroc_path.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {euroc_path}")

    # Load first CSV (or combine multiple)
    df = pd.read_csv(csv_files[0])

    print(f"Loaded {len(df)} samples from {csv_files[0].name}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic attacks")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to clean EuRoC data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/attack_datasets/synthetic",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Synthetic Attack Generator")
    print("=" * 60)

    # Load clean data
    print("\nLoading clean EuRoC data...")
    clean_data = load_euroc_data(input_path)

    # Add missing columns if needed
    if 'label' not in clean_data.columns:
        clean_data['label'] = 0
    if 'attack_type' not in clean_data.columns:
        clean_data['attack_type'] = 'Normal'

    # Generate attacks
    print("\nGenerating attacks...")
    generator = SyntheticAttackGenerator(clean_data, seed=args.seed)
    attacks = generator.generate_all_attacks()

    # Save each attack type
    for attack_name, attack_data in attacks.items():
        output_file = output_path / f"{attack_name}.csv"
        attack_data.to_csv(output_file, index=False)

        # Create metadata
        meta = {
            "attack_type": attack_name,
            "n_samples": len(attack_data),
            "n_attack_samples": int(attack_data['label'].sum()),
            "attack_ratio": float(attack_data['label'].mean()),
        }

        meta_file = output_path / f"{attack_name}_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  ✓ {attack_name}: {len(attack_data)} samples "
              f"({meta['attack_ratio']*100:.1f}% attack)")

    print("\n" + "=" * 60)
    print("Attack generation complete!")
    print(f"Output: {output_path.absolute()}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train PINN: python scripts/security/train_detector.py")
    print("2. Evaluate: python scripts/security/evaluate_detector.py")


if __name__ == "__main__":
    main()
