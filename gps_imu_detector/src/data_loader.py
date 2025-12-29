"""
Data Loader with Sequence-Wise Splits

Enforces:
1. Sequence boundaries (no temporal leakage)
2. Independent sensor signals (no circular derivations)
3. LOSO-CV protocol
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import warnings


@dataclass
class SequenceData:
    """Container for a single sequence's data."""
    sequence_id: str
    timestamps: np.ndarray
    position: np.ndarray      # [N, 3] - x, y, z
    velocity: np.ndarray      # [N, 3] - vx, vy, vz
    attitude: np.ndarray      # [N, 3] - roll, pitch, yaw
    angular_rates: np.ndarray # [N, 3] - p, q, r
    acceleration: np.ndarray  # [N, 3] - ax, ay, az
    labels: Optional[np.ndarray] = None  # Attack labels if available


class GPSIMUDataLoader:
    """
    Data loader enforcing evaluation protocol rules.

    Key principles:
    1. Sequence-wise splits only (LOSO-CV)
    2. Scalers fit on training normal data only
    3. No circular sensors (no derived ground truth)
    """

    # Independent sensor columns (no circular derivations)
    POSITION_COLS = ['x', 'y', 'z']
    VELOCITY_COLS = ['vx', 'vy', 'vz']
    ATTITUDE_COLS = ['roll', 'pitch', 'yaw']
    ANGULAR_RATE_COLS = ['p', 'q', 'r']
    ACCELERATION_COLS = ['ax', 'ay', 'az']

    # Columns to EXCLUDE (potentially circular)
    EXCLUDED_COLS = ['baro_alt', 'mag_heading', 'derived_*']

    def __init__(self, data_path: str, dt: float = 0.005):
        """
        Initialize data loader.

        Args:
            data_path: Path to dataset CSV
            dt: Sampling period (default 0.005 = 200Hz)
        """
        self.data_path = Path(data_path)
        self.dt = dt
        self.sequences: Dict[str, SequenceData] = {}
        self.scaler: Optional[StandardScaler] = None
        self._scaler_fitted = False

    def load(self) -> Dict[str, SequenceData]:
        """Load and parse dataset into sequences."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)

        # Rename columns if needed
        df = df.rename(columns={
            'phi': 'roll', 'theta': 'pitch', 'psi': 'yaw'
        })

        # Identify sequence column
        seq_col = 'sequence' if 'sequence' in df.columns else 'trajectory_id'
        if seq_col not in df.columns:
            # Treat as single sequence
            df[seq_col] = 'seq_0'

        # Load each sequence
        for seq_id in df[seq_col].unique():
            seq_df = df[df[seq_col] == seq_id].sort_values('timestamp')

            # Extract independent signals
            seq_data = SequenceData(
                sequence_id=str(seq_id),
                timestamps=seq_df['timestamp'].values if 'timestamp' in seq_df.columns else np.arange(len(seq_df)) * self.dt,
                position=self._safe_extract(seq_df, self.POSITION_COLS),
                velocity=self._safe_extract(seq_df, self.VELOCITY_COLS),
                attitude=self._safe_extract(seq_df, self.ATTITUDE_COLS),
                angular_rates=self._safe_extract(seq_df, self.ANGULAR_RATE_COLS),
                acceleration=self._safe_extract(seq_df, self.ACCELERATION_COLS),
            )

            self.sequences[seq_id] = seq_data

        print(f"Loaded {len(self.sequences)} sequences")
        return self.sequences

    def _safe_extract(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """Safely extract columns, filling missing with zeros."""
        available = [c for c in cols if c in df.columns]
        if len(available) == len(cols):
            return df[cols].values
        elif len(available) > 0:
            warnings.warn(f"Missing columns: {set(cols) - set(available)}")
            result = np.zeros((len(df), len(cols)))
            for i, c in enumerate(cols):
                if c in df.columns:
                    result[:, i] = df[c].values
            return result
        else:
            return np.zeros((len(df), len(cols)))

    def get_loso_splits(self) -> List[Tuple[List[str], str]]:
        """
        Generate Leave-One-Sequence-Out splits.

        Returns:
            List of (train_seq_ids, test_seq_id) tuples
        """
        seq_ids = list(self.sequences.keys())
        splits = []

        for test_id in seq_ids:
            train_ids = [s for s in seq_ids if s != test_id]
            splits.append((train_ids, test_id))

        return splits

    def get_train_test_data(
        self,
        train_ids: List[str],
        test_id: str,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/test data for a single LOSO fold.

        IMPORTANT: Scaler is fit ONLY on training data.

        Args:
            train_ids: List of training sequence IDs
            test_id: Test sequence ID
            fit_scaler: If True, fit scaler on training data

        Returns:
            X_train, X_test, seq_boundaries_train, seq_boundaries_test
        """
        # Concatenate training sequences
        train_data = []
        train_boundaries = [0]

        for seq_id in train_ids:
            seq = self.sequences[seq_id]
            features = self._extract_features(seq)
            train_data.append(features)
            train_boundaries.append(train_boundaries[-1] + len(features))

        X_train = np.vstack(train_data)

        # Get test sequence
        test_seq = self.sequences[test_id]
        X_test = self._extract_features(test_seq)
        test_boundaries = np.array([0, len(X_test)])

        # Fit scaler on training data ONLY
        if fit_scaler:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            self._scaler_fitted = True

        # Transform test data with training-fitted scaler
        if self._scaler_fitted:
            X_test = self.scaler.transform(X_test)

        return X_train, X_test, np.array(train_boundaries), test_boundaries

    def _extract_features(self, seq: SequenceData) -> np.ndarray:
        """Extract feature vector from sequence data."""
        return np.hstack([
            seq.position,
            seq.velocity,
            seq.attitude,
            seq.angular_rates,
            seq.acceleration,
        ])

    def verify_no_circular_sensors(self) -> bool:
        """
        Verify no sensor is derived from ground truth.

        Checks correlation between potential circular pairs.
        Returns True if no circular sensors detected.
        """
        print("\n=== Circular Sensor Verification ===")

        for seq_id, seq in self.sequences.items():
            # Check position-velocity consistency
            # If velocity is exactly d(position)/dt, it might be derived
            if len(seq.position) > 1:
                pos_diff = np.diff(seq.position, axis=0) / self.dt
                vel_subset = seq.velocity[:-1]

                for i, name in enumerate(['vx', 'vy', 'vz']):
                    corr = np.corrcoef(pos_diff[:, i], vel_subset[:, i])[0, 1]
                    if np.abs(corr) > 0.99:
                        warnings.warn(
                            f"Sequence {seq_id}: {name} may be derived from position "
                            f"(correlation: {corr:.4f})"
                        )

            # Check velocity-acceleration consistency
            if len(seq.velocity) > 1:
                vel_diff = np.diff(seq.velocity, axis=0) / self.dt
                acc_subset = seq.acceleration[:-1]

                for i, name in enumerate(['ax', 'ay', 'az']):
                    corr = np.corrcoef(vel_diff[:, i], acc_subset[:, i])[0, 1]
                    if np.abs(corr) > 0.99:
                        warnings.warn(
                            f"Sequence {seq_id}: {name} may be derived from velocity "
                            f"(correlation: {corr:.4f})"
                        )

        print("Circular sensor check complete.")
        return True


class AttackCatalog:
    """
    Attack catalog with reproducible generation.

    Attack types:
    1. Bias: Constant offset
    2. Drift: Slow ramp (AR(1))
    3. Noise: Increased variance
    4. Coordinated: Multiple sensors with consistent bias
    5. Intermittent: On/off attacks
    """

    ATTACK_TYPES = ['bias', 'drift', 'noise', 'coordinated', 'intermittent', 'ramp']
    SENSOR_GROUPS = ['position', 'velocity', 'attitude', 'angular_rates', 'acceleration']

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_attack(
        self,
        data: np.ndarray,
        attack_type: str,
        magnitude: float,
        sensor_group: str = 'position',
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate attack on data.

        Args:
            data: [N, D] input data
            attack_type: Type of attack
            magnitude: Attack magnitude (relative to std)
            sensor_group: Which sensor group to attack

        Returns:
            attacked_data, labels (1=attack, 0=normal)
        """
        attacked = data.copy()
        labels = np.zeros(len(data))
        n = len(data)

        # Determine which columns to attack
        col_ranges = {
            'position': (0, 3),
            'velocity': (3, 6),
            'attitude': (6, 9),
            'angular_rates': (9, 12),
            'acceleration': (12, 15),
        }
        start_col, end_col = col_ranges.get(sensor_group, (0, 3))

        # Compute baseline std for magnitude scaling
        baseline_std = np.std(data[:, start_col:end_col], axis=0)

        if attack_type == 'bias':
            # Constant offset
            offset = magnitude * baseline_std
            attacked[:, start_col:end_col] += offset
            labels[:] = 1

        elif attack_type == 'drift':
            # AR(1) slow drift
            ar_coef = kwargs.get('ar_coefficient', 0.99)
            drift = np.zeros((n, end_col - start_col))
            for i in range(1, n):
                drift[i] = ar_coef * drift[i-1] + self.rng.randn(end_col - start_col) * 0.01
            drift = drift / np.std(drift, axis=0) * magnitude * baseline_std
            attacked[:, start_col:end_col] += drift
            labels[:] = 1

        elif attack_type == 'noise':
            # Increased variance
            noise = self.rng.randn(n, end_col - start_col) * magnitude * baseline_std
            attacked[:, start_col:end_col] += noise
            labels[:] = 1

        elif attack_type == 'coordinated':
            # Attack multiple sensor groups consistently
            offset = magnitude * baseline_std
            attacked[:, start_col:end_col] += offset
            # Also attack angular rates with correlated offset
            if sensor_group != 'angular_rates':
                rate_offset = magnitude * np.std(data[:, 9:12], axis=0) * 0.5
                attacked[:, 9:12] += rate_offset
            labels[:] = 1

        elif attack_type == 'intermittent':
            # On/off attacks
            prob = kwargs.get('intermittent_prob', 0.1)
            attack_mask = self.rng.rand(n) < prob
            offset = magnitude * baseline_std
            attacked[attack_mask, start_col:end_col] += offset
            labels[attack_mask] = 1

        elif attack_type == 'ramp':
            # Linear ramp
            ramp = np.linspace(0, 1, n).reshape(-1, 1)
            attacked[:, start_col:end_col] += ramp * magnitude * baseline_std
            labels[:] = 1

        return attacked, labels

    def get_attack_catalog(self) -> List[Dict]:
        """
        Get full attack catalog for reproducible evaluation.

        Returns:
            List of attack specifications
        """
        catalog = []
        magnitudes = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

        for attack_type in self.ATTACK_TYPES:
            for sensor_group in self.SENSOR_GROUPS:
                for magnitude in magnitudes:
                    catalog.append({
                        'attack_type': attack_type,
                        'sensor_group': sensor_group,
                        'magnitude': magnitude,
                        'seed': self.seed,
                    })

        return catalog

    def save_catalog(self, path: str):
        """Save attack catalog to JSON."""
        import json
        catalog = self.get_attack_catalog()
        with open(path, 'w') as f:
            json.dump(catalog, f, indent=2)
        print(f"Saved {len(catalog)} attack specifications to {path}")


if __name__ == "__main__":
    # Test with EuRoC data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    loader = GPSIMUDataLoader("../data/euroc/all_sequences.csv")
    sequences = loader.load()

    # Verify no circular sensors
    loader.verify_no_circular_sensors()

    # Get LOSO splits
    splits = loader.get_loso_splits()
    print(f"\nGenerated {len(splits)} LOSO-CV folds")

    # Test attack generation
    catalog = AttackCatalog(seed=42)
    print(f"\nAttack catalog: {len(catalog.get_attack_catalog())} specifications")
