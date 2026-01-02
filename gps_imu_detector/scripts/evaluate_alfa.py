#!/usr/bin/env python3
"""
GPS Spoofing Evaluation on REAL ALFA Flight Data

UNSUPERVISED approach:
- Train on NORMAL ALFA flights (label=0)
- Inject GPS spoofing attacks on test flights
- Evaluate detection with Mahalanobis distance

This uses REAL flight data, unlike the synthetic evaluation.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


class ALFAEvaluator:
    """Evaluate GPS spoofing detection on real ALFA + synthetic flight data."""

    def __init__(self, data_dir: str, seed: int = 42, use_synthetic: bool = True):
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.use_synthetic = use_synthetic
        np.random.seed(seed)
        self.gps_noise_std = 1.5  # meters (realistic)

    def load_synthetic_normal_flights(self, n_flights: int = 50) -> List[Tuple[str, np.ndarray]]:
        """Load synthetic normal flights from pinn_ready_attacks.csv."""
        synthetic_path = self.data_dir.parent.parent / "attack_datasets" / "synthetic" / "pinn_ready_attacks.csv"

        if not synthetic_path.exists():
            print(f"  Warning: Synthetic data not found at {synthetic_path}")
            return []

        try:
            df = pd.read_csv(synthetic_path, nrows=100000)  # Limit for speed
        except Exception as e:
            print(f"  Warning: Could not read synthetic data: {e}")
            return []

        if 'label' not in df.columns:
            return []

        # Get normal data only
        normal_df = df[df['label'] == 0]

        required_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']
        if not all(c in normal_df.columns for c in required_cols):
            return []

        # Split into chunks of ~500 samples each to simulate "flights"
        chunk_size = 500
        flights = []

        for i in range(min(n_flights, len(normal_df) // chunk_size)):
            start = i * chunk_size
            end = start + chunk_size
            chunk = normal_df.iloc[start:end][required_cols].values

            # Add realistic GPS noise to synthetic data
            chunk[:, :3] += np.random.randn(len(chunk), 3) * self.gps_noise_std
            chunk[:, 3:6] += np.random.randn(len(chunk), 3) * self.gps_noise_std * 0.1

            flights.append((f"synthetic_{i:03d}", chunk))

        print(f"  Loaded {len(flights)} synthetic normal flights")
        return flights

    def load_normal_flights(self) -> List[Tuple[str, np.ndarray]]:
        """Load normal flight segments from ALFA (label=0 only)."""
        all_files = list(self.data_dir.glob("*.csv"))

        flights = []
        total_samples = 0

        for f in all_files:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"  Warning: Could not read {f.name}: {e}")
                continue

            if 'label' not in df.columns:
                continue

            # Extract NORMAL segments only (label=0)
            normal_df = df[df['label'] == 0]
            if len(normal_df) < 50:
                continue

            # Extract GPS-IMU features
            required_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']
            if not all(c in df.columns for c in required_cols):
                continue

            features = normal_df[required_cols].values
            flights.append((f.stem, features))
            total_samples += len(features)

        print(f"Loaded {len(flights)} normal flight segments ({total_samples} samples)")
        return flights

    def inject_attack(self, data: np.ndarray, attack_type: str,
                      magnitude: float) -> Tuple[np.ndarray, np.ndarray]:
        """Inject GPS spoofing attack."""
        attacked = data.copy()
        n = len(data)
        labels = np.zeros(n)

        # Attack window: 30% to 80%
        start = int(n * 0.3)
        end = int(n * 0.8)
        labels[start:end] = 1

        # Offset in meters
        offset = magnitude * self.gps_noise_std

        if attack_type == 'bias':
            # Constant GPS offset
            direction = np.random.randn(3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            attacked[start:end, :3] += direction * offset

        elif attack_type == 'drift':
            # Slow AR(1) drift
            drift = np.zeros(end - start)
            for i in range(1, len(drift)):
                drift[i] = 0.995 * drift[i-1] + np.random.randn() * 0.01
            drift = drift / (np.std(drift) + 1e-8) * offset
            attacked[start:end, 0] += drift
            attacked[start:end, 1] += drift * 0.5

        elif attack_type == 'noise_injection':
            # Increased GPS noise
            noise = np.random.randn(end - start, 3) * offset
            attacked[start:end, :3] += noise

        elif attack_type == 'coordinated':
            # Position + velocity coordinated
            direction = np.random.randn(3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            attacked[start:end, :3] += direction * offset
            attacked[start:end, 3:6] += direction * offset * 0.01

        elif attack_type == 'intermittent':
            # Random on/off
            attack_mask = np.random.rand(end - start) < 0.2
            direction = np.random.randn(3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            attacked[start:end, :3][attack_mask] += direction * offset
            labels[start:end] = attack_mask.astype(float)

        elif attack_type == 'step':
            # Sudden step
            direction = np.random.randn(3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            attacked[start:end, :3] += direction * offset

        return attacked, labels

    def extract_features(self, data: np.ndarray, window_size: int = 20) -> np.ndarray:
        """Extract windowed features."""
        n = len(data) - window_size + 1
        if n <= 0:
            return np.zeros((1, window_size * data.shape[1]))
        features = np.zeros((n, window_size * data.shape[1]))
        for i in range(n):
            features[i] = data[i:i+window_size].flatten()
        return features

    def train_detector(self, flights: List[Tuple[str, np.ndarray]]) -> Tuple:
        """Train unsupervised Mahalanobis detector on normal flights."""
        all_features = []
        for name, data in flights:
            features = self.extract_features(data)
            all_features.append(features)

        X_train = np.vstack(all_features)
        print(f"Training on {len(X_train)} windows from {len(flights)} flights")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        mean = np.mean(X_scaled, axis=0)
        cov = np.cov(X_scaled.T) + np.eye(X_scaled.shape[1]) * 1e-4
        cov_inv = np.linalg.inv(cov)

        # Threshold at p99
        distances = np.array([np.sqrt((x - mean) @ cov_inv @ (x - mean)) for x in X_scaled[:1000]])
        threshold = np.percentile(distances, 99)

        return scaler, mean, cov_inv, threshold

    def score(self, data: np.ndarray, scaler, mean, cov_inv) -> np.ndarray:
        """Score data using Mahalanobis distance."""
        features = self.extract_features(data)
        X_scaled = scaler.transform(features)
        scores = np.array([np.sqrt((x - mean) @ cov_inv @ (x - mean)) for x in X_scaled])
        # Pad to match original length
        pad_len = len(data) - len(scores)
        return np.concatenate([np.zeros(pad_len), scores])

    def evaluate(self) -> Dict:
        """Run evaluation on ALFA + synthetic data."""
        print("=" * 70)
        print("GPS SPOOFING EVALUATION ON ALFA + SYNTHETIC FLIGHT DATA")
        print("=" * 70)

        # Load ALFA flights
        alfa_flights = self.load_normal_flights()
        print(f"  ALFA: {len(alfa_flights)} flights")

        # Load synthetic flights to balance dataset
        synthetic_flights = []
        if self.use_synthetic:
            synthetic_flights = self.load_synthetic_normal_flights(n_flights=50)
            print(f"  Synthetic: {len(synthetic_flights)} flights")

        # Combine
        flights = alfa_flights + synthetic_flights
        np.random.shuffle(flights)

        print(f"  Total: {len(flights)} flights")

        if len(flights) < 3:
            print("ERROR: Not enough normal flights")
            return {}

        attack_types = ['bias', 'drift', 'noise_injection', 'coordinated', 'intermittent', 'step']
        magnitudes = [1.0, 2.0, 5.0, 10.0, 20.0]

        # Split: 70% train, 30% test
        np.random.shuffle(flights)
        n_train = int(len(flights) * 0.7)
        train_flights = flights[:n_train]
        test_flights = flights[n_train:]

        print(f"Train: {len(train_flights)} flights, Test: {len(test_flights)} flights")

        # Train detector
        print("\nTraining detector on normal flights...")
        scaler, mean, cov_inv, threshold = self.train_detector(train_flights)
        print(f"Threshold (p99): {threshold:.4f}")

        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset': 'ALFA + Synthetic',
            'n_alfa_flights': len(alfa_flights),
            'n_synthetic_flights': len(synthetic_flights),
            'n_train_flights': len(train_flights),
            'n_test_flights': len(test_flights),
            'gps_noise_std': self.gps_noise_std,
            'by_attack': {}
        }

        print("\nEvaluating attacks...")
        print("-" * 70)
        print(f"{'Attack':<15}", end="")
        for m in magnitudes:
            print(f"{m}x".center(10), end="")
        print()
        print("-" * 70)

        for attack_type in attack_types:
            results['by_attack'][attack_type] = {}
            print(f"{attack_type:<15}", end="")

            for magnitude in magnitudes:
                all_labels = []
                all_scores = []

                for name, data in test_flights:
                    # Inject attack
                    attacked, labels = self.inject_attack(data, attack_type, magnitude)
                    scores = self.score(attacked, scaler, mean, cov_inv)

                    all_labels.extend(labels)
                    all_scores.extend(scores)

                all_labels = np.array(all_labels)
                all_scores = np.array(all_scores)

                if len(np.unique(all_labels)) > 1:
                    auroc = roc_auc_score(all_labels, all_scores)
                else:
                    auroc = 0.5

                results['by_attack'][attack_type][f'{magnitude}x'] = {
                    'auroc': float(auroc),
                    'magnitude_m': float(magnitude * self.gps_noise_std)
                }

                print(f"{auroc:>6.1%}".center(10), end="")
            print()

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY AT 10x (15m offset)")
        print("=" * 70)

        detectable = 0
        for attack_type in attack_types:
            auroc = results['by_attack'][attack_type].get('10.0x', {}).get('auroc', 0.5)
            status = "DETECTABLE" if auroc > 0.70 else "MARGINAL" if auroc > 0.55 else "UNDETECTABLE"
            if auroc > 0.55:
                detectable += 1
            print(f"  {attack_type:<15}: {auroc:>6.1%} - {status}")

        results['summary'] = {
            'detectable_count': detectable,
            'total_attacks': 6
        }

        # Save
        output_path = Path(__file__).parent.parent / "results" / "alfa_results.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data" / "alfa" / "preprocessed"
    evaluator = ALFAEvaluator(str(data_dir))
    evaluator.evaluate()
