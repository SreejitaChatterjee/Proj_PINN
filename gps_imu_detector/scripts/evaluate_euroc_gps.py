#!/usr/bin/env python3
"""
GPS Spoofing Evaluation on REAL EuRoC Flight Data

UNSUPERVISED approach:
- Train on normal EuRoC flights (no attacks)
- Inject GPS spoofing attacks
- Evaluate detection

This gives REALISTIC, DEPLOYABLE numbers.
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


class EuRoCGPSSpoofingEvaluator:
    """Evaluate GPS spoofing detection on real EuRoC flight data."""

    def __init__(self, data_path: str, seed: int = 42):
        self.data_path = Path(data_path)
        self.seed = seed
        np.random.seed(seed)
        self.gps_noise_std = 1.5  # meters (realistic)

    def load_data(self) -> Dict[str, np.ndarray]:
        """Load EuRoC sequences."""
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} samples")

        sequences = {}
        for seq_name in df['sequence'].unique():
            seq_df = df[df['sequence'] == seq_name]
            features = seq_df[['x', 'y', 'z', 'vx', 'vy', 'vz',
                              'roll', 'pitch', 'yaw', 'p', 'q', 'r']].values
            sequences[seq_name] = features
            print(f"  {seq_name}: {len(features)} samples")

        return sequences

    def inject_attack(self, data: np.ndarray, attack_type: str,
                      magnitude: float) -> Tuple[np.ndarray, np.ndarray]:
        """Inject GPS spoofing attack."""
        attacked = data.copy()
        n = len(data)
        labels = np.zeros(n)

        start = int(n * 0.3)
        end = int(n * 0.8)
        labels[start:end] = 1

        offset = magnitude * self.gps_noise_std

        if attack_type == 'bias':
            attacked[start:end, :3] += offset

        elif attack_type == 'drift':
            drift = np.zeros(end - start)
            for i in range(1, len(drift)):
                drift[i] = 0.995 * drift[i-1] + np.random.randn() * 0.01
            drift = drift / (np.std(drift) + 1e-8) * offset
            attacked[start:end, 0] += drift
            attacked[start:end, 1] += drift * 0.5

        elif attack_type == 'noise_injection':
            noise = np.random.randn(end - start, 3) * offset
            attacked[start:end, :3] += noise

        elif attack_type == 'coordinated':
            # Position + velocity coordinated
            attacked[start:end, :3] += offset
            attacked[start:end, 3:6] += offset * 0.01

        elif attack_type == 'intermittent':
            attack_mask = np.random.rand(end - start) < 0.2
            attacked[start:end, :3][attack_mask] += offset
            labels[start:end] = attack_mask.astype(float)

        elif attack_type == 'step':
            attacked[start:end, :3] += offset

        return attacked, labels

    def extract_features(self, data: np.ndarray, window_size: int = 20) -> np.ndarray:
        """Extract windowed features."""
        n = len(data) - window_size + 1
        features = np.zeros((n, window_size * data.shape[1]))
        for i in range(n):
            features[i] = data[i:i+window_size].flatten()
        return features

    def train_detector(self, train_sequences: Dict[str, np.ndarray]) -> Tuple:
        """Train unsupervised Mahalanobis detector."""
        all_features = []
        for seq in train_sequences.values():
            features = self.extract_features(seq)
            all_features.append(features)

        X_train = np.vstack(all_features)
        print(f"Training on {len(X_train)} windows from {len(train_sequences)} sequences")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        mean = np.mean(X_scaled, axis=0)
        cov = np.cov(X_scaled.T) + np.eye(X_scaled.shape[1]) * 1e-4
        cov_inv = np.linalg.inv(cov)

        # Threshold at p99
        distances = np.array([np.sqrt((x - mean) @ cov_inv @ (x - mean)) for x in X_scaled])
        threshold = np.percentile(distances, 99)

        return scaler, mean, cov_inv, threshold

    def score(self, data: np.ndarray, scaler, mean, cov_inv) -> np.ndarray:
        """Score data using Mahalanobis distance."""
        features = self.extract_features(data)
        X_scaled = scaler.transform(features)
        scores = np.array([np.sqrt((x - mean) @ cov_inv @ (x - mean)) for x in X_scaled])
        # Pad
        return np.concatenate([np.zeros(len(data) - len(scores)), scores])

    def evaluate(self) -> Dict:
        """Run LOSO-CV evaluation."""
        print("=" * 70)
        print("GPS SPOOFING EVALUATION ON EUROC DATA (UNSUPERVISED)")
        print("=" * 70)

        sequences = self.load_data()
        seq_names = list(sequences.keys())

        attack_types = ['bias', 'drift', 'noise_injection', 'coordinated', 'intermittent', 'step']
        magnitudes = [1.0, 2.0, 5.0, 10.0, 20.0]

        # Use LOSO-CV: train on N-1 sequences, test on 1
        all_results = {at: {f'{m}x': [] for m in magnitudes} for at in attack_types}

        print(f"\nRunning LOSO-CV with {len(seq_names)} folds...")

        for test_seq in seq_names:
            train_seqs = {k: v for k, v in sequences.items() if k != test_seq}
            test_data = sequences[test_seq]

            # Train on normal data only
            scaler, mean, cov_inv, threshold = self.train_detector(train_seqs)

            for attack_type in attack_types:
                for magnitude in magnitudes:
                    attacked, labels = self.inject_attack(test_data, attack_type, magnitude)
                    scores = self.score(attacked, scaler, mean, cov_inv)

                    if len(np.unique(labels)) > 1:
                        auroc = roc_auc_score(labels, scores)
                    else:
                        auroc = 0.5

                    all_results[attack_type][f'{magnitude}x'].append(auroc)

        # Aggregate results
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'LOSO-CV',
            'n_folds': len(seq_names),
            'gps_noise_std': self.gps_noise_std,
            'by_attack': {}
        }

        print("\n" + "=" * 70)
        print("RESULTS (Mean AUROC across LOSO folds)")
        print("=" * 70)
        print(f"\n{'Attack':<15}", end="")
        for m in magnitudes:
            print(f"{m}x".center(10), end="")
        print()
        print("-" * 65)

        for attack_type in attack_types:
            results['by_attack'][attack_type] = {}
            print(f"{attack_type:<15}", end="")

            for magnitude in magnitudes:
                aurocs = all_results[attack_type][f'{magnitude}x']
                mean_auroc = np.mean(aurocs)
                std_auroc = np.std(aurocs)
                results['by_attack'][attack_type][f'{magnitude}x'] = {
                    'auroc_mean': float(mean_auroc),
                    'auroc_std': float(std_auroc),
                    'auroc_values': [float(a) for a in aurocs]
                }
                print(f"{mean_auroc:>6.1%}".center(10), end="")
            print()

        # Summary at 10x
        print("\n" + "=" * 70)
        print("SUMMARY AT 10x (15m offset)")
        print("=" * 70)

        detectable = 0
        for attack_type in attack_types:
            auroc = results['by_attack'][attack_type]['10.0x']['auroc_mean']
            status = "DETECTABLE" if auroc > 0.70 else "UNDETECTABLE"
            if auroc > 0.70:
                detectable += 1
            print(f"  {attack_type:<15}: {auroc:>6.1%} - {status}")

        print(f"\n  DETECTABLE: {detectable}/6 attacks")

        results['summary'] = {
            'detectable_count': detectable,
            'total_attacks': 6
        }

        # Save
        output_path = Path(__file__).parent.parent / "results" / "euroc_gps_results.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent / "data" / "euroc" / "all_sequences.csv"
    evaluator = EuRoCGPSSpoofingEvaluator(str(data_path))
    evaluator.evaluate()
