#!/usr/bin/env python3
"""
GPS Spoofing Evaluation on REAL ALFA Flight Data

Uses real UAV flight data (ALFA dataset) as base, injects GPS spoofing attacks,
and evaluates detection with UNSUPERVISED approach (train on normal only).

This gives REALISTIC numbers for publication.
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

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ALFAGPSSpoofingEvaluator:
    """Evaluate GPS spoofing detection on real ALFA flight data."""

    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.seed = seed
        np.random.seed(seed)

        # GPS noise from real sensors
        self.gps_noise_std = 1.5  # meters (realistic)

    def load_normal_flights(self) -> List[np.ndarray]:
        """Load normal segments from ALL ALFA flights (label=0)."""
        all_files = list(self.data_dir.glob("*.csv"))

        flights = []
        total_samples = 0

        for f in all_files:
            df = pd.read_csv(f)
            if 'label' not in df.columns:
                continue

            # Extract NORMAL segments (label=0)
            normal_df = df[df['label'] == 0]
            if len(normal_df) < 100:
                continue

            # Extract GPS-IMU features
            features = normal_df[['x', 'y', 'z', 'vx', 'vy', 'vz',
                                  'phi', 'theta', 'psi', 'p', 'q', 'r']].values
            flights.append(features)
            total_samples += len(features)

        print(f"Loaded {len(flights)} normal flight segments ({total_samples} samples)")
        return flights

    def inject_attack(self, flight: np.ndarray, attack_type: str,
                      magnitude: float) -> Tuple[np.ndarray, np.ndarray]:
        """Inject GPS spoofing attack on real flight data."""
        attacked = flight.copy()
        n = len(flight)
        labels = np.zeros(n)

        # Attack starts at 30%, lasts 50% of flight
        start = int(n * 0.3)
        end = int(n * 0.8)
        labels[start:end] = 1

        # Scale magnitude by GPS noise
        offset = magnitude * self.gps_noise_std

        if attack_type == 'bias':
            # Constant GPS offset (undetectable by physics)
            attacked[start:end, 0] += offset  # x position
            attacked[start:end, 1] += offset * 0.5  # y position

        elif attack_type == 'drift':
            # Slow AR(1) drift (physics-consistent)
            drift = np.zeros(end - start)
            for i in range(1, len(drift)):
                drift[i] = 0.995 * drift[i-1] + np.random.randn() * 0.01
            drift = drift / (np.std(drift) + 1e-8) * offset
            attacked[start:end, 0] += drift
            attacked[start:end, 1] += drift * 0.5

        elif attack_type == 'noise_injection':
            # Increased GPS noise (variance-breaking)
            noise = np.random.randn(end - start, 3) * offset
            attacked[start:end, :3] += noise

        elif attack_type == 'coordinated':
            # Position + velocity coordinated (physics-consistent)
            attacked[start:end, 0] += offset
            attacked[start:end, 3] += offset * 0.01  # Consistent velocity

        elif attack_type == 'intermittent':
            # Random on/off attacks
            attack_mask = np.random.rand(end - start) < 0.2
            attacked[start:end, :3][attack_mask] += offset
            labels[start:end] = attack_mask.astype(float)

        elif attack_type == 'step':
            # Sudden step change
            attacked[start:end, :3] += offset

        return attacked, labels

    def extract_windows(self, data: np.ndarray, window_size: int = 20) -> np.ndarray:
        """Extract sliding windows for detection."""
        n_windows = len(data) - window_size + 1
        windows = np.zeros((n_windows, window_size * data.shape[1]))

        for i in range(n_windows):
            windows[i] = data[i:i+window_size].flatten()

        return windows

    def train_detector(self, normal_flights: List[np.ndarray]) -> Tuple:
        """Train unsupervised detector on normal flights only."""
        # Concatenate all normal flights
        all_normal = []
        for flight in normal_flights:
            windows = self.extract_windows(flight)
            all_normal.append(windows)

        X_train = np.vstack(all_normal)

        # Fit scaler on normal data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Compute normal statistics for anomaly detection
        mean = np.mean(X_train_scaled, axis=0)
        cov = np.cov(X_train_scaled.T) + np.eye(X_train_scaled.shape[1]) * 1e-6
        cov_inv = np.linalg.inv(cov)

        # Threshold at 99th percentile of training Mahalanobis distances
        distances = []
        for x in X_train_scaled:
            d = np.sqrt((x - mean) @ cov_inv @ (x - mean))
            distances.append(d)
        threshold = np.percentile(distances, 99)

        return scaler, mean, cov_inv, threshold

    def score_flight(self, flight: np.ndarray, scaler, mean, cov_inv) -> np.ndarray:
        """Score a flight using Mahalanobis distance."""
        windows = self.extract_windows(flight)
        X_scaled = scaler.transform(windows)

        scores = []
        for x in X_scaled:
            d = np.sqrt((x - mean) @ cov_inv @ (x - mean))
            scores.append(d)

        # Pad to match original length
        pad_len = len(flight) - len(scores)
        scores = np.concatenate([np.zeros(pad_len), np.array(scores)])

        return scores

    def evaluate(self) -> Dict:
        """Run full evaluation."""
        print("=" * 70)
        print("GPS SPOOFING EVALUATION ON REAL ALFA FLIGHT DATA")
        print("=" * 70)

        # Load normal flights
        normal_flights = self.load_normal_flights()

        if len(normal_flights) < 3:
            print("ERROR: Not enough normal flights for evaluation")
            return {}

        # Split: 70% train, 30% test
        n_train = int(len(normal_flights) * 0.7)
        train_flights = normal_flights[:n_train]
        test_flights = normal_flights[n_train:]

        print(f"Train flights: {len(train_flights)}, Test flights: {len(test_flights)}")

        # Train detector on normal data only (UNSUPERVISED)
        print("\nTraining unsupervised detector on normal flights...")
        scaler, mean, cov_inv, threshold = self.train_detector(train_flights)
        print(f"Threshold (p99): {threshold:.4f}")

        # Attack types and magnitudes
        attack_types = ['bias', 'drift', 'noise_injection', 'coordinated',
                       'intermittent', 'step']
        magnitudes = [1.0, 2.0, 5.0, 10.0]  # multiples of GPS noise

        results = {
            'timestamp': datetime.now().isoformat(),
            'n_train_flights': len(train_flights),
            'n_test_flights': len(test_flights),
            'gps_noise_std': self.gps_noise_std,
            'by_attack': {},
            'by_magnitude': {},
        }

        print("\nEvaluating attacks...")
        print("-" * 70)

        for attack_type in attack_types:
            results['by_attack'][attack_type] = {}

            for magnitude in magnitudes:
                all_labels = []
                all_scores = []

                for flight in test_flights:
                    # Inject attack
                    attacked, labels = self.inject_attack(flight, attack_type, magnitude)

                    # Score attacked flight
                    scores = self.score_flight(attacked, scaler, mean, cov_inv)

                    all_labels.extend(labels)
                    all_scores.extend(scores)

                # Compute AUROC
                all_labels = np.array(all_labels)
                all_scores = np.array(all_scores)

                if len(np.unique(all_labels)) > 1:
                    auroc = roc_auc_score(all_labels, all_scores)
                else:
                    auroc = 0.5

                results['by_attack'][attack_type][f'{magnitude}x'] = {
                    'auroc': float(auroc),
                    'magnitude_m': float(magnitude * self.gps_noise_std),
                }

                print(f"  {attack_type:15s} @ {magnitude:4.1f}x ({magnitude*self.gps_noise_std:5.1f}m): "
                      f"AUROC = {auroc:.1%}")

        # Summary by attack type
        print("\n" + "=" * 70)
        print("SUMMARY BY ATTACK TYPE (at 10x = 15m)")
        print("=" * 70)

        summary = {}
        for attack_type in attack_types:
            auroc_10x = results['by_attack'][attack_type].get('10.0x', {}).get('auroc', 0.5)
            status = "DETECTABLE" if auroc_10x > 0.70 else "UNDETECTABLE"
            summary[attack_type] = {'auroc': auroc_10x, 'status': status}
            print(f"  {attack_type:15s}: {auroc_10x:.1%} - {status}")

        results['summary'] = summary

        # Count detectable
        n_detectable = sum(1 for v in summary.values() if v['status'] == 'DETECTABLE')
        print(f"\n  DETECTABLE: {n_detectable}/6 attacks")

        # Save results
        output_path = Path(__file__).parent.parent / "results" / "alfa_gps_results.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data" / "alfa" / "preprocessed"
    evaluator = ALFAGPSSpoofingEvaluator(str(data_dir))
    results = evaluator.evaluate()
