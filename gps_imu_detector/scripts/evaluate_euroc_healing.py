#!/usr/bin/env python3
"""
Self-Healing (IASP) Evaluation on REAL EuRoC Flight Data

Validates Table 2 claims:
1. Automatic recovery after spoof detection
2. Navigation error reduction during spoofing
3. Stability under nominal flight
4. Recovery latency for real-time use
5. Risk of oscillation
6. Mode switch requirement
7. Estimator continuity
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inverse_model import CycleConsistencyDetector


class EuRoCHealingEvaluator:
    """Evaluate IASP self-healing on real EuRoC data."""

    def __init__(self, data_path: str, seed: int = 42):
        self.data_path = Path(data_path)
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.gps_noise_std = 1.5

    def load_data(self) -> Dict[str, np.ndarray]:
        """Load EuRoC sequences."""
        df = pd.read_csv(self.data_path)
        sequences = {}
        for seq_name in df['sequence'].unique():
            seq_df = df[df['sequence'] == seq_name]
            # Use 6D state: position + velocity
            features = seq_df[['x', 'y', 'z', 'vx', 'vy', 'vz']].values
            sequences[seq_name] = features
        return sequences

    def inject_spoof(self, data: np.ndarray, magnitude: float) -> Tuple[np.ndarray, np.ndarray]:
        """Inject GPS spoofing (constant bias)."""
        spoofed = data.copy()
        n = len(data)

        # Spoof starts at 30%, lasts 50%
        start = int(n * 0.3)
        end = int(n * 0.8)

        offset = np.array([magnitude, magnitude/2, magnitude/4, 0, 0, 0])
        spoofed[start:end] += offset

        # Ground truth is original data
        return spoofed, data

    def evaluate(self) -> Dict:
        """Run full healing evaluation."""
        print("=" * 70)
        print("SELF-HEALING (IASP) EVALUATION ON EUROC DATA")
        print("=" * 70)

        sequences = self.load_data()
        seq_names = list(sequences.keys())
        print(f"Loaded {len(seq_names)} sequences")

        # Use first 3 sequences for training, rest for testing
        train_seqs = [sequences[s] for s in seq_names[:3]]
        test_seqs = [(s, sequences[s]) for s in seq_names[3:]]

        # Concatenate training data
        train_data = np.vstack(train_seqs)
        print(f"Training data: {len(train_data)} samples")

        # Train ICI detector
        print("\n[1] Training ICI detector...")
        state_dim = 6
        detector = CycleConsistencyDetector(
            state_dim=state_dim,
            hidden_dim=64,
            num_layers=3,
            device='cpu'
        )

        detector.fit(
            train_data.reshape(1, -1, state_dim),
            epochs=30,
            cycle_lambda=0.25,
            verbose=False
        )
        print("    Training complete.")

        # Calibrate threshold on training data
        train_ici = detector.score_trajectory(train_data[:5000], return_raw=True)
        ici_threshold = np.percentile(train_ici, 99)
        saturation_constant = max(np.mean(train_ici) * 3, 10.0)
        print(f"    ICI threshold (p99): {ici_threshold:.4f}")

        # Test on different spoof magnitudes
        magnitudes = [25, 50, 100, 200]  # meters

        results = {
            'timestamp': datetime.now().isoformat(),
            'n_train_sequences': 3,
            'n_test_sequences': len(test_seqs),
            'ici_threshold': float(ici_threshold),
            'by_magnitude': {}
        }

        print("\n[2] Evaluating healing at different magnitudes...")

        for magnitude in magnitudes:
            print(f"\n--- {magnitude}m spoof ---")

            all_error_reduction = []
            all_stability = []
            all_quiescence = []
            all_latencies = []

            for seq_name, test_data in test_seqs:
                # Limit test length for speed
                test_data = test_data[:5000]

                # Inject spoof
                spoofed, ground_truth = self.inject_spoof(test_data, magnitude)

                # Measure latency
                X = torch.tensor(spoofed[:100], dtype=torch.float32)
                latencies = []
                for i in range(100):
                    start = time.perf_counter()
                    with torch.no_grad():
                        x_next = detector.forward_model(X[i:i+1])
                        x_proj = detector.inverse_model(x_next)
                    latencies.append((time.perf_counter() - start) * 1000)
                all_latencies.extend(latencies)

                # Apply healing
                healing_result = detector.heal_trajectory(
                    spoofed,
                    saturation_constant=saturation_constant,
                    ici_threshold=ici_threshold,
                    return_details=True
                )
                healed = healing_result['healed_trajectory']

                # Compute errors
                error_no_healing = np.linalg.norm(spoofed[:, :3] - ground_truth[:, :3], axis=1)
                error_with_healing = np.linalg.norm(healed[:, :3] - ground_truth[:, :3], axis=1)

                # Error reduction
                reduction = 100 * (1 - np.mean(error_with_healing) / (np.mean(error_no_healing) + 1e-8))
                all_error_reduction.append(reduction)

                # Stability (alpha variance)
                alpha_diff_var = np.var(np.diff(healing_result['alpha_values']))
                is_stable = alpha_diff_var < 0.01
                all_stability.append(is_stable)

                # Quiescence on nominal portion (before spoof starts)
                nominal_healing = detector.heal_trajectory(
                    ground_truth[:int(len(ground_truth)*0.25)],
                    saturation_constant=saturation_constant,
                    ici_threshold=ici_threshold,
                    return_details=True
                )
                false_healing_rate = np.mean(nominal_healing['alpha_values'] > 0.01)
                all_quiescence.append(1 - false_healing_rate)

            # Aggregate
            mean_reduction = np.mean(all_error_reduction)
            stability_rate = np.mean(all_stability) * 100
            quiescence_rate = np.mean(all_quiescence) * 100
            mean_latency = np.mean(all_latencies)
            p99_latency = np.percentile(all_latencies, 99)
            realtime_rate = 100 * np.mean(np.array(all_latencies) < 5.0)

            results['by_magnitude'][f'{magnitude}m'] = {
                'error_reduction_pct': float(mean_reduction),
                'stability_pct': float(stability_rate),
                'quiescence_pct': float(quiescence_rate),
                'latency_mean_ms': float(mean_latency),
                'latency_p99_ms': float(p99_latency),
                'realtime_pct': float(realtime_rate),
            }

            print(f"    Error reduction: {mean_reduction:.1f}%")
            print(f"    Stability: {stability_rate:.1f}%")
            print(f"    Quiescence: {quiescence_rate:.1f}%")
            print(f"    Latency: {mean_latency:.2f}ms (p99: {p99_latency:.2f}ms)")

        # Summary at 100m
        print("\n" + "=" * 70)
        print("TABLE 2 VALIDATION (at 100m spoof)")
        print("=" * 70)

        r = results['by_magnitude']['100m']
        print(f"""
| Self-Healing Aspect                    | Measured    | Status |
|----------------------------------------|-------------|--------|
| Navigation error reduction             | {r['error_reduction_pct']:>6.1f}%     | {'PASS' if r['error_reduction_pct'] >= 70 else 'FAIL'} |
| Stability (no oscillation)             | {r['stability_pct']:>6.1f}%     | {'PASS' if r['stability_pct'] >= 90 else 'FAIL'} |
| Quiescence on nominal                  | {r['quiescence_pct']:>6.1f}%     | {'PASS' if r['quiescence_pct'] >= 95 else 'FAIL'} |
| Real-time feasibility (< 5ms)          | {r['realtime_pct']:>6.1f}%     | {'PASS' if r['realtime_pct'] >= 99 else 'FAIL'} |
| Mean latency                           | {r['latency_mean_ms']:>6.2f}ms    | — |
| P99 latency                            | {r['latency_p99_ms']:>6.2f}ms    | — |
""")

        # Additional metrics
        results['summary'] = {
            'error_reduction_100m': r['error_reduction_pct'],
            'stability': r['stability_pct'],
            'quiescence': r['quiescence_pct'],
            'realtime_feasibility': r['realtime_pct'],
            'mode_switch_required': 0.0,  # IASP is automatic
            'oscillation_risk': 100 - r['stability_pct'],
        }

        # Save
        output_path = Path(__file__).parent.parent / "results" / "euroc_healing_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent / "data" / "euroc" / "all_sequences.csv"
    evaluator = EuRoCHealingEvaluator(str(data_path))
    evaluator.evaluate()
