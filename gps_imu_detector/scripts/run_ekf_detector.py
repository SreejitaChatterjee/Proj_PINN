#!/usr/bin/env python3
"""
Standalone EKF-NIS Detector Evaluation.

This script evaluates the EKF-based anomaly detector independently,
providing baseline metrics for the hybrid fusion.

Usage:
    python scripts/run_ekf_detector.py --output results/ekf_results.json

Targets:
    - EKF AUROC: 0.65-0.75
    - Worst-case Recall@5%FPR: >= 25-35%
    - CPU cost: < 1 ms per sample
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ekf import NISAnomalyDetector


def generate_synthetic_trajectory(
    n_samples: int = 1000,
    dt: float = 0.01,
    velocity: float = 5.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic GPS trajectory.

    Simulates a UAV flying in a figure-8 pattern.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n_samples) * dt
    omega = 0.5  # rad/s

    # Figure-8 trajectory
    x = velocity * np.sin(omega * t)
    y = velocity * np.sin(2 * omega * t) / 2
    z = 10 + 0.5 * np.sin(0.3 * omega * t)  # Slight altitude variation

    # Add GPS noise
    positions = np.stack([x, y, z], axis=1)
    positions += rng.normal(0, noise_std, positions.shape)

    return positions


def inject_spoofing_attack(
    trajectory: np.ndarray,
    attack_type: str = 'offset',
    magnitude: float = 10.0,
    start_frac: float = 0.3,
    duration_frac: float = 0.5,
    seed: int = 42,
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject GPS spoofing attack into trajectory.

    Returns:
        (spoofed_trajectory, attack_mask): Spoofed data and boolean mask
    """
    rng = np.random.default_rng(seed)
    n = len(trajectory)
    spoofed = trajectory.copy()

    start_idx = int(n * start_frac)
    end_idx = int(n * (start_frac + duration_frac))
    end_idx = min(end_idx, n)

    mask = np.zeros(n, dtype=bool)
    mask[start_idx:end_idx] = True

    if attack_type == 'offset':
        # Constant offset
        offset = rng.uniform(-1, 1, 3)
        offset = offset / np.linalg.norm(offset) * magnitude
        spoofed[start_idx:end_idx] += offset

    elif attack_type == 'drift':
        # Gradual drift
        drift_samples = end_idx - start_idx
        drift = np.linspace(0, 1, drift_samples)[:, None]
        direction = rng.uniform(-1, 1, 3)
        direction = direction / np.linalg.norm(direction) * magnitude
        spoofed[start_idx:end_idx] += drift * direction

    elif attack_type == 'ramp':
        # Ramp attack (consistent with ICI experiments)
        drift_samples = end_idx - start_idx
        ramp = np.arange(drift_samples)[:, None] * dt * magnitude / 10
        spoofed[start_idx:end_idx, :2] += ramp[:, :2]  # X-Y only

    elif attack_type == 'oscillation':
        # Oscillating spoofing
        t = np.arange(end_idx - start_idx)
        osc = magnitude * np.sin(2 * np.pi * t / 50)[:, None]
        spoofed[start_idx:end_idx] += osc * np.array([1, 1, 0])

    elif attack_type == 'jump':
        # Sudden jump
        spoofed[start_idx:end_idx] += magnitude * np.array([1, 0, 0])

    return spoofed, mask


def evaluate_on_attack(
    detector: NISAnomalyDetector,
    normal_traj: np.ndarray,
    attack_type: str,
    magnitude: float,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate detector on single attack type."""
    # Create spoofed trajectory
    spoofed, mask = inject_spoofing_attack(
        normal_traj,
        attack_type=attack_type,
        magnitude=magnitude,
        seed=seed,
    )

    # Process normal
    detector.initialize(normal_traj[0])
    normal_scores = []
    for pos in normal_traj[1:]:
        _, score, _ = detector.step(pos)
        normal_scores.append(score)

    # Process spoofed
    detector.initialize(spoofed[0])
    spoofed_scores = []
    for pos in spoofed[1:]:
        _, score, _ = detector.step(pos)
        spoofed_scores.append(score)

    # Only compare during attack window
    attack_mask = mask[1:]  # Skip first sample
    attack_scores = np.array(spoofed_scores)[attack_mask]
    normal_scores = np.array(normal_scores)

    if len(attack_scores) == 0:
        return {'auroc': 0.5, 'recall_5pct': 0.0}

    # Compute AUROC
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(attack_scores))
    ])
    y_scores = np.concatenate([normal_scores, attack_scores])

    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5

    # Recall at 5% FPR
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    recall_5pct = float(tpr[np.searchsorted(fpr, 0.05)])

    return {
        'auroc': float(auroc),
        'recall_5pct': recall_5pct,
        'mean_attack_score': float(np.mean(attack_scores)),
        'mean_normal_score': float(np.mean(normal_scores)),
    }


def benchmark_latency(
    detector: NISAnomalyDetector,
    n_samples: int = 10000,
) -> Dict[str, float]:
    """Benchmark EKF detector latency."""
    # Generate test data
    positions = generate_synthetic_trajectory(n_samples, seed=999)

    detector.initialize(positions[0])

    # Warmup
    for pos in positions[1:100]:
        detector.step(pos)

    # Timed run
    times = []
    for pos in positions[100:]:
        start = time.perf_counter()
        detector.step(pos)
        times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # Convert to ms

    return {
        'mean_ms': float(np.mean(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'max_ms': float(np.max(times)),
    }


def main():
    parser = argparse.ArgumentParser(description='EKF-NIS Detector Evaluation')
    parser.add_argument('--output', type=Path, default='results/ekf_results.json')
    parser.add_argument('--n-trajectories', type=int, default=10)
    parser.add_argument('--trajectory-length', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("EKF-NIS DETECTOR EVALUATION")
    print("=" * 60)

    # Initialize detector
    detector = NISAnomalyDetector(
        dt=0.01,
        process_noise_pos=0.01,
        process_noise_vel=0.1,
        measurement_noise=1.0,
    )

    # Generate calibration trajectories
    print("\n[1] Generating calibration data...")
    cal_trajectories = [
        generate_synthetic_trajectory(
            n_samples=args.trajectory_length,
            seed=args.seed + i,
        )
        for i in range(args.n_trajectories)
    ]

    # Calibrate
    print("[2] Calibrating detector...")
    cal_stats = detector.calibrate(cal_trajectories, target_fpr=0.05)
    print(f"    Mean NIS: {cal_stats['mean_nis']:.3f}")
    print(f"    Std NIS:  {cal_stats['std_nis']:.3f}")
    print(f"    Threshold (5% FPR): {cal_stats['threshold']:.3f}")

    # Evaluate on different attack types
    print("\n[3] Evaluating on attacks...")

    attack_types = ['offset', 'drift', 'ramp', 'oscillation', 'jump']
    magnitudes = [5.0, 10.0, 25.0, 50.0, 100.0]

    results_by_attack = {}
    all_aurocs = []

    for attack in attack_types:
        results_by_attack[attack] = {}
        for mag in magnitudes:
            # Generate fresh test trajectory
            test_traj = generate_synthetic_trajectory(
                n_samples=args.trajectory_length,
                seed=args.seed + 1000,
            )

            metrics = evaluate_on_attack(
                detector,
                test_traj,
                attack_type=attack,
                magnitude=mag,
                seed=args.seed,
            )

            results_by_attack[attack][str(mag)] = metrics
            all_aurocs.append(metrics['auroc'])

            print(f"    {attack:12} @ {mag:5}m: AUROC={metrics['auroc']:.3f}, Recall@5%={metrics['recall_5pct']:.3f}")

    # Compute worst case
    worst_auroc = min(all_aurocs)
    mean_auroc = np.mean(all_aurocs)

    print(f"\n    Overall AUROC: {mean_auroc:.3f} (worst: {worst_auroc:.3f})")

    # Latency benchmark
    print("\n[4] Latency benchmark...")
    latency = benchmark_latency(detector)
    print(f"    Mean: {latency['mean_ms']:.3f} ms")
    print(f"    P95:  {latency['p95_ms']:.3f} ms")
    print(f"    P99:  {latency['p99_ms']:.3f} ms")

    # Target check
    print("\n[5] Target Check...")
    targets = {
        'AUROC >= 0.65': bool(mean_auroc >= 0.65),
        'Worst AUROC >= 0.50': bool(worst_auroc >= 0.50),
        'P95 latency < 1ms': bool(latency['p95_ms'] < 1.0),
    }

    for target, passed in targets.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {target}")

    # Save results
    results = {
        'calibration': cal_stats,
        'by_attack': results_by_attack,
        'summary': {
            'mean_auroc': mean_auroc,
            'worst_auroc': worst_auroc,
            'n_attacks': len(attack_types) * len(magnitudes),
        },
        'latency': latency,
        'targets': targets,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"EKF-NIS Mean AUROC:    {mean_auroc:.3f}")
    print(f"EKF-NIS Worst AUROC:   {worst_auroc:.3f}")
    print(f"Latency (P95):         {latency['p95_ms']:.3f} ms")

    all_passed = all(targets.values())
    print(f"\nAll targets: {'PASS' if all_passed else 'FAIL'}")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
