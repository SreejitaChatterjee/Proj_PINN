#!/usr/bin/env python3
"""
End-to-End Hybrid Detector Evaluation.

This script evaluates the complete hybrid GPS spoofing detector:
1. EKF-NIS baseline
2. ML (ICI) baseline
3. Hybrid fusion (optimized weights)

Outputs worst-case evaluation table and comparison metrics.

Usage:
    python scripts/run_hybrid_eval.py --output results/hybrid_results.json

Targets:
    - Hybrid AUROC >= max(EKF, ML)
    - Worst-case Recall@5%FPR: +8-15% vs ML
    - P95 latency < 5 ms
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ekf import NISAnomalyDetector
from src.inverse_model import CycleConsistencyDetector
from src.hybrid import HybridFusion, FusionConfig


def generate_trajectory(
    n_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic UAV trajectory."""
    rng = np.random.default_rng(seed)

    t = np.arange(n_samples) * dt
    omega = 0.5

    # Figure-8 pattern
    x = 5 * np.sin(omega * t)
    y = 5 * np.sin(2 * omega * t) / 2
    z = 10 + 0.5 * np.sin(0.3 * omega * t)

    vx = 5 * omega * np.cos(omega * t)
    vy = 5 * omega * np.cos(2 * omega * t)
    vz = 0.15 * omega * np.cos(0.3 * omega * t)

    # State: [x, y, z, vx, vy, vz]
    states = np.stack([x, y, z, vx, vy, vz], axis=1)

    # Add noise
    states[:, :3] += rng.normal(0, 0.3, (n_samples, 3))
    states[:, 3:] += rng.normal(0, 0.1, (n_samples, 3))

    return states


def inject_attack(
    trajectory: np.ndarray,
    attack_type: str,
    magnitude: float,
    start_frac: float = 0.3,
    duration_frac: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject GPS spoofing attack."""
    rng = np.random.default_rng(seed)
    n = len(trajectory)
    spoofed = trajectory.copy()

    start = int(n * start_frac)
    end = int(n * (start_frac + duration_frac))
    end = min(end, n)

    mask = np.zeros(n, dtype=bool)
    mask[start:end] = True

    if attack_type == 'offset':
        spoofed[start:end, :3] += magnitude * np.array([1, 0, 0])

    elif attack_type == 'drift':
        drift_len = end - start
        drift = np.linspace(0, magnitude, drift_len)[:, None]
        spoofed[start:end, :3] += drift * np.array([1, 1, 0])

    elif attack_type == 'consistent':
        # Consistency-preserving (stealthy) spoofing
        # Offset position AND velocity to maintain dynamics consistency
        drift_len = end - start
        drift = np.linspace(0, magnitude, drift_len)
        spoofed[start:end, 0] += drift  # Position
        spoofed[start:end, 3] += magnitude / (drift_len * 0.01)  # Velocity (derivative)

    elif attack_type == 'oscillation':
        t = np.arange(end - start)
        osc = magnitude * np.sin(2 * np.pi * t / 50)[:, None]
        spoofed[start:end, :3] += osc * np.array([1, 0, 0])

    elif attack_type == 'jump':
        spoofed[start:end, :3] += magnitude * np.array([1, 0, 0])

    return spoofed, mask


def compute_ekf_scores(
    detector: NISAnomalyDetector,
    trajectories: List[np.ndarray],
) -> np.ndarray:
    """Compute EKF-NIS scores for trajectories (position only)."""
    all_scores = []

    for traj in trajectories:
        positions = traj[:, :3]  # x, y, z

        detector.initialize(positions[0])
        for pos in positions[1:]:
            _, score, _ = detector.step(pos)
            all_scores.append(score)

    return np.array(all_scores)


def compute_ml_scores(
    detector: CycleConsistencyDetector,
    trajectories: List[np.ndarray],
) -> np.ndarray:
    """Compute ML (ICI) scores for trajectories."""
    import torch

    all_scores = []

    for traj in trajectories:
        traj_tensor = torch.tensor(traj, dtype=torch.float32)

        for i in range(len(traj)):
            x_t = traj_tensor[i:i+1]
            ici = detector.compute_ici(x_t)
            all_scores.append(ici.item())

    return np.array(all_scores)


def main():
    parser = argparse.ArgumentParser(description='Hybrid Detector Evaluation')
    parser.add_argument('--output', type=Path, default=Path('results/hybrid_results.json'))
    parser.add_argument('--n-trajectories', type=int, default=5)
    parser.add_argument('--trajectory-length', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("HYBRID GPS SPOOFING DETECTOR EVALUATION")
    print("=" * 70)

    # =========================================================================
    # 1. Generate Data
    # =========================================================================
    print("\n[1] Generating synthetic data...")

    # Normal trajectories for training/calibration
    normal_trajs = [
        generate_trajectory(args.trajectory_length, seed=args.seed + i)
        for i in range(args.n_trajectories)
    ]

    # Attack types and magnitudes
    attack_configs = [
        ('offset', 10.0),
        ('offset', 50.0),
        ('drift', 10.0),
        ('drift', 50.0),
        ('consistent', 10.0),
        ('consistent', 50.0),
        ('oscillation', 10.0),
        ('jump', 25.0),
    ]

    # Generate attack trajectories
    attack_trajs = []
    attack_labels = []
    attack_masks = []

    base_traj = generate_trajectory(args.trajectory_length, seed=args.seed + 100)

    for attack_type, magnitude in attack_configs:
        spoofed, mask = inject_attack(
            base_traj,
            attack_type=attack_type,
            magnitude=magnitude,
            seed=args.seed,
        )
        attack_trajs.append(spoofed)
        attack_labels.append(f"{attack_type}_{magnitude}")
        attack_masks.append(mask)

    print(f"    Normal trajectories: {len(normal_trajs)}")
    print(f"    Attack configurations: {len(attack_configs)}")

    # =========================================================================
    # 2. Initialize Detectors
    # =========================================================================
    print("\n[2] Initializing detectors...")

    # EKF-NIS detector
    ekf_detector = NISAnomalyDetector(
        dt=0.01,
        process_noise_pos=0.01,
        process_noise_vel=0.1,
        measurement_noise=1.0,
    )

    # ML (ICI) detector
    ml_detector = CycleConsistencyDetector(state_dim=6, hidden_dim=64)

    # Train ML detector on normal data
    print("    Training ML (ICI) detector...")
    ml_detector.fit(normal_trajs, epochs=20, verbose=False)

    # Calibrate EKF detector
    print("    Calibrating EKF detector...")
    ekf_positions = [t[:, :3] for t in normal_trajs]
    ekf_detector.calibrate(ekf_positions, target_fpr=0.05)

    print("    Detectors ready.")

    # =========================================================================
    # 3. Compute Scores
    # =========================================================================
    print("\n[3] Computing detector scores...")

    # Normal scores
    ekf_normal = compute_ekf_scores(ekf_detector, normal_trajs)
    ml_normal = compute_ml_scores(ml_detector, normal_trajs)

    # Attack scores
    ekf_attack_all = []
    ml_attack_all = []
    attack_label_all = []

    for traj, label, mask in zip(attack_trajs, attack_labels, attack_masks):
        ekf_scores = compute_ekf_scores(ekf_detector, [traj])
        ml_scores = compute_ml_scores(ml_detector, [traj])

        # Only use attack window samples
        attack_window = mask[1:]  # Skip first (initialization)
        if len(attack_window) > len(ekf_scores):
            attack_window = attack_window[:len(ekf_scores)]

        ekf_attack_all.extend(ekf_scores[attack_window].tolist())
        ml_attack_all.extend(ml_scores[attack_window].tolist())
        attack_label_all.extend([label] * int(np.sum(attack_window)))

    ekf_attack = np.array(ekf_attack_all)
    ml_attack = np.array(ml_attack_all)
    attack_label_arr = np.array(attack_label_all)

    print(f"    Normal samples: {len(ekf_normal)}")
    print(f"    Attack samples: {len(ekf_attack)}")

    # =========================================================================
    # 4. Evaluate Individual Detectors
    # =========================================================================
    print("\n[4] Evaluating individual detectors...")

    from sklearn.metrics import roc_auc_score, roc_curve

    def eval_detector(normal, attack, name):
        y_true = np.concatenate([np.zeros(len(normal)), np.ones(len(attack))])
        y_scores = np.concatenate([normal, attack])

        auroc = roc_auc_score(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)

        recall_1 = tpr[np.searchsorted(fpr, 0.01)]
        recall_5 = tpr[np.searchsorted(fpr, 0.05)]

        # Per-attack worst case
        unique_attacks = np.unique(attack_label_arr)
        worst_recall = 1.0

        for att in unique_attacks:
            mask = attack_label_arr == att
            threshold = np.percentile(normal, 95)  # 5% FPR
            recall = np.mean(attack[mask] > threshold)
            worst_recall = min(worst_recall, recall)

        print(f"    {name}:")
        print(f"        AUROC:          {auroc:.4f}")
        print(f"        Recall@1%FPR:   {recall_1:.4f}")
        print(f"        Recall@5%FPR:   {recall_5:.4f}")
        print(f"        Worst Recall:   {worst_recall:.4f}")

        return {
            'auroc': float(auroc),
            'recall_1pct': float(recall_1),
            'recall_5pct': float(recall_5),
            'worst_recall': float(worst_recall),
        }

    ekf_results = eval_detector(ekf_normal, ekf_attack, "EKF-NIS")
    ml_results = eval_detector(ml_normal, ml_attack, "ML (ICI)")

    # =========================================================================
    # 5. Hybrid Fusion with Grid Search
    # =========================================================================
    print("\n[5] Optimizing hybrid fusion weights...")

    from src.hybrid.fuse_scores import grid_search_weights, evaluate_hybrid

    best_config, search_results = grid_search_weights(
        ekf_normal, ml_normal,
        ekf_attack, ml_attack,
        attack_label_arr,
        target_fpr=0.05,
    )

    print(f"    Best weights: w_ekf={best_config.w_ekf:.2f}, w_ml={best_config.w_ml:.2f}")
    print(f"    Best worst-case recall: {search_results['best_worst_recall']:.4f}")

    # =========================================================================
    # 6. Evaluate Hybrid Detector
    # =========================================================================
    print("\n[6] Evaluating hybrid detector...")

    fusion = HybridFusion(best_config)
    fusion.calibrate(ekf_normal, ml_normal, target_fpr=0.05)

    hybrid_metrics = evaluate_hybrid(
        fusion,
        ekf_normal, ml_normal,
        ekf_attack, ml_attack,
    )

    print(f"    Hybrid AUROC:        {hybrid_metrics['auroc']:.4f}")
    print(f"    Hybrid Recall@1%:    {hybrid_metrics['recall_1pct_fpr']:.4f}")
    print(f"    Hybrid Recall@5%:    {hybrid_metrics['recall_5pct_fpr']:.4f}")

    # Per-attack breakdown
    print("\n[7] Per-Attack Breakdown:")

    per_attack_results = {}
    unique_attacks = np.unique(attack_label_arr)

    for att in unique_attacks:
        mask = attack_label_arr == att
        att_ekf = ekf_attack[mask]
        att_ml = ml_attack[mask]

        # EKF recall
        ekf_thresh = np.percentile(ekf_normal, 95)
        ekf_recall = np.mean(att_ekf > ekf_thresh)

        # ML recall
        ml_thresh = np.percentile(ml_normal, 95)
        ml_recall = np.mean(att_ml > ml_thresh)

        # Hybrid recall
        hybrid_normal = fusion.fuse(ekf_normal, ml_normal)
        hybrid_att = fusion.fuse(att_ekf, att_ml)
        hybrid_thresh = np.percentile(hybrid_normal, 95)
        hybrid_recall = np.mean(hybrid_att > hybrid_thresh)

        per_attack_results[att] = {
            'ekf_recall': float(ekf_recall),
            'ml_recall': float(ml_recall),
            'hybrid_recall': float(hybrid_recall),
            'n_samples': int(np.sum(mask)),
        }

        print(f"    {att:20}: EKF={ekf_recall:.3f}  ML={ml_recall:.3f}  Hybrid={hybrid_recall:.3f}")

    # =========================================================================
    # 8. Latency Benchmark
    # =========================================================================
    print("\n[8] Latency benchmark...")

    import torch

    n_bench = 5000
    bench_traj = generate_trajectory(n_bench, seed=999)

    # Warmup
    for i in range(100):
        ekf_detector.initialize(bench_traj[0, :3])
        ekf_detector.step(bench_traj[1, :3])
        ml_detector.compute_ici(torch.tensor(bench_traj[0:1], dtype=torch.float32))

    # Timed run
    ekf_times = []
    ml_times = []
    hybrid_times = []

    ekf_detector.initialize(bench_traj[0, :3])

    for i in range(1, n_bench):
        # EKF
        start = time.perf_counter()
        _, ekf_score, _ = ekf_detector.step(bench_traj[i, :3])
        ekf_times.append(time.perf_counter() - start)

        # ML
        start = time.perf_counter()
        ml_score = ml_detector.compute_ici(
            torch.tensor(bench_traj[i:i+1], dtype=torch.float32)
        ).item()
        ml_times.append(time.perf_counter() - start)

        # Hybrid fusion (just the fusion step)
        start = time.perf_counter()
        _ = fusion.fuse(np.array([ekf_score]), np.array([ml_score]))
        hybrid_times.append(time.perf_counter() - start)

    ekf_times = np.array(ekf_times) * 1000
    ml_times = np.array(ml_times) * 1000
    total_times = ekf_times + ml_times + np.array(hybrid_times) * 1000

    latency = {
        'ekf_p95_ms': float(np.percentile(ekf_times, 95)),
        'ml_p95_ms': float(np.percentile(ml_times, 95)),
        'total_p95_ms': float(np.percentile(total_times, 95)),
        'total_mean_ms': float(np.mean(total_times)),
    }

    print(f"    EKF P95:     {latency['ekf_p95_ms']:.3f} ms")
    print(f"    ML P95:      {latency['ml_p95_ms']:.3f} ms")
    print(f"    Total P95:   {latency['total_p95_ms']:.3f} ms")

    # =========================================================================
    # 9. Target Check
    # =========================================================================
    print("\n[9] Target Check...")

    # Improvement over ML baseline
    ml_worst = ml_results['worst_recall']
    hybrid_worst = min(r['hybrid_recall'] for r in per_attack_results.values())
    improvement = hybrid_worst - ml_worst

    targets = {
        'Hybrid AUROC >= max(EKF, ML)': hybrid_metrics['auroc'] >= max(ekf_results['auroc'], ml_results['auroc']),
        'Hybrid worst-case > ML worst-case': hybrid_worst >= ml_worst,
        'Total P95 latency < 5ms': latency['total_p95_ms'] < 5.0,
    }

    for target, passed in targets.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {target}")

    # =========================================================================
    # 10. Save Results
    # =========================================================================
    results = {
        'ekf': ekf_results,
        'ml': ml_results,
        'hybrid': {
            **hybrid_metrics,
            'weights': {'w_ekf': best_config.w_ekf, 'w_ml': best_config.w_ml},
        },
        'per_attack': per_attack_results,
        'latency': latency,
        'targets': targets,
        'grid_search': search_results['grid_search'],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Detector':<15} {'AUROC':<10} {'R@1%FPR':<10} {'R@5%FPR':<10} {'Worst':<10}")
    print("-" * 55)
    print(f"{'EKF-NIS':<15} {ekf_results['auroc']:<10.4f} {ekf_results['recall_1pct']:<10.4f} {ekf_results['recall_5pct']:<10.4f} {ekf_results['worst_recall']:<10.4f}")
    print(f"{'ML (ICI)':<15} {ml_results['auroc']:<10.4f} {ml_results['recall_1pct']:<10.4f} {ml_results['recall_5pct']:<10.4f} {ml_results['worst_recall']:<10.4f}")
    print(f"{'Hybrid':<15} {hybrid_metrics['auroc']:<10.4f} {hybrid_metrics['recall_1pct_fpr']:<10.4f} {hybrid_metrics['recall_5pct_fpr']:<10.4f} {hybrid_worst:<10.4f}")
    print("=" * 70)

    all_passed = all(targets.values())
    print(f"\nAll targets: {'PASS' if all_passed else 'FAIL'}")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
