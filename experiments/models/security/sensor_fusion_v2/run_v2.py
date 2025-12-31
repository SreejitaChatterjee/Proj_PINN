"""
Run V2 Detector - Physics-First Approach

This uses pure physics consistency as the primary detector,
with optional learned residual for hard cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
from datetime import datetime

from detector_v2 import HybridDetector, DetectorConfigV2, evaluate_detector
from scripts.security.generate_synthetic_attacks import SyntheticAttackGenerator


def main():
    print("=" * 70)
    print("SENSOR FUSION DETECTOR V2 (Physics-First)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[1/4] Loading EuRoC data...")
    df = pd.read_csv("data/euroc/all_sequences.csv")

    # Normalize
    for old, new in [("roll", "phi"), ("pitch", "theta"), ("yaw", "psi")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "thrust" not in df.columns:
        df["thrust"] = df["az"] + 9.81 if "az" in df.columns else 9.81
    for col in ["torque_x", "torque_y", "torque_z"]:
        if col not in df.columns:
            df[col] = 0.0

    # Split by sequence
    print("\n[2/4] Splitting by sequence...")
    sequences = df["sequence"].unique()
    np.random.seed(42)
    np.random.shuffle(sequences)

    train_seqs = list(sequences[:3])
    test_seqs = list(sequences[4:])

    train_df = df[df["sequence"].isin(train_seqs)].reset_index(drop=True)
    test_df = df[df["sequence"].isin(test_seqs)].reset_index(drop=True)

    print(f"  Train: {train_seqs}")
    print(f"  Test:  {test_seqs}")

    # Create detector with calibrated thresholds
    print("\n[3/4] Calibrating physics thresholds...")

    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]
    train_data = train_df[state_cols + control_cols].values

    # Compute thresholds from training data
    dt = 0.005
    pos = train_data[:, 0:3]
    att = train_data[:, 3:6]
    rate = train_data[:, 6:9]
    vel = train_data[:, 9:12]

    pos_deriv = (pos[1:] - pos[:-1]) / dt
    pos_vel_score = np.linalg.norm(pos_deriv - vel[1:], axis=1)

    att_deriv = (att[1:] - att[:-1]) / dt
    att_rate_score = np.linalg.norm(att_deriv - rate[1:], axis=1)

    window = 20
    kinematic_score = np.zeros(len(train_data) - 1)
    for i in range(window, len(train_data) - 1):
        vel_integral = vel[i-window+1:i+1].sum(axis=0) * dt
        pos_change = pos[i+1] - pos[i-window+1]
        kinematic_score[i] = np.linalg.norm(vel_integral - pos_change)

    config = DetectorConfigV2(
        dt=dt,
        pos_vel_threshold=np.percentile(pos_vel_score, 99),
        att_rate_threshold=np.percentile(att_rate_score[~np.isnan(att_rate_score)], 99),
        kinematic_threshold=np.percentile(kinematic_score[kinematic_score > 0], 99)
    )

    print(f"  pos_vel_threshold:   {config.pos_vel_threshold:.6f}")
    print(f"  att_rate_threshold:  {config.att_rate_threshold:.6f}")
    print(f"  kinematic_threshold: {config.kinematic_threshold:.6f}")

    detector = HybridDetector(config)

    # Optional: Train residual learner
    print("\n  Training residual learner (for hard cases)...")
    detector.train_residual(train_data, epochs=20)

    # Generate attacks and evaluate
    print("\n[4/4] Evaluating on attacks...")
    generator = SyntheticAttackGenerator(test_df, seed=42, randomize=False)
    attacks = generator.generate_all_attacks(handle_nan=True)

    results = evaluate_detector(detector, test_df, attacks)

    # Compute overall metrics
    attack_results = [v for k, v in results.items() if k != "clean"]
    overall_recall = np.mean([r['recall'] for r in attack_results])
    overall_precision = np.mean([r['precision'] for r in attack_results])
    overall_f1 = np.mean([r['f1'] for r in attack_results])

    # Save results
    output_dir = Path("sensor_fusion_detector/results_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_out = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
    results_out['overall'] = {
        'recall': float(overall_recall),
        'precision': float(overall_precision),
        'f1': float(overall_f1)
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_out, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nOverall Metrics:")
    print(f"  Recall:    {overall_recall*100:.1f}%")
    print(f"  Precision: {overall_precision*100:.1f}%")
    print(f"  F1:        {overall_f1*100:.1f}%")

    # Compare
    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)
    print(f"  PINN Baseline:        18.7%")
    print(f"  Learned Model (v1):   29.8%")
    print(f"  Physics-First (v2):   {overall_recall*100:.1f}%")

    # Categorize results
    easily_detected = [(k, v['recall']) for k, v in results.items() if k != 'clean' and v['recall'] > 0.5]
    hard_to_detect = [(k, v['recall']) for k, v in results.items() if k != 'clean' and v['recall'] < 0.1]

    print(f"\n[OK] Easily Detected (>50%): {len(easily_detected)}/30")
    print(f"[X]  Hard to Detect (<10%):  {len(hard_to_detect)}/30")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
