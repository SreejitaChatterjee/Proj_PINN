"""
Run Detector V3 - Comprehensive multi-signal detection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
from datetime import datetime

from detector_v3 import ComprehensiveDetector, DetectorV3Config, evaluate_detector
from scripts.security.generate_synthetic_attacks import SyntheticAttackGenerator


def main():
    print("=" * 70)
    print("SENSOR FUSION DETECTOR V3 (Comprehensive)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[1/4] Loading EuRoC data...")
    df = pd.read_csv("data/euroc/all_sequences.csv")

    for old, new in [("roll", "phi"), ("pitch", "theta"), ("yaw", "psi")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "thrust" not in df.columns:
        df["thrust"] = df["az"] + 9.81 if "az" in df.columns else 9.81
    for col in ["torque_x", "torque_y", "torque_z"]:
        if col not in df.columns:
            df[col] = 0.0

    # Split
    print("\n[2/4] Splitting data...")
    sequences = df["sequence"].unique()
    np.random.seed(42)
    np.random.shuffle(sequences)

    train_seqs = list(sequences[:3])
    test_seqs = list(sequences[4:])

    train_df = df[df["sequence"].isin(train_seqs)].reset_index(drop=True)
    test_df = df[df["sequence"].isin(test_seqs)].reset_index(drop=True)

    print(f"  Train: {train_seqs}")
    print(f"  Test:  {test_seqs}")

    # Prepare data
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]
    train_data = train_df[state_cols + control_cols].values

    # Create and calibrate detector
    print("\n[3/4] Calibrating detector...")
    detector = ComprehensiveDetector()
    detector.calibrate(train_data)

    # Evaluate
    print("\n[4/4] Evaluating on attacks...")
    generator = SyntheticAttackGenerator(test_df, seed=42, randomize=False)
    attacks = generator.generate_all_attacks(handle_nan=True)

    results = evaluate_detector(detector, attacks)

    # Compute overall
    attack_results = [v for k, v in results.items() if k != "clean"]
    overall_recall = np.mean([r['recall'] for r in attack_results])
    overall_precision = np.mean([r['precision'] for r in attack_results])
    overall_f1 = np.mean([r['f1'] for r in attack_results])

    # Breakdown by detection difficulty
    print("\n" + "=" * 70)
    print("DETECTION BREAKDOWN")
    print("=" * 70)

    high_recall = [(k, v['recall']) for k, v in results.items() if k != 'clean' and v['recall'] > 0.8]
    medium_recall = [(k, v['recall']) for k, v in results.items() if k != 'clean' and 0.3 < v['recall'] <= 0.8]
    low_recall = [(k, v['recall']) for k, v in results.items() if k != 'clean' and v['recall'] <= 0.3]

    print(f"\n[HIGH] >80% recall: {len(high_recall)}/30")
    for name, recall in sorted(high_recall, key=lambda x: -x[1]):
        print(f"  {name}: {recall*100:.1f}%")

    print(f"\n[MED] 30-80% recall: {len(medium_recall)}/30")
    for name, recall in sorted(medium_recall, key=lambda x: -x[1]):
        print(f"  {name}: {recall*100:.1f}%")

    print(f"\n[LOW] <30% recall: {len(low_recall)}/30")
    for name, recall in sorted(low_recall, key=lambda x: -x[1]):
        print(f"  {name}: {recall*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"\n{'Method':<30} {'Recall':>10}")
    print("-" * 45)
    print(f"{'PINN Baseline':<30} {'18.7%':>10}")
    print(f"{'Learned Model (v1)':<30} {'29.8%':>10}")
    print(f"{'Physics-First (v2)':<30} {'67.4%':>10}")
    print(f"{'Comprehensive (v3)':<30} {overall_recall*100:>9.1f}%")

    # Save results
    output_dir = Path("sensor_fusion_detector/results_v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_out = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
    results_out['overall'] = {
        'recall': float(overall_recall),
        'precision': float(overall_precision),
        'f1': float(overall_f1)
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_out, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
