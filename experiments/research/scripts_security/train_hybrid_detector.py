"""
Train Hybrid Attack Detector with Data-Driven Routing.

This script trains a hybrid detector that routes samples to whichever
detector (baseline or PINN) performs better, learned from training data.

Expected improvements:
- control_hijack: 22% -> 99.9% (routed to PINN)
- temporal attacks: keep 30.2% (routed to baseline, not degraded)
- Overall: 74% -> 78-80%

Usage:
    python scripts/security/train_hybrid_detector.py
    python scripts/security/train_hybrid_detector.py --model models/security/pinn_synthetic_detector.pth
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinn_dynamics.security.hybrid_detector import HybridAttackDetector

sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic_attacks import SyntheticAttackGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/euroc")
    parser.add_argument("--output", default="models/security/hybrid_detector")
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--model", default="models/security/pinn_synthetic_detector.pth")
    parser.add_argument("--scalers", default="models/security/scalers_synthetic.pkl")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HYBRID ATTACK DETECTOR TRAINING")
    print(f"Target Recall: {args.target_recall*100:.0f}%")
    print("=" * 70)

    # Load PINN predictor
    print("\n[0/5] Loading PINN model...")
    predictor = None
    model_path = Path(args.model)
    if model_path.exists():
        from pinn_dynamics import Predictor, QuadrotorPINN

        model = QuadrotorPINN()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        scalers_path = Path(args.scalers)
        scaler_X, scaler_y = None, None
        if scalers_path.exists():
            with open(scalers_path, "rb") as f:
                scalers = pickle.load(f)
                if isinstance(scalers, dict):
                    scaler_X = scalers.get("scaler_X")
                    scaler_y = scalers.get("scaler_y")
                elif isinstance(scalers, (list, tuple)):
                    scaler_X, scaler_y = scalers

        predictor = Predictor(model, scaler_X=scaler_X, scaler_y=scaler_y)
        print(f"  Loaded PINN from {model_path}")
    else:
        print(f"  WARNING: PINN model not found at {model_path}")
        print("  Hybrid detector will use baseline only (no routing benefit)")

    # Load EuRoC data
    print("\n[1/5] Loading EuRoC data...")
    csv_path = data_path / "all_sequences.csv"
    if not csv_path.exists():
        print(f"  ERROR: Data not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} samples")

    # Standardize column names
    for old, new in [("roll", "phi"), ("pitch", "theta"), ("yaw", "psi")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Create thrust from acceleration
    if "thrust" not in df.columns:
        if "az" in df.columns:
            df["thrust"] = df["az"] + 9.81
        else:
            df["thrust"] = 9.81

    # Create torques if missing
    for col in ["torque_x", "torque_y", "torque_z"]:
        if col not in df.columns:
            df[col] = 0.0

    # Generate synthetic attacks
    print("\n[2/5] Generating synthetic attacks...")
    generator = SyntheticAttackGenerator(df, seed=42, randomize=True)
    attack_dfs = generator.generate_all_attacks()
    print(f"  Generated {len(attack_dfs)} attack types")

    # Prepare training data
    print("\n[3/5] Preparing training data...")
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    # Normal data
    normal_data = df[state_cols].values
    normal_controls = df[control_cols].values

    # 70% train, 30% test for normal
    n_train_normal = int(0.7 * len(normal_data))
    normal_train = normal_data[:n_train_normal]
    normal_train_ctrl = normal_controls[:n_train_normal]
    normal_test = normal_data[n_train_normal:]
    normal_test_ctrl = normal_controls[n_train_normal:]

    # Collect attack data
    attack_train_states = []
    attack_train_controls = []
    attack_test_data = {}

    for attack_name, attack_df in attack_dfs.items():
        if attack_df is None or len(attack_df) == 0:
            continue

        states = attack_df[state_cols].values
        controls = attack_df[control_cols].values
        labels = attack_df["label"].values

        # Get only the attack samples for training
        attack_mask = labels == 1
        attack_states = states[attack_mask]
        attack_controls = controls[attack_mask]

        if len(attack_states) < 100:
            print(f"    Skipping {attack_name}: only {len(attack_states)} attack samples")
            continue

        # 70% train, 30% test
        n = len(attack_states)
        n_train = int(0.7 * n)

        attack_train_states.append(attack_states[:n_train])
        attack_train_controls.append(attack_controls[:n_train])

        # Store test data with proper composition
        n_test_attack = n - n_train
        if n_test_attack > 0:
            normal_mask = labels == 0
            normal_states_seq = states[normal_mask]
            normal_controls_seq = controls[normal_mask]

            n_normal_test = min(len(normal_states_seq), n_test_attack * 2)

            if n_normal_test > 0:
                test_states = np.vstack(
                    [normal_states_seq[:n_normal_test], attack_states[n_train:]]
                )
                test_controls = np.vstack(
                    [normal_controls_seq[:n_normal_test], attack_controls[n_train:]]
                )
                test_labels = np.concatenate([np.zeros(n_normal_test), np.ones(n_test_attack)])
            else:
                test_states = attack_states[n_train:]
                test_controls = attack_controls[n_train:]
                test_labels = np.ones(n_test_attack)

            attack_test_data[attack_name] = {
                "states": test_states,
                "controls": test_controls,
                "labels": test_labels,
            }

    if not attack_train_states:
        print("  ERROR: No attack data collected!")
        return

    attack_train = np.vstack(attack_train_states)
    attack_train_ctrl = np.vstack(attack_train_controls)

    print(f"  Normal train: {len(normal_train):,}")
    print(
        f"  Attack train: {len(attack_train):,} samples from {len(attack_train_states)} attack types"
    )

    # Create and train hybrid detector
    print("\n[4/5] Training Hybrid Detector...")
    detector = HybridAttackDetector(
        predictor=predictor,
        target_recall=args.target_recall,
        n_estimators=200,
    )

    train_metrics = detector.train(
        normal_states=normal_train,
        attack_states=attack_train,
        normal_controls=normal_train_ctrl,
        attack_controls=attack_train_ctrl,
        validation_split=0.3,
    )

    # Evaluate per attack type
    print("\n[5/5] Evaluating on held-out test data...")
    print()

    results = {}
    categories = {
        "GPS": [
            "gps_gradual_drift",
            "gps_sudden_jump",
            "gps_oscillating",
            "gps_meaconing",
            "gps_jamming",
            "gps_freeze",
            "gps_multipath",
        ],
        "IMU": [
            "imu_constant_bias",
            "imu_gradual_drift",
            "imu_sinusoidal",
            "imu_noise_injection",
            "imu_scale_factor",
            "gyro_saturation",
            "accel_saturation",
        ],
        "Mag/Baro": ["magnetometer_spoofing", "barometer_spoofing"],
        "Actuator": [
            "actuator_stuck",
            "actuator_degraded",
            "control_hijack",
            "thrust_manipulation",
        ],
        "Coordinated": ["coordinated_gps_imu", "stealthy_coordinated"],
        "Temporal": ["replay_attack", "time_delay", "sensor_dropout"],
        "Stealth": [
            "adaptive_attack",
            "intermittent_attack",
            "slow_ramp",
            "resonance_attack",
            "false_data_injection",
        ],
    }

    # Test on normal data
    preds, probs = detector.predict_batch(normal_test, controls=normal_test_ctrl)
    fpr_normal = np.mean(preds) if len(preds) > 0 else 0
    print(f"  {'clean':30s} | FPR: {fpr_normal*100:5.1f}% | (baseline)")
    results["clean"] = {"fpr": float(fpr_normal)}

    # Test on each attack type
    for attack_name, test_data in attack_test_data.items():
        test_states = test_data["states"]
        test_labels = test_data["labels"]
        test_controls = test_data["controls"]

        if len(test_states) < 250:
            continue

        eval_metrics = detector.evaluate(
            test_states,
            test_labels,
            controls=test_controls,
        )
        results[attack_name] = eval_metrics

        print(
            f"  {attack_name:30s} | Recall: {eval_metrics['recall']*100:5.1f}% | "
            f"FPR: {eval_metrics['fpr']*100:5.1f}% | F1: {eval_metrics['f1']*100:5.1f}%"
        )

    # Category results
    print("\n" + "=" * 70)
    print("RESULTS BY CATEGORY")
    print("=" * 70)

    category_results = {}
    for cat_name, attack_list in categories.items():
        recalls = [
            results[a]["recall"] for a in attack_list if a in results and "recall" in results[a]
        ]
        if recalls:
            avg_recall = np.mean(recalls)
            min_recall = np.min(recalls)
            category_results[cat_name] = {"avg": avg_recall, "min": min_recall}
            print(f"  {cat_name:15s}: {avg_recall*100:5.1f}% avg | {min_recall*100:5.1f}% min")

    # Overall
    all_recalls = [r["recall"] for k, r in results.items() if k != "clean" and "recall" in r]
    all_fprs = [r["fpr"] for k, r in results.items() if k != "clean" and "fpr" in r]

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    overall_recall = np.mean(all_recalls) if all_recalls else 0
    overall_min = np.min(all_recalls) if all_recalls else 0
    overall_fpr = np.mean(all_fprs) if all_fprs else 0

    print(f"\nTarget Recall: {args.target_recall*100:.0f}%")
    print(f"Achieved:")
    print(f"  Overall Avg Recall: {overall_recall*100:.1f}%")
    print(f"  Overall Min Recall: {overall_min*100:.1f}%")
    print(f"  Overall Avg FPR:    {overall_fpr*100:.1f}%")
    print(f"  Clean Data FPR:     {fpr_normal*100:.1f}%")

    if overall_recall >= args.target_recall:
        print(f"\n  Target met!")
    else:
        print(f"\n  Target not met: {overall_recall*100:.1f}% < {args.target_recall*100:.0f}%")

    # Save model
    detector.save(str(output_path))
    print(f"\nModel saved to: {output_path}")

    # Save results
    final_results = {
        "target_recall": args.target_recall,
        "training_metrics": train_metrics,
        "category_results": category_results,
        "per_attack_results": results,
        "overall_recall": float(overall_recall),
        "overall_min_recall": float(overall_min),
        "overall_fpr": float(overall_fpr),
        "clean_fpr": float(fpr_normal),
    }

    with open(output_path / "evaluation_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Results saved to: {output_path / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
