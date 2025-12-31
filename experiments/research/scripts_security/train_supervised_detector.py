"""
Train and Evaluate Supervised Attack Classifier.

This script:
1. Loads normal EuRoC data
2. Generates synthetic attacks (all 30 types)
3. Trains a Random Forest classifier on features
4. Evaluates per-attack-type performance

Usage:
    python scripts/security/train_supervised_detector.py
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinn_dynamics.security.supervised_detector import SupervisedAttackClassifier

sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic_attacks import SyntheticAttackGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/euroc")
    parser.add_argument("--output", default="models/security/supervised_detector")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--class-weight", type=float, default=3.0)
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SUPERVISED ATTACK CLASSIFIER TRAINING")
    print("=" * 70)

    # Load EuRoC data
    print("\n[1/5] Loading EuRoC data...")
    csv_file = None
    for name in ["all_sequences.csv", "euroc_processed.csv"]:
        if (data_path / name).exists():
            csv_file = data_path / name
            break
    if csv_file is None:
        csv_files = list(data_path.glob("*.csv"))
        if csv_files:
            csv_file = csv_files[0]

    if csv_file is None:
        print(f"  ERROR: No CSV files found in {data_path}")
        return

    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df):,} samples")

    # Normalize columns
    for old, new in [("roll", "phi"), ("pitch", "theta"), ("yaw", "psi")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "thrust" not in df.columns:
        df["thrust"] = df["az"] + 9.81 if "az" in df.columns else 9.81
    for col in ["torque_x", "torque_y", "torque_z"]:
        if col not in df.columns:
            df[col] = 0.0

    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]

    # Generate attacks
    print("\n[2/5] Generating synthetic attacks...")
    generator = SyntheticAttackGenerator(df, seed=42, randomize=False)
    attacks = generator.generate_all_attacks(handle_nan=True)

    # Prepare training data
    print("\n[3/5] Preparing training data...")

    # Normal data from clean
    normal_data = attacks["clean"][state_cols].values

    # Split normal data: 70% train, 30% test
    n_normal = len(normal_data)
    n_train_normal = int(0.7 * n_normal)
    normal_train = normal_data[:n_train_normal]
    normal_test = normal_data[n_train_normal:]

    print(f"  Normal train: {len(normal_train):,}")
    print(f"  Normal test: {len(normal_test):,}")

    # Collect attack data
    attack_train_data = []
    attack_test_data = {}

    for attack_name, attack_df in attacks.items():
        if attack_name == "clean":
            continue

        # Get attack samples only (label == 1)
        attack_mask = attack_df["label"] == 1
        attack_states = attack_df[attack_mask][state_cols].values

        if len(attack_states) < 100:
            continue

        # Split: 70% train, 30% test - split attack samples only
        n_attack = len(attack_states)
        n_train = int(0.7 * n_attack)
        n_test_attack = n_attack - n_train

        attack_train_data.append(attack_states[:n_train])

        # Store test data with proper attack-only samples + some normals for FPR
        if n_test_attack > 0:
            normal_mask = attack_df["label"] == 0
            normal_states_seq = attack_df[normal_mask][state_cols].values

            # Take similar amount of normal samples for balanced evaluation
            n_normal_test = min(len(normal_states_seq), n_test_attack * 2)

            if n_normal_test > 0:
                test_states = np.vstack(
                    [normal_states_seq[:n_normal_test], attack_states[n_train:]]
                )
                test_labels = np.concatenate([np.zeros(n_normal_test), np.ones(n_test_attack)])
            else:
                test_states = attack_states[n_train:]
                test_labels = np.ones(n_test_attack)

            attack_test_data[attack_name] = {
                "states": test_states,
                "labels": test_labels,
            }

    # Combine attack training data
    attack_train = np.vstack(attack_train_data)
    print(
        f"  Attack train: {len(attack_train):,} samples from {len(attack_train_data)} attack types"
    )

    # Create classifier
    print("\n[4/5] Training classifier...")
    classifier = SupervisedAttackClassifier(
        window_size=args.window_size,
        class_weight=args.class_weight,
    )

    # Train
    metrics = classifier.train(normal_train, attack_train)

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

    # Test on normal data first
    normal_preds, normal_probs = classifier.predict_batch(normal_test)
    fpr_normal = np.mean(normal_preds) if len(normal_preds) > 0 else 0
    print(f"  {'clean':30s} | FPR: {fpr_normal*100:5.1f}% | (baseline)")
    results["clean"] = {"fpr": float(fpr_normal)}

    # Test on each attack type
    for attack_name, test_data in attack_test_data.items():
        test_states = test_data["states"]
        test_labels = test_data["labels"]

        if len(test_states) < args.window_size + 10:
            continue

        eval_metrics = classifier.evaluate(test_states, test_labels)
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
        recalls = [results[a]["recall"] for a in attack_list if a in results]
        if recalls:
            avg_recall = np.mean(recalls)
            category_results[cat_name] = avg_recall
            print(f"  {cat_name:15s}: {avg_recall*100:5.1f}% avg recall")

    # Overall metrics
    all_recalls = [r["recall"] for k, r in results.items() if k != "clean" and "recall" in r]
    all_fprs = [r["fpr"] for k, r in results.items() if k != "clean" and "fpr" in r]

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nOverall Avg Recall: {np.mean(all_recalls)*100:.1f}%")
    print(f"Overall Avg FPR:    {np.mean(all_fprs)*100:.1f}%")
    print(f"Clean Data FPR:     {fpr_normal*100:.1f}%")

    # Save model and results
    model_path = output_path / "classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "classifier": classifier.classifier,
                "scaler": classifier.scaler,
                "feature_extractor": classifier.feature_extractor,
                "window_size": classifier.window_size,
            },
            f,
        )
    print(f"\nModel saved to: {model_path}")

    results_path = output_path / "evaluation_results.json"
    summary = {
        "training_metrics": metrics,
        "category_results": category_results,
        "per_attack_results": results,
        "overall_recall": float(np.mean(all_recalls)),
        "overall_fpr": float(np.mean(all_fprs)),
        "clean_fpr": float(fpr_normal),
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
