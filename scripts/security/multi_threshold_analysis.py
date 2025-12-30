"""
Multi-threshold analysis for PINN detector.

Tests different detection thresholds and shows performance tradeoffs.
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pinn_dynamics import QuadrotorPINN

sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic_attacks import SyntheticAttackGenerator


def main():
    # Load model
    model_dir = Path("models/security")
    model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)
    model.load_state_dict(torch.load(model_dir / "pinn_synthetic_detector.pth", map_location="cpu"))
    model.eval()

    # Load scalers
    with open(model_dir / "scalers_synthetic.pkl", "rb") as f:
        scalers = pickle.load(f)

    # Load EuRoC data
    df = pd.read_csv("data/euroc/all_sequences.csv")
    for old, new in [("roll", "phi"), ("pitch", "theta"), ("yaw", "psi")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "thrust" not in df.columns:
        df["thrust"] = df["az"] + 9.81 if "az" in df.columns else 9.81
    for col in ["torque_x", "torque_y", "torque_z"]:
        if col not in df.columns:
            df[col] = 0.0

    # Generate attacks
    generator = SyntheticAttackGenerator(df, seed=42, randomize=False)
    attacks = generator.generate_all_attacks(handle_nan=True)

    # Compute predictions for all attacks
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    def compute_errors(attack_data):
        states = attack_data[state_cols].values
        controls = attack_data[control_cols].values
        X = np.concatenate([states[:-1], controls[:-1]], axis=1)
        y = states[1:]
        labels = attack_data["label"].values[1:]

        X_scaled = scalers["scaler_X"].transform(X)

        with torch.no_grad():
            X_t = torch.FloatTensor(X_scaled)
            y_pred_scaled = model(X_t).numpy()

        y_pred = scalers["scaler_y"].inverse_transform(y_pred_scaled)
        errors = np.linalg.norm(y - y_pred, axis=1)

        return errors, labels

    # Collect all errors and labels
    all_attack_errors = {}
    for name, data in attacks.items():
        errors, labels = compute_errors(data)
        all_attack_errors[name] = {"errors": errors, "labels": labels}

    # Get clean data errors for threshold calibration
    clean_errors = all_attack_errors["clean"]["errors"]

    # Test multiple threshold percentiles
    print("=" * 70)
    print("MULTI-THRESHOLD ANALYSIS")
    print("=" * 70)

    for percentile in [50, 75, 90, 95, 99]:
        threshold = np.percentile(clean_errors, percentile)

        # Compute metrics for each attack
        recalls = {}
        fprs = {}

        for name, data in all_attack_errors.items():
            errors = data["errors"]
            labels = data["labels"]
            preds = (errors > threshold).astype(int)

            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            tn = np.sum((preds == 0) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            recalls[name] = recall
            fprs[name] = fpr

        # Compute average recall (excluding clean)
        attack_recalls = [v for k, v in recalls.items() if k != "clean"]
        avg_recall = np.mean(attack_recalls)

        print(f"\nThreshold: {percentile}th percentile ({threshold:.4f})")
        print(f"  Avg Attack Recall: {avg_recall*100:.1f}%")
        print(f'  Clean FPR: {fprs["clean"]*100:.1f}%')

        # Show top and bottom
        sorted_recalls = sorted(
            [(k, v) for k, v in recalls.items() if k != "clean"],
            key=lambda x: x[1],
            reverse=True,
        )
        print(
            f"  Best: {sorted_recalls[0][0]} ({sorted_recalls[0][1]*100:.1f}%), "
            f"{sorted_recalls[1][0]} ({sorted_recalls[1][1]*100:.1f}%)"
        )
        print(
            f"  Worst: {sorted_recalls[-1][0]} ({sorted_recalls[-1][1]*100:.1f}%), "
            f"{sorted_recalls[-2][0]} ({sorted_recalls[-2][1]*100:.1f}%)"
        )

    # Best threshold analysis
    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLD: 75th percentile")
    print("=" * 70)

    threshold = np.percentile(clean_errors, 75)
    print(f"\nDetailed results at 75th percentile threshold ({threshold:.4f}):")
    print()

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
        "Mag_Baro": ["magnetometer_spoofing", "barometer_spoofing"],
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

    for cat_name, attacks_list in categories.items():
        print(f"{cat_name}:")
        cat_recalls = []
        for attack_name in attacks_list:
            if attack_name in all_attack_errors:
                data = all_attack_errors[attack_name]
                errors = data["errors"]
                labels = data["labels"]
                preds = (errors > threshold).astype(int)
                tp = np.sum((preds == 1) & (labels == 1))
                fn = np.sum((preds == 0) & (labels == 1))
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                cat_recalls.append(recall)
                print(f"  {attack_name:30s} | Recall: {recall*100:5.1f}%")
        print(f"  Category Average: {np.mean(cat_recalls)*100:.1f}%")
        print()

    # Clean FPR
    clean_preds = (clean_errors > threshold).astype(int)
    clean_fpr = np.mean(clean_preds)
    print(f"Clean Data FPR: {clean_fpr*100:.1f}%")


if __name__ == "__main__":
    main()
