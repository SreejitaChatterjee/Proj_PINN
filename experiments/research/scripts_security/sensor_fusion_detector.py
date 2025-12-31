"""
Sensor Fusion Anomaly Detection for Bias Attack Detection

Key Insight: Bias attacks affect ONE sensor reading but NOT its derivative/integral.
Cross-modal consistency checking detects this discrepancy.

Consistency Checks:
1. Attitude-Rate: d(attitude)/dt should match angular rates (p, q, r)
2. Velocity-Acceleration: d(velocity)/dt should match accelerations (ax, ay, az)
3. Position-Velocity: d(position)/dt should match velocities (vx, vy, vz)

If roll is biased but p (roll rate) is not, the integral of p won't match roll change.
This INCONSISTENCY is detectable!
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
EUROC_PATH = PROJECT_ROOT / "data" / "euroc" / "all_sequences.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "security" / "sensor_fusion_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Contamination from domain knowledge (not tuned on attacks)
CONTAMINATION = 0.05


def compute_consistency_features(data, dt=0.005):
    """
    Compute cross-modal consistency features.

    Args:
        data: DataFrame with columns [x,y,z,roll,pitch,yaw,p,q,r,vx,vy,vz,ax,ay,az]
        dt: sampling period (EuRoC is 200Hz = 0.005s)

    Returns:
        features: [N-1, n_features] consistency violation features
    """
    # Extract signals
    pos = data[["x", "y", "z"]].values
    att = data[["roll", "pitch", "yaw"]].values
    rates = data[["p", "q", "r"]].values
    vel = data[["vx", "vy", "vz"]].values
    acc = data[["ax", "ay", "az"]].values

    n = len(data) - 1
    features = []

    for i in range(n):
        feat = []

        # === 1. ATTITUDE-RATE CONSISTENCY ===
        # Actual attitude change
        att_change = att[i + 1] - att[i]

        # Expected change from integrating angular rates (simplified Euler)
        # For small angles: d(roll)/dt ≈ p, d(pitch)/dt ≈ q, d(yaw)/dt ≈ r
        expected_att_change = rates[i] * dt

        # Consistency error
        att_error = att_change - expected_att_change
        feat.extend(
            [
                np.abs(att_error[0]),  # roll consistency error
                np.abs(att_error[1]),  # pitch consistency error
                np.abs(att_error[2]),  # yaw consistency error
                np.linalg.norm(att_error),  # total attitude error
            ]
        )

        # === 2. VELOCITY-ACCELERATION CONSISTENCY ===
        # Actual velocity change
        vel_change = vel[i + 1] - vel[i]

        # Expected change from integrating accelerations
        # Note: Need to account for gravity (az includes gravity ~9.81)
        expected_vel_change = acc[i] * dt

        # Consistency error (may have offset due to gravity/bias)
        vel_error = vel_change - expected_vel_change
        feat.extend(
            [
                np.abs(vel_error[0]),  # vx consistency error
                np.abs(vel_error[1]),  # vy consistency error
                np.abs(vel_error[2]),  # vz consistency error
                np.linalg.norm(vel_error),  # total velocity error
            ]
        )

        # === 3. POSITION-VELOCITY CONSISTENCY ===
        # Actual position change
        pos_change = pos[i + 1] - pos[i]

        # Expected change from integrating velocities
        expected_pos_change = vel[i] * dt

        # Consistency error
        pos_error = pos_change - expected_pos_change
        feat.extend(
            [
                np.abs(pos_error[0]),  # x consistency error
                np.abs(pos_error[1]),  # y consistency error
                np.abs(pos_error[2]),  # z consistency error
                np.linalg.norm(pos_error),  # total position error
            ]
        )

        features.append(feat)

    return np.array(features)


def extract_windowed_features(consistency_feats, windows=[5, 10, 25, 50]):
    """
    Extract multi-scale features from consistency errors.

    For bias detection, we look for PERSISTENT errors (mean, cumsum)
    not just variance changes.
    """
    if len(consistency_feats) < max(windows):
        return np.array([])

    all_features = []
    max_window = max(windows)
    n_base = consistency_feats.shape[1]

    for i in range(max_window, len(consistency_feats)):
        feat_list = []

        for w_size in windows:
            window = consistency_feats[i - w_size : i]

            # Mean error (detects persistent bias)
            feat_list.extend(np.mean(window, axis=0))

            # Std error (detects noise/instability)
            feat_list.extend(np.std(window, axis=0))

            # Max error (detects spikes)
            feat_list.extend(np.max(np.abs(window), axis=0))

            # Cumulative sum magnitude (detects drift/bias accumulation)
            cumsum = np.cumsum(window, axis=0)
            feat_list.append(np.max(np.abs(cumsum[:, 3])))  # attitude cumsum
            feat_list.append(np.max(np.abs(cumsum[:, 7])))  # velocity cumsum
            feat_list.append(np.max(np.abs(cumsum[:, 11])))  # position cumsum

        all_features.append(feat_list)

    return np.array(all_features)


def generate_attack(clean_df, attack_type, magnitude):
    """
    Generate attack on specific sensor channels.

    Key: Attack ONE modality, leave derivatives unchanged.
    This creates detectable inconsistency.
    """
    attacked = clean_df.copy()
    n = len(clean_df)

    if attack_type == "bias_attitude":
        # Bias roll/pitch but NOT angular rates p/q
        # This creates attitude-rate inconsistency
        attacked["roll"] = attacked["roll"] + magnitude * 0.05
        attacked["pitch"] = attacked["pitch"] + magnitude * 0.05
        # p, q, r remain unchanged -> inconsistency!

    elif attack_type == "bias_velocity":
        # Bias velocity but NOT accelerations
        attacked["vx"] = attacked["vx"] + magnitude * 0.5
        attacked["vy"] = attacked["vy"] + magnitude * 0.5
        # ax, ay, az remain unchanged -> inconsistency!

    elif attack_type == "bias_position":
        # Bias position but NOT velocity
        attacked["x"] = attacked["x"] + magnitude * 1.0
        attacked["y"] = attacked["y"] + magnitude * 1.0
        # vx, vy, vz remain unchanged -> inconsistency!

    elif attack_type == "bias_rates":
        # Bias angular rates but NOT attitudes
        attacked["p"] = attacked["p"] + magnitude * 0.1
        attacked["q"] = attacked["q"] + magnitude * 0.1
        # roll, pitch remain unchanged -> inconsistency!

    elif attack_type == "noise":
        # Add noise to all channels
        for col in ["x", "y", "z", "roll", "pitch", "yaw", "p", "q", "r", "vx", "vy", "vz"]:
            attacked[col] = attacked[col] + np.random.normal(0, magnitude * 0.1, n)

    elif attack_type == "coordinated_bias":
        # Sophisticated attack: bias BOTH attitude AND rates consistently
        # This is HARD to detect (maintains consistency)
        attacked["roll"] = attacked["roll"] + magnitude * 0.05
        attacked["pitch"] = attacked["pitch"] + magnitude * 0.05
        # Also bias rates to maintain consistency (sophisticated attacker)
        attacked["p"] = attacked["p"] + magnitude * 0.001  # Small rate to match
        attacked["q"] = attacked["q"] + magnitude * 0.001

    return attacked


def run_sensor_fusion_evaluation():
    """Main evaluation using sensor fusion consistency checking."""
    print("=" * 70)
    print("SENSOR FUSION ANOMALY DETECTION")
    print("Cross-Modal Consistency Checking for Bias Detection")
    print("=" * 70)

    # Load EuRoC data
    print("\nLoading EuRoC data...")
    df = pd.read_csv(EUROC_PATH)
    sequences = df["sequence"].unique()
    print(f"Found {len(sequences)} sequences")

    # Leave-One-Sequence-Out CV
    all_results = []

    for test_seq in sequences:
        print(f"\n--- Testing on {test_seq} ---")

        # Train/test split
        train_df = df[df["sequence"] != test_seq].copy()
        test_df = df[df["sequence"] == test_seq].copy()

        # Step 1: Compute consistency features on training data
        print("  Computing consistency features on training data...")
        train_consistency = compute_consistency_features(train_df.iloc[:50000])
        train_features = extract_windowed_features(train_consistency)
        print(f"    Got {len(train_features)} feature vectors")

        if len(train_features) < 100:
            print("    Insufficient features, skipping...")
            continue

        # Step 2: Train anomaly detector
        print("  Training IsolationForest...")
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)

        detector = IsolationForest(
            n_estimators=200, contamination=CONTAMINATION, random_state=42, n_jobs=-1
        )
        detector.fit(train_scaled)

        # Step 3: Evaluate on clean test data (FPR)
        print("  Evaluating on clean test data...")
        test_consistency = compute_consistency_features(test_df.iloc[:5000])
        test_features = extract_windowed_features(test_consistency)

        if len(test_features) == 0:
            continue

        test_scaled = scaler.transform(test_features)
        clean_preds = detector.predict(test_scaled)

        fp = np.sum(clean_preds == -1)
        tn = np.sum(clean_preds == 1)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"    FPR on clean: {fpr*100:.1f}%")

        # Step 4: Evaluate on attacks
        attack_types = [
            "bias_attitude",  # Bias attitude, not rates
            "bias_velocity",  # Bias velocity, not accelerations
            "bias_position",  # Bias position, not velocity
            "bias_rates",  # Bias rates, not attitude
            "noise",  # Random noise
            "coordinated_bias",  # Sophisticated attack (maintains consistency)
        ]
        magnitudes = [0.25, 0.5, 1.0, 2.0, 4.0]

        seq_results = {"sequence": test_seq, "fpr": fpr, "attacks": {}}

        for attack_type in attack_types:
            attack_recalls = []

            for magnitude in magnitudes:
                # Generate attack
                test_base = test_df.iloc[:1000].copy()
                attacked_df = generate_attack(test_base, attack_type, magnitude)

                # Compute consistency features on attacked data
                attack_consistency = compute_consistency_features(attacked_df)
                attack_features = extract_windowed_features(attack_consistency)

                if len(attack_features) == 0:
                    continue

                attack_scaled = scaler.transform(attack_features)
                attack_preds = detector.predict(attack_scaled)

                # Recall = fraction detected as anomaly
                recall = np.sum(attack_preds == -1) / len(attack_preds)
                attack_recalls.append(recall)

            avg_recall = np.mean(attack_recalls) if attack_recalls else 0
            seq_results["attacks"][attack_type] = avg_recall
            print(f"    {attack_type:20s}: {avg_recall*100:.1f}% recall")

        all_results.append(seq_results)

    # Summary
    print("\n" + "=" * 70)
    print("SENSOR FUSION DETECTION SUMMARY")
    print("=" * 70)

    if not all_results:
        print("No results collected!")
        return

    avg_fpr = np.mean([r["fpr"] for r in all_results])
    print(f"\nAverage FPR: {avg_fpr*100:.1f}%")

    print("\nPer-Attack Recall (averaged across sequences):")
    attack_types = [
        "bias_attitude",
        "bias_velocity",
        "bias_position",
        "bias_rates",
        "noise",
        "coordinated_bias",
    ]

    for attack_type in attack_types:
        recalls = [r["attacks"].get(attack_type, 0) for r in all_results]
        avg = np.mean(recalls)
        print(f"  {attack_type:20s}: {avg*100:.1f}%")

    # Compute average of bias attacks
    bias_types = ["bias_attitude", "bias_velocity", "bias_position", "bias_rates"]
    bias_recalls = []
    for r in all_results:
        for bt in bias_types:
            if bt in r["attacks"]:
                bias_recalls.append(r["attacks"][bt])

    avg_bias_recall = np.mean(bias_recalls) if bias_recalls else 0

    # Overall (excluding coordinated_bias which is designed to evade)
    detectable_attacks = ["bias_attitude", "bias_velocity", "bias_position", "bias_rates", "noise"]
    all_recalls = []
    for r in all_results:
        for attack in detectable_attacks:
            if attack in r["attacks"]:
                all_recalls.append(r["attacks"][attack])

    overall_avg = np.mean(all_recalls) if all_recalls else 0

    print(f"\nAverage Bias Attack Recall: {avg_bias_recall*100:.1f}%")
    print(f"Overall Average Recall (excl. coordinated): {overall_avg*100:.1f}%")

    # Comparison
    print("\n" + "-" * 70)
    print("COMPARISON: Previous Approaches vs Sensor Fusion")
    print("-" * 70)
    print("Raw multi-scale features:     22.7% recall, 6.5% FPR, 3.1% bias")
    print("PINN-residual features:       11.2% recall, 14.0% FPR, 0.1% bias")
    print(
        f"Sensor Fusion (this):         {overall_avg*100:.1f}% recall, {avg_fpr*100:.1f}% FPR, {avg_bias_recall*100:.1f}% bias"
    )

    if avg_bias_recall > 0.031:
        print("\n*** IMPROVEMENT on bias attacks! ***")

    # Save results
    report = f"""
================================================================================
SENSOR FUSION ANOMALY DETECTION RESULTS
================================================================================

METHODOLOGY:
Cross-modal consistency checking exploits the fact that bias attacks
typically affect ONE sensor modality but NOT its derivative/integral.

Consistency Checks:
1. Attitude-Rate: d(attitude)/dt should match angular rates (p, q, r)
2. Velocity-Acceleration: d(velocity)/dt should match accelerations
3. Position-Velocity: d(position)/dt should match velocities

Attack Types Tested:
- bias_attitude: Bias roll/pitch, leave rates unchanged -> DETECTABLE
- bias_velocity: Bias vx/vy, leave accelerations unchanged -> DETECTABLE
- bias_position: Bias x/y, leave velocities unchanged -> DETECTABLE
- bias_rates: Bias p/q, leave attitudes unchanged -> DETECTABLE
- noise: Random noise on all channels
- coordinated_bias: Sophisticated attack maintaining consistency -> HARD

RESULTS (LOSO-CV):
- Average FPR: {avg_fpr*100:.1f}%
- Overall Recall: {overall_avg*100:.1f}%
- Average Bias Recall: {avg_bias_recall*100:.1f}%

Per-Attack Type:
"""
    for attack_type in attack_types:
        recalls = [r["attacks"].get(attack_type, 0) for r in all_results]
        avg = np.mean(recalls)
        report += f"  {attack_type:20s}: {avg*100:.1f}%\n"

    report += f"""
COMPARISON:
                        Raw Features    PINN-Residual    Sensor Fusion
  Overall Recall:           22.7%           11.2%          {overall_avg*100:.1f}%
  Bias Recall:               3.1%            0.1%          {avg_bias_recall*100:.1f}%
  FPR:                       6.5%           14.0%          {avg_fpr*100:.1f}%

KEY INSIGHT:
Sensor fusion can detect bias attacks that evade single-sensor detectors
by exploiting INCONSISTENCY between sensor modalities.

LIMITATION:
Coordinated attacks that maintain cross-modal consistency are still hard
to detect without external reference (e.g., GPS, visual odometry).
"""

    report_path = OUTPUT_DIR / "SENSOR_FUSION_RESULTS.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nResults saved to: {report_path}")

    return all_results


if __name__ == "__main__":
    run_sensor_fusion_evaluation()
