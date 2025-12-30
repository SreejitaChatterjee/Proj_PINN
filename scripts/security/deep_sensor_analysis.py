"""
Deep analysis of sensor consistency for attack detection.

This script analyzes:
1. Cross-sensor consistency relationships in normal flight
2. How different attacks break different consistency relationships
3. Which consistency signals can detect which attacks

Goal: Design an architecture based on physical sensor relationships.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.security.generate_synthetic_attacks import SyntheticAttackGenerator


def load_data():
    """Load EuRoC data."""
    data_path = Path("data/euroc/all_sequences.csv")
    df = pd.read_csv(data_path)

    # Normalize columns
    for old, new in [("roll", "phi"), ("pitch", "theta"), ("yaw", "psi")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "thrust" not in df.columns:
        df["thrust"] = df["az"] + 9.81 if "az" in df.columns else 9.81
    for col in ["torque_x", "torque_y", "torque_z"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


def compute_consistency_metrics(df: pd.DataFrame, dt: float = 0.005) -> pd.DataFrame:
    """
    Compute cross-sensor consistency metrics.

    These metrics exploit physical relationships between sensors:
    - GPS position derivative should match velocity
    - IMU acceleration integral should match velocity changes
    - Gyro integral should match attitude changes
    - Barometer should match GPS altitude
    - etc.
    """
    metrics = pd.DataFrame(index=df.index[1:])  # Skip first row (need derivatives)

    # ==========================================================================
    # 1. POSITION-VELOCITY CONSISTENCY (GPS internal consistency)
    # ==========================================================================
    # d(position)/dt should equal velocity
    pos_cols = ["x", "y", "z"]
    vel_cols = ["vx", "vy", "vz"]

    for p, v in zip(pos_cols, vel_cols):
        # Numerical derivative of position
        pos_deriv = df[p].diff() / dt
        # Reported velocity
        reported_vel = df[v]
        # Inconsistency
        metrics[f"pos_vel_inconsistency_{p}"] = np.abs(pos_deriv.values[1:] - reported_vel.values[1:])

    # Combined position-velocity inconsistency (L2 norm)
    metrics["pos_vel_inconsistency"] = np.sqrt(
        metrics["pos_vel_inconsistency_x"]**2 +
        metrics["pos_vel_inconsistency_y"]**2 +
        metrics["pos_vel_inconsistency_z"]**2
    )

    # ==========================================================================
    # 2. VELOCITY-ACCELERATION CONSISTENCY (IMU integration check)
    # ==========================================================================
    # d(velocity)/dt should approximately equal acceleration (minus gravity)
    accel_cols = ["ax", "ay", "az"] if "ax" in df.columns else None

    if accel_cols:
        for v, a in zip(vel_cols, ["ax", "ay", "az"]):
            vel_deriv = df[v].diff() / dt
            reported_accel = df[a].values[1:]
            # For az, need to account for gravity
            if a == "az":
                reported_accel = reported_accel + 9.81
            metrics[f"vel_accel_inconsistency_{v}"] = np.abs(vel_deriv.values[1:] - reported_accel)

        metrics["vel_accel_inconsistency"] = np.sqrt(
            metrics["vel_accel_inconsistency_vx"]**2 +
            metrics["vel_accel_inconsistency_vy"]**2 +
            metrics["vel_accel_inconsistency_vz"]**2
        )

    # ==========================================================================
    # 3. ATTITUDE-ANGULAR RATE CONSISTENCY (Gyro integration check)
    # ==========================================================================
    # d(attitude)/dt should equal angular rates (simplified, ignoring rotation matrices)
    att_cols = ["phi", "theta", "psi"]
    rate_cols = ["p", "q", "r"]

    for att, rate in zip(att_cols, rate_cols):
        att_deriv = df[att].diff() / dt
        reported_rate = df[rate]
        metrics[f"att_rate_inconsistency_{att}"] = np.abs(att_deriv.values[1:] - reported_rate.values[1:])

    metrics["att_rate_inconsistency"] = np.sqrt(
        metrics["att_rate_inconsistency_phi"]**2 +
        metrics["att_rate_inconsistency_theta"]**2 +
        metrics["att_rate_inconsistency_psi"]**2
    )

    # ==========================================================================
    # 4. KINEMATIC CONSISTENCY (Position from double integration)
    # ==========================================================================
    # x(t) = x(0) + integral(v dt) should match reported position
    # Compute over sliding windows
    window = 20  # 100ms window at 200Hz

    for p, v in zip(pos_cols, vel_cols):
        # Integrated position change over window
        vel_integral = df[v].rolling(window).sum() * dt
        # Actual position change over window
        pos_change = df[p].diff(window)
        metrics[f"kinematic_inconsistency_{p}"] = np.abs(
            vel_integral.values[1:] - pos_change.values[1:]
        )

    metrics["kinematic_inconsistency"] = np.sqrt(
        metrics["kinematic_inconsistency_x"]**2 +
        metrics["kinematic_inconsistency_y"]**2 +
        metrics["kinematic_inconsistency_z"]**2
    )

    # ==========================================================================
    # 5. ENERGY CONSISTENCY (Physics-based)
    # ==========================================================================
    # Kinetic energy change should relate to thrust and gravity
    speed_sq = df["vx"]**2 + df["vy"]**2 + df["vz"]**2
    kinetic_energy_change = speed_sq.diff() / dt

    # Expected energy change from thrust (simplified)
    if "thrust" in df.columns:
        thrust_power = df["thrust"] * df["vz"]  # Power = Force * velocity
        metrics["energy_inconsistency"] = np.abs(
            kinetic_energy_change.values[1:] - thrust_power.values[1:]
        )

    # ==========================================================================
    # 6. SMOOTHNESS METRICS (Jerk and angular acceleration)
    # ==========================================================================
    # High jerk indicates discontinuities (possible attack injection point)
    for v in vel_cols:
        accel = df[v].diff() / dt
        jerk = accel.diff() / dt
        metrics[f"jerk_{v}"] = np.abs(jerk.values[1:])

    metrics["jerk_magnitude"] = np.sqrt(
        metrics["jerk_vx"]**2 +
        metrics["jerk_vy"]**2 +
        metrics["jerk_vz"]**2
    )

    # Angular acceleration (should be bounded)
    for r in rate_cols:
        angular_accel = df[r].diff() / dt
        metrics[f"angular_accel_{r}"] = np.abs(angular_accel.values[1:])

    metrics["angular_accel_magnitude"] = np.sqrt(
        metrics["angular_accel_p"]**2 +
        metrics["angular_accel_q"]**2 +
        metrics["angular_accel_r"]**2
    )

    # ==========================================================================
    # 7. STATISTICAL ANOMALY METRICS
    # ==========================================================================
    # Z-scores for key variables (how many std devs from mean)
    for col in ["vx", "vy", "vz", "p", "q", "r", "phi", "theta"]:
        rolling_mean = df[col].rolling(100).mean()
        rolling_std = df[col].rolling(100).std()
        z_score = (df[col] - rolling_mean) / (rolling_std + 1e-6)
        metrics[f"zscore_{col}"] = np.abs(z_score.values[1:])

    # ==========================================================================
    # 8. CROSS-CORRELATION ANOMALY
    # ==========================================================================
    # Correlation between position and velocity should be consistent
    window = 50
    for p, v in zip(pos_cols, vel_cols):
        rolling_corr = df[p].rolling(window).corr(df[v])
        # Correlation breakdown indicates inconsistency
        metrics[f"correlation_{p}_{v}"] = rolling_corr.values[1:]

    return metrics


def analyze_attack_signatures(normal_metrics: pd.DataFrame,
                               attack_metrics: dict,
                               attack_labels: dict) -> dict:
    """
    Analyze which consistency metrics detect which attacks.

    Returns a matrix of [attack_type x metric] detection power.
    """
    results = {}

    # Compute normal thresholds (99th percentile)
    normal_thresholds = {}
    for col in normal_metrics.columns:
        clean_vals = normal_metrics[col].dropna()
        if len(clean_vals) > 0:
            normal_thresholds[col] = np.percentile(clean_vals, 99)

    # For each attack, compute detection rate per metric
    for attack_name, attack_metric_df in attack_metrics.items():
        if attack_name == "clean":
            continue

        labels = attack_labels[attack_name]
        attack_mask = labels[1:] == 1  # Align with metrics (which skip first row)
        attack_mask = attack_mask[:len(attack_metric_df)]  # Ensure same length

        if attack_mask.sum() == 0:
            continue

        attack_results = {}
        for col in attack_metric_df.columns:
            if col not in normal_thresholds:
                continue

            threshold = normal_thresholds[col]
            attack_vals = attack_metric_df[col].values

            # Detection rate: % of attack samples exceeding threshold
            attack_samples = attack_vals[attack_mask]
            attack_samples = attack_samples[~np.isnan(attack_samples)]

            if len(attack_samples) > 0:
                detection_rate = np.mean(attack_samples > threshold)
                separation = (np.mean(attack_samples) - np.mean(normal_metrics[col].dropna())) / (np.std(normal_metrics[col].dropna()) + 1e-6)
            else:
                detection_rate = 0
                separation = 0

            attack_results[col] = {
                "detection_rate": float(detection_rate),
                "separation": float(separation)
            }

        results[attack_name] = attack_results

    return results


def main():
    print("=" * 80)
    print("DEEP SENSOR CONSISTENCY ANALYSIS")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    df = load_data()

    # Use one sequence for analysis
    test_seq = "V1_01_easy"
    df_test = df[df["sequence"] == test_seq].reset_index(drop=True)
    print(f"  Using sequence: {test_seq} ({len(df_test):,} samples)")

    # Compute consistency metrics on clean data
    print("\n[2/5] Computing consistency metrics on CLEAN data...")
    clean_metrics = compute_consistency_metrics(df_test)
    print(f"  Computed {len(clean_metrics.columns)} consistency metrics")

    # Print metric statistics
    print("\n  Metric statistics (clean data):")
    key_metrics = [
        "pos_vel_inconsistency",
        "att_rate_inconsistency",
        "kinematic_inconsistency",
        "jerk_magnitude",
        "angular_accel_magnitude"
    ]
    for m in key_metrics:
        if m in clean_metrics.columns:
            vals = clean_metrics[m].dropna()
            print(f"    {m:30s}: mean={vals.mean():.4f}, std={vals.std():.4f}, 99th={np.percentile(vals, 99):.4f}")

    # Generate attacks and compute metrics
    print("\n[3/5] Generating attacks and computing metrics...")
    generator = SyntheticAttackGenerator(df_test, seed=42, randomize=False)
    attacks = generator.generate_all_attacks(handle_nan=True)

    attack_metrics = {}
    attack_labels = {}

    for attack_name, attack_data in attacks.items():
        metrics = compute_consistency_metrics(attack_data)
        attack_metrics[attack_name] = metrics
        attack_labels[attack_name] = attack_data["label"].values

    print(f"  Analyzed {len(attacks)} attack types")

    # Analyze attack signatures
    print("\n[4/5] Analyzing attack signatures...")
    signatures = analyze_attack_signatures(clean_metrics, attack_metrics, attack_labels)

    # Find best metric for each attack
    print("\n" + "=" * 80)
    print("ATTACK DETECTION ANALYSIS")
    print("=" * 80)

    attack_detection_summary = {}

    for attack_name, metrics_results in signatures.items():
        # Sort metrics by detection rate
        sorted_metrics = sorted(
            metrics_results.items(),
            key=lambda x: x[1]["detection_rate"],
            reverse=True
        )

        best_metric = sorted_metrics[0] if sorted_metrics else (None, {"detection_rate": 0})
        best_metric_name = best_metric[0]
        best_detection_rate = best_metric[1]["detection_rate"]

        attack_detection_summary[attack_name] = {
            "best_metric": best_metric_name,
            "best_detection_rate": best_detection_rate,
            "top_5_metrics": [(m, r["detection_rate"]) for m, r in sorted_metrics[:5]]
        }

        status = "[OK]" if best_detection_rate > 0.5 else "[..] " if best_detection_rate > 0.1 else "[X]"
        print(f"\n{status} {attack_name}")
        print(f"  Best metric: {best_metric_name} (detection: {best_detection_rate*100:.1f}%)")
        if len(sorted_metrics) > 1:
            print(f"  Runner-up:   {sorted_metrics[1][0]} (detection: {sorted_metrics[1][1]['detection_rate']*100:.1f}%)")

    # Categorize attacks by detectability
    print("\n" + "=" * 80)
    print("DETECTION CATEGORIES")
    print("=" * 80)

    easily_detected = []
    partially_detected = []
    hard_to_detect = []

    for attack, summary in attack_detection_summary.items():
        rate = summary["best_detection_rate"]
        if rate > 0.5:
            easily_detected.append((attack, rate, summary["best_metric"]))
        elif rate > 0.1:
            partially_detected.append((attack, rate, summary["best_metric"]))
        else:
            hard_to_detect.append((attack, rate, summary["best_metric"]))

    print(f"\n[OK] EASILY DETECTED (>50% with single metric): {len(easily_detected)}")
    for attack, rate, metric in sorted(easily_detected, key=lambda x: -x[1]):
        print(f"    {attack:30s} {rate*100:5.1f}% via {metric}")

    print(f"\n[..] PARTIALLY DETECTED (10-50%): {len(partially_detected)}")
    for attack, rate, metric in sorted(partially_detected, key=lambda x: -x[1]):
        print(f"    {attack:30s} {rate*100:5.1f}% via {metric}")

    print(f"\n[X] HARD TO DETECT (<10%): {len(hard_to_detect)}")
    for attack, rate, metric in sorted(hard_to_detect, key=lambda x: -x[1]):
        print(f"    {attack:30s} {rate*100:5.1f}% via {metric}")

    # Identify key metrics
    print("\n" + "=" * 80)
    print("KEY CONSISTENCY METRICS")
    print("=" * 80)

    metric_effectiveness = {}
    for attack, metrics_results in signatures.items():
        for metric, results in metrics_results.items():
            if metric not in metric_effectiveness:
                metric_effectiveness[metric] = []
            metric_effectiveness[metric].append(results["detection_rate"])

    # Average detection rate per metric
    metric_avg = {m: np.mean(rates) for m, rates in metric_effectiveness.items()}
    sorted_metrics = sorted(metric_avg.items(), key=lambda x: -x[1])

    print("\nTop 15 most effective metrics (avg detection rate across all attacks):")
    for i, (metric, avg_rate) in enumerate(sorted_metrics[:15]):
        print(f"  {i+1:2d}. {metric:40s} {avg_rate*100:.1f}%")

    # Save results
    print("\n[5/5] Saving analysis results...")
    output_dir = Path("research/security/sensor_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "attack_detection_summary": attack_detection_summary,
        "metric_effectiveness": {m: float(v) for m, v in metric_avg.items()},
        "top_metrics": [m for m, _ in sorted_metrics[:15]],
        "easily_detected_attacks": [a for a, _, _ in easily_detected],
        "hard_to_detect_attacks": [a for a, _, _ in hard_to_detect],
    }

    with open(output_dir / "sensor_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {output_dir / 'sensor_analysis_results.json'}")

    # Architecture recommendations
    print("\n" + "=" * 80)
    print("ARCHITECTURE RECOMMENDATIONS")
    print("=" * 80)

    print("""
Based on this analysis, the detector should:

1. PRIMARY SIGNALS (High detection power):
   - Position-velocity inconsistency (GPS internal check)
   - Attitude-rate inconsistency (Gyro integration check)
   - Kinematic consistency (Double integration check)
   - Jerk magnitude (Smoothness check)

2. SECONDARY SIGNALS (For stealthy attacks):
   - Z-scores for velocity/rates (Statistical anomaly)
   - Rolling correlations (Pattern consistency)
   - Angular acceleration bounds (Physical limits)

3. ARCHITECTURE INSIGHT:
   - Attacks that fool ONE consistency metric often fail others
   - Need ENSEMBLE of consistency checks, not single metric
   - Temporal context is crucial (attacks often have sharp onset)

4. HARD-TO-DETECT ATTACKS need:
   - Longer temporal context (replay, slow drift)
   - Cross-sequence learning (what's normal trajectory shape)
   - Multi-scale analysis (different window sizes)
""")


if __name__ == "__main__":
    main()
