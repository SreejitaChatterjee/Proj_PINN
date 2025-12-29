"""
Cross-Dataset Transfer Test for Sensor Fusion Approach

Key Question: Does sensor fusion generalize across platforms?
- Train on EuRoC (MAV platform)
- Test on PADRE (Bebop2, Solo drones)

If physics-based consistency checking works, it SHOULD transfer.
This is the real test of whether the approach has value.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
EUROC_PATH = PROJECT_ROOT / "data" / "euroc" / "all_sequences.csv"
PADRE_PATH = PROJECT_ROOT / "data" / "padre"
OUTPUT_DIR = PROJECT_ROOT / "models" / "security" / "sensor_fusion_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTAMINATION = 0.05


def compute_consistency_features(data, dt=0.005):
    """Compute cross-modal consistency features."""
    pos = data[['x', 'y', 'z']].values
    att = data[['roll', 'pitch', 'yaw']].values
    rates = data[['p', 'q', 'r']].values
    vel = data[['vx', 'vy', 'vz']].values
    acc = data[['ax', 'ay', 'az']].values

    n = len(data) - 1
    features = []

    for i in range(n):
        feat = []

        # Attitude-Rate consistency
        att_change = att[i+1] - att[i]
        expected_att_change = rates[i] * dt
        att_error = att_change - expected_att_change
        feat.extend([np.abs(att_error[0]), np.abs(att_error[1]),
                     np.abs(att_error[2]), np.linalg.norm(att_error)])

        # Velocity-Acceleration consistency
        vel_change = vel[i+1] - vel[i]
        expected_vel_change = acc[i] * dt
        vel_error = vel_change - expected_vel_change
        feat.extend([np.abs(vel_error[0]), np.abs(vel_error[1]),
                     np.abs(vel_error[2]), np.linalg.norm(vel_error)])

        # Position-Velocity consistency
        pos_change = pos[i+1] - pos[i]
        expected_pos_change = vel[i] * dt
        pos_error = pos_change - expected_pos_change
        feat.extend([np.abs(pos_error[0]), np.abs(pos_error[1]),
                     np.abs(pos_error[2]), np.linalg.norm(pos_error)])

        features.append(feat)

    return np.array(features)


def extract_windowed_features(consistency_feats, windows=[5, 10, 25, 50]):
    """Extract multi-scale features from consistency errors."""
    if len(consistency_feats) < max(windows):
        return np.array([])

    all_features = []
    max_window = max(windows)

    for i in range(max_window, len(consistency_feats)):
        feat_list = []
        for w_size in windows:
            window = consistency_feats[i-w_size:i]
            feat_list.extend(np.mean(window, axis=0))
            feat_list.extend(np.std(window, axis=0))
            feat_list.extend(np.max(np.abs(window), axis=0))
            cumsum = np.cumsum(window, axis=0)
            feat_list.append(np.max(np.abs(cumsum[:, 3])))
            feat_list.append(np.max(np.abs(cumsum[:, 7])))
            feat_list.append(np.max(np.abs(cumsum[:, 11])))
        all_features.append(feat_list)

    return np.array(all_features)


def load_padre_as_euroc_format(padre_file):
    """
    Convert PADRE format to EuRoC-like format.

    PADRE has per-motor IMU data. We need to:
    1. Average/aggregate to get single IMU reading
    2. Map to EuRoC column names
    """
    df = pd.read_csv(padre_file)

    # PADRE columns typically include per-motor data
    # We'll try to extract relevant columns
    result = pd.DataFrame()

    # Check available columns
    cols = df.columns.tolist()

    # Try to find position (might not exist in PADRE)
    if 'x' in cols:
        result['x'] = df['x']
        result['y'] = df['y']
        result['z'] = df['z']
    else:
        # Integrate from velocity if available, else zeros
        result['x'] = 0
        result['y'] = 0
        result['z'] = 0

    # Attitude - PADRE might have different naming
    if 'roll' in cols:
        result['roll'] = df['roll']
        result['pitch'] = df['pitch']
        result['yaw'] = df['yaw']
    elif 'phi' in cols:
        result['roll'] = df['phi']
        result['pitch'] = df['theta']
        result['yaw'] = df['psi']
    else:
        result['roll'] = 0
        result['pitch'] = 0
        result['yaw'] = 0

    # Angular rates - average across motors if multiple
    gyro_cols = [c for c in cols if 'gyro' in c.lower() or c in ['p', 'q', 'r']]
    if 'p' in cols:
        result['p'] = df['p']
        result['q'] = df['q']
        result['r'] = df['r']
    elif 'gx_1' in cols:
        # Average across motors
        result['p'] = df[[c for c in cols if 'gx' in c]].mean(axis=1)
        result['q'] = df[[c for c in cols if 'gy' in c]].mean(axis=1)
        result['r'] = df[[c for c in cols if 'gz' in c]].mean(axis=1)
    else:
        result['p'] = 0
        result['q'] = 0
        result['r'] = 0

    # Velocities - might need to integrate or use available
    if 'vx' in cols:
        result['vx'] = df['vx']
        result['vy'] = df['vy']
        result['vz'] = df['vz']
    else:
        result['vx'] = 0
        result['vy'] = 0
        result['vz'] = 0

    # Accelerations - average across motors
    if 'ax' in cols:
        result['ax'] = df['ax']
        result['ay'] = df['ay']
        result['az'] = df['az']
    elif 'ax_1' in cols:
        result['ax'] = df[[c for c in cols if 'ax' in c]].mean(axis=1)
        result['ay'] = df[[c for c in cols if 'ay' in c]].mean(axis=1)
        result['az'] = df[[c for c in cols if 'az' in c]].mean(axis=1)
    else:
        result['ax'] = 0
        result['ay'] = 0
        result['az'] = 0

    return result


def run_cross_dataset_test():
    """Test sensor fusion generalization: EuRoC -> PADRE."""
    print("=" * 70)
    print("CROSS-DATASET TRANSFER TEST: EuRoC -> PADRE")
    print("Testing if sensor fusion generalizes across platforms")
    print("=" * 70)

    # Step 1: Train on EuRoC
    print("\n--- Step 1: Train detector on EuRoC ---")
    print("Loading EuRoC training data...")
    euroc_df = pd.read_csv(EUROC_PATH)
    print(f"  Loaded {len(euroc_df)} samples from EuRoC")

    print("  Computing consistency features...")
    euroc_consistency = compute_consistency_features(euroc_df.iloc[:50000])
    euroc_features = extract_windowed_features(euroc_consistency)
    print(f"  Extracted {len(euroc_features)} feature vectors")

    print("  Training detector...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(euroc_features)

    detector = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )
    detector.fit(train_scaled)
    print("  Training complete.")

    # Step 2: Test on PADRE
    print("\n--- Step 2: Test on PADRE dataset ---")
    print("Loading PADRE dataset...")

    if not PADRE_PATH.exists():
        print(f"  ERROR: PADRE path not found: {PADRE_PATH}")
        return

    padre_files = list(PADRE_PATH.glob("*.csv"))
    if not padre_files:
        print("  ERROR: No PADRE CSV files found")
        return

    print(f"  Found {len(padre_files)} PADRE files")

    # Classify files
    normal_files = [f for f in padre_files if '_0000' in f.name]
    fault_files = [f for f in padre_files if '_0000' not in f.name and f.name != 'all_sequences.csv']

    print(f"  Normal flights: {len(normal_files)}")
    print(f"  Fault flights: {len(fault_files)}")

    # Test on normal flights (should have low FPR)
    print("\n  Testing on PADRE normal flights...")
    normal_fprs = []
    for f in normal_files[:5]:
        try:
            padre_df = load_padre_as_euroc_format(f)
            if len(padre_df) < 100:
                continue

            # Check if we have meaningful data
            if padre_df['ax'].abs().max() < 0.01:
                print(f"    {f.name}: No meaningful acceleration data, skipping")
                continue

            consistency = compute_consistency_features(padre_df, dt=0.01)  # PADRE might be 100Hz
            features = extract_windowed_features(consistency)

            if len(features) == 0:
                continue

            features_scaled = scaler.transform(features)
            preds = detector.predict(features_scaled)
            fpr = np.sum(preds == -1) / len(preds)
            normal_fprs.append(fpr)
            print(f"    {f.name}: FPR = {fpr*100:.1f}%")
        except Exception as e:
            print(f"    {f.name}: Error - {e}")

    # Test on fault flights (should have high recall)
    print("\n  Testing on PADRE fault flights...")
    fault_recalls = []
    for f in fault_files[:15]:
        try:
            padre_df = load_padre_as_euroc_format(f)
            if len(padre_df) < 100:
                continue

            if padre_df['ax'].abs().max() < 0.01:
                continue

            consistency = compute_consistency_features(padre_df, dt=0.01)
            features = extract_windowed_features(consistency)

            if len(features) == 0:
                continue

            features_scaled = scaler.transform(features)
            preds = detector.predict(features_scaled)
            recall = np.sum(preds == -1) / len(preds)
            fault_recalls.append(recall)
            print(f"    {f.name}: Recall = {recall*100:.1f}%")
        except Exception as e:
            print(f"    {f.name}: Error - {e}")

    # Summary
    print("\n" + "=" * 70)
    print("CROSS-DATASET TRANSFER RESULTS")
    print("=" * 70)

    if normal_fprs:
        avg_fpr = np.mean(normal_fprs)
        print(f"\nFPR on PADRE Normal: {avg_fpr*100:.1f}% +/- {np.std(normal_fprs)*100:.1f}%")
    else:
        avg_fpr = 0
        print("\nNo normal flight results")

    if fault_recalls:
        avg_recall = np.mean(fault_recalls)
        print(f"Recall on PADRE Faults: {avg_recall*100:.1f}% +/- {np.std(fault_recalls)*100:.1f}%")
    else:
        avg_recall = 0
        print("No fault flight results")

    # Comparison
    print("\n" + "-" * 70)
    print("COMPARISON: Within-Dataset vs Cross-Dataset")
    print("-" * 70)
    print("Within EuRoC (Sensor Fusion):  49.0% recall, 7.5% FPR")
    print(f"Cross to PADRE:                {avg_recall*100:.1f}% recall, {avg_fpr*100:.1f}% FPR")

    if avg_recall > 0.10:
        print("\n*** SENSOR FUSION TRANSFERS BETTER THAN RAW FEATURES! ***")
        print("(Raw features: 3.3% cross-dataset recall)")
    else:
        print("\n Cross-dataset transfer still challenging.")

    # Save results
    report = f"""
================================================================================
SENSOR FUSION CROSS-DATASET TRANSFER RESULTS
================================================================================

METHODOLOGY:
- Train: Sensor fusion detector on EuRoC (MAV platform)
- Test: PADRE dataset (Bebop2, Solo drones)
- Physics-based consistency should be platform-independent

RESULTS:
- FPR on PADRE Normal: {avg_fpr*100:.1f}%
- Recall on PADRE Faults: {avg_recall*100:.1f}%

COMPARISON:
                        Within EuRoC    Cross to PADRE
  Raw Features:             22.7%           3.3%
  Sensor Fusion:            49.0%          {avg_recall*100:.1f}%

CONCLUSION:
{"Sensor fusion shows better cross-dataset transfer than raw features!" if avg_recall > 0.10 else "Cross-dataset transfer remains challenging."}
"""

    report_path = OUTPUT_DIR / "CROSS_DATASET_RESULTS.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nResults saved to: {report_path}")


if __name__ == "__main__":
    run_cross_dataset_test()
