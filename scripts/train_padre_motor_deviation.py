"""
PADRE Fault Detection via Motor Deviation Analysis.

Key Insight: Don't learn what "normal" looks like.
Instead, detect when ANY motor deviates from the others.

Normal flight: All 4 motors behave similarly
Faulty flight: Damaged motor(s) behave differently

This approach should generalize across drones because it's
based on RELATIVE motor comparison, not absolute patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from scipy.stats import kurtosis, skew
import re


def extract_motor_deviation_features(window):
    """
    Extract features based on motor-to-motor comparison.

    The key idea: faults cause ONE motor to deviate from the group.
    We measure how much each motor differs from the average of all motors.
    """
    n_motors = 4
    n_sensors_per_motor = 6  # 3 accel + 3 gyro

    features = []

    # Compute per-motor statistics
    motor_stats = []
    for m in range(n_motors):
        start = m * n_sensors_per_motor
        end = start + n_sensors_per_motor
        motor_data = window[:, start:end]

        # Basic stats
        motor_mean = motor_data.mean()
        motor_std = motor_data.std()
        motor_energy = np.sum(motor_data ** 2)
        motor_range = motor_data.max() - motor_data.min()

        # Vibration intensity (RMS)
        motor_rms = np.sqrt(np.mean(motor_data ** 2))

        # Frequency content
        motor_fft_energy = 0
        for col in range(n_sensors_per_motor):
            fft = np.abs(np.fft.rfft(motor_data[:, col]))
            motor_fft_energy += fft.sum()

        # Kurtosis (peakedness - faults often have spiky signals)
        motor_kurtosis = np.mean([kurtosis(motor_data[:, c]) for c in range(n_sensors_per_motor)])

        motor_stats.append({
            'mean': motor_mean,
            'std': motor_std,
            'energy': motor_energy,
            'range': motor_range,
            'rms': motor_rms,
            'fft_energy': motor_fft_energy,
            'kurtosis': motor_kurtosis
        })

    # Compute group average
    avg_std = np.mean([m['std'] for m in motor_stats])
    avg_energy = np.mean([m['energy'] for m in motor_stats])
    avg_rms = np.mean([m['rms'] for m in motor_stats])
    avg_fft = np.mean([m['fft_energy'] for m in motor_stats])
    avg_kurtosis = np.mean([m['kurtosis'] for m in motor_stats])

    # DEVIATION FEATURES: How much does each motor differ from the group?
    for m in range(n_motors):
        ms = motor_stats[m]

        # Deviation from group average (normalized)
        std_dev = (ms['std'] - avg_std) / (avg_std + 1e-8)
        energy_dev = (ms['energy'] - avg_energy) / (avg_energy + 1e-8)
        rms_dev = (ms['rms'] - avg_rms) / (avg_rms + 1e-8)
        fft_dev = (ms['fft_energy'] - avg_fft) / (avg_fft + 1e-8)
        kurtosis_dev = (ms['kurtosis'] - avg_kurtosis) / (np.abs(avg_kurtosis) + 1e-8)

        features.extend([std_dev, energy_dev, rms_dev, fft_dev, kurtosis_dev])

    # MAX DEVIATION: The most anomalous motor (key fault indicator)
    std_devs = [(motor_stats[m]['std'] - avg_std) / (avg_std + 1e-8) for m in range(n_motors)]
    energy_devs = [(motor_stats[m]['energy'] - avg_energy) / (avg_energy + 1e-8) for m in range(n_motors)]
    rms_devs = [(motor_stats[m]['rms'] - avg_rms) / (avg_rms + 1e-8) for m in range(n_motors)]

    features.append(max(np.abs(std_devs)))      # Max std deviation
    features.append(max(np.abs(energy_devs)))   # Max energy deviation
    features.append(max(np.abs(rms_devs)))      # Max RMS deviation
    features.append(np.std(std_devs))           # Spread of deviations
    features.append(np.std(energy_devs))

    # PAIRWISE MOTOR CORRELATIONS
    # Normal: high correlation between motors
    # Faulty: damaged motor has low correlation with others
    for m1 in range(n_motors):
        for m2 in range(m1 + 1, n_motors):
            s1, e1 = m1 * n_sensors_per_motor, (m1 + 1) * n_sensors_per_motor
            s2, e2 = m2 * n_sensors_per_motor, (m2 + 1) * n_sensors_per_motor

            # Flatten and correlate
            d1 = window[:, s1:e1].flatten()
            d2 = window[:, s2:e2].flatten()

            corr = np.corrcoef(d1, d2)[0, 1]
            features.append(corr if not np.isnan(corr) else 0)

    # MIN CORRELATION (faulty motor will have low correlation)
    # This is computed from the 6 pairwise correlations above
    pairwise_corrs = features[-6:]  # Last 6 features are correlations
    features.append(min(pairwise_corrs))
    features.append(np.std(pairwise_corrs))

    # ASYMMETRY FEATURES
    # Compare opposite motors (A vs C, B vs D)
    for pair in [(0, 2), (1, 3)]:
        m1, m2 = pair
        s1, e1 = m1 * n_sensors_per_motor, (m1 + 1) * n_sensors_per_motor
        s2, e2 = m2 * n_sensors_per_motor, (m2 + 1) * n_sensors_per_motor

        asymmetry = np.abs(window[:, s1:e1].std() - window[:, s2:e2].std())
        asymmetry_norm = asymmetry / (avg_std + 1e-8)
        features.append(asymmetry_norm)

    return np.array(features)


def load_drone_data(data_dir, window_size=256, stride=128):
    X, y = [], []
    for csv_file in sorted(Path(data_dir).glob('*.csv')):
        match = re.search(r'_(\d{4})\.csv', csv_file.name)
        if not match:
            continue
        codes = match.group(1)
        is_faulty = 1 if any(int(c) > 0 for c in codes) else 0

        df = pd.read_csv(csv_file)
        data = df.values.astype(np.float32)[:, :24]

        for i in range((len(data) - window_size) // stride + 1):
            window = data[i * stride: i * stride + window_size]
            X.append(extract_motor_deviation_features(window))
            y.append(is_faulty)

    return np.array(X), np.array(y)


def main():
    print("=" * 70)
    print("MOTOR DEVIATION ANALYSIS - Drone-Agnostic Fault Detection")
    print("=" * 70)
    print("\nKey Idea: Detect when ANY motor deviates from the group")
    print("This should generalize because it's RELATIVE, not ABSOLUTE\n")

    bebop_dir = Path('data/PADRE_dataset/Parrot_Bebop_2/Normalized_data')
    solo_dir = Path('data/PADRE_dataset/3DR_Solo/Normalized_data/extracted')

    X_bebop, y_bebop = load_drone_data(bebop_dir)
    X_solo, y_solo = load_drone_data(solo_dir)

    print(f"Features per sample: {X_bebop.shape[1]}")
    print(f"Bebop: {len(X_bebop)} samples ({sum(y_bebop==0)} normal, {sum(y_bebop==1)} faulty)")
    print(f"Solo:  {len(X_solo)} samples ({sum(y_solo==0)} normal, {sum(y_solo==1)} faulty)")

    # Test 1: Train on Bebop, test on Solo
    print("\n" + "=" * 70)
    print("TEST 1: Train on Bebop -> Test on Solo")
    print("=" * 70)

    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                  random_state=42, n_jobs=-1, max_depth=15)
    clf.fit(X_bebop, y_bebop)
    y_pred = clf.predict(X_solo)

    cm = confusion_matrix(y_solo, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Accuracy:  {accuracy_score(y_solo, y_pred)*100:.1f}%")
    print(f"F1 Score:  {f1_score(y_solo, y_pred)*100:.1f}%")
    print(f"Normal:    {tn}/{tn+fp} = {tn/(tn+fp)*100:.1f}%")
    print(f"Faulty:    {tp}/{tp+fn} = {tp/(tp+fn)*100:.1f}%")
    print(f"FP={fp}, FN={fn}")

    # Test 2: Train on Solo, test on Bebop
    print("\n" + "=" * 70)
    print("TEST 2: Train on Solo -> Test on Bebop")
    print("=" * 70)

    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                  random_state=42, n_jobs=-1, max_depth=15)
    clf.fit(X_solo, y_solo)
    y_pred = clf.predict(X_bebop)

    cm = confusion_matrix(y_bebop, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Accuracy:  {accuracy_score(y_bebop, y_pred)*100:.1f}%")
    print(f"F1 Score:  {f1_score(y_bebop, y_pred)*100:.1f}%")
    print(f"Normal:    {tn}/{tn+fp} = {tn/(tn+fp)*100:.1f}%")
    print(f"Faulty:    {tp}/{tp+fn} = {tp/(tp+fn)*100:.1f}%")
    print(f"FP={fp}, FN={fn}")

    # Feature importance analysis
    print("\n" + "=" * 70)
    print("TOP FEATURES (from Bebop model)")
    print("=" * 70)

    clf.fit(X_bebop, y_bebop)
    feature_names = (
        [f"Motor{m}_{stat}_dev" for m in ['A','B','C','D']
         for stat in ['std', 'energy', 'rms', 'fft', 'kurtosis']] +
        ['max_std_dev', 'max_energy_dev', 'max_rms_dev', 'std_spread', 'energy_spread'] +
        [f"corr_{p}" for p in ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']] +
        ['min_corr', 'corr_spread', 'asymm_AC', 'asymm_BD']
    )

    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(top_idx):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"  {i+1}. {name}: {importances[idx]:.4f}")


if __name__ == "__main__":
    main()
