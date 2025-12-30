#!/usr/bin/env python3
"""
Test all 4 fixes for generalization + additional measures.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

print('='*70)
print('COMPREHENSIVE FIX COMPARISON + ADDITIONAL MEASURES')
print('='*70)

# Load data
df = pd.read_csv(Path(__file__).parent.parent.parent / 'data/euroc/all_sequences.csv')
state_cols = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vx', 'vy', 'vz']
print(f'Loaded {len(df):,} samples')

# Standard feature extraction
def extract_features(data, window_size=50):
    features = []
    for i in range(window_size, len(data)):
        window = data[i-window_size:i]
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        diff = np.diff(window, axis=0)
        diff_mean = np.mean(diff, axis=0)
        diff_std = np.std(diff, axis=0)
        feat = np.concatenate([mean, std, diff_mean, diff_std])
        features.append(feat)
    return np.array(features)

# Physics-based feature extraction (residuals)
def extract_physics_features(data, dt=0.005):
    features = []
    for i in range(3, len(data)-1):
        # Velocity from position difference
        vel = (data[i, :3] - data[i-1, :3]) / dt
        vel_prev = (data[i-1, :3] - data[i-2, :3]) / dt

        # Acceleration
        acc = (vel - vel_prev) / dt
        acc_prev = (vel_prev - (data[i-2, :3] - data[i-3, :3])/dt) / dt

        # Jerk
        jerk = (acc - acc_prev) / dt

        # Position prediction residual
        pos_pred = data[i-1, :3] + vel_prev * dt
        residual = data[i, :3] - pos_pred

        # Attitude rate consistency
        att_diff = data[i, 3:6] - data[i-1, 3:6]
        att_rate = data[i, 6:9]
        att_residual = att_diff/dt - att_rate

        feat = np.concatenate([
            residual,
            np.abs(residual),
            jerk,
            np.abs(jerk),
            att_residual,
            [np.linalg.norm(residual)],
            [np.linalg.norm(jerk)],
            [np.linalg.norm(att_residual)],
            acc,
            [np.linalg.norm(acc)],
        ])
        features.append(feat)
    return np.array(features)

# Normalized deviation features (scale-invariant)
def extract_normalized_features(data, window_size=50):
    features = []
    for i in range(window_size, len(data)):
        window = data[i-window_size:i]
        current = data[i]

        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0) + 1e-8

        # Z-score of current point
        z_score = (current - mean) / std

        # Max deviation in window
        max_dev = np.max(np.abs(window - mean), axis=0) / std

        # Rate of change normalized
        diff = np.diff(window, axis=0)
        diff_std = np.std(diff, axis=0) + 1e-8
        current_diff = window[-1] - window[-2]
        diff_z = current_diff / diff_std

        feat = np.concatenate([z_score, max_dev, diff_z, np.abs(z_score), np.abs(diff_z)])
        features.append(feat)
    return np.array(features)

# Test attacks with varying parameters
def generate_test_attacks(base_data, magnitudes=[0.25, 0.5, 1.0, 2.0, 4.0]):
    attacks = {}
    for mag in magnitudes:
        # GPS drift
        drift = np.linspace(0, 5*mag, len(base_data)).reshape(-1,1)
        attacks[f'gps_drift_{mag}x'] = base_data + drift * np.array([1,1,0.5,0,0,0,0,0,0,0,0,0])

        # IMU bias
        bias = 0.05 * mag
        attacks[f'imu_bias_{mag}x'] = base_data + np.array([0,0,0,bias,bias,0,bias/2,bias/2,0,0,0,0])

        # Noise injection
        np.random.seed(int(mag*100))
        noise = 0.1 * mag
        attacks[f'noise_{mag}x'] = base_data + np.random.normal(0, noise, base_data.shape)

        # Sudden jump
        jump_data = base_data.copy()
        jump_idx = len(base_data) // 2
        jump_data[jump_idx:, :3] += 2.0 * mag
        attacks[f'jump_{mag}x'] = jump_data

    return attacks

# Prepare base data
clean_data = df[state_cols].values[:10000]
test_base = df[state_cols].values[50000:50500]

# Store all results
all_results = {}

# ============================================================================
print('\n' + '='*70)
print('FIX 1: DATA AUGMENTATION')
print('='*70)

train_X_aug = []
train_y_aug = []

clean_feat = extract_features(clean_data[:5000])
train_X_aug.append(clean_feat)
train_y_aug.append(np.zeros(len(clean_feat)))

np.random.seed(42)
for mag in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0]:
    for attack_type in ['drift', 'bias', 'noise', 'jump']:
        base = clean_data[:500].copy()
        if attack_type == 'drift':
            attack = base + np.linspace(0, 5*mag, 500).reshape(-1,1) * np.array([1,1,0.5,0,0,0,0,0,0,0,0,0])
        elif attack_type == 'bias':
            attack = base + np.array([0,0,0,0.05*mag,0.05*mag,0,0.025*mag,0.025*mag,0,0,0,0])
        elif attack_type == 'noise':
            attack = base + np.random.normal(0, 0.1*mag, (500, 12))
        else:
            attack[250:, :3] += 2.0 * mag

        feat = extract_features(attack)
        train_X_aug.append(feat)
        train_y_aug.append(np.ones(len(feat)))

train_X_aug = np.vstack(train_X_aug)
train_y_aug = np.concatenate(train_y_aug)

scaler_aug = StandardScaler()
train_X_aug_scaled = scaler_aug.fit_transform(train_X_aug)

clf_aug = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
clf_aug.fit(train_X_aug_scaled, train_y_aug)

test_attacks = generate_test_attacks(test_base)
results_aug = {}
for name, attack in test_attacks.items():
    feat = extract_features(attack)
    feat_scaled = scaler_aug.transform(feat)
    preds = clf_aug.predict(feat_scaled)
    results_aug[name] = np.mean(preds)
    print(f'  {name:20s}: {np.mean(preds)*100:5.1f}%')

# FPR
clean_test = extract_features(clean_data[8000:9000])
clean_test_scaled = scaler_aug.transform(clean_test)
fpr_aug = np.mean(clf_aug.predict(clean_test_scaled))

print(f'\n  AVERAGE RECALL: {np.mean(list(results_aug.values()))*100:.1f}%')
print(f'  FALSE POSITIVE RATE: {fpr_aug*100:.1f}%')
all_results['Data Augmentation'] = {'recall': np.mean(list(results_aug.values())), 'fpr': fpr_aug, 'per_attack': results_aug}

# ============================================================================
print('\n' + '='*70)
print('FIX 2: PHYSICS-BASED FEATURES')
print('='*70)

train_X_phys = []
train_y_phys = []

clean_phys = extract_physics_features(clean_data[:5000])
clean_phys = np.nan_to_num(clean_phys, nan=0, posinf=1e6, neginf=-1e6)
train_X_phys.append(clean_phys)
train_y_phys.append(np.zeros(len(clean_phys)))

for mag in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for attack_type in ['drift', 'bias', 'noise', 'jump']:
        base = clean_data[:500].copy()
        if attack_type == 'drift':
            attack = base + np.linspace(0, 5*mag, 500).reshape(-1,1) * np.array([1,1,0.5,0,0,0,0,0,0,0,0,0])
        elif attack_type == 'bias':
            attack = base + np.array([0,0,0,0.05*mag,0.05*mag,0,0.025*mag,0.025*mag,0,0,0,0])
        elif attack_type == 'noise':
            attack = base + np.random.normal(0, 0.1*mag, (500, 12))
        else:
            attack[250:, :3] += 2.0 * mag

        feat = extract_physics_features(attack)
        feat = np.nan_to_num(feat, nan=0, posinf=1e6, neginf=-1e6)
        if len(feat) > 0:
            train_X_phys.append(feat)
            train_y_phys.append(np.ones(len(feat)))

train_X_phys = np.vstack(train_X_phys)
train_y_phys = np.concatenate(train_y_phys)

scaler_phys = RobustScaler()
train_X_phys_scaled = scaler_phys.fit_transform(train_X_phys)

clf_phys = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
clf_phys.fit(train_X_phys_scaled, train_y_phys)

results_phys = {}
for name, attack in test_attacks.items():
    feat = extract_physics_features(attack)
    feat = np.nan_to_num(feat, nan=0, posinf=1e6, neginf=-1e6)
    if len(feat) > 0:
        feat_scaled = scaler_phys.transform(feat)
        preds = clf_phys.predict(feat_scaled)
        results_phys[name] = np.mean(preds)
        print(f'  {name:20s}: {np.mean(preds)*100:5.1f}%')

clean_phys_test = extract_physics_features(clean_data[8000:9000])
clean_phys_test = np.nan_to_num(clean_phys_test, nan=0, posinf=1e6, neginf=-1e6)
clean_phys_scaled = scaler_phys.transform(clean_phys_test)
fpr_phys = np.mean(clf_phys.predict(clean_phys_scaled))

print(f'\n  AVERAGE RECALL: {np.mean(list(results_phys.values()))*100:.1f}%')
print(f'  FALSE POSITIVE RATE: {fpr_phys*100:.1f}%')
all_results['Physics Features'] = {'recall': np.mean(list(results_phys.values())), 'fpr': fpr_phys, 'per_attack': results_phys}

# ============================================================================
print('\n' + '='*70)
print('FIX 3: ANOMALY DETECTION (Isolation Forest)')
print('='*70)

clean_feat_anom = extract_features(clean_data[:8000])
scaler_anom = StandardScaler()
clean_feat_scaled = scaler_anom.fit_transform(clean_feat_anom)

iso_forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso_forest.fit(clean_feat_scaled)

results_anom = {}
for name, attack in test_attacks.items():
    feat = extract_features(attack)
    feat_scaled = scaler_anom.transform(feat)
    preds = iso_forest.predict(feat_scaled)
    recall = np.mean(preds == -1)
    results_anom[name] = recall
    print(f'  {name:20s}: {recall*100:5.1f}%')

clean_test_anom = scaler_anom.transform(extract_features(clean_data[8000:9000]))
fpr_anom = np.mean(iso_forest.predict(clean_test_anom) == -1)

print(f'\n  AVERAGE RECALL: {np.mean(list(results_anom.values()))*100:.1f}%')
print(f'  FALSE POSITIVE RATE: {fpr_anom*100:.1f}%')
all_results['Anomaly (IsoForest)'] = {'recall': np.mean(list(results_anom.values())), 'fpr': fpr_anom, 'per_attack': results_anom}

# ============================================================================
print('\n' + '='*70)
print('FIX 4: THRESHOLD-BASED RULES (3-sigma)')
print('='*70)

clean_stats = extract_features(clean_data[:8000])
mean_clean = np.mean(clean_stats, axis=0)
std_clean = np.std(clean_stats, axis=0) + 1e-8

def threshold_detector(features, mean, std, n_sigma=3):
    z_scores = np.abs((features - mean) / std)
    violations = np.any(z_scores > n_sigma, axis=1)
    return violations.astype(int)

results_thresh = {}
for name, attack in test_attacks.items():
    feat = extract_features(attack)
    preds = threshold_detector(feat, mean_clean, std_clean, n_sigma=3)
    results_thresh[name] = np.mean(preds)
    print(f'  {name:20s}: {np.mean(preds)*100:5.1f}%')

clean_test_feat = extract_features(clean_data[8000:9000])
fpr_thresh = np.mean(threshold_detector(clean_test_feat, mean_clean, std_clean, n_sigma=3))

print(f'\n  AVERAGE RECALL: {np.mean(list(results_thresh.values()))*100:.1f}%')
print(f'  FALSE POSITIVE RATE: {fpr_thresh*100:.1f}%')
all_results['Threshold (3-sigma)'] = {'recall': np.mean(list(results_thresh.values())), 'fpr': fpr_thresh, 'per_attack': results_thresh}

# ============================================================================
print('\n' + '='*70)
print('FIX 5: NORMALIZED FEATURES (Scale-Invariant)')
print('='*70)

train_X_norm = []
train_y_norm = []

clean_norm = extract_normalized_features(clean_data[:5000])
clean_norm = np.nan_to_num(clean_norm, nan=0, posinf=10, neginf=-10)
train_X_norm.append(clean_norm)
train_y_norm.append(np.zeros(len(clean_norm)))

for mag in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for attack_type in ['drift', 'bias', 'noise', 'jump']:
        base = clean_data[:500].copy()
        if attack_type == 'drift':
            attack = base + np.linspace(0, 5*mag, 500).reshape(-1,1) * np.array([1,1,0.5,0,0,0,0,0,0,0,0,0])
        elif attack_type == 'bias':
            attack = base + np.array([0,0,0,0.05*mag,0.05*mag,0,0.025*mag,0.025*mag,0,0,0,0])
        elif attack_type == 'noise':
            attack = base + np.random.normal(0, 0.1*mag, (500, 12))
        else:
            attack[250:, :3] += 2.0 * mag

        feat = extract_normalized_features(attack)
        feat = np.nan_to_num(feat, nan=0, posinf=10, neginf=-10)
        train_X_norm.append(feat)
        train_y_norm.append(np.ones(len(feat)))

train_X_norm = np.vstack(train_X_norm)
train_y_norm = np.concatenate(train_y_norm)

clf_norm = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
clf_norm.fit(train_X_norm, train_y_norm)

results_norm = {}
for name, attack in test_attacks.items():
    feat = extract_normalized_features(attack)
    feat = np.nan_to_num(feat, nan=0, posinf=10, neginf=-10)
    preds = clf_norm.predict(feat)
    results_norm[name] = np.mean(preds)
    print(f'  {name:20s}: {np.mean(preds)*100:5.1f}%')

clean_norm_test = extract_normalized_features(clean_data[8000:9000])
clean_norm_test = np.nan_to_num(clean_norm_test, nan=0, posinf=10, neginf=-10)
fpr_norm = np.mean(clf_norm.predict(clean_norm_test))

print(f'\n  AVERAGE RECALL: {np.mean(list(results_norm.values()))*100:.1f}%')
print(f'  FALSE POSITIVE RATE: {fpr_norm*100:.1f}%')
all_results['Normalized Features'] = {'recall': np.mean(list(results_norm.values())), 'fpr': fpr_norm, 'per_attack': results_norm}

# ============================================================================
print('\n' + '='*70)
print('FIX 6: GRADIENT BOOSTING + PHYSICS')
print('='*70)

clf_gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
clf_gb.fit(train_X_phys_scaled, train_y_phys)

results_gb = {}
for name, attack in test_attacks.items():
    feat = extract_physics_features(attack)
    feat = np.nan_to_num(feat, nan=0, posinf=1e6, neginf=-1e6)
    if len(feat) > 0:
        feat_scaled = scaler_phys.transform(feat)
        preds = clf_gb.predict(feat_scaled)
        results_gb[name] = np.mean(preds)
        print(f'  {name:20s}: {np.mean(preds)*100:5.1f}%')

fpr_gb = np.mean(clf_gb.predict(clean_phys_scaled))

print(f'\n  AVERAGE RECALL: {np.mean(list(results_gb.values()))*100:.1f}%')
print(f'  FALSE POSITIVE RATE: {fpr_gb*100:.1f}%')
all_results['GradientBoost+Physics'] = {'recall': np.mean(list(results_gb.values())), 'fpr': fpr_gb, 'per_attack': results_gb}

# ============================================================================
print('\n' + '='*70)
print('FIX 7: ENSEMBLE OF ALL METHODS')
print('='*70)

results_ensemble = {}
for name, attack in test_attacks.items():
    votes = []

    # Method 1: Data Aug
    feat1 = extract_features(attack)
    votes.append(clf_aug.predict(scaler_aug.transform(feat1)))

    # Method 2: Physics
    feat2 = extract_physics_features(attack)
    feat2 = np.nan_to_num(feat2, nan=0, posinf=1e6, neginf=-1e6)
    if len(feat2) > 0:
        phys_preds = clf_phys.predict(scaler_phys.transform(feat2))
        # Align lengths
        if len(phys_preds) < len(feat1):
            phys_preds = np.pad(phys_preds, (0, len(feat1)-len(phys_preds)), constant_values=0)
        else:
            phys_preds = phys_preds[:len(feat1)]
        votes.append(phys_preds)

    # Method 3: Threshold
    votes.append(threshold_detector(feat1, mean_clean, std_clean, n_sigma=3))

    # Method 4: Normalized
    feat4 = extract_normalized_features(attack)
    feat4 = np.nan_to_num(feat4, nan=0, posinf=10, neginf=-10)
    votes.append(clf_norm.predict(feat4))

    # Majority vote (>= 2 detectors agree)
    vote_sum = np.sum(votes, axis=0)
    final_pred = (vote_sum >= 2).astype(int)
    results_ensemble[name] = np.mean(final_pred)
    print(f'  {name:20s}: {np.mean(final_pred)*100:5.1f}%')

# FPR for ensemble
clean_feat_ens = extract_features(clean_data[8000:9000])
clean_norm_ens = extract_normalized_features(clean_data[8000:9000])
clean_norm_ens = np.nan_to_num(clean_norm_ens, nan=0, posinf=10, neginf=-10)

votes_clean = [
    clf_aug.predict(scaler_aug.transform(clean_feat_ens)),
    threshold_detector(clean_feat_ens, mean_clean, std_clean, n_sigma=3),
    clf_norm.predict(clean_norm_ens),
]
vote_sum_clean = np.sum(votes_clean, axis=0)
fpr_ensemble = np.mean(vote_sum_clean >= 2)

print(f'\n  AVERAGE RECALL: {np.mean(list(results_ensemble.values()))*100:.1f}%')
print(f'  FALSE POSITIVE RATE: {fpr_ensemble*100:.1f}%')
all_results['Ensemble (Majority)'] = {'recall': np.mean(list(results_ensemble.values())), 'fpr': fpr_ensemble, 'per_attack': results_ensemble}

# ============================================================================
print('\n' + '='*70)
print('FINAL COMPARISON TABLE')
print('='*70)

print('\n  {:30s} | {:>10s} | {:>10s} | {:>12s}'.format('Method', 'Recall', 'FPR', 'Status'))
print('  ' + '-'*70)
print('  {:30s} | {:>9.1f}% | {:>9.1f}% | {:>12s}'.format('Original (no fix)', 0.0, 0.0, 'FAILED'))

for name, res in sorted(all_results.items(), key=lambda x: -x[1]['recall']):
    recall = res['recall'] * 100
    fpr = res['fpr'] * 100
    if recall > 80 and fpr < 10:
        status = 'EXCELLENT'
    elif recall > 60 and fpr < 20:
        status = 'GOOD'
    elif recall > 40:
        status = 'MODERATE'
    else:
        status = 'POOR'
    print('  {:30s} | {:>9.1f}% | {:>9.1f}% | {:>12s}'.format(name, recall, fpr, status))

# Best method
best_name = max(all_results.keys(), key=lambda x: all_results[x]['recall'] - all_results[x]['fpr']*0.5)
print(f'\n  RECOMMENDED: {best_name}')
print(f'    Recall: {all_results[best_name]["recall"]*100:.1f}%')
print(f'    FPR: {all_results[best_name]["fpr"]*100:.1f}%')

# Per magnitude analysis
print('\n' + '='*70)
print('RECALL BY ATTACK MAGNITUDE (Best Method)')
print('='*70)
best_results = all_results[best_name]['per_attack']
for mag in [0.25, 0.5, 1.0, 2.0, 4.0]:
    mag_results = [v for k, v in best_results.items() if f'{mag}x' in k]
    if mag_results:
        print(f'  Magnitude {mag}x: {np.mean(mag_results)*100:.1f}%')
