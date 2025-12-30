#!/usr/bin/env python3
"""
Final Security Detection Pipeline - Using Best Generalized Detector

Combines:
1. Generalized Multi-Scale IsolationForest (81.8% recall, 10.7% FPR)
2. Supervised Random Forest Classifier
3. Ensemble voting for production use
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic_attacks import SyntheticAttackGenerator

print('='*70)
print('FINAL SECURITY DETECTION PIPELINE')
print('='*70)

# Configuration
WINDOWS = [5, 10, 25, 50, 100, 200]
CONTAMINATION = 0.07
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'models/security/final_pipeline'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(Path(__file__).parent.parent.parent / 'data/euroc/all_sequences.csv')
state_cols = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vx', 'vy', 'vz']
clean_data = df[state_cols].values
print(f'Loaded {len(clean_data):,} samples')

# Feature extraction
def extract_multiscale_features(data, windows=WINDOWS):
    """Extract multi-scale statistical features."""
    all_features = []
    for i in range(max(windows), len(data)):
        feat_list = []
        for w_size in windows:
            w = data[i-w_size:i]
            feat_list.extend([
                np.mean(w, axis=0).mean(),
                np.std(w, axis=0).mean(),
                np.max(np.abs(np.diff(w, axis=0))),
            ])
        all_features.append(feat_list)
    return np.array(all_features)

# Test attack generation with varying magnitudes
def generate_test_attacks(base_data, magnitudes=[0.25, 0.5, 1.0, 2.0, 4.0]):
    attacks = {}
    for mag in magnitudes:
        # GPS drift
        drift = np.linspace(0, 5*mag, len(base_data)).reshape(-1,1)
        attacks[f'gps_drift_{mag}x'] = base_data + drift * np.array([1,1,0.5,0,0,0,0,0,0,0,0,0])
        # IMU bias
        bias = 0.05 * mag
        attacks[f'imu_bias_{mag}x'] = base_data + np.array([0,0,0,bias,bias,0,bias/2,bias/2,0,0,0,0])
        # Noise
        np.random.seed(int(mag*100))
        attacks[f'noise_{mag}x'] = base_data + np.random.normal(0, 0.1*mag, base_data.shape)
        # Jump
        jump_data = base_data.copy()
        jump_data[len(base_data)//2:, :3] += 2.0 * mag
        attacks[f'jump_{mag}x'] = jump_data
        # Oscillation
        t = np.arange(len(base_data)).reshape(-1,1)
        attacks[f'osc_{mag}x'] = base_data + np.sin(2*np.pi*mag*t/100) * mag * np.array([1,1,0,0,0,0,0,0,0,0,0,0])
    return attacks

# Split data
train_data = clean_data[:100000]
test_clean = clean_data[110000:115000]
test_base = clean_data[120000:120500]

print('\n' + '='*70)
print('STEP 1: TRAIN GENERALIZED DETECTOR (IsolationForest)')
print('='*70)

print('Extracting training features...')
train_features = extract_multiscale_features(train_data)
print(f'Feature shape: {train_features.shape}')

scaler_iso = StandardScaler()
train_features_scaled = scaler_iso.fit_transform(train_features)

print(f'Training IsolationForest (c={CONTAMINATION})...')
iso_detector = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=42,
    n_jobs=-1
)
iso_detector.fit(train_features_scaled)

print('\n' + '='*70)
print('STEP 2: TRAIN SUPERVISED CLASSIFIER (with augmented attacks)')
print('='*70)

# Generate augmented training attacks
print('Generating augmented attack data...')
train_X_clf = [train_features]
train_y_clf = [np.zeros(len(train_features))]

# Add attacks with multiple magnitudes for generalization
for mag in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5]:
    attack_base = train_data[:1000]
    for attack_type in ['drift', 'bias', 'noise', 'jump']:
        if attack_type == 'drift':
            attack = attack_base + np.linspace(0, 5*mag, 1000).reshape(-1,1) * np.array([1,1,0.5,0,0,0,0,0,0,0,0,0])
        elif attack_type == 'bias':
            attack = attack_base + np.array([0,0,0,0.05*mag,0.05*mag,0,0.025*mag,0.025*mag,0,0,0,0])
        elif attack_type == 'noise':
            np.random.seed(int(mag*100))
            attack = attack_base + np.random.normal(0, 0.1*mag, attack_base.shape)
        else:
            attack = attack_base.copy()
            attack[500:, :3] += 2.0 * mag

        feat = extract_multiscale_features(attack)
        train_X_clf.append(feat)
        train_y_clf.append(np.ones(len(feat)))

train_X_clf = np.vstack(train_X_clf)
train_y_clf = np.concatenate(train_y_clf)

scaler_clf = StandardScaler()
train_X_clf_scaled = scaler_clf.fit_transform(train_X_clf)

print(f'Training data: {len(train_X_clf)} samples ({int(np.sum(train_y_clf))} attacks)')

print('Training Random Forest classifier...')
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight={0: 1, 1: 3},
    random_state=42,
    n_jobs=-1
)
clf.fit(train_X_clf_scaled, train_y_clf)

print('\n' + '='*70)
print('STEP 3: EVALUATE ON TEST DATA')
print('='*70)

# Generate test attacks
test_attacks = generate_test_attacks(test_base)

results = {
    'iso_forest': {'per_attack': {}, 'clean_fpr': 0},
    'classifier': {'per_attack': {}, 'clean_fpr': 0},
    'ensemble': {'per_attack': {}, 'clean_fpr': 0},
}

# Evaluate on clean data (FPR)
print('\nEvaluating on clean data...')
clean_features = extract_multiscale_features(test_clean)
clean_scaled_iso = scaler_iso.transform(clean_features)
clean_scaled_clf = scaler_clf.transform(clean_features)

iso_clean_pred = (iso_detector.predict(clean_scaled_iso) == -1).astype(int)
clf_clean_pred = clf.predict(clean_scaled_clf)
ens_clean_pred = ((iso_clean_pred + clf_clean_pred) >= 1).astype(int)

results['iso_forest']['clean_fpr'] = np.mean(iso_clean_pred)
results['classifier']['clean_fpr'] = np.mean(clf_clean_pred)
results['ensemble']['clean_fpr'] = np.mean(ens_clean_pred)

print(f'  IsoForest FPR: {results["iso_forest"]["clean_fpr"]*100:.1f}%')
print(f'  Classifier FPR: {results["classifier"]["clean_fpr"]*100:.1f}%')
print(f'  Ensemble FPR: {results["ensemble"]["clean_fpr"]*100:.1f}%')

# Evaluate on attacks
print('\nEvaluating on attacks...')
print(f'{"Attack":<25} | {"IsoForest":>10} | {"Classifier":>10} | {"Ensemble":>10}')
print('-'*65)

for attack_name, attack_data in test_attacks.items():
    attack_features = extract_multiscale_features(attack_data)
    attack_scaled_iso = scaler_iso.transform(attack_features)
    attack_scaled_clf = scaler_clf.transform(attack_features)

    iso_pred = (iso_detector.predict(attack_scaled_iso) == -1).astype(int)
    clf_pred = clf.predict(attack_scaled_clf)
    ens_pred = ((iso_pred + clf_pred) >= 1).astype(int)

    results['iso_forest']['per_attack'][attack_name] = np.mean(iso_pred)
    results['classifier']['per_attack'][attack_name] = np.mean(clf_pred)
    results['ensemble']['per_attack'][attack_name] = np.mean(ens_pred)

    print(f'{attack_name:<25} | {np.mean(iso_pred)*100:>9.1f}% | {np.mean(clf_pred)*100:>9.1f}% | {np.mean(ens_pred)*100:>9.1f}%')

# Summary
print('\n' + '='*70)
print('SUMMARY')
print('='*70)

for method in ['iso_forest', 'classifier', 'ensemble']:
    recalls = list(results[method]['per_attack'].values())
    avg_recall = np.mean(recalls)
    min_recall = np.min(recalls)
    fpr = results[method]['clean_fpr']

    results[method]['avg_recall'] = avg_recall
    results[method]['min_recall'] = min_recall

    print(f'\n{method.upper()}:')
    print(f'  Avg Recall: {avg_recall*100:.1f}%')
    print(f'  Min Recall: {min_recall*100:.1f}%')
    print(f'  FPR: {fpr*100:.1f}%')
    print(f'  F1 (approx): {2*avg_recall*(1-fpr)/(avg_recall + (1-fpr)):.3f}')

print('\n' + '='*70)
print('STEP 4: SAVE FINAL MODELS')
print('='*70)

# Save IsolationForest
with open(OUTPUT_DIR / 'isolation_forest.pkl', 'wb') as f:
    pickle.dump(iso_detector, f)
print('Saved: isolation_forest.pkl')

with open(OUTPUT_DIR / 'scaler_iso.pkl', 'wb') as f:
    pickle.dump(scaler_iso, f)
print('Saved: scaler_iso.pkl')

# Save Classifier
with open(OUTPUT_DIR / 'classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
print('Saved: classifier.pkl')

with open(OUTPUT_DIR / 'scaler_clf.pkl', 'wb') as f:
    pickle.dump(scaler_clf, f)
print('Saved: scaler_clf.pkl')

# Save config and results
config = {
    'windows': WINDOWS,
    'contamination': CONTAMINATION,
    'n_estimators_iso': 200,
    'n_estimators_clf': 200,
    'state_columns': state_cols,
}
with open(OUTPUT_DIR / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('Saved: config.json')

# Convert numpy values to Python floats for JSON
results_json = {}
for method, data in results.items():
    results_json[method] = {
        'avg_recall': float(data['avg_recall']),
        'min_recall': float(data['min_recall']),
        'clean_fpr': float(data['clean_fpr']),
        'per_attack': {k: float(v) for k, v in data['per_attack'].items()}
    }

with open(OUTPUT_DIR / 'evaluation_results.json', 'w') as f:
    json.dump(results_json, f, indent=2)
print('Saved: evaluation_results.json')

print('\n' + '='*70)
print('FINAL RESULTS')
print('='*70)

best_method = max(['iso_forest', 'classifier', 'ensemble'],
                  key=lambda x: results[x]['avg_recall'] - 0.5*results[x]['clean_fpr'])

print(f'''
BEST METHOD: {best_method.upper()}
  Average Recall: {results[best_method]["avg_recall"]*100:.1f}%
  Minimum Recall: {results[best_method]["min_recall"]*100:.1f}%
  False Positive Rate: {results[best_method]["clean_fpr"]*100:.1f}%

Models saved to: {OUTPUT_DIR}
''')
