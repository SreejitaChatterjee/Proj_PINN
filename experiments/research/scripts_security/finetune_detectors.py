#!/usr/bin/env python3
"""Fine-tune the best detection methods."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest

print("=" * 70)
print("FINE-TUNING BEST METHODS")
print("=" * 70)

# Load data
df = pd.read_csv(Path(__file__).parent.parent.parent / "data/euroc/all_sequences.csv")
state_cols = ["x", "y", "z", "roll", "pitch", "yaw", "p", "q", "r", "vx", "vy", "vz"]
clean_data = df[state_cols].values[:15000]
test_base = df[state_cols].values[50000:50500]


def extract_multiscale_features(data, windows=[10, 25, 50, 100]):
    all_features = []
    for i in range(max(windows), len(data)):
        feat_list = []
        for w_size in windows:
            w = data[i - w_size : i]
            feat_list.extend(
                [
                    np.mean(w, axis=0).mean(),
                    np.std(w, axis=0).mean(),
                    np.max(np.abs(np.diff(w, axis=0))),
                ]
            )
        all_features.append(feat_list)
    return np.array(all_features)


def extract_enhanced_multiscale(data, windows=[10, 25, 50, 100]):
    all_features = []
    for i in range(max(windows), len(data)):
        feat_list = []
        for w_size in windows:
            w = data[i - w_size : i]
            feat_list.extend(
                [
                    np.mean(w, axis=0).mean(),
                    np.std(w, axis=0).mean(),
                    np.median(w, axis=0).mean(),
                ]
            )
            diff = np.diff(w, axis=0)
            feat_list.extend(
                [
                    np.mean(np.abs(diff)),
                    np.max(np.abs(diff)),
                    np.std(diff.flatten()),
                ]
            )
            feat_list.append(np.max(w) - np.min(w))
            t = np.arange(w_size)
            slopes = [np.polyfit(t, w[:, j], 1)[0] for j in range(min(3, w.shape[1]))]
            feat_list.extend(slopes)
        all_features.append(feat_list)
    return np.array(all_features)


def generate_test_attacks(base_data, magnitudes=[0.25, 0.5, 1.0, 2.0, 4.0]):
    attacks = {}
    for mag in magnitudes:
        drift = np.linspace(0, 5 * mag, len(base_data)).reshape(-1, 1)
        attacks[f"gps_drift_{mag}x"] = base_data + drift * np.array(
            [1, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        bias = 0.05 * mag
        attacks[f"imu_bias_{mag}x"] = base_data + np.array(
            [0, 0, 0, bias, bias, 0, bias / 2, bias / 2, 0, 0, 0, 0]
        )
        np.random.seed(int(mag * 100))
        attacks[f"noise_{mag}x"] = base_data + np.random.normal(0, 0.1 * mag, base_data.shape)
        jump_data = base_data.copy()
        jump_data[len(base_data) // 2 :, :3] += 2.0 * mag
        attacks[f"jump_{mag}x"] = jump_data
        t = np.arange(len(base_data)).reshape(-1, 1)
        attacks[f"osc_{mag}x"] = base_data + np.sin(2 * np.pi * mag * t / 100) * mag * np.array(
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
    return attacks


train_data = clean_data[:10000]
clean_test = clean_data[12000:13000]
test_attacks = generate_test_attacks(test_base)

all_results = []

# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: CONTAMINATION TUNING (Multi-scale + IsoForest)")
print("=" * 70)

for contamination in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]:
    train_feat = extract_multiscale_features(train_data)
    det = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    det.fit(train_feat)

    recalls = []
    for name, attack in test_attacks.items():
        feat = extract_multiscale_features(attack)
        preds = det.predict(feat)
        recalls.append(np.mean(preds == -1))

    clean_feat = extract_multiscale_features(clean_test)
    clean_preds = det.predict(clean_feat)
    fpr = np.mean(clean_preds == -1)

    recall = np.mean(recalls)
    score = recall - 0.5 * fpr
    print(
        f"  contamination={contamination:.2f}: Recall={recall*100:5.1f}%, FPR={fpr*100:5.1f}%, Score={score*100:.1f}"
    )
    all_results.append(("MultiScale-IsoForest", f"c={contamination}", recall, fpr, score))

# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: WINDOW SIZE COMBINATIONS")
print("=" * 70)

window_configs = [
    [5, 10, 20, 40],
    [10, 25, 50, 100],
    [5, 15, 30, 60, 120],
    [10, 20, 40, 80],
    [8, 16, 32, 64, 128],
    [5, 10, 25, 50, 100, 200],
]

for windows in window_configs:
    train_feat = extract_multiscale_features(train_data, windows=windows)
    det = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1)
    det.fit(train_feat)

    recalls = []
    for name, attack in test_attacks.items():
        feat = extract_multiscale_features(attack, windows=windows)
        preds = det.predict(feat)
        recalls.append(np.mean(preds == -1))

    clean_feat = extract_multiscale_features(clean_test, windows=windows)
    fpr = np.mean(det.predict(clean_feat) == -1)

    recall = np.mean(recalls)
    score = recall - 0.5 * fpr
    print(
        f"  windows={windows}: Recall={recall*100:5.1f}%, FPR={fpr*100:5.1f}%, Score={score*100:.1f}"
    )
    all_results.append(("MultiScale-Windows", str(windows), recall, fpr, score))

# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: ENHANCED MULTI-SCALE FEATURES")
print("=" * 70)

for contamination in [0.02, 0.03, 0.05, 0.07]:
    train_feat = extract_enhanced_multiscale(train_data)
    det = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    det.fit(train_feat)

    recalls = []
    for name, attack in test_attacks.items():
        feat = extract_enhanced_multiscale(attack)
        preds = det.predict(feat)
        recalls.append(np.mean(preds == -1))

    clean_feat = extract_enhanced_multiscale(clean_test)
    fpr = np.mean(det.predict(clean_feat) == -1)

    recall = np.mean(recalls)
    score = recall - 0.5 * fpr
    print(
        f"  Enhanced (c={contamination:.2f}): Recall={recall*100:5.1f}%, FPR={fpr*100:5.1f}%, Score={score*100:.1f}"
    )
    all_results.append(("Enhanced-MultiScale", f"c={contamination}", recall, fpr, score))

# ============================================================================
print("\n" + "=" * 70)
print("TEST 4: KALMAN + MULTI-SCALE HYBRID")
print("=" * 70)


class KalmanDetector:
    def __init__(self, threshold_pct=95):
        self.threshold_pct = threshold_pct
        self.threshold = None

    def fit(self, data):
        residuals = self._compute_residuals(data)
        self.threshold = np.percentile(residuals, self.threshold_pct)

    def _compute_residuals(self, data):
        residuals = []
        x = data[0].copy()
        for i in range(1, len(data)):
            pred = x
            residual = np.linalg.norm(data[i] - pred)
            residuals.append(residual)
            x = data[i]
        return np.array(residuals)

    def predict(self, data):
        residuals = self._compute_residuals(data)
        residuals = np.concatenate([[0], residuals])
        return (residuals > self.threshold).astype(int)


train_multi = extract_multiscale_features(train_data)
iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1)
iso.fit(train_multi)

for kalman_pct in [95, 97, 99]:
    kalman = KalmanDetector(threshold_pct=kalman_pct)
    kalman.fit(train_data)

    for vote_thresh in [1, 2]:
        recalls = []
        for name, attack in test_attacks.items():
            multi_feat = extract_multiscale_features(attack)
            iso_pred = (iso.predict(multi_feat) == -1).astype(int)
            kalman_pred = kalman.predict(attack)[: len(iso_pred)]

            combined = (iso_pred + kalman_pred) >= vote_thresh
            recalls.append(np.mean(combined))

        clean_multi = extract_multiscale_features(clean_test)
        iso_clean = (iso.predict(clean_multi) == -1).astype(int)
        kalman_clean = kalman.predict(clean_test)[: len(iso_clean)]
        fpr = np.mean((iso_clean + kalman_clean) >= vote_thresh)

        recall = np.mean(recalls)
        score = recall - 0.5 * fpr
        mode = "ANY" if vote_thresh == 1 else "BOTH"
        print(
            f"  Kalman({kalman_pct})+MultiScale ({mode}): Recall={recall*100:5.1f}%, FPR={fpr*100:5.1f}%, Score={score*100:.1f}"
        )
        all_results.append(("Kalman+MultiScale", f"k={kalman_pct},{mode}", recall, fpr, score))

# ============================================================================
print("\n" + "=" * 70)
print("TEST 5: ADAPTIVE THRESHOLD (per-feature)")
print("=" * 70)


def adaptive_threshold_detector(train_data, test_data, n_sigma=3):
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0) + 1e-8
    z_scores = np.abs((test_data - mean) / std)
    return np.any(z_scores > n_sigma, axis=1).astype(int)


for n_sigma in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    train_feat = extract_multiscale_features(train_data)

    recalls = []
    for name, attack in test_attacks.items():
        feat = extract_multiscale_features(attack)
        preds = adaptive_threshold_detector(train_feat, feat, n_sigma)
        recalls.append(np.mean(preds))

    clean_feat = extract_multiscale_features(clean_test)
    fpr = np.mean(adaptive_threshold_detector(train_feat, clean_feat, n_sigma))

    recall = np.mean(recalls)
    score = recall - 0.5 * fpr
    print(
        f"  n_sigma={n_sigma:.1f}: Recall={recall*100:5.1f}%, FPR={fpr*100:5.1f}%, Score={score*100:.1f}"
    )
    all_results.append(("AdaptiveThreshold", f"sigma={n_sigma}", recall, fpr, score))

# ============================================================================
print("\n" + "=" * 70)
print("BEST CONFIGURATIONS - SORTED BY SCORE")
print("=" * 70)

all_results.sort(key=lambda x: x[4], reverse=True)

print(f'\n{"Method":<25} {"Config":<25} {"Recall":>8} {"FPR":>8} {"Score":>8}')
print("-" * 80)
for method, config, recall, fpr, score in all_results[:15]:
    status = "GOOD" if recall >= 0.7 and fpr <= 0.15 else "OK" if recall >= 0.6 else ""
    print(
        f"{method:<25} {config:<25} {recall*100:>7.1f}% {fpr*100:>7.1f}% {score*100:>7.1f} {status}"
    )

# Best overall
best = all_results[0]
print(f'\n{"="*70}')
print(f"BEST: {best[0]} ({best[1]})")
print(f"  Recall: {best[2]*100:.1f}%")
print(f"  FPR: {best[3]*100:.1f}%")
print(f"  Score: {best[4]*100:.1f}")
print(f'{"="*70}')
