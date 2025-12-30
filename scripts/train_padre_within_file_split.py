"""
PADRE Classifier with Within-File Splitting.

When you have limited files per class, sample windows from WITHIN each file
to ensure both classes are represented in train and test.

Strategy: Take first 70% of windows from each file for training,
         last 30% for testing (respects time ordering).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re


def extract_features(window):
    """Extract time + frequency domain features."""
    feat = []
    for col in range(window.shape[1]):
        ch = window[:, col]
        feat.extend([ch.mean(), ch.std(), ch.max() - ch.min()])
        fft = np.abs(np.fft.rfft(ch))
        feat.extend([fft[1:10].sum(), fft[10:50].sum(), fft[50:].sum(), np.argmax(fft[1:]) if len(fft) > 1 else 0])
    return feat


def load_data_with_temporal_split(data_dir, window_size=256, stride=128, train_ratio=0.7):
    """
    Load data with temporal split within each file.

    For each file:
    - First train_ratio% of windows go to training
    - Last (1-train_ratio)% go to testing

    This ensures:
    1. Both classes in train and test
    2. No data leakage (train is always before test in time)
    3. Model must generalize to later portions of each flight
    """
    X_train, y_train, X_test, y_test = [], [], [], []
    file_stats = []

    for csv_file in sorted(Path(data_dir).glob('*.csv')):
        match = re.search(r'normalized_(\d{4})\.csv', csv_file.name)
        if not match:
            continue

        codes = match.group(1)
        is_faulty = 1 if any(int(c) > 0 for c in codes) else 0

        df = pd.read_csv(csv_file)
        data = df.values.astype(np.float32)

        # Extract all windows
        windows = []
        for i in range((len(data) - window_size) // stride + 1):
            window = data[i * stride: i * stride + window_size]
            windows.append(extract_features(window))

        # Temporal split
        n_train = int(len(windows) * train_ratio)
        train_windows = windows[:n_train]
        test_windows = windows[n_train:]

        X_train.extend(train_windows)
        y_train.extend([is_faulty] * len(train_windows))
        X_test.extend(test_windows)
        y_test.extend([is_faulty] * len(test_windows))

        label = "Normal" if is_faulty == 0 else "Faulty"
        file_stats.append({
            'file': csv_file.name,
            'label': label,
            'train': len(train_windows),
            'test': len(test_windows)
        })

    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test),
            file_stats)


def main():
    print("=" * 70)
    print("PADRE CLASSIFIER - WITHIN-FILE TEMPORAL SPLIT")
    print("=" * 70)

    data_dir = Path('data/PADRE_dataset/Parrot_Bebop_2/Normalized_data')

    for train_ratio in [0.7, 0.8, 0.5]:
        print(f"\n{'='*70}")
        print(f"TRAIN RATIO: {train_ratio:.0%}")
        print(f"{'='*70}")

        X_train, y_train, X_test, y_test, file_stats = load_data_with_temporal_split(
            data_dir, train_ratio=train_ratio
        )

        print(f"\nData Split:")
        print(f"  Train: {len(X_train)} samples ({sum(y_train==0)} normal, {sum(y_train==1)} faulty)")
        print(f"  Test:  {len(X_test)} samples ({sum(y_test==0)} normal, {sum(y_test==1)} faulty)")

        print(f"\nPer-File Distribution:")
        for s in file_stats:
            print(f"  {s['file']}: train={s['train']}, test={s['test']} ({s['label']})")

        # Train
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                     random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\n{'='*50}")
        print(f"RESULTS (train={train_ratio:.0%})")
        print(f"{'='*50}")
        print(f"Accuracy:    {accuracy_score(y_test, y_pred):.2%}")
        print(f"Precision:   {precision_score(y_test, y_pred):.2%}")
        print(f"Recall:      {recall_score(y_test, y_pred):.2%}")
        print(f"F1 Score:    {f1_score(y_test, y_pred):.2%}")
        print(f"\nConfusion Matrix:")
        print(f"  TN (Normal correct):  {tn:5d}")
        print(f"  FP (Normal → Faulty): {fp:5d}")
        print(f"  FN (Faulty → Normal): {fn:5d}")
        print(f"  TP (Faulty correct):  {tp:5d}")

        # Per-class accuracy
        print(f"\nPer-Class Accuracy:")
        print(f"  Normal:  {tn/(tn+fp)*100:.1f}% ({tn}/{tn+fp})")
        print(f"  Faulty:  {tp/(tp+fn)*100:.1f}% ({tp}/{tp+fn})")


if __name__ == "__main__":
    main()
