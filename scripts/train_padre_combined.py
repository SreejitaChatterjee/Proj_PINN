"""
PADRE Classifier - Combined Bebop + Solo Data.

Uses both Parrot Bebop 2 and 3DR Solo datasets:
- Bebop 2: 20 files (1 normal)
- 3DR Solo: 9 files (1 normal)
- Total: 29 files, 2 normal

Within-file temporal split ensures both normal files contribute to train and test.
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
    n_cols = window.shape[1]
    for col in range(n_cols):
        ch = window[:, col]
        feat.extend([ch.mean(), ch.std(), ch.max() - ch.min()])
        fft = np.abs(np.fft.rfft(ch))
        feat.extend([fft[1:10].sum(), fft[10:50].sum(), fft[50:].sum(), np.argmax(fft[1:]) if len(fft) > 1 else 0])
    return feat


def parse_fault_code(filename):
    """Extract 4-digit fault code from filename."""
    # Match patterns like _0000.csv, _1022.csv, etc.
    match = re.search(r'_(\d{4})\.csv$', filename)
    if match:
        return match.group(1)
    return None


def load_all_padre_data(base_dir, window_size=256, stride=128, train_ratio=0.7):
    """Load both Bebop and Solo datasets with temporal split."""

    datasets = [
        ('Bebop2', base_dir / 'Parrot_Bebop_2' / 'Normalized_data'),
        ('Solo', base_dir / 'data/PADRE_dataset/3DR_Solo/Normalized_data/extracted' if 'GitHub' not in str(base_dir)
                else Path('C:/Users/sreej/OneDrive/Documents/GitHub/Proj_PINN/data/PADRE_dataset/3DR_Solo/Normalized_data/extracted'))
    ]

    X_train, y_train, X_test, y_test = [], [], [], []
    file_stats = []

    # Bebop 2 data
    bebop_dir = Path('C:/Users/sreej/OneDrive/Documents/GitHub/Proj_PINN/data/PADRE_dataset/Parrot_Bebop_2/Normalized_data')
    solo_dir = Path('C:/Users/sreej/OneDrive/Documents/GitHub/Proj_PINN/data/PADRE_dataset/3DR_Solo/Normalized_data/extracted')

    for drone_name, data_dir in [('Bebop2', bebop_dir), ('Solo', solo_dir)]:
        if not data_dir.exists():
            print(f"Skipping {drone_name}: {data_dir} not found")
            continue

        for csv_file in sorted(data_dir.glob('*.csv')):
            fault_code = parse_fault_code(csv_file.name)
            if not fault_code:
                continue

            is_faulty = 1 if any(int(c) > 0 for c in fault_code) else 0

            df = pd.read_csv(csv_file)
            data = df.values.astype(np.float32)[:, :24]  # Use first 24 columns (common to both)
            n_cols = 24

            # Extract all windows
            windows = []
            for i in range((len(data) - window_size) // stride + 1):
                window = data[i * stride: i * stride + window_size]
                windows.append(extract_features(window))

            if not windows:
                continue

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
                'drone': drone_name,
                'file': csv_file.name,
                'fault_code': fault_code,
                'label': label,
                'cols': n_cols,
                'train': len(train_windows),
                'test': len(test_windows)
            })

    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test),
            file_stats)


def main():
    print("=" * 80)
    print("PADRE CLASSIFIER - COMBINED BEBOP 2 + 3DR SOLO")
    print("=" * 80)

    base_dir = Path('C:/Users/sreej/OneDrive/Documents/GitHub/Proj_PINN/data/PADRE_dataset')

    X_train, y_train, X_test, y_test, file_stats = load_all_padre_data(base_dir)

    print(f"\nDataset Summary:")
    print(f"  Total files: {len(file_stats)}")
    print(f"  Bebop 2:     {sum(1 for s in file_stats if s['drone'] == 'Bebop2')}")
    print(f"  3DR Solo:    {sum(1 for s in file_stats if s['drone'] == 'Solo')}")
    print(f"  Normal:      {sum(1 for s in file_stats if s['label'] == 'Normal')}")
    print(f"  Faulty:      {sum(1 for s in file_stats if s['label'] == 'Faulty')}")

    print(f"\nData Split:")
    print(f"  Train: {len(X_train)} samples ({sum(y_train==0)} normal, {sum(y_train==1)} faulty)")
    print(f"  Test:  {len(X_test)} samples ({sum(y_test==0)} normal, {sum(y_test==1)} faulty)")

    print(f"\nPer-File Details:")
    for s in file_stats:
        print(f"  [{s['drone']:6s}] {s['file'][-30:]}: {s['cols']}ch, train={s['train']}, test={s['test']} ({s['label']})")

    # Check feature dimensions
    if len(set(X_train[i].shape[0] if hasattr(X_train[i], 'shape') else len(X_train[i]) for i in range(len(X_train)))) > 1:
        print("\nWARNING: Inconsistent feature dimensions - drones have different sensor counts!")
        print("Training on Bebop 2 only (24 channels = 168 features)")

        # Filter to Bebop only
        bebop_indices_train = [i for i, s in enumerate(file_stats) for _ in range(s['train']) if s['drone'] == 'Bebop2']
        bebop_indices_test = [i for i, s in enumerate(file_stats) for _ in range(s['test']) if s['drone'] == 'Bebop2']

        # Recalculate
        X_train_bebop, y_train_bebop = [], []
        X_test_bebop, y_test_bebop = [], []

        bebop_stats = [s for s in file_stats if s['drone'] == 'Bebop2']
        idx_train = 0
        idx_test = 0

        for s in file_stats:
            if s['drone'] == 'Bebop2':
                for _ in range(s['train']):
                    if idx_train < len(X_train):
                        X_train_bebop.append(X_train[idx_train])
                        y_train_bebop.append(y_train[idx_train])
                    idx_train += 1
                for _ in range(s['test']):
                    if idx_test < len(X_test):
                        X_test_bebop.append(X_test[idx_test])
                        y_test_bebop.append(y_test[idx_test])
                    idx_test += 1
            else:
                idx_train += s['train']
                idx_test += s['test']

    # Train classifier
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        print(f"Unexpected confusion matrix shape: {cm.shape}")
        tn, fp, fn, tp = 0, 0, 0, 0

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:    {accuracy_score(y_test, y_pred):.2%}")
    print(f"Precision:   {precision_score(y_test, y_pred, zero_division=0):.2%}")
    print(f"Recall:      {recall_score(y_test, y_pred, zero_division=0):.2%}")
    print(f"F1 Score:    {f1_score(y_test, y_pred, zero_division=0):.2%}")
    print(f"\nConfusion Matrix:")
    print(f"  TN (Normal correct):  {tn:5d}")
    print(f"  FP (Normal->Faulty):  {fp:5d}")
    print(f"  FN (Faulty->Normal):  {fn:5d}")
    print(f"  TP (Faulty correct):  {tp:5d}")

    if tn + fp > 0:
        print(f"\nNormal class accuracy:  {tn/(tn+fp)*100:.1f}%")
    if tp + fn > 0:
        print(f"Faulty class accuracy:  {tp/(tp+fn)*100:.1f}%")


if __name__ == "__main__":
    main()
