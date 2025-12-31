"""
PADRE Classifier with Proper Grouped Cross-Validation.

This script avoids overfitting by:
1. Grouping windows by source file (no data leakage)
2. Using stratified sampling to maintain class balance
3. Reporting per-file and per-fold metrics

Key insight: Adjacent windows from the same file overlap by 50%,
so they MUST stay together in train or test, never split.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def extract_features(window):
    """Extract time + frequency domain features."""
    feat = []
    for col in range(window.shape[1]):
        ch = window[:, col]
        # Time domain
        feat.extend([ch.mean(), ch.std(), ch.max() - ch.min()])
        # Frequency domain
        fft = np.abs(np.fft.rfft(ch))
        feat.extend(
            [
                fft[1:10].sum(),
                fft[10:50].sum(),
                fft[50:].sum(),
                np.argmax(fft[1:]) if len(fft) > 1 else 0,
            ]
        )
    return feat


def load_data_with_groups(data_dir, window_size=256, stride=128):
    """Load data and track which file each window came from."""
    X, y, groups = [], [], []
    file_info = {}

    for file_id, csv_file in enumerate(sorted(Path(data_dir).glob("*.csv"))):
        match = re.search(r"normalized_(\d{4})\.csv", csv_file.name)
        if not match:
            continue

        codes = match.group(1)
        is_faulty = 1 if any(int(c) > 0 for c in codes) else 0

        df = pd.read_csv(csv_file)
        data = df.values.astype(np.float32)

        n_windows = 0
        for i in range((len(data) - window_size) // stride + 1):
            window = data[i * stride : i * stride + window_size]
            X.append(extract_features(window))
            y.append(is_faulty)
            groups.append(file_id)  # Track source file
            n_windows += 1

        file_info[file_id] = {
            "name": csv_file.name,
            "fault_code": codes,
            "is_faulty": is_faulty,
            "n_windows": n_windows,
        }

    return np.array(X), np.array(y), np.array(groups), file_info


def main():
    print("=" * 70)
    print("PADRE CLASSIFIER WITH GROUPED CROSS-VALIDATION")
    print("=" * 70)

    data_dir = Path("data/PADRE_dataset/Parrot_Bebop_2/Normalized_data")
    X, y, groups, file_info = load_data_with_groups(data_dir)

    print(f"\nDataset: {len(X)} windows from {len(file_info)} files")
    print(f"Normal files: {sum(1 for f in file_info.values() if f['is_faulty'] == 0)}")
    print(f"Faulty files: {sum(1 for f in file_info.values() if f['is_faulty'] == 1)}")

    # Show file distribution
    print("\nFile Distribution:")
    for fid, info in file_info.items():
        label = "Normal" if info["is_faulty"] == 0 else "Faulty"
        print(f"  [{fid:2d}] {info['name']}: {info['n_windows']:4d} windows ({label})")

    # ============================================================
    # Method 1: Stratified Group K-Fold (Recommended)
    # ============================================================
    print("\n" + "=" * 70)
    print("METHOD 1: Stratified Group K-Fold (5 folds)")
    print("=" * 70)
    print("Each fold keeps all windows from the same file together.")
    print("Stratification ensures each fold has both normal and faulty samples.\n")

    # Get unique groups and their labels for stratification
    unique_groups = np.unique(groups)
    group_labels = np.array([file_info[g]["is_faulty"] for g in unique_groups])

    # Check if we can stratify
    n_normal_files = sum(1 for l in group_labels if l == 0)
    n_faulty_files = sum(1 for l in group_labels if l == 1)
    n_folds = min(5, n_normal_files, n_faulty_files)

    if n_folds < 2:
        print(f"WARNING: Only {n_normal_files} normal files - cannot do proper stratified CV!")
        print("Using simple Group K-Fold instead.\n")
        cv = GroupKFold(n_splits=min(5, len(unique_groups)))
    else:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Which files are in test set?
        test_files = set(groups[test_idx])
        test_file_names = [file_info[f]["name"] for f in test_files]

        # Train and predict
        clf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        fold_results.append(
            {
                "fold": fold + 1,
                "test_files": test_file_names,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "fp": fp,
                "fn": fn,
            }
        )

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"Fold {fold+1}: Acc={acc:.2%}, F1={f1:.2%}, FP={fp}, FN={fn}")
        print(f"         Test files: {', '.join(test_file_names)}")

    # Aggregate metrics
    print("\n" + "-" * 70)
    print("AGGREGATE RESULTS (Stratified Group K-Fold)")
    print("-" * 70)

    cm_total = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    if cm_total.size == 4:
        tn, fp, fn, tp = cm_total.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    print(f"Accuracy:  {accuracy_score(all_y_true, all_y_pred):.2%}")
    print(f"Precision: {precision_score(all_y_true, all_y_pred, zero_division=0):.2%}")
    print(f"Recall:    {recall_score(all_y_true, all_y_pred, zero_division=0):.2%}")
    print(f"F1 Score:  {f1_score(all_y_true, all_y_pred, zero_division=0):.2%}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:5d}  FP={fp:5d}")
    print(f"  FN={fn:5d}  TP={tp:5d}")

    # ============================================================
    # Method 2: Leave-One-File-Out (Most Rigorous)
    # ============================================================
    print("\n" + "=" * 70)
    print("METHOD 2: Leave-One-File-Out Cross-Validation")
    print("=" * 70)
    print("Tests on each file independently (most rigorous but high variance).\n")

    lofo_results = []
    all_y_true_lofo, all_y_pred_lofo = [], []

    for test_file_id in unique_groups:
        # Train on all files except this one
        train_mask = groups != test_file_id
        test_mask = groups == test_file_id

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Skip if no samples
        if len(X_test) == 0:
            continue

        clf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        info = file_info[test_file_id]

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif info["is_faulty"] == 0:
            tn, fp = cm[0, 0], len(y_pred) - cm[0, 0]
            fn, tp = 0, 0
        else:
            tn, fp = 0, 0
            fn = len(y_pred) - (y_pred == 1).sum()
            tp = (y_pred == 1).sum()

        label = "Normal" if info["is_faulty"] == 0 else "Faulty"
        print(f"{info['name']}: Acc={acc:.2%} ({label}, {len(y_test)} samples, FP={fp}, FN={fn})")

        lofo_results.append(
            {
                "file": info["name"],
                "is_faulty": info["is_faulty"],
                "accuracy": acc,
                "fp": fp,
                "fn": fn,
            }
        )

        all_y_true_lofo.extend(y_test)
        all_y_pred_lofo.extend(y_pred)

    print("\n" + "-" * 70)
    print("AGGREGATE RESULTS (Leave-One-File-Out)")
    print("-" * 70)

    cm_total = confusion_matrix(all_y_true_lofo, all_y_pred_lofo, labels=[0, 1])
    if cm_total.size == 4:
        tn, fp, fn, tp = cm_total.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    print(f"Accuracy:  {accuracy_score(all_y_true_lofo, all_y_pred_lofo):.2%}")
    print(f"Precision: {precision_score(all_y_true_lofo, all_y_pred_lofo, zero_division=0):.2%}")
    print(f"Recall:    {recall_score(all_y_true_lofo, all_y_pred_lofo, zero_division=0):.2%}")
    print(f"F1 Score:  {f1_score(all_y_true_lofo, all_y_pred_lofo, zero_division=0):.2%}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:5d}  FP={fp:5d}")
    print(f"  FN={fn:5d}  TP={tp:5d}")

    # Per-class accuracy
    normal_results = [r for r in lofo_results if r["is_faulty"] == 0]
    faulty_results = [r for r in lofo_results if r["is_faulty"] == 1]

    if normal_results:
        normal_acc = np.mean([r["accuracy"] for r in normal_results])
        print(f"\nNormal files mean accuracy: {normal_acc:.2%} ({len(normal_results)} files)")
    if faulty_results:
        faulty_acc = np.mean([r["accuracy"] for r in faulty_results])
        print(f"Faulty files mean accuracy: {faulty_acc:.2%} ({len(faulty_results)} files)")

    # ============================================================
    # Recommendations
    # ============================================================
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print(
        """
1. USE STRATIFIED GROUP K-FOLD for your final model evaluation
   - Keeps file integrity (no data leakage)
   - Maintains class balance in each fold
   - More stable estimates than leave-one-file-out

2. LEAVE-ONE-FILE-OUT reveals worst-case performance
   - Shows how model performs on truly unseen flights
   - High variance due to limited files

3. YOUR DATASET LIMITATION
   - Only 2 normal files means normal class is under-represented
   - Consider collecting more normal flight data
   - Or use data augmentation for normal class

4. PROPER TRAIN/TEST SPLIT FOR DEPLOYMENT
   - Hold out 1 normal + 3-4 faulty files as final test set
   - Train on remaining files
   - Never tune hyperparameters on test set
"""
    )


if __name__ == "__main__":
    main()
