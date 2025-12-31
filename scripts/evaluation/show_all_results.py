"""Show all metrics for PADRE fault detection models."""

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
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputClassifier

# Load and prepare data
data_dir = Path("data/PADRE_dataset/Parrot_Bebop_2/Normalized_data")
X, y_binary, y_motor, y_multilabel = [], [], [], []

for csv_file in sorted(data_dir.glob("*.csv")):
    match = re.search(r"normalized_(\d{4})\.csv", csv_file.name)
    if not match:
        continue
    codes = match.group(1)
    is_faulty = 1 if any(int(c) > 0 for c in codes) else 0
    faulty_motors = [i for i, c in enumerate(codes) if int(c) > 0]
    motor_id = 0 if not faulty_motors else (faulty_motors[0] + 1 if len(faulty_motors) == 1 else 5)
    multilabel = [1 if int(c) > 0 else 0 for c in codes]

    df = pd.read_csv(csv_file)
    data = df.values.astype(np.float32)
    for i in range((len(data) - 256) // 128 + 1):
        window = data[i * 128 : i * 128 + 256]
        feat = []
        for col in range(24):
            feat.extend(
                [
                    window[:, col].mean(),
                    window[:, col].std(),
                    window[:, col].max() - window[:, col].min(),
                ]
            )
        X.append(feat)
        y_binary.append(is_faulty)
        y_motor.append(motor_id)
        y_multilabel.append(multilabel)

X = np.array(X)
y_binary = np.array(y_binary)
y_motor = np.array(y_motor)
y_multilabel = np.array(y_multilabel)

np.random.seed(42)
idx = np.random.permutation(len(X))
split = int(0.8 * len(idx))
train_idx, test_idx = idx[:split], idx[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_bin_train, y_bin_test = y_binary[train_idx], y_binary[test_idx]
y_mot_train, y_mot_test = y_motor[train_idx], y_motor[test_idx]
y_multi_train, y_multi_test = y_multilabel[train_idx], y_multilabel[test_idx]

# Train models
rf1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_bin_train)
rf2 = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
).fit(X_train, y_bin_train)
rf3 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_mot_train)
rf4 = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
).fit(X_train, y_mot_train)
rf5 = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
).fit(X_train, y_multi_train)

p1, p2, p3, p4, p5 = (
    rf1.predict(X_test),
    rf2.predict(X_test),
    rf3.predict(X_test),
    rf4.predict(X_test),
    rf5.predict(X_test),
)

print("=" * 90)
print("ALL METRICS - PADRE MOTOR FAULT DETECTION")
print("=" * 90)

# Binary Unbalanced
print("\n" + "-" * 90)
print("MODEL 1: Binary Classification (Unbalanced)")
print("-" * 90)
cm = confusion_matrix(y_bin_test, p1)
tn, fp, fn, tp = cm.ravel()
print(
    f"Accuracy:    {accuracy_score(y_bin_test, p1):.4f} ({accuracy_score(y_bin_test, p1) * 100:.2f}%)"
)
print(f"Precision:   {precision_score(y_bin_test, p1):.4f}")
print(f"Recall:      {recall_score(y_bin_test, p1):.4f}")
print(f"F1 Score:    {f1_score(y_bin_test, p1):.4f}")
print(f"ROC AUC:     {roc_auc_score(y_bin_test, rf1.predict_proba(X_test)[:, 1]):.4f}")
print(f"Specificity: {tn / (tn + fp):.4f}")
print(f"\nConfusion Matrix:")
print(f"  True Negatives (TN):   {tn:5d}  (Normal correctly classified)")
print(f"  False Positives (FP):  {fp:5d}  (Normal misclassified as Faulty)")
print(f"  False Negatives (FN):  {fn:5d}  (Faulty missed)")
print(f"  True Positives (TP):   {tp:5d}  (Faulty correctly classified)")
print(f"\nPer-Class Accuracy:")
print(f"  Normal class:  {tn / (tn + fp) * 100:.2f}%")
print(f"  Faulty class:  {tp / (tp + fn) * 100:.2f}%")

# Binary Balanced
print("\n" + "-" * 90)
print("MODEL 2: Binary Classification (Balanced)")
print("-" * 90)
cm = confusion_matrix(y_bin_test, p2)
tn, fp, fn, tp = cm.ravel()
print(
    f"Accuracy:    {accuracy_score(y_bin_test, p2):.4f} ({accuracy_score(y_bin_test, p2) * 100:.2f}%)"
)
print(f"Precision:   {precision_score(y_bin_test, p2):.4f}")
print(f"Recall:      {recall_score(y_bin_test, p2):.4f}")
print(f"F1 Score:    {f1_score(y_bin_test, p2):.4f}")
print(f"ROC AUC:     {roc_auc_score(y_bin_test, rf2.predict_proba(X_test)[:, 1]):.4f}")
print(f"Specificity: {tn / (tn + fp):.4f}")
print(f"\nConfusion Matrix:")
print(f"  True Negatives (TN):   {tn:5d}")
print(f"  False Positives (FP):  {fp:5d}")
print(f"  False Negatives (FN):  {fn:5d}")
print(f"  True Positives (TP):   {tp:5d}")
print(f"\nPer-Class Accuracy:")
print(f"  Normal class:  {tn / (tn + fp) * 100:.2f}%")
print(f"  Faulty class:  {tp / (tp + fn) * 100:.2f}%")

# Motor ID Unbalanced
print("\n" + "-" * 90)
print("MODEL 3: Motor Identification (6-class, Unbalanced)")
print("-" * 90)
motor_names = ["Normal", "Motor A", "Motor B", "Motor C", "Motor D", "Multiple"]
print(f"Accuracy:    {accuracy_score(y_mot_test, p3):.4f}")
print(f'Macro F1:    {f1_score(y_mot_test, p3, average="macro"):.4f}')
print(f'Weighted F1: {f1_score(y_mot_test, p3, average="weighted"):.4f}')
cm = confusion_matrix(y_mot_test, p3)
print(f"\nPer-Class Metrics:")
header = f'{"Class":<12} {"Precision":<12} {"Recall":<12} {"F1":<12} {"Support":<10} {"Errors":<10}'
print(header)
for i, name in enumerate(motor_names):
    support = (y_mot_test == i).sum()
    if support > 0:
        prec = precision_score(y_mot_test == i, p3 == i, zero_division=0)
        rec = recall_score(y_mot_test == i, p3 == i, zero_division=0)
        f1 = f1_score(y_mot_test == i, p3 == i, zero_division=0)
        errors = support - cm[i, i]
        print(f"{name:<12} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {support:<10} {errors:<10}")

# Motor ID Balanced
print("\n" + "-" * 90)
print("MODEL 4: Motor Identification (6-class, Balanced)")
print("-" * 90)
print(f"Accuracy:    {accuracy_score(y_mot_test, p4):.4f}")
print(f'Macro F1:    {f1_score(y_mot_test, p4, average="macro"):.4f}')
print(f'Weighted F1: {f1_score(y_mot_test, p4, average="weighted"):.4f}')
cm = confusion_matrix(y_mot_test, p4)
print(f"\nPer-Class Metrics:")
print(header)
for i, name in enumerate(motor_names):
    support = (y_mot_test == i).sum()
    if support > 0:
        prec = precision_score(y_mot_test == i, p4 == i, zero_division=0)
        rec = recall_score(y_mot_test == i, p4 == i, zero_division=0)
        f1 = f1_score(y_mot_test == i, p4 == i, zero_division=0)
        errors = support - cm[i, i]
        print(f"{name:<12} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {support:<10} {errors:<10}")

# Multi-Label
print("\n" + "-" * 90)
print("MODEL 5: Multi-Label (4 binary classifiers, per-motor)")
print("-" * 90)
print(f"\nPer-Motor Metrics:")
header2 = f'{"Motor":<10} {"Accuracy":<12} {"Precision":<12} {"Recall":<12} {"F1":<12} {"FP":<8} {"FN":<8}'
print(header2)
for i, motor in enumerate(["A", "B", "C", "D"]):
    acc = accuracy_score(y_multi_test[:, i], p5[:, i])
    prec = precision_score(y_multi_test[:, i], p5[:, i], zero_division=0)
    rec = recall_score(y_multi_test[:, i], p5[:, i], zero_division=0)
    f1 = f1_score(y_multi_test[:, i], p5[:, i], zero_division=0)
    cm = confusion_matrix(y_multi_test[:, i], p5[:, i])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        fp, fn = 0, 0
    print(f"{motor:<10} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {fp:<8} {fn:<8}")

exact = (p5 == y_multi_test).all(axis=1).mean()
hamming = (p5 == y_multi_test).mean()
print(f"\nExact Match Accuracy (all 4 motors correct): {exact:.4f}")
print(f"Hamming Accuracy (per-motor average):        {hamming:.4f}")

# Summary Table
print("\n" + "=" * 90)
print("SUMMARY TABLE")
print("=" * 90)
header3 = f'{"Model":<40} {"Accuracy":<12} {"F1":<12} {"Precision":<12} {"Recall":<12}'
print(header3)
print("-" * 88)
print(
    f'{"Binary (Unbalanced)":<40} {accuracy_score(y_bin_test, p1):<12.4f} {f1_score(y_bin_test, p1):<12.4f} {precision_score(y_bin_test, p1):<12.4f} {recall_score(y_bin_test, p1):<12.4f}'
)
print(
    f'{"Binary (Balanced)":<40} {accuracy_score(y_bin_test, p2):<12.4f} {f1_score(y_bin_test, p2):<12.4f} {precision_score(y_bin_test, p2):<12.4f} {recall_score(y_bin_test, p2):<12.4f}'
)
print(
    f'{"Motor ID (Unbalanced)":<40} {accuracy_score(y_mot_test, p3):<12.4f} {f1_score(y_mot_test, p3, average="weighted"):<12.4f} {precision_score(y_mot_test, p3, average="weighted"):<12.4f} {recall_score(y_mot_test, p3, average="weighted"):<12.4f}'
)
print(
    f'{"Motor ID (Balanced)":<40} {accuracy_score(y_mot_test, p4):<12.4f} {f1_score(y_mot_test, p4, average="weighted"):<12.4f} {precision_score(y_mot_test, p4, average="weighted"):<12.4f} {recall_score(y_mot_test, p4, average="weighted"):<12.4f}'
)
print(f'{"Multi-Label (Hamming Avg)":<40} {hamming:<12.4f} {"-":<12} {"-":<12} {"-":<12}')
print(f'{"Multi-Label (Exact Match)":<40} {exact:<12.4f} {"-":<12} {"-":<12} {"-":<12}')
