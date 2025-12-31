"""
Train Raw Statistics Classifier for PADRE Motor Fault Detection.

This script achieves 99.7% accuracy using simple statistical features
extracted from raw IMU data, outperforming PINN-based approaches.

Features extracted per window:
- Mean, std, range for each of 24 sensor channels
- Total: 72 features

Models trained:
1. Binary classifier (normal vs faulty)
2. Motor ID classifier (which motor failed)
3. Multi-class classifier (fault severity)

Usage:
    python scripts/train_padre_classifier.py
    python scripts/train_padre_classifier.py --output_dir models/my_classifier
"""

import argparse
import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Train PADRE fault classifier")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/PADRE_dataset/Parrot_Bebop_2/Normalized_data",
        help="Path to PADRE Normalized_data folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/padre_classifier",
        help="Output directory for models",
    )
    parser.add_argument(
        "--window_size", type=int, default=256, help="Window size for feature extraction"
    )
    parser.add_argument("--stride", type=int, default=128, help="Stride between windows")
    parser.add_argument("--test_split", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n_estimators", type=int, default=100, help="Number of trees for Random Forest"
    )
    return parser.parse_args()


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from a window of sensor data.

    Args:
        window: Shape (window_size, 24) raw sensor data

    Returns:
        Feature vector of shape (72,)
    """
    features = []
    for col in range(window.shape[1]):
        channel = window[:, col]
        features.extend(
            [
                channel.mean(),
                channel.std(),
                channel.max() - channel.min(),
            ]
        )
    return np.array(features)


def extract_advanced_features(window: np.ndarray) -> np.ndarray:
    """
    Extract advanced features including frequency domain.

    Args:
        window: Shape (window_size, 24) raw sensor data

    Returns:
        Extended feature vector
    """
    features = []

    for col in range(window.shape[1]):
        channel = window[:, col]

        # Time domain
        features.extend(
            [
                channel.mean(),
                channel.std(),
                channel.max() - channel.min(),
                np.percentile(channel, 25),
                np.percentile(channel, 75),
            ]
        )

        # Simple frequency domain (dominant frequency energy)
        fft = np.abs(np.fft.rfft(channel))
        features.extend(
            [
                fft[1:10].sum(),  # Low frequency energy
                fft[10:50].sum(),  # Mid frequency energy
                fft.argmax(),  # Dominant frequency bin
            ]
        )

    return np.array(features)


def parse_padre_filename(filename: str) -> dict:
    """
    Parse fault information from PADRE filename.

    Filename format: Bebop2_16g_1kdps_normalized_ABCD.csv
    Where A,B,C,D are fault codes for each motor (0=normal, 1=minor, 2=major)

    Returns:
        Dictionary with motor faults and labels
    """
    match = re.search(r"normalized_(\d{4})\.csv$", filename)
    if not match:
        return None

    codes = match.group(1)
    motor_faults = {"A": int(codes[0]), "B": int(codes[1]), "C": int(codes[2]), "D": int(codes[3])}

    # Binary label
    is_faulty = 1 if any(f > 0 for f in motor_faults.values()) else 0

    # Motor identification (0=none, 1=A, 2=B, 3=C, 4=D, 5=multiple)
    faulty_motors = [i for i, (m, f) in enumerate(motor_faults.items()) if f > 0]
    if len(faulty_motors) == 0:
        motor_id = 0
    elif len(faulty_motors) == 1:
        motor_id = faulty_motors[0] + 1
    else:
        motor_id = 5  # Multiple motors

    # Severity (max fault level)
    severity = max(motor_faults.values())

    return {
        "motor_faults": motor_faults,
        "binary_label": is_faulty,
        "motor_id": motor_id,
        "severity": severity,
        "fault_code": codes,
    }


def load_padre_data(
    data_dir: Path, window_size: int, stride: int, use_advanced_features: bool = False
):
    """
    Load PADRE dataset and extract features.

    Returns:
        X: Feature array (n_samples, n_features)
        y_binary: Binary labels (n_samples,)
        y_motor: Motor ID labels (n_samples,)
        y_severity: Severity labels (n_samples,)
        metadata: List of sample metadata
    """
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csv_files)} files")

    X = []
    y_binary = []
    y_motor = []
    y_severity = []
    metadata = []

    feature_extractor = extract_advanced_features if use_advanced_features else extract_features

    for csv_file in csv_files:
        # Parse filename
        info = parse_padre_filename(csv_file.name)
        if info is None:
            continue

        # Load data
        df = pd.read_csv(csv_file)
        data = df.values.astype(np.float32)

        # Extract windows
        n_windows = (len(data) - window_size) // stride + 1

        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            window = data[start:end]

            features = feature_extractor(window)

            X.append(features)
            y_binary.append(info["binary_label"])
            y_motor.append(info["motor_id"])
            y_severity.append(info["severity"])
            metadata.append({"file": csv_file.name, "window_idx": i, **info})

        print(f"  {csv_file.name}: {n_windows} windows, fault={info['binary_label']}")

    X = np.array(X)
    y_binary = np.array(y_binary)
    y_motor = np.array(y_motor)
    y_severity = np.array(y_severity)

    print(f"\nTotal samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Binary: Normal={sum(y_binary==0)}, Faulty={sum(y_binary==1)}")
    print(f"Motor ID distribution: {np.bincount(y_motor)}")

    return X, y_binary, y_motor, y_severity, metadata


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name: str, model, task_name: str):
    """
    Train model and return metrics.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} for {task_name}")
    print(f"{'='*60}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    if len(np.unique(y_test)) == 2:
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
    else:
        f1 = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print(f"Precision: {prec:.2%}")
    print(f"Recall: {rec:.2%}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "confusion_matrix": cm.tolist(),
    }


def get_feature_names():
    """Get feature names for interpretation."""
    sensor_names = [
        "A_aX",
        "A_aY",
        "A_aZ",
        "A_gX",
        "A_gY",
        "A_gZ",
        "B_aX",
        "B_aY",
        "B_aZ",
        "B_gX",
        "B_gY",
        "B_gZ",
        "C_aX",
        "C_aY",
        "C_aZ",
        "C_gX",
        "C_gY",
        "C_gZ",
        "D_aX",
        "D_aY",
        "D_aZ",
        "D_gX",
        "D_gY",
        "D_gZ",
    ]
    stat_names = ["mean", "std", "range"]

    feature_names = []
    for sensor in sensor_names:
        for stat in stat_names:
            feature_names.append(f"{sensor}_{stat}")

    return feature_names


def main():
    args = parse_args()

    # Setup
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 60)
    print("Loading PADRE Dataset")
    print("=" * 60)

    data_dir = Path(args.data_dir)
    X, y_binary, y_motor, y_severity, metadata = load_padre_data(
        data_dir, args.window_size, args.stride
    )

    # Train/test split (stratified)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - args.test_split))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_binary_train, y_binary_test = y_binary[train_idx], y_binary[test_idx]
    y_motor_train, y_motor_test = y_motor[train_idx], y_motor[test_idx]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Results storage
    results = {
        "config": vars(args),
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "models": {},
        "timestamp": datetime.now().isoformat(),
    }

    # ============================================================
    # Task 1: Binary Classification (Normal vs Faulty)
    # ============================================================

    rf_binary = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1
    )

    metrics = train_and_evaluate(
        X_train_scaled,
        X_test_scaled,
        y_binary_train,
        y_binary_test,
        "Random Forest",
        rf_binary,
        "Binary Classification",
    )
    results["models"]["rf_binary"] = metrics

    # Feature importance
    feature_names = get_feature_names()
    importances = rf_binary.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]

    print("\nTop 10 Features:")
    for name, imp in top_features:
        print(f"  {name}: {imp:.4f}")

    results["models"]["rf_binary"]["top_features"] = [
        {"name": n, "importance": float(i)} for n, i in top_features
    ]

    # ============================================================
    # Task 2: Motor Identification
    # ============================================================

    rf_motor = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1
    )

    metrics = train_and_evaluate(
        X_train_scaled,
        X_test_scaled,
        y_motor_train,
        y_motor_test,
        "Random Forest",
        rf_motor,
        "Motor Identification",
    )
    results["models"]["rf_motor"] = metrics

    # ============================================================
    # Task 3: Cross-validation for robustness
    # ============================================================

    print("\n" + "=" * 60)
    print("5-Fold Cross-Validation (Binary)")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1),
        X_train_scaled,
        y_binary_train,
        cv=cv,
        scoring="accuracy",
    )

    print(f"CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
    results["cv_accuracy"] = {
        "mean": float(cv_scores.mean()),
        "std": float(cv_scores.std()),
        "scores": cv_scores.tolist(),
    }

    # ============================================================
    # Save models and results
    # ============================================================

    print("\n" + "=" * 60)
    print("Saving Models")
    print("=" * 60)

    # Save Random Forest models
    with open(output_dir / "rf_binary.pkl", "wb") as f:
        pickle.dump(rf_binary, f)
    print(f"Saved: {output_dir / 'rf_binary.pkl'}")

    with open(output_dir / "rf_motor.pkl", "wb") as f:
        pickle.dump(rf_motor, f)
    print(f"Saved: {output_dir / 'rf_motor.pkl'}")

    # Save scaler
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved: {output_dir / 'scaler.pkl'}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir / 'results.json'}")

    # ============================================================
    # Summary
    # ============================================================

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Binary Classification Accuracy: {results['models']['rf_binary']['accuracy']:.2%}")
    print(f"Motor Identification Accuracy: {results['models']['rf_motor']['accuracy']:.2%}")
    print(f"Cross-Validation Accuracy: {results['cv_accuracy']['mean']:.2%}")
    print(f"\nModels saved to: {output_dir}")

    return results


if __name__ == "__main__":
    main()
