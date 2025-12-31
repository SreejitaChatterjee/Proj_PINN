"""
PADRE Motor Fault Detection - Production Training Script.

Implements within-file temporal split to avoid overfitting:
- First 70% of each file for training
- Last 30% for testing
- Both normal files contribute to train AND test

Usage:
    python scripts/train_padre_final.py
    python scripts/train_padre_final.py --train_ratio 0.8
"""

import argparse
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract time + frequency domain features from sensor window.

    Args:
        window: Shape (window_size, 24) - 24 sensor channels

    Returns:
        Feature vector of shape (168,) - 7 features per channel
    """
    features = []
    for col in range(window.shape[1]):
        ch = window[:, col]

        # Time domain (3 features)
        features.extend([ch.mean(), ch.std(), ch.max() - ch.min()])

        # Frequency domain (4 features)
        fft = np.abs(np.fft.rfft(ch))
        features.extend(
            [
                fft[1:10].sum(),  # Low frequency energy
                fft[10:50].sum(),  # Mid frequency energy
                fft[50:].sum(),  # High frequency energy
                np.argmax(fft[1:]) if len(fft) > 1 else 0,  # Dominant frequency
            ]
        )

    return np.array(features)


def parse_fault_code(filename: str) -> str:
    """Extract 4-digit fault code from PADRE filename."""
    match = re.search(r"_(\d{4})\.csv$", filename)
    return match.group(1) if match else None


def load_padre_data(
    data_dirs: list, window_size: int = 256, stride: int = 128, train_ratio: float = 0.7
):
    """
    Load PADRE data with within-file temporal split.

    Args:
        data_dirs: List of (name, path) tuples for each dataset
        window_size: Samples per window
        stride: Samples between windows
        train_ratio: Fraction of each file for training

    Returns:
        X_train, y_train, X_test, y_test, file_stats
    """
    X_train, y_train, X_test, y_test = [], [], [], []
    groups_train, groups_test = [], []
    file_stats = []
    file_id = 0

    for drone_name, data_dir in data_dirs:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"  Skipping {drone_name}: {data_dir} not found")
            continue

        for csv_file in sorted(data_dir.glob("*.csv")):
            fault_code = parse_fault_code(csv_file.name)
            if not fault_code:
                continue

            # Parse labels
            is_faulty = 1 if any(int(c) > 0 for c in fault_code) else 0
            motor_faults = [int(c) > 0 for c in fault_code]  # Per-motor binary

            # Load data (use first 24 columns for compatibility)
            df = pd.read_csv(csv_file)
            data = df.values.astype(np.float32)[:, :24]

            # Extract windows
            windows = []
            for i in range((len(data) - window_size) // stride + 1):
                window = data[i * stride : i * stride + window_size]
                windows.append(extract_features(window))

            if not windows:
                continue

            # Temporal split
            n_train = int(len(windows) * train_ratio)

            X_train.extend(windows[:n_train])
            y_train.extend([is_faulty] * n_train)
            groups_train.extend([file_id] * n_train)

            X_test.extend(windows[n_train:])
            y_test.extend([is_faulty] * (len(windows) - n_train))
            groups_test.extend([file_id] * (len(windows) - n_train))

            file_stats.append(
                {
                    "id": file_id,
                    "drone": drone_name,
                    "file": csv_file.name,
                    "fault_code": fault_code,
                    "is_faulty": is_faulty,
                    "motor_faults": motor_faults,
                    "n_train": n_train,
                    "n_test": len(windows) - n_train,
                }
            )
            file_id += 1

    return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), file_stats)


def get_feature_names():
    """Get descriptive names for all 168 features."""
    sensors = [
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
    stats = ["mean", "std", "range", "lowFreq", "midFreq", "highFreq", "domFreq"]

    names = []
    for sensor in sensors:
        for stat in stats:
            names.append(f"{sensor}_{stat}")
    return names


def main():
    parser = argparse.ArgumentParser(description="Train PADRE fault detector")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction of each file for training (default: 0.7)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/padre_classifier", help="Output directory"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=100, help="Number of trees in Random Forest"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (increase to reduce FP)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("PADRE MOTOR FAULT DETECTION - PRODUCTION TRAINING")
    print("=" * 80)
    print(f"Train ratio: {args.train_ratio:.0%}")
    print(f"Output: {args.output_dir}")

    # Data directories
    base = Path("C:/Users/sreej/OneDrive/Documents/GitHub/Proj_PINN/data/PADRE_dataset")
    data_dirs = [
        ("Bebop2", base / "Parrot_Bebop_2" / "Normalized_data"),
        ("Solo", base / "3DR_Solo" / "Normalized_data" / "extracted"),
    ]

    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    X_train, y_train, X_test, y_test, file_stats = load_padre_data(
        data_dirs, train_ratio=args.train_ratio
    )

    n_normal = sum(1 for f in file_stats if f["is_faulty"] == 0)
    n_faulty = sum(1 for f in file_stats if f["is_faulty"] == 1)

    print(f"\nDataset: {len(file_stats)} files ({n_normal} normal, {n_faulty} faulty)")
    print(f"Train: {len(X_train)} samples ({sum(y_train==0)} normal, {sum(y_train==1)} faulty)")
    print(f"Test:  {len(X_test)} samples ({sum(y_test==0)} normal, {sum(y_test==1)} faulty)")
    print(f"Features: {X_train.shape[1]}")

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # Predict with threshold
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= args.threshold).astype(int)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nThreshold: {args.threshold}")
    print(f"\nOverall Metrics:")
    print(
        f"  Accuracy:    {accuracy_score(y_test, y_pred):.4f} ({accuracy_score(y_test, y_pred)*100:.2f}%)"
    )
    print(f"  Precision:   {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:      {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score:    {f1_score(y_test, y_pred):.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (TN):   {tn:5d}  (Normal correctly classified)")
    print(f"  False Positives (FP):  {fp:5d}  (Normal misclassified as Faulty)")
    print(f"  False Negatives (FN):  {fn:5d}  (Faulty missed - DANGEROUS)")
    print(f"  True Positives (TP):   {tp:5d}  (Faulty correctly classified)")

    print(f"\nPer-Class Accuracy:")
    print(f"  Normal class:  {tn/(tn+fp)*100:.1f}% ({tn}/{tn+fp})")
    print(f"  Faulty class:  {tp/(tp+fn)*100:.1f}% ({tp}/{tp+fn})")

    # Feature importance
    feature_names = get_feature_names()
    importances = clf.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]

    print(f"\nTop 10 Features:")
    for name, imp in top_features:
        print(f"  {name}: {imp:.4f}")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    # Save classifier
    model_path = output_dir / "rf_binary_final.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model: {model_path}")

    # Save scaler
    scaler_path = output_dir / "scaler_final.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler: {scaler_path}")

    # Save results
    results = {
        "config": {
            "train_ratio": args.train_ratio,
            "n_estimators": args.n_estimators,
            "threshold": args.threshold,
            "n_features": X_train.shape[1],
            "window_size": 256,
            "stride": 128,
        },
        "dataset": {
            "n_files": len(file_stats),
            "n_normal_files": n_normal,
            "n_faulty_files": n_faulty,
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "normal_accuracy": float(tn / (tn + fp)),
            "faulty_accuracy": float(tp / (tp + fn)),
        },
        "top_features": [{"name": n, "importance": float(i)} for n, i in top_features],
        "files": file_stats,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = output_dir / "results_final.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {results_path}")

    # Save inference code
    inference_code = '''"""
PADRE Fault Detection - Inference Module.

Usage:
    from padre_inference import PADREDetector

    detector = PADREDetector('models/padre_classifier')
    is_faulty, confidence = detector.predict(sensor_window)
"""

import pickle
import numpy as np
from pathlib import Path


class PADREDetector:
    """Production fault detector for PADRE sensor data."""

    def __init__(self, model_dir: str, threshold: float = 0.5):
        model_dir = Path(model_dir)

        with open(model_dir / 'rf_binary_final.pkl', 'rb') as f:
            self.clf = pickle.load(f)
        with open(model_dir / 'scaler_final.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        self.threshold = threshold
        self.window_size = 256

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """Extract features from sensor window."""
        features = []
        for col in range(min(window.shape[1], 24)):
            ch = window[:, col]
            features.extend([ch.mean(), ch.std(), ch.max() - ch.min()])
            fft = np.abs(np.fft.rfft(ch))
            features.extend([fft[1:10].sum(), fft[10:50].sum(), fft[50:].sum(),
                           np.argmax(fft[1:]) if len(fft) > 1 else 0])
        return np.array(features)

    def predict(self, window: np.ndarray) -> tuple:
        """
        Predict if sensor window indicates motor fault.

        Args:
            window: Shape (256, 24) sensor data

        Returns:
            (is_faulty: bool, confidence: float)
        """
        if window.shape[0] < self.window_size:
            raise ValueError(f"Window must have at least {self.window_size} samples")

        features = self.extract_features(window[:self.window_size])
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        proba = self.clf.predict_proba(features_scaled)[0, 1]
        is_faulty = proba >= self.threshold

        return bool(is_faulty), float(proba)

    def predict_stream(self, data: np.ndarray, stride: int = 128):
        """
        Process streaming sensor data.

        Args:
            data: Shape (n_samples, 24) sensor data
            stride: Samples between predictions

        Yields:
            (window_idx, is_faulty, confidence)
        """
        for i in range((len(data) - self.window_size) // stride + 1):
            window = data[i * stride: i * stride + self.window_size]
            is_faulty, conf = self.predict(window)
            yield i, is_faulty, conf


if __name__ == "__main__":
    # Test
    detector = PADREDetector('models/padre_classifier')

    # Simulate normal data
    test_window = np.random.randn(256, 24).astype(np.float32)
    is_faulty, conf = detector.predict(test_window)
    print(f"Test prediction: faulty={is_faulty}, confidence={conf:.3f}")
'''

    inference_path = output_dir / "padre_inference.py"
    with open(inference_path, "w") as f:
        f.write(inference_code)
    print(f"Inference: {inference_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"\nModel ready for deployment!")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")


if __name__ == "__main__":
    main()
