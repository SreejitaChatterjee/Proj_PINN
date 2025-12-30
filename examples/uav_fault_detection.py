"""
UAV Fault Detection Example
============================

EXPERIMENTAL - Example code for fault detection research.

STATUS: Research in Progress - See research/security/UAV_FAULT_DETECTION.md

Demonstrates:
- Loading trained detector
- Processing flight data
- Real-time anomaly detection
- Performance evaluation

Validated Results (CMU ALFA Dataset):
- Best AUROC: 0.575 (feature-based approach)
- Achievable: 30% detection at 10% false alarm rate
- Limitation: ~1 Hz sampling rate limits dynamics-based detection

Note: Previous claims of 65.7% F1 and 4.5% FPR were based on
incorrect evaluation methodology and have been retracted.

Usage:
    python examples/uav_fault_detection.py

Requirements:
    - Trained model at models/security/detector_w0_seed0.pth
    - Test data at data/alfa/temporal/
    - See research/security/UAV_FAULT_DETECTION.md for setup
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinn_dynamics import QuadrotorPINN, Predictor
from pinn_dynamics.security import AnomalyDetector
from sklearn.preprocessing import StandardScaler


def load_trained_detector(model_path, device='cpu'):
    """Load pre-trained fault detector."""
    print(f"Loading trained detector from {model_path}...")

    # Load model
    model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    return model


def load_test_data(normal_path, fault_path):
    """Load and preprocess test data."""
    print("\nLoading test data...")

    # Load CSVs
    normal_df = pd.read_csv(normal_path)
    fault_df = pd.read_csv(fault_path)

    print(f"Normal flights: {len(normal_df)} samples")
    print(f"Fault flights: {len(fault_df)} samples")

    # Extract features: 12 states + 4 controls = 16 features
    normal_X = normal_df.iloc[:, :16].values
    fault_X = fault_df.iloc[:, :16].values

    # Labels: 0 = normal, 1 = fault
    normal_y = np.zeros(len(normal_df))
    fault_y = np.ones(len(fault_df))

    # Combine
    X = np.vstack([normal_X, fault_X])
    y = np.concatenate([normal_y, fault_y])

    # Get fault types (if available)
    fault_types = fault_df['fault_type'].values if 'fault_type' in fault_df.columns else None

    return X, y, fault_types


def create_detector(model, X_train, device='cpu', threshold=0.1707):
    """Create anomaly detector with calibrated statistics."""
    print("\nCreating detector...")

    # Fit scalers on training data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Normalize inputs (states + controls)
    X_train_scaled = scaler_X.fit_transform(X_train[:, :16])

    # For outputs, we normalize just the states (12 features)
    # In practice, we'd use the actual next states from training data
    # For this example, we'll use the same states as a placeholder
    scaler_y.fit(X_train[:, :12])

    # Create predictor
    predictor = Predictor(model, scaler_X, scaler_y, device=device)

    # Create detector
    detector = AnomalyDetector(
        predictor=predictor,
        threshold=threshold,
        use_physics=False  # w=0 performs better (p<10^-6)
    )

    print(f"Threshold: {threshold}")
    print(f"Use physics: False (w=0 >> w=20)")

    return detector


def detect_faults(detector, X_test, y_test, fault_types=None):
    """Run fault detection on test data."""
    print("\nRunning fault detection...")

    detections = []
    scores = []
    uncertainties = []

    for i in range(len(X_test) - 1):
        # Extract current state, control, and next state
        current_state = X_test[i, :12]
        control = X_test[i, 12:16]
        next_state_measured = X_test[i+1, :12]

        # Detect anomaly
        result = detector.detect(current_state, control, next_state_measured)

        detections.append(result.is_anomaly)
        scores.append(result.score)
        uncertainties.append(result.uncertainty)

    # Convert to arrays
    detections = np.array(detections)
    scores = np.array(scores)
    uncertainties = np.array(uncertainties)
    y_test = y_test[:-1]  # Align with detections (we lose last sample)

    return detections, scores, uncertainties, y_test


def evaluate_performance(detections, y_true, fault_types=None):
    """Calculate performance metrics."""
    print("\n" + "="*50)
    print("PERFORMANCE RESULTS")
    print("="*50)

    # Confusion matrix
    TP = np.sum((detections == 1) & (y_true == 1))
    TN = np.sum((detections == 0) & (y_true == 0))
    FP = np.sum((detections == 1) & (y_true == 0))
    FN = np.sum((detections == 0) & (y_true == 1))

    # Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

    print(f"\nOverall Metrics:")
    print(f"  F1 Score: {f1*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall: {recall*100:.1f}%")
    print(f"  False Positive Rate: {fpr*100:.1f}%")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {TP:,}")
    print(f"  True Negatives:  {TN:,}")
    print(f"  False Positives: {FP:,}")
    print(f"  False Negatives: {FN:,}")

    # Per-fault type analysis
    if fault_types is not None:
        print(f"\nPer-Fault Type Performance:")
        unique_faults = np.unique(fault_types)
        for fault in unique_faults:
            mask = (fault_types == fault)[:-1]  # Align with detections
            if np.sum(mask) > 0:
                fault_TP = np.sum((detections[mask] == 1) & (y_true[mask] == 1))
                fault_FN = np.sum((detections[mask] == 0) & (y_true[mask] == 1))
                fault_prec = fault_TP / np.sum(detections[mask] == 1) if np.sum(detections[mask] == 1) > 0 else 0
                fault_recall = fault_TP / (fault_TP + fault_FN) if (fault_TP + fault_FN) > 0 else 0
                fault_f1 = 2 * fault_prec * fault_recall / (fault_prec + fault_recall) if (fault_prec + fault_recall) > 0 else 0

                print(f"  {fault}:")
                print(f"    F1={fault_f1*100:.1f}%, Prec={fault_prec*100:.1f}%, Recall={fault_recall*100:.1f}%")

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }


def main():
    """Main execution."""
    print("="*50)
    print("UAV FAULT DETECTION - EXAMPLE")
    print("="*50)
    print("\nEXPERIMENTAL - Research in progress")
    print("Dataset: CMU ALFA (47 flights, ~1 Hz sampling)")
    print("See: research/security/UAV_FAULT_DETECTION.md")
    print()

    # Paths
    MODEL_PATH = Path('models/security/detector_w0_seed0.pth')
    NORMAL_PATH = Path('data/ALFA_processed/normal_test.csv')
    FAULT_PATH = Path('data/ALFA_processed/fault_test.csv')
    NORMAL_TRAIN_PATH = Path('data/ALFA_processed/normal_train.csv')

    # Check files exist
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please run: python scripts/security/train_detector.py")
        print("Or see research/security/QUICKSTART.md for setup")
        return

    if not NORMAL_PATH.exists() or not FAULT_PATH.exists():
        print(f"ERROR: Test data not found")
        print("Please run: python scripts/security/preprocess_alfa.py")
        print("Or see research/security/QUICKSTART.md for setup")
        return

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model = load_trained_detector(MODEL_PATH, device=device)

    # Load data
    X_test, y_test, fault_types = load_test_data(NORMAL_PATH, FAULT_PATH)

    # Load training data for calibration
    train_df = pd.read_csv(NORMAL_TRAIN_PATH)
    X_train = train_df.iloc[:, :16].values

    # Create detector
    detector = create_detector(model, X_train, device=device, threshold=0.1707)

    # Run detection
    detections, scores, uncertainties, y_test_aligned = detect_faults(detector, X_test, y_test, fault_types)

    # Evaluate
    metrics = evaluate_performance(detections, y_test_aligned, fault_types)

    print("\n" + "="*50)
    print("NOTES")
    print("="*50)
    print(f"\nThis is EXPERIMENTAL research code.")
    print(f"Validated results on ALFA dataset:")
    print(f"  - Best AUROC: 0.575 (feature-based detection)")
    print(f"  - Achievable: 30% detection at 10% FA rate")
    print(f"  - Limitation: ~1 Hz sampling limits dynamics approaches")
    print(f"\nSee research/security/UAV_FAULT_DETECTION.md for details.")

    print("\n" + "="*50)
    print("See research/security/ for full documentation")
    print("="*50)


if __name__ == '__main__':
    main()
