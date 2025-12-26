"""
Train and evaluate baseline anomaly detectors on ALFA dataset.

Usage:
    python scripts/security/evaluate_baselines_fixed.py \
        --data data/attack_datasets/processed/alfa \
        --output research/security/baselines
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from collections import defaultdict
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinn_dynamics.security.baselines import (
    KalmanResidualDetector,
    Chi2Detector,
    IsolationForestDetector,
    OneClassSVMDetector
)
from pinn_dynamics.security.evaluation import DetectionEvaluator, BenchmarkSuite


def load_normal_data(data_dir: Path):
    """Load normal (no-failure) flights for training."""
    state_cols = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz']

    normal_flights = []
    for csv_file in data_dir.glob("*no_failure*.csv"):
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            normal_flights.append(df)

    if not normal_flights:
        raise ValueError("No normal flights found!")

    normal_df = pd.concat(normal_flights, ignore_index=True)
    states = normal_df[state_cols].values

    return states


def load_fault_scenarios(data_dir: Path):
    """Load all fault scenarios."""
    scenarios = defaultdict(list)

    for csv_file in data_dir.glob("*.csv"):
        if csv_file.name.startswith("summary"):
            continue

        df = pd.read_csv(csv_file)
        if len(df) == 0:
            continue

        fault_type = df['fault_type'].iloc[0]
        scenarios[fault_type].append({
            'name': csv_file.stem,
            'data': df,
            'fault_type': fault_type
        })

    return scenarios


def prepare_test_data(df: pd.DataFrame):
    """Convert dataframe to test format."""
    state_cols = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    states = df[state_cols].values
    labels = df['label'].values
    timestamps = df['timestamp'].values
    return states, labels, timestamps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/attack_datasets/processed/alfa")
    parser.add_argument("--output", type=str, default="research/security/baselines")
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BASELINE DETECTOR EVALUATION - ALFA Dataset")
    print("=" * 80)

    # Load data
    print("\n[1/3] Loading training data...")
    normal_states = load_normal_data(data_dir)
    state_changes = np.diff(normal_states, axis=0)
    print(f"  Normal states: {len(normal_states)}")
    print(f"  State changes: {len(state_changes)}")

    print("\n[2/3] Training baseline detectors...")

    detectors = {}
    results_by_method = defaultdict(list)

    # 1. Chi-squared Test
    print("\n[2.1/4] Chi-squared Test")
    chi2 = Chi2Detector(state_dim=12, alpha=0.05)
    chi2.fit(state_changes)
    detectors['Chi2'] = chi2

    # 2. Isolation Forest
    print("\n[2.2/4] Isolation Forest")
    iforest = IsolationForestDetector(contamination=0.1, n_estimators=100)
    iforest.fit(normal_states)
    detectors['IForest'] = iforest

    # 3. One-Class SVM
    print("\n[2.3/4] One-Class SVM (this may take a while)...")
    # Subsample for SVM training (too slow otherwise)
    svm_sample_size = min(5000, len(normal_states))
    svm_indices = np.random.choice(len(normal_states), svm_sample_size, replace=False)
    svm = OneClassSVMDetector(nu=0.1, kernel='rbf', gamma='auto')
    svm.fit(normal_states[svm_indices])
    detectors['SVM'] = svm

    print(f"\n[2.4/4] Trained {len(detectors)} baselines")

    # Save trained models
    models_path = output_dir / 'trained_baselines.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump(detectors, f)
    print(f"  Saved models: {models_path}")

    # Evaluate
    print("\n[3/3] Evaluating on fault scenarios...")
    scenarios = load_fault_scenarios(data_dir)

    benchmark = BenchmarkSuite(output_dir=str(output_dir))

    for method_name, detector in detectors.items():
        print(f"\n  Evaluating {method_name}...")

        all_predictions = []
        all_labels = []
        all_scores = []
        all_timestamps = []

        for fault_type, flights in scenarios.items():
            for flight in flights:
                states, labels, timestamps = prepare_test_data(flight['data'])

                predictions = []
                scores = []

                if method_name == 'Chi2':
                    # Chi2 needs state changes
                    for i in range(1, len(states)):
                        state_change = states[i] - states[i-1]
                        result = detector.detect(state_change)
                        predictions.append(int(result.is_anomaly))
                        scores.append(result.score)
                    # Pad first timestep
                    predictions = [0] + predictions
                    scores = [0.0] + scores
                else:
                    # IForest and SVM work on states directly
                    for i in range(len(states)):
                        result = detector.detect(states[i])
                        predictions.append(int(result.is_anomaly))
                        scores.append(result.score)

                # Store results
                results_by_method[method_name].append({
                    'flight': flight['name'],
                    'fault_type': fault_type,
                    'predictions': predictions,
                    'labels': labels.tolist(),
                    'scores': scores
                })

                all_predictions.extend(predictions)
                all_labels.extend(labels)
                all_scores.extend(scores)
                all_timestamps.extend(timestamps)

        # Compute overall metrics
        predictions_np = np.array(all_predictions)
        labels_np = np.array(all_labels)
        scores_np = np.array(all_scores)
        timestamps_np = np.array(all_timestamps)

        evaluator = DetectionEvaluator(method_name=method_name)
        metrics = evaluator.evaluate(predictions_np, labels_np, scores_np, timestamps_np)

        print(f"    F1: {metrics.f1:.3f}, Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}")

        benchmark.add_result(method_name, metrics)

        # Save per-flight results
        flight_results = []
        for r in results_by_method[method_name]:
            preds = np.array(r['predictions'])
            labs = np.array(r['labels'])

            TP = np.sum((preds == 1) & (labs == 1))
            TN = np.sum((preds == 0) & (labs == 0))
            FP = np.sum((preds == 1) & (labs == 0))
            FN = np.sum((preds == 0) & (labs == 1))

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            flight_results.append({
                'flight': r['flight'],
                'fault_type': r['fault_type'],
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'TP': int(TP),
                'TN': int(TN),
                'FP': int(FP),
                'FN': int(FN)
            })

        df = pd.DataFrame(flight_results)
        csv_path = output_dir / f"{method_name}_per_flight.csv"
        df.to_csv(csv_path, index=False)

    # Save and print summary
    print("\n[4/4] Saving results...")
    benchmark.save_results("baseline_results.json")
    benchmark.print_summary()
    benchmark.export_latex_table("baseline_comparison.tex")

    print("\n" + "=" * 80)
    print("BASELINE EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
