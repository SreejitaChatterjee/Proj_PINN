"""
Train and evaluate baseline anomaly detectors on ALFA dataset.

Compares 5 baseline methods with PINN detector:
1. Kalman Filter Residuals
2. LSTM Autoencoder
3. Chi-squared Test
4. Isolation Forest
5. One-Class SVM

Usage:
    python scripts/security/evaluate_baselines.py \
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
    LSTMAutoencoder,
    Chi2Detector,
    IsolationForestDetector,
    OneClassSVMDetector
)
from pinn_dynamics.security.evaluation import DetectionEvaluator, BenchmarkSuite


def load_fault_scenarios(data_dir: Path):
    """Load all fault scenarios from ALFA dataset."""
    scenarios = defaultdict(list)

    csv_files = list(data_dir.glob("*.csv"))

    for csv_file in csv_files:
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

    print(f"Loaded fault scenarios:")
    for fault_type, flights in scenarios.items():
        print(f"  {fault_type}: {len(flights)} flights")

    return scenarios


def prepare_training_data(data_dir: Path):
    """Prepare training data from normal flights."""
    state_cols = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']

    normal_flights = []
    for csv_file in data_dir.glob("*no_failure*.csv"):
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            normal_flights.append(df)

    if not normal_flights:
        raise ValueError("No normal flights found for training!")

    normal_df = pd.concat(normal_flights, ignore_index=True)

    states = normal_df[state_cols].values[:-1]
    controls = normal_df[control_cols].values[:-1]
    next_states = normal_df[state_cols].values[1:]

    return states, controls, next_states


def prepare_test_data(df: pd.DataFrame):
    """Convert dataframe to test format."""
    state_cols = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']

    states = df[state_cols].values[:-1]
    controls = df[control_cols].values[:-1]
    next_states = df[state_cols].values[1:]
    labels = df['label'].values[:-1]
    timestamps = df['timestamp'].values[:-1]

    return states, controls, next_states, labels, timestamps


def evaluate_detector(detector, scenario, threshold=None):
    """Evaluate a detector on a single scenario."""
    df = scenario['data']
    states, controls, next_states, labels, timestamps = prepare_test_data(df)

    # Detect anomalies
    start_time = time.time()
    scores = []
    for i in range(len(states)):
        score = detector.detect(states[i], controls[i], next_states[i])
        scores.append(score)
    inference_time = (time.time() - start_time) / len(states) * 1000  # ms per sample

    scores = np.array(scores)

    # Apply threshold
    if threshold is None:
        # Auto-tune threshold on this flight
        threshold = np.percentile(scores, 75)

    predictions = (scores > threshold).astype(int)

    # Compute metrics
    evaluator = DetectionEvaluator(method_name=detector.__class__.__name__)
    metrics = evaluator.evaluate(predictions, labels, scores, timestamps)

    # Add inference time
    metrics.mean_inference_time = inference_time
    metrics.std_inference_time = 0.0

    return metrics, scores, threshold


def tune_threshold_on_validation(detector, val_scenarios, metric='f1'):
    """Tune threshold on validation set."""
    print(f"  Tuning threshold...")

    all_scores = []
    all_labels = []

    for scenario in val_scenarios[:5]:  # Use first 5 flights for tuning
        df = scenario['data']
        states, controls, next_states, labels, _ = prepare_test_data(df)

        scores = []
        for i in range(len(states)):
            score = detector.detect(states[i], controls[i], next_states[i])
            scores.append(score)

        all_scores.extend(scores)
        all_labels.extend(labels)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Try different thresholds
    thresholds = np.percentile(all_scores, np.linspace(10, 90, 50))
    best_threshold = thresholds[0]
    best_metric_value = 0

    for thresh in thresholds:
        predictions = (all_scores > thresh).astype(int)

        TP = np.sum((predictions == 1) & (all_labels == 1))
        TN = np.sum((predictions == 0) & (all_labels == 0))
        FP = np.sum((predictions == 1) & (all_labels == 0))
        FN = np.sum((predictions == 0) & (all_labels == 1))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if metric == 'f1' and f1 > best_metric_value:
            best_metric_value = f1
            best_threshold = thresh

    print(f"    Optimal threshold: {best_threshold:.4f} (F1={best_metric_value:.3f})")
    return best_threshold


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline detectors")
    parser.add_argument(
        "--data",
        type=str,
        default="data/attack_datasets/processed/alfa",
        help="Path to ALFA dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="research/security/baselines",
        help="Output directory",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BASELINE DETECTOR EVALUATION - ALFA Dataset")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading data...")
    scenarios = load_fault_scenarios(data_dir)

    print("\n[2/4] Preparing training data from normal flights...")
    states_train, controls_train, next_states_train = prepare_training_data(data_dir)
    print(f"  Training samples: {len(states_train)}")

    # Initialize and train detectors
    print("\n[3/4] Training baseline detectors...")

    detectors = {}

    # 1. Kalman Filter (no training needed)
    print("\n[3.1/5] Kalman Filter Residuals")
    kalman = KalmanResidualDetector(state_dim=12, control_dim=4)
    kalman.fit(states_train, controls_train, next_states_train)
    detectors['Kalman'] = kalman

    # 2. Chi-squared (no training needed, just compute covariance)
    print("\n[3.2/5] Chi-squared Test")
    chi2 = Chi2Detector(state_dim=12, control_dim=4)
    chi2.fit(states_train, controls_train, next_states_train)
    detectors['Chi2'] = chi2

    # 3. LSTM Autoencoder
    print("\n[3.3/5] LSTM Autoencoder")
    print("  Training LSTM (this may take a while)...")
    lstm = LSTMAutoencoder(state_dim=12, control_dim=4, hidden_size=64, num_layers=2)
    lstm.fit(states_train, controls_train, next_states_train, epochs=50, batch_size=32)
    detectors['LSTM'] = lstm

    # 4. Isolation Forest
    print("\n[3.4/5] Isolation Forest")
    iforest = IsolationForestDetector(contamination=0.1)
    iforest.fit(states_train, controls_train, next_states_train)
    detectors['IForest'] = iforest

    # 5. One-Class SVM
    print("\n[3.5/5] One-Class SVM")
    print("  Training SVM (this may take a while)...")
    svm = OneClassSVMDetector(nu=0.1, kernel='rbf')
    svm.fit(states_train, controls_train, next_states_train)
    detectors['SVM'] = svm

    # Save trained models
    models_path = output_dir / 'trained_baselines.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump(detectors, f)
    print(f"\n  Saved trained models: {models_path}")

    # Evaluate on all scenarios
    print("\n[4/4] Evaluating on fault scenarios...")

    benchmark = BenchmarkSuite(output_dir=str(output_dir))
    results_by_method = defaultdict(list)

    for method_name, detector in detectors.items():
        print(f"\n  Evaluating {method_name}...")

        # Tune threshold on validation set
        all_scenarios = []
        for fault_type, flights in scenarios.items():
            all_scenarios.extend(flights)

        threshold = tune_threshold_on_validation(detector, all_scenarios)

        # Evaluate on each fault type
        for fault_type, flights in scenarios.items():
            print(f"    Testing on {fault_type} ({len(flights)} flights)...")

            for flight in flights:
                metrics, scores, _ = evaluate_detector(detector, flight, threshold)
                results_by_method[method_name].append({
                    'flight': flight['name'],
                    'fault_type': fault_type,
                    **metrics.to_dict()
                })

        # Compute average metrics
        avg_metrics = {
            'accuracy': np.mean([r['accuracy'] for r in results_by_method[method_name]]),
            'precision': np.mean([r['precision'] for r in results_by_method[method_name]]),
            'recall': np.mean([r['recall'] for r in results_by_method[method_name]]),
            'f1': np.mean([r['f1'] for r in results_by_method[method_name]]),
            'fpr': np.mean([r['fpr'] for r in results_by_method[method_name]]),
            'mean_inference_time': np.mean([r['mean_inference_time'] for r in results_by_method[method_name]]),
        }

        print(f"    Average F1: {avg_metrics['f1']:.3f}, "
              f"Precision: {avg_metrics['precision']:.3f}, "
              f"Recall: {avg_metrics['recall']:.3f}")

        # Add to benchmark suite
        from pinn_dynamics.security.evaluation import DetectionMetrics
        avg_metrics_obj = DetectionMetrics(
            accuracy=avg_metrics['accuracy'],
            precision=avg_metrics['precision'],
            recall=avg_metrics['recall'],
            f1=avg_metrics['f1'],
            fpr=avg_metrics['fpr'],
            tpr=avg_metrics['recall'],
            specificity=1 - avg_metrics['fpr'],
            balanced_accuracy=(avg_metrics['recall'] + (1 - avg_metrics['fpr'])) / 2,
            TP=0, TN=0, FP=0, FN=0,  # Aggregated, so individual counts not meaningful
            mean_inference_time=avg_metrics['mean_inference_time']
        )
        benchmark.add_result(method_name, avg_metrics_obj)

    # Save results
    print("\n[5/5] Saving results...")
    benchmark.save_results("baseline_results.json")
    benchmark.print_summary()

    # Export LaTeX table
    benchmark.export_latex_table("baseline_comparison.tex")

    # Save per-flight results for each method
    for method_name, results in results_by_method.items():
        df = pd.DataFrame(results)
        csv_path = output_dir / f"{method_name}_per_flight.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {method_name} results: {csv_path}")

    print("\n" + "=" * 80)
    print("BASELINE EVALUATION COMPLETE!")
    print("=" * 80)
    print("\nNext step: Compare with PINN detector results")


if __name__ == "__main__":
    main()
