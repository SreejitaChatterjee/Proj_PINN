"""
Evaluate PINN fault detector on ALFA dataset.

Tests trained models on 37 real fault scenarios and compares with baselines.

Usage:
    python scripts/security/evaluate_detector.py \\
        --model models/security/pinn_w0_best.pth \\
        --data data/attack_datasets/processed/alfa/ \\
        --output research/security/results/
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Import PINN framework (install with: pip install -e .)
try:
    from pinn_dynamics import Predictor, QuadrotorPINN
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pinn_dynamics import QuadrotorPINN, Predictor

from pinn_dynamics.security import AnomalyDetector
from pinn_dynamics.security.baselines import (
    Chi2Detector,
    IsolationForestDetector,
    KalmanResidualDetector,
    OneClassSVMDetector,
)
from pinn_dynamics.security.evaluation import BenchmarkSuite, DetectionEvaluator


def load_trained_model(model_path: Path, scalers_path: Path, device: str = "cpu"):
    """Load trained PINN model and scalers."""
    # Load model
    model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load scalers
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    scaler_X = scalers["scaler_X"]
    scaler_y = scalers["scaler_y"]

    print(f"Loaded model from {model_path}")

    return model, scaler_X, scaler_y


def load_fault_scenarios(data_dir: Path):
    """
    Load all fault scenarios from ALFA dataset.

    Returns:
        dict: {fault_type: list of dataframes}
    """
    scenarios = defaultdict(list)

    csv_files = list(data_dir.glob("*.csv"))

    for csv_file in csv_files:
        if csv_file.name.startswith("summary"):
            continue

        df = pd.read_csv(csv_file)

        if len(df) == 0:
            continue

        fault_type = df["fault_type"].iloc[0]
        scenarios[fault_type].append({"name": csv_file.stem, "data": df, "fault_type": fault_type})

    print(f"\nLoaded fault scenarios:")
    for fault_type, flights in scenarios.items():
        print(f"  {fault_type}: {len(flights)} flights")

    return scenarios


def prepare_test_data(df: pd.DataFrame):
    """Convert dataframe to test format (states, controls, labels)."""
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    states = df[state_cols].values[:-1]
    controls = df[control_cols].values[:-1]
    next_states = df[state_cols].values[1:]
    labels = df["label"].values[:-1]
    timestamps = df["timestamp"].values[:-1]

    return states, controls, next_states, labels, timestamps


def evaluate_on_scenario(detector: AnomalyDetector, scenario: dict, scaler_X, scaler_y):
    """
    Evaluate detector on a single fault scenario.

    Returns:
        Detection metrics
    """
    df = scenario["data"]
    states, controls, next_states, labels, timestamps = prepare_test_data(df)

    # Prepare inputs (scale them)
    X = np.concatenate([states, controls], axis=1)
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(next_states)

    # Detect anomalies
    results = []
    scores = []

    for i in range(len(states)):
        result = detector.detect(states[i], controls[i], next_states[i])
        results.append(result.is_anomaly)
        scores.append(result.total_score)

    predictions = np.array(results, dtype=int)
    scores = np.array(scores)

    # Compute metrics
    evaluator = DetectionEvaluator(method_name=f"PINN_{scenario['fault_type']}")
    metrics = evaluator.evaluate(predictions, labels, scores, timestamps)

    return metrics, scores


def run_full_evaluation(model_path: Path, data_dir: Path, output_dir: Path, threshold: float = 3.0):
    """
    Run complete evaluation on all fault scenarios.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PINN FAULT DETECTOR EVALUATION - ALFA Dataset")
    print("=" * 60)

    # Load model
    print("\n[1/5] Loading trained model...")
    scalers_path = model_path.parent / "scalers.pkl"
    model, scaler_X, scaler_y = load_trained_model(model_path, scalers_path)

    # Create predictor
    predictor = Predictor(model, scaler_X, scaler_y, device="cpu")

    # Create detector (use simplified version: prediction error only, no physics)
    detector = AnomalyDetector(predictor, threshold=threshold, use_physics=False)

    # Calibrate detector on normal flights
    print("\n[2/5] Calibrating detector on normal flights...")
    normal_flights = []
    for csv_file in data_dir.glob("*no_failure*.csv"):
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            normal_flights.append(df)

    if normal_flights:
        normal_df = pd.concat(normal_flights, ignore_index=True)
        states, controls, next_states, _, _ = prepare_test_data(normal_df)
        detector.calibrate(states, controls, next_states)
    else:
        print("  Warning: No normal flights found for calibration!")

    # Load fault scenarios
    print("\n[3/5] Loading fault scenarios...")
    scenarios = load_fault_scenarios(data_dir)

    # Evaluate on each fault type
    print("\n[4/5] Evaluating on fault scenarios...")
    results_by_fault = {}
    all_results = []

    for fault_type, flights in scenarios.items():
        print(f"\n  Testing {fault_type} ({len(flights)} flights)...")

        fault_metrics = []
        for flight in flights:
            metrics, scores = evaluate_on_scenario(detector, flight, scaler_X, scaler_y)
            fault_metrics.append(metrics)
            all_results.append(
                {
                    "flight": flight["name"],
                    "fault_type": fault_type,
                    **metrics.to_dict(),
                }
            )

        # Average metrics across flights of this type
        if fault_metrics:
            avg_metrics = {
                "accuracy": np.mean([m.accuracy for m in fault_metrics]),
                "precision": np.mean([m.precision for m in fault_metrics]),
                "recall": np.mean([m.recall for m in fault_metrics]),
                "f1": np.mean([m.f1 for m in fault_metrics]),
                "fpr": np.mean([m.fpr for m in fault_metrics]),
                "tpr": np.mean([m.tpr for m in fault_metrics]),
                "n_flights": len(flights),
            }
            results_by_fault[fault_type] = avg_metrics

            print(
                f"    F1: {avg_metrics['f1']:.3f}, "
                f"Precision: {avg_metrics['precision']:.3f}, "
                f"Recall: {avg_metrics['recall']:.3f}"
            )

    # Save results
    print("\n[5/5] Saving results...")

    # Save per-flight results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "per_flight_results.csv", index=False)
    print(f"  Saved: {output_dir / 'per_flight_results.csv'}")

    # Save per-fault-type results
    with open(output_dir / "per_fault_type_results.json", "w") as f:
        json.dump(results_by_fault, f, indent=2)
    print(f"  Saved: {output_dir / 'per_fault_type_results.json'}")

    # Overall statistics
    overall = {
        "mean_f1": np.mean([m["f1"] for m in results_by_fault.values()]),
        "mean_precision": np.mean([m["precision"] for m in results_by_fault.values()]),
        "mean_recall": np.mean([m["recall"] for m in results_by_fault.values()]),
        "mean_fpr": np.mean([m["fpr"] for m in results_by_fault.values()]),
        "total_flights_tested": sum(m["n_flights"] for m in results_by_fault.values()),
        "fault_types_tested": len(results_by_fault),
    }

    with open(output_dir / "overall_results.json", "w") as f:
        json.dump(overall, f, indent=2)
    print(f"  Saved: {output_dir / 'overall_results.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total fault types tested: {overall['fault_types_tested']}")
    print(f"Total flights tested: {overall['total_flights_tested']}")
    print(f"\nOverall Performance:")
    print(f"  F1-Score:   {overall['mean_f1']:.3f}")
    print(f"  Precision:  {overall['mean_precision']:.3f}")
    print(f"  Recall:     {overall['mean_recall']:.3f}")
    print(f"  FPR:        {overall['mean_fpr']:.3f}")

    print("\nPer-Fault-Type Performance:")
    for fault_type, metrics in sorted(results_by_fault.items()):
        print(
            f"  {fault_type:20s}: F1={metrics['f1']:.3f}, "
            f"Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f}"
        )

    print(f"\nResults saved to: {output_dir.absolute()}")

    return results_by_fault, overall


def main():
    parser = argparse.ArgumentParser(description="Evaluate PINN detector")
    parser.add_argument(
        "--model",
        type=str,
        default="models/security/pinn_w0_best.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/attack_datasets/processed/alfa",
        help="Path to ALFA dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="research/security/results",
        help="Output directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Detection threshold",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    data_dir = Path(args.data)
    output_dir = Path(args.output)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return

    # Run evaluation
    results_by_fault, overall = run_full_evaluation(
        model_path, data_dir, output_dir, args.threshold
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review results in research/security/results/")
    print("2. Compare with baselines")
    print("3. Generate paper figures (ROC curves, tables)")


if __name__ == "__main__":
    main()
