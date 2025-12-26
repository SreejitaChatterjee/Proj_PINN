"""
Comprehensive evaluation framework for anomaly detection methods.

Provides metrics, visualizations, and statistical tests for comparing
physics-informed detection with baselines.

Publication-ready outputs:
- ROC curves (TPR vs FPR)
- Precision-Recall curves
- Confusion matrices
- Detection delay (time-to-detect)
- Statistical significance tests
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class DetectionMetrics:
    """Container for detection performance metrics."""

    # Classification metrics
    accuracy: float
    precision: float
    recall: float  # = TPR = detection rate
    f1: float
    fpr: float  # False positive rate
    tpr: float  # True positive rate = recall
    specificity: float  # = 1 - FPR
    balanced_accuracy: float  # = (TPR + TNR) / 2

    # Confusion matrix
    TP: int
    TN: int
    FP: int
    FN: int

    # Detection delay
    mean_detection_delay: Optional[float] = None  # seconds
    median_detection_delay: Optional[float] = None

    # ROC/PR metrics
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None

    # Computational cost
    mean_inference_time: Optional[float] = None  # milliseconds
    std_inference_time: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in self.__dict__.items()
        }


class DetectionEvaluator:
    """
    Comprehensive evaluator for anomaly detection methods.

    Computes all metrics needed for research paper comparison tables.
    """

    def __init__(self, method_name: str = "Detector"):
        self.method_name = method_name

    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        scores: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> DetectionMetrics:
        """
        Compute comprehensive detection metrics.

        Args:
            predictions: [N] binary predictions (0=normal, 1=anomaly)
            ground_truth: [N] binary ground truth labels
            scores: [N] anomaly scores (for ROC/PR curves)
            timestamps: [N] timestamps (for detection delay)

        Returns:
            DetectionMetrics with all computed metrics
        """
        # Confusion matrix
        TP = np.sum((predictions == 1) & (ground_truth == 1))
        TN = np.sum((predictions == 0) & (ground_truth == 0))
        FP = np.sum((predictions == 1) & (ground_truth == 0))
        FN = np.sum((predictions == 0) & (ground_truth == 1))

        # Basic metrics
        accuracy = (TP + TN) / len(predictions) if len(predictions) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # = TPR
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        specificity = 1 - fpr
        balanced_accuracy = (recall + specificity) / 2

        # Detection delay
        mean_delay, median_delay = None, None
        if timestamps is not None:
            delay = self._compute_detection_delay(predictions, ground_truth, timestamps)
            if delay:
                mean_delay = np.mean(delay)
                median_delay = np.median(delay)

        # ROC/PR AUC (if scores provided)
        roc_auc, pr_auc = None, None
        if scores is not None:
            roc_auc = self._compute_roc_auc(scores, ground_truth)
            pr_auc = self._compute_pr_auc(scores, ground_truth)

        return DetectionMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            fpr=fpr,
            tpr=recall,
            specificity=specificity,
            balanced_accuracy=balanced_accuracy,
            TP=int(TP),
            TN=int(TN),
            FP=int(FP),
            FN=int(FN),
            mean_detection_delay=mean_delay,
            median_detection_delay=median_delay,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
        )

    def _compute_detection_delay(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[float]:
        """
        Compute time-to-detect (TTD) for each attack.

        TTD = time from attack start to first detection.

        Args:
            predictions: Binary predictions
            ground_truth: Binary labels
            timestamps: Timestamps (seconds)

        Returns:
            List of detection delays (seconds) for each attack
        """
        delays = []

        # Find attack intervals
        attack_starts = np.where(np.diff(ground_truth, prepend=0) == 1)[0]
        attack_ends = np.where(np.diff(ground_truth, append=0) == -1)[0]

        for start, end in zip(attack_starts, attack_ends):
            # Find first detection in this interval
            detections_in_interval = np.where(predictions[start : end + 1] == 1)[0]

            if len(detections_in_interval) > 0:
                # Detection delay = time from attack start to first detection
                first_detection = start + detections_in_interval[0]
                delay = timestamps[first_detection] - timestamps[start]
                delays.append(delay)
            # else: missed detection (already counted in FN)

        return delays

    def _compute_roc_auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute ROC-AUC using trapezoidal rule."""
        fpr, tpr, _ = self._compute_roc_curve(scores, labels)
        from scipy.integrate import trapezoid
        return trapezoid(tpr, fpr)

    def _compute_pr_auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute Precision-Recall AUC."""
        precision, recall, _ = self._compute_pr_curve(scores, labels)
        from scipy.integrate import trapezoid
        return trapezoid(precision, recall)

    def _compute_roc_curve(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve (TPR vs FPR).

        Returns:
            (fpr, tpr, thresholds)
        """
        # Sort by score (descending)
        sorted_idx = np.argsort(-scores)
        sorted_labels = labels[sorted_idx]
        sorted_scores = scores[sorted_idx]

        # Compute TPR and FPR at each threshold
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)

        tpr = np.zeros(len(labels) + 1)
        fpr = np.zeros(len(labels) + 1)
        thresholds = np.zeros(len(labels) + 1)

        tp, fp = 0, 0
        for i in range(len(sorted_labels)):
            if sorted_labels[i] == 1:
                tp += 1
            else:
                fp += 1

            tpr[i + 1] = tp / n_pos if n_pos > 0 else 0
            fpr[i + 1] = fp / n_neg if n_neg > 0 else 0
            thresholds[i + 1] = sorted_scores[i]

        return fpr, tpr, thresholds

    def _compute_pr_curve(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Precision-Recall curve.

        Returns:
            (precision, recall, thresholds)
        """
        sorted_idx = np.argsort(-scores)
        sorted_labels = labels[sorted_idx]
        sorted_scores = scores[sorted_idx]

        precision = np.zeros(len(labels) + 1)
        recall = np.zeros(len(labels) + 1)
        thresholds = np.zeros(len(labels) + 1)

        tp, fp = 0, 0
        n_pos = np.sum(labels == 1)

        for i in range(len(sorted_labels)):
            if sorted_labels[i] == 1:
                tp += 1
            else:
                fp += 1

            precision[i + 1] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i + 1] = tp / n_pos if n_pos > 0 else 0
            thresholds[i + 1] = sorted_scores[i]

        return precision, recall, thresholds


class BenchmarkSuite:
    """
    Complete benchmark suite for comparing detection methods.

    Runs all baselines + PINN detector on same datasets, generates
    comparison tables and figures for paper.
    """

    def __init__(self, output_dir: str = "research/security/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def add_result(self, method_name: str, metrics: DetectionMetrics):
        """Add evaluation results for a method."""
        self.results[method_name] = metrics.to_dict()

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save all results to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_path}")

    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table (LaTeX-ready).

        Returns:
            DataFrame with methods as rows, metrics as columns
        """
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(self.results, orient="index")

        # Select key metrics for table
        key_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "fpr",
            "roc_auc",
            "mean_detection_delay",
            "mean_inference_time",
        ]

        df_table = df[key_metrics].copy()

        # Format percentages
        for col in ["accuracy", "precision", "recall", "f1", "fpr"]:
            if col in df_table.columns:
                df_table[col] = df_table[col].apply(lambda x: f"{x*100:.2f}")

        # Format delays
        if "mean_detection_delay" in df_table.columns:
            df_table["mean_detection_delay"] = df_table["mean_detection_delay"].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )

        # Format inference time
        if "mean_inference_time" in df_table.columns:
            df_table["mean_inference_time"] = df_table["mean_inference_time"].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )

        return df_table

    def export_latex_table(self, filename: str = "comparison_table.tex"):
        """Export comparison table as LaTeX."""
        df = self.generate_comparison_table()
        latex = df.to_latex(
            caption="Detection Performance Comparison",
            label="tab:comparison",
            column_format="l" + "c" * len(df.columns),
            float_format="%.2f",
        )

        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"LaTeX table exported to {output_path}")

    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "=" * 80)
        print("DETECTION PERFORMANCE COMPARISON")
        print("=" * 80)

        df = self.generate_comparison_table()
        print(df.to_string())

        # Find best method for each metric
        print("\n" + "=" * 80)
        print("BEST PERFORMERS")
        print("=" * 80)

        for metric in ["f1", "roc_auc", "mean_detection_delay"]:
            if metric in df.columns:
                # Convert back to float for comparison
                values = pd.to_numeric(df[metric], errors="coerce")
                if metric == "mean_detection_delay":
                    best_idx = values.idxmin()  # Lower is better
                else:
                    best_idx = values.idxmax()  # Higher is better

                print(f"{metric.upper()}: {best_idx} ({df.loc[best_idx, metric]})")


class StatisticalSignificanceTest:
    """
    Statistical significance testing for detection results.

    Uses McNemar's test for comparing binary classifiers.
    """

    @staticmethod
    def mcnemar_test(
        predictions_a: np.ndarray,
        predictions_b: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Tuple[float, bool]:
        """
        McNemar's test for comparing two detectors.

        H0: The two detectors have the same error rate.

        Args:
            predictions_a: Predictions from detector A
            predictions_b: Predictions from detector B
            ground_truth: True labels

        Returns:
            (p_value, is_significant)
        """
        # Build contingency table
        # n01: A correct, B incorrect
        # n10: A incorrect, B correct
        correct_a = predictions_a == ground_truth
        correct_b = predictions_b == ground_truth

        n01 = np.sum(correct_a & ~correct_b)
        n10 = np.sum(~correct_a & correct_b)

        # McNemar's test statistic (with continuity correction)
        if (n01 + n10) == 0:
            return 1.0, False  # No disagreement

        chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)

        # p-value from chi-squared distribution (df=1)
        from scipy.stats import chi2

        p_value = 1 - chi2.cdf(chi2_stat, df=1)

        is_significant = p_value < 0.05

        return p_value, is_significant


def create_attack_scenario_report(
    results_by_scenario: Dict[str, Dict[str, DetectionMetrics]],
    output_dir: str = "research/security/results",
):
    """
    Create detailed report for different attack scenarios.

    Args:
        results_by_scenario: {scenario_name: {method_name: metrics}}
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create summary table
    summary_data = []
    for scenario, methods in results_by_scenario.items():
        for method, metrics in methods.items():
            summary_data.append(
                {
                    "Scenario": scenario,
                    "Method": method,
                    "F1": metrics.f1,
                    "Precision": metrics.precision,
                    "Recall": metrics.recall,
                    "FPR": metrics.fpr,
                    "Detection Delay (s)": metrics.mean_detection_delay,
                }
            )

    df = pd.DataFrame(summary_data)

    # Save as CSV
    csv_path = output_dir / "scenario_breakdown.csv"
    df.to_csv(csv_path, index=False)
    print(f"Scenario breakdown saved to {csv_path}")

    # Pivot table for each metric
    for metric in ["F1", "Precision", "Recall"]:
        pivot = df.pivot(index="Scenario", columns="Method", values=metric)
        print(f"\n{metric} by Scenario:")
        print(pivot.to_string())
