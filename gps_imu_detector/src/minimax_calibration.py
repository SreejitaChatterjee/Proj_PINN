"""
Minimax Calibration for Worst-Case Recall Optimization

Optimizes detection thresholds and fusion weights to maximize
WORST-CASE recall across all attack types, subject to FPR constraint.

Key insight: Standard calibration maximizes average performance, but
minimax calibration ensures no attack type has catastrophically low recall.

Usage:
    calibrator = MinimaxCalibrator(target_fpr=0.05)
    optimal_weights = calibrator.calibrate(scores_dict, labels_dict)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.metrics import roc_curve, precision_recall_curve
import warnings


@dataclass
class CalibrationResult:
    """Result of minimax calibration."""
    weights: np.ndarray
    threshold: float
    worst_case_recall: float
    worst_case_attack: str
    per_attack_recall: Dict[str, float]
    achieved_fpr: float
    optimization_success: bool


class MinimaxCalibrator:
    """
    Minimax threshold and weight calibration.

    Maximizes: min_{attack} Recall(attack)
    Subject to: FPR <= target_fpr
    """

    def __init__(
        self,
        target_fpr: float = 0.05,
        n_components: int = 4,
        method: str = 'differential_evolution'
    ):
        """
        Initialize calibrator.

        Args:
            target_fpr: Maximum allowed false positive rate
            n_components: Number of score components (PINN, EKF, ML, temporal)
            method: Optimization method ('grid', 'scipy', 'differential_evolution')
        """
        self.target_fpr = target_fpr
        self.n_components = n_components
        self.method = method

    def calibrate(
        self,
        component_scores: Dict[str, np.ndarray],
        attack_labels: Dict[str, np.ndarray],
        normal_scores: Dict[str, np.ndarray]
    ) -> CalibrationResult:
        """
        Calibrate weights and threshold for worst-case recall.

        Args:
            component_scores: Dict mapping attack_type -> [n_samples, n_components]
            attack_labels: Dict mapping attack_type -> binary labels
            normal_scores: Dict mapping component -> scores on normal data

        Returns:
            CalibrationResult with optimal weights and metrics
        """
        # Stack normal scores for threshold setting
        normal_combined = self._stack_normal_scores(normal_scores)

        if self.method == 'grid':
            return self._grid_search(component_scores, attack_labels, normal_combined)
        elif self.method == 'differential_evolution':
            return self._differential_evolution(component_scores, attack_labels, normal_combined)
        else:
            return self._scipy_optimize(component_scores, attack_labels, normal_combined)

    def _stack_normal_scores(self, normal_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """Stack normal scores into [n_samples, n_components] array."""
        components = ['pinn', 'ekf', 'ml', 'temporal']
        n_samples = len(list(normal_scores.values())[0])

        stacked = np.zeros((n_samples, self.n_components))
        for i, comp in enumerate(components[:self.n_components]):
            if comp in normal_scores:
                stacked[:, i] = normal_scores[comp]

        return stacked

    def _compute_fused_score(
        self,
        component_scores: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """Compute weighted fusion of component scores."""
        # Normalize weights to sum to 1
        weights = weights / (weights.sum() + 1e-8)
        return np.dot(component_scores, weights)

    def _compute_threshold_for_fpr(
        self,
        normal_fused: np.ndarray,
        target_fpr: float
    ) -> float:
        """Compute threshold to achieve target FPR on normal data."""
        return np.percentile(normal_fused, 100 * (1 - target_fpr))

    def _compute_recalls(
        self,
        component_scores: Dict[str, np.ndarray],
        attack_labels: Dict[str, np.ndarray],
        weights: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """Compute recall for each attack type."""
        recalls = {}

        for attack_type, scores in component_scores.items():
            labels = attack_labels[attack_type]
            fused = self._compute_fused_score(scores, weights)

            # Compute recall at threshold
            attack_mask = labels == 1
            if attack_mask.sum() > 0:
                attack_scores = fused[attack_mask]
                recall = (attack_scores > threshold).mean()
                recalls[attack_type] = recall
            else:
                recalls[attack_type] = 0.0

        return recalls

    def _objective(
        self,
        weights: np.ndarray,
        component_scores: Dict[str, np.ndarray],
        attack_labels: Dict[str, np.ndarray],
        normal_combined: np.ndarray
    ) -> float:
        """
        Objective function: negative of worst-case recall.

        We minimize this, so we get maximum worst-case recall.
        """
        # Compute fused normal scores and threshold
        normal_fused = self._compute_fused_score(normal_combined, weights)
        threshold = self._compute_threshold_for_fpr(normal_fused, self.target_fpr)

        # Compute recalls
        recalls = self._compute_recalls(
            component_scores, attack_labels, weights, threshold
        )

        if not recalls:
            return 1.0  # Worst case

        # Worst-case recall (minimize negative = maximize)
        worst_recall = min(recalls.values())
        return -worst_recall

    def _grid_search(
        self,
        component_scores: Dict[str, np.ndarray],
        attack_labels: Dict[str, np.ndarray],
        normal_combined: np.ndarray
    ) -> CalibrationResult:
        """Grid search over weight combinations."""
        best_worst_recall = -1.0
        best_weights = None
        best_recalls = None

        # Grid over simplex (weights sum to 1)
        n_steps = 10
        for w1 in np.linspace(0.1, 0.7, n_steps):
            for w2 in np.linspace(0.1, 0.7 - w1, n_steps):
                for w3 in np.linspace(0.1, 0.7 - w1 - w2, n_steps):
                    w4 = 1.0 - w1 - w2 - w3
                    if w4 < 0.05:
                        continue

                    weights = np.array([w1, w2, w3, w4])

                    # Evaluate
                    obj = self._objective(
                        weights, component_scores, attack_labels, normal_combined
                    )
                    worst_recall = -obj

                    if worst_recall > best_worst_recall:
                        best_worst_recall = worst_recall
                        best_weights = weights.copy()

                        # Get detailed recalls
                        normal_fused = self._compute_fused_score(normal_combined, weights)
                        threshold = self._compute_threshold_for_fpr(normal_fused, self.target_fpr)
                        best_recalls = self._compute_recalls(
                            component_scores, attack_labels, weights, threshold
                        )

        # Compute final threshold
        normal_fused = self._compute_fused_score(normal_combined, best_weights)
        threshold = self._compute_threshold_for_fpr(normal_fused, self.target_fpr)
        achieved_fpr = (normal_fused > threshold).mean()

        # Find worst attack
        worst_attack = min(best_recalls.keys(), key=lambda k: best_recalls[k])

        return CalibrationResult(
            weights=best_weights,
            threshold=threshold,
            worst_case_recall=best_worst_recall,
            worst_case_attack=worst_attack,
            per_attack_recall=best_recalls,
            achieved_fpr=achieved_fpr,
            optimization_success=True
        )

    def _differential_evolution(
        self,
        component_scores: Dict[str, np.ndarray],
        attack_labels: Dict[str, np.ndarray],
        normal_combined: np.ndarray
    ) -> CalibrationResult:
        """Use differential evolution for global optimization."""
        bounds = [(0.05, 0.8)] * self.n_components

        def objective_wrapper(weights):
            return self._objective(
                weights, component_scores, attack_labels, normal_combined
            )

        result = differential_evolution(
            objective_wrapper,
            bounds,
            seed=42,
            maxiter=100,
            polish=True
        )

        best_weights = result.x
        best_weights = best_weights / best_weights.sum()  # Normalize

        # Compute final metrics
        normal_fused = self._compute_fused_score(normal_combined, best_weights)
        threshold = self._compute_threshold_for_fpr(normal_fused, self.target_fpr)
        recalls = self._compute_recalls(
            component_scores, attack_labels, best_weights, threshold
        )

        worst_recall = min(recalls.values()) if recalls else 0.0
        worst_attack = min(recalls.keys(), key=lambda k: recalls[k]) if recalls else "none"
        achieved_fpr = (normal_fused > threshold).mean()

        return CalibrationResult(
            weights=best_weights,
            threshold=threshold,
            worst_case_recall=worst_recall,
            worst_case_attack=worst_attack,
            per_attack_recall=recalls,
            achieved_fpr=achieved_fpr,
            optimization_success=result.success
        )

    def _scipy_optimize(
        self,
        component_scores: Dict[str, np.ndarray],
        attack_labels: Dict[str, np.ndarray],
        normal_combined: np.ndarray
    ) -> CalibrationResult:
        """Use scipy.optimize.minimize with constraints."""
        # Initial guess: equal weights
        x0 = np.ones(self.n_components) / self.n_components

        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}

        # Bounds: each weight between 0.05 and 0.8
        bounds = [(0.05, 0.8)] * self.n_components

        def objective_wrapper(weights):
            return self._objective(
                weights, component_scores, attack_labels, normal_combined
            )

        result = minimize(
            objective_wrapper,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200}
        )

        best_weights = result.x
        best_weights = best_weights / best_weights.sum()

        # Compute final metrics
        normal_fused = self._compute_fused_score(normal_combined, best_weights)
        threshold = self._compute_threshold_for_fpr(normal_fused, self.target_fpr)
        recalls = self._compute_recalls(
            component_scores, attack_labels, best_weights, threshold
        )

        worst_recall = min(recalls.values()) if recalls else 0.0
        worst_attack = min(recalls.keys(), key=lambda k: recalls[k]) if recalls else "none"
        achieved_fpr = (normal_fused > threshold).mean()

        return CalibrationResult(
            weights=best_weights,
            threshold=threshold,
            worst_case_recall=worst_recall,
            worst_case_attack=worst_attack,
            per_attack_recall=recalls,
            achieved_fpr=achieved_fpr,
            optimization_success=result.success
        )


def compare_calibration_methods(
    component_scores: Dict[str, np.ndarray],
    attack_labels: Dict[str, np.ndarray],
    normal_scores: Dict[str, np.ndarray],
    target_fpr: float = 0.05
) -> Dict[str, CalibrationResult]:
    """
    Compare minimax vs standard (average) calibration.

    Returns results for both methods to show improvement.
    """
    results = {}

    # Minimax calibration
    minimax = MinimaxCalibrator(target_fpr=target_fpr, method='differential_evolution')
    results['minimax'] = minimax.calibrate(component_scores, attack_labels, normal_scores)

    # Standard calibration (maximize average recall)
    standard = StandardCalibrator(target_fpr=target_fpr)
    results['standard'] = standard.calibrate(component_scores, attack_labels, normal_scores)

    # Print comparison
    print("\n" + "=" * 60)
    print("CALIBRATION COMPARISON")
    print("=" * 60)
    print(f"\n{'Method':<15} {'Worst Recall':<15} {'Worst Attack':<20}")
    print("-" * 50)
    for method, result in results.items():
        print(f"{method:<15} {result.worst_case_recall:.3f}         {result.worst_case_attack:<20}")

    print(f"\nMinimax improvement: {results['minimax'].worst_case_recall - results['standard'].worst_case_recall:.3f}")

    return results


class StandardCalibrator:
    """Standard calibration that maximizes average recall."""

    def __init__(self, target_fpr: float = 0.05, n_components: int = 4):
        self.target_fpr = target_fpr
        self.n_components = n_components

    def calibrate(
        self,
        component_scores: Dict[str, np.ndarray],
        attack_labels: Dict[str, np.ndarray],
        normal_scores: Dict[str, np.ndarray]
    ) -> CalibrationResult:
        """Calibrate for average performance."""
        # Simple grid search maximizing average recall
        best_avg_recall = -1.0
        best_weights = None

        normal_combined = self._stack_normal_scores(normal_scores)

        n_steps = 10
        for w1 in np.linspace(0.1, 0.7, n_steps):
            for w2 in np.linspace(0.1, 0.7 - w1, n_steps):
                for w3 in np.linspace(0.1, 0.7 - w1 - w2, n_steps):
                    w4 = 1.0 - w1 - w2 - w3
                    if w4 < 0.05:
                        continue

                    weights = np.array([w1, w2, w3, w4])

                    # Compute average recall
                    normal_fused = np.dot(normal_combined, weights)
                    threshold = np.percentile(normal_fused, 100 * (1 - self.target_fpr))

                    recalls = []
                    for attack_type, scores in component_scores.items():
                        labels = attack_labels[attack_type]
                        fused = np.dot(scores, weights)
                        attack_mask = labels == 1
                        if attack_mask.sum() > 0:
                            recall = (fused[attack_mask] > threshold).mean()
                            recalls.append(recall)

                    avg_recall = np.mean(recalls) if recalls else 0.0

                    if avg_recall > best_avg_recall:
                        best_avg_recall = avg_recall
                        best_weights = weights.copy()

        # Compute final metrics
        normal_fused = np.dot(normal_combined, best_weights)
        threshold = np.percentile(normal_fused, 100 * (1 - self.target_fpr))

        per_attack_recall = {}
        for attack_type, scores in component_scores.items():
            labels = attack_labels[attack_type]
            fused = np.dot(scores, weights)
            attack_mask = labels == 1
            if attack_mask.sum() > 0:
                per_attack_recall[attack_type] = (fused[attack_mask] > threshold).mean()

        worst_recall = min(per_attack_recall.values()) if per_attack_recall else 0.0
        worst_attack = min(per_attack_recall.keys(), key=lambda k: per_attack_recall[k]) if per_attack_recall else "none"
        achieved_fpr = (normal_fused > threshold).mean()

        return CalibrationResult(
            weights=best_weights,
            threshold=threshold,
            worst_case_recall=worst_recall,
            worst_case_attack=worst_attack,
            per_attack_recall=per_attack_recall,
            achieved_fpr=achieved_fpr,
            optimization_success=True
        )

    def _stack_normal_scores(self, normal_scores: Dict[str, np.ndarray]) -> np.ndarray:
        components = ['pinn', 'ekf', 'ml', 'temporal']
        n_samples = len(list(normal_scores.values())[0])
        stacked = np.zeros((n_samples, self.n_components))
        for i, comp in enumerate(components[:self.n_components]):
            if comp in normal_scores:
                stacked[:, i] = normal_scores[comp]
        return stacked


if __name__ == "__main__":
    # Demo
    np.random.seed(42)

    # Generate synthetic scores
    n = 500
    n_components = 4

    # Normal data scores
    normal_scores = {
        'pinn': np.random.randn(n) * 0.5,
        'ekf': np.random.randn(n) * 0.5,
        'ml': np.random.randn(n) * 0.5,
        'temporal': np.random.randn(n) * 0.5
    }

    # Attack scores (different attacks have different detectability)
    attack_types = ['bias', 'drift', 'noise', 'coordinated', 'intermittent']
    component_scores = {}
    attack_labels = {}

    for i, attack in enumerate(attack_types):
        n_attack = 200
        # Each attack type has different detectability per component
        scores = np.zeros((n_attack, n_components))
        scores[:, 0] = np.random.randn(n_attack) * 0.5 + 1.0 + i * 0.1  # PINN
        scores[:, 1] = np.random.randn(n_attack) * 0.5 + 0.8 - i * 0.1  # EKF
        scores[:, 2] = np.random.randn(n_attack) * 0.5 + 0.5 + i * 0.2  # ML
        scores[:, 3] = np.random.randn(n_attack) * 0.5 + 0.7           # Temporal

        component_scores[attack] = scores
        attack_labels[attack] = np.ones(n_attack)

    # Compare methods
    results = compare_calibration_methods(
        component_scores, attack_labels, normal_scores, target_fpr=0.05
    )

    print("\n\nDetailed Minimax Result:")
    print(f"  Weights: {results['minimax'].weights}")
    print(f"  Threshold: {results['minimax'].threshold:.3f}")
    print(f"  Achieved FPR: {results['minimax'].achieved_fpr:.3f}")
    print(f"  Per-attack recalls: {results['minimax'].per_attack_recall}")
