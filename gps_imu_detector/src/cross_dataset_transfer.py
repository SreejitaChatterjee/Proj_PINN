"""
Cross-Dataset Transfer Evaluation for GPS-IMU Detector.

Evaluates detector generalization across different datasets to prevent
dataset-specific overfitting and validate transferability claims.

Supported datasets:
- Synthetic (generated attacks on simulated trajectories)
- ALFA (CMU real UAV faults)
- EuRoC (ETH Zurich MAV)

Key insight: A detector that only works on one dataset has no scientific value.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class TransferResult:
    """Result of cross-dataset transfer evaluation."""
    source_dataset: str
    target_dataset: str
    source_auroc: float
    target_auroc: float
    transfer_ratio: float  # target/source
    domain_shift: float  # MMD or similar

    @property
    def transfers_well(self) -> bool:
        """Transfer is successful if ratio > 0.8"""
        return self.transfer_ratio > 0.8

    def __str__(self):
        status = "✓" if self.transfers_well else "✗"
        return (
            f"{self.source_dataset} → {self.target_dataset}: "
            f"{self.source_auroc:.3f} → {self.target_auroc:.3f} "
            f"(ratio={self.transfer_ratio:.2f}) {status}"
        )


def compute_mmd(
    X_source: np.ndarray,
    X_target: np.ndarray,
    kernel: str = 'rbf',
    gamma: float = None
) -> float:
    """
    Compute Maximum Mean Discrepancy between datasets.

    MMD measures the distance between feature distributions.
    Higher MMD = larger domain shift = harder transfer.

    Args:
        X_source: Source dataset features (N, D)
        X_target: Target dataset features (M, D)
        kernel: 'rbf' or 'linear'
        gamma: RBF kernel width (auto if None)

    Returns:
        MMD value (0 = identical distributions)
    """
    if gamma is None:
        # Median heuristic
        combined = np.vstack([X_source[:1000], X_target[:1000]])
        dists = np.linalg.norm(
            combined[:, None, :] - combined[None, :, :],
            axis=2
        )
        gamma = 1.0 / (2 * np.median(dists) ** 2 + 1e-8)

    def rbf_kernel(X, Y):
        dists = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * dists)

    # Subsample for efficiency
    n = min(500, len(X_source))
    m = min(500, len(X_target))
    idx_s = np.random.choice(len(X_source), n, replace=False)
    idx_t = np.random.choice(len(X_target), m, replace=False)

    Xs = X_source[idx_s]
    Xt = X_target[idx_t]

    K_ss = rbf_kernel(Xs, Xs)
    K_tt = rbf_kernel(Xt, Xt)
    K_st = rbf_kernel(Xs, Xt)

    mmd = (
        np.mean(K_ss) +
        np.mean(K_tt) -
        2 * np.mean(K_st)
    )

    return max(0, mmd)  # Numerical stability


def evaluate_transfer(
    model,
    source_data: Tuple[np.ndarray, np.ndarray],
    target_data: Tuple[np.ndarray, np.ndarray],
    source_name: str,
    target_name: str,
) -> TransferResult:
    """
    Evaluate how well a model trained on source transfers to target.

    Args:
        model: Trained detector with predict() method
        source_data: (X_source, y_source) from training dataset
        target_data: (X_target, y_target) from new dataset
        source_name: Name of source dataset
        target_name: Name of target dataset

    Returns:
        TransferResult with metrics
    """
    from sklearn.metrics import roc_auc_score

    X_source, y_source = source_data
    X_target, y_target = target_data

    # Evaluate on source (should be good)
    y_pred_source = model.predict(X_source)
    source_auroc = roc_auc_score(y_source, y_pred_source)

    # Evaluate on target (test transfer)
    y_pred_target = model.predict(X_target)

    # Handle case where target has only one class
    if len(np.unique(y_target)) < 2:
        target_auroc = 0.5  # Random baseline
    else:
        target_auroc = roc_auc_score(y_target, y_pred_target)

    # Compute domain shift
    domain_shift = compute_mmd(X_source, X_target)

    # Transfer ratio
    transfer_ratio = target_auroc / source_auroc if source_auroc > 0 else 0

    return TransferResult(
        source_dataset=source_name,
        target_dataset=target_name,
        source_auroc=source_auroc,
        target_auroc=target_auroc,
        transfer_ratio=transfer_ratio,
        domain_shift=domain_shift
    )


def cross_dataset_matrix(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    train_fn,
    seed: int = 42
) -> Dict[str, Dict[str, TransferResult]]:
    """
    Compute full transfer matrix across all dataset pairs.

    Args:
        datasets: Dict of {name: (X, y)} for each dataset
        train_fn: Function that takes (X, y) and returns trained model
        seed: Random seed

    Returns:
        Nested dict of TransferResults: results[source][target]
    """
    np.random.seed(seed)
    results = {}

    for source_name, (X_source, y_source) in datasets.items():
        results[source_name] = {}

        # Train on source
        model = train_fn(X_source, y_source)

        for target_name, (X_target, y_target) in datasets.items():
            result = evaluate_transfer(
                model,
                (X_source, y_source),
                (X_target, y_target),
                source_name,
                target_name
            )
            results[source_name][target_name] = result

    return results


def format_transfer_matrix(results: Dict[str, Dict[str, TransferResult]]) -> str:
    """Format transfer matrix as markdown table."""
    datasets = list(results.keys())

    lines = []
    header = "| Train \\ Test | " + " | ".join(datasets) + " |"
    separator = "|" + "|".join(["---"] * (len(datasets) + 1)) + "|"

    lines.append(header)
    lines.append(separator)

    for source in datasets:
        row = f"| {source} |"
        for target in datasets:
            r = results[source][target]
            cell = f" {r.target_auroc:.2f} "
            if source == target:
                cell = f" **{r.target_auroc:.2f}** "
            elif r.transfers_well:
                cell = f" {r.target_auroc:.2f}✓ "
            else:
                cell = f" {r.target_auroc:.2f}✗ "
            row += cell + "|"
        lines.append(row)

    return '\n'.join(lines)


def domain_adaptation_coral(
    X_source: np.ndarray,
    X_target: np.ndarray,
    lambda_coral: float = 1.0
) -> np.ndarray:
    """
    CORAL (Correlation Alignment) domain adaptation.

    Aligns source covariance to target covariance.

    Args:
        X_source: Source features (N, D)
        X_target: Target features (M, D)
        lambda_coral: Regularization weight

    Returns:
        Aligned source features
    """
    # Center
    X_s = X_source - X_source.mean(axis=0)
    X_t = X_target - X_target.mean(axis=0)

    # Covariance
    C_s = np.cov(X_s.T) + np.eye(X_s.shape[1]) * 1e-6
    C_t = np.cov(X_t.T) + np.eye(X_t.shape[1]) * 1e-6

    # Whiten source, color with target
    # A_s = C_s^(-1/2), A_t = C_t^(1/2)
    U_s, S_s, _ = np.linalg.svd(C_s)
    A_s = U_s @ np.diag(1.0 / np.sqrt(S_s + 1e-6)) @ U_s.T

    U_t, S_t, _ = np.linalg.svd(C_t)
    A_t = U_t @ np.diag(np.sqrt(S_t)) @ U_t.T

    # Transform
    X_aligned = X_s @ A_s @ A_t

    # Interpolate based on lambda
    X_aligned = lambda_coral * X_aligned + (1 - lambda_coral) * X_s

    # Restore mean
    X_aligned = X_aligned + X_target.mean(axis=0)

    return X_aligned


class TransferEvaluator:
    """
    High-level API for cross-dataset transfer evaluation.

    Usage:
        evaluator = TransferEvaluator()
        evaluator.register_dataset('synthetic', X_syn, y_syn)
        evaluator.register_dataset('alfa', X_alfa, y_alfa)
        results = evaluator.evaluate_all(train_fn)
        print(evaluator.summary())
    """

    def __init__(self):
        self.datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.results: Optional[Dict] = None

    def register_dataset(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray
    ):
        """Register a dataset for transfer evaluation."""
        self.datasets[name] = (X, y)

    def evaluate_all(self, train_fn, seed: int = 42) -> Dict:
        """Evaluate all pairwise transfers."""
        self.results = cross_dataset_matrix(self.datasets, train_fn, seed)
        return self.results

    def summary(self) -> str:
        """Generate summary report."""
        if self.results is None:
            return "No results yet. Call evaluate_all() first."

        lines = ["# Cross-Dataset Transfer Summary\n"]

        # Matrix
        lines.append("## Transfer Matrix (AUROC)\n")
        lines.append(format_transfer_matrix(self.results))
        lines.append("")

        # Domain shifts
        lines.append("\n## Domain Shifts (MMD)\n")
        for source in self.results:
            for target in self.results[source]:
                if source != target:
                    r = self.results[source][target]
                    lines.append(
                        f"- {source} → {target}: MMD={r.domain_shift:.4f}"
                    )

        # Verdict
        lines.append("\n## Verdict\n")
        all_transfers = [
            r for s in self.results.values()
            for t, r in s.items()
            if r.source_dataset != r.target_dataset
        ]

        good = sum(1 for r in all_transfers if r.transfers_well)
        total = len(all_transfers)

        if good == total:
            lines.append("✓ ALL transfers successful (ratio > 0.8)")
        elif good > total / 2:
            lines.append(f"◐ PARTIAL transfer: {good}/{total} successful")
        else:
            lines.append(f"✗ POOR transfer: {good}/{total} successful")

        return '\n'.join(lines)


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)

    print("=" * 60)
    print("Cross-Dataset Transfer Demo")
    print("=" * 60)

    # Create synthetic datasets with different distributions
    def make_dataset(n, shift=0):
        X = np.random.randn(n, 10) + shift
        y = (X[:, 0] + X[:, 1] > shift).astype(int)
        return X, y

    evaluator = TransferEvaluator()
    evaluator.register_dataset('synthetic', *make_dataset(1000, 0))
    evaluator.register_dataset('shifted', *make_dataset(1000, 2))
    evaluator.register_dataset('very_shifted', *make_dataset(1000, 5))

    # Simple logistic regression trainer
    def train_fn(X, y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        return model

    evaluator.evaluate_all(train_fn)
    print(evaluator.summary())

    print("\n✓ Cross-dataset transfer module ready")
