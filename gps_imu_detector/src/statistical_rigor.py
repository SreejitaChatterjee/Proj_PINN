"""
Statistical Rigor Module for GPS-IMU Detector.

Provides bootstrap confidence intervals and statistical tests
required for publication-quality claims.

Key functions:
- bootstrap_auroc_ci: Confidence interval for AUROC
- bootstrap_recall_ci: Confidence interval for recall@FPR
- compare_methods: Statistical comparison (paired test)
- per_flight_variability: Flight-level ROC analysis
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


@dataclass
class ConfidenceInterval:
    """Confidence interval with point estimate."""
    point: float
    lower: float
    upper: float
    confidence: float = 0.95
    n_bootstrap: int = 1000

    def __str__(self):
        return f"{self.point:.3f} [{self.lower:.3f}, {self.upper:.3f}]"

    def to_latex(self):
        return f"${self.point:.3f}$ (${self.lower:.3f}$--${self.upper:.3f}$)"


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> ConfidenceInterval:
    """
    Compute bootstrap confidence interval for AUROC.

    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted scores/probabilities
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        ConfidenceInterval with point estimate and bounds
    """
    np.random.seed(seed)

    n = len(y_true)
    aurocs = []

    # Point estimate
    point_auroc = roc_auc_score(y_true, y_scores)

    # Bootstrap
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]

        # Skip if only one class in sample
        if len(np.unique(y_true_boot)) < 2:
            continue

        aurocs.append(roc_auc_score(y_true_boot, y_scores_boot))

    aurocs = np.array(aurocs)
    alpha = 1 - confidence
    lower = np.percentile(aurocs, 100 * alpha / 2)
    upper = np.percentile(aurocs, 100 * (1 - alpha / 2))

    return ConfidenceInterval(
        point=point_auroc,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n_bootstrap=n_bootstrap
    )


def bootstrap_recall_at_fpr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_fpr: float = 0.05,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> ConfidenceInterval:
    """
    Compute bootstrap CI for recall at fixed FPR threshold.

    This is the operationally relevant metric for anomaly detection.

    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted scores
        target_fpr: Target false positive rate (e.g., 0.05 for 5%)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed

    Returns:
        ConfidenceInterval for recall@FPR
    """
    np.random.seed(seed)

    def recall_at_fpr(y_t, y_s, fpr_target):
        fpr, tpr, thresholds = roc_curve(y_t, y_s)
        # Find threshold that gives target FPR
        idx = np.searchsorted(fpr, fpr_target)
        if idx >= len(tpr):
            idx = len(tpr) - 1
        return tpr[idx]

    n = len(y_true)
    recalls = []

    # Point estimate
    point_recall = recall_at_fpr(y_true, y_scores, target_fpr)

    # Bootstrap
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        recalls.append(recall_at_fpr(y_true_boot, y_scores_boot, target_fpr))

    recalls = np.array(recalls)
    alpha = 1 - confidence
    lower = np.percentile(recalls, 100 * alpha / 2)
    upper = np.percentile(recalls, 100 * (1 - alpha / 2))

    return ConfidenceInterval(
        point=point_recall,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n_bootstrap=n_bootstrap
    )


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two methods."""
    method_a_mean: float
    method_b_mean: float
    difference: float
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool
    test_name: str

    def __str__(self):
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (
            f"A={self.method_a_mean:.3f}, B={self.method_b_mean:.3f}, "
            f"diff={self.difference:+.3f}, p={self.p_value:.4f}{sig}, d={self.effect_size:.2f}"
        )


def compare_methods(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    paired: bool = True,
    test: str = 'auto'
) -> ComparisonResult:
    """
    Statistical comparison of two methods.

    Args:
        scores_a: Scores from method A (e.g., AUROC values across folds)
        scores_b: Scores from method B
        paired: Whether samples are paired (same folds)
        test: 'ttest', 'wilcoxon', or 'auto'

    Returns:
        ComparisonResult with p-value and effect size
    """
    from scipy import stats

    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    diff = mean_a - mean_b

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
    effect_size = diff / pooled_std if pooled_std > 0 else 0

    # Choose test
    if test == 'auto':
        # Use Wilcoxon for small samples, t-test otherwise
        test = 'wilcoxon' if len(scores_a) < 20 else 'ttest'

    if paired:
        if test == 'ttest':
            stat, p_value = stats.ttest_rel(scores_a, scores_b)
        else:
            stat, p_value = stats.wilcoxon(scores_a, scores_b)
    else:
        if test == 'ttest':
            stat, p_value = stats.ttest_ind(scores_a, scores_b)
        else:
            stat, p_value = stats.mannwhitneyu(scores_a, scores_b)

    return ComparisonResult(
        method_a_mean=mean_a,
        method_b_mean=mean_b,
        difference=diff,
        p_value=p_value,
        effect_size=effect_size,
        significant=p_value < 0.05,
        test_name=test
    )


def per_flight_roc(
    flight_ids: np.ndarray,
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, Dict]:
    """
    Compute per-flight ROC metrics to assess variability.

    Args:
        flight_ids: Flight identifier for each sample
        y_true: Ground truth labels
        y_scores: Predicted scores

    Returns:
        Dictionary with per-flight metrics and summary statistics
    """
    unique_flights = np.unique(flight_ids)
    flight_metrics = {}

    aurocs = []

    for flight in unique_flights:
        mask = flight_ids == flight
        y_t = y_true[mask]
        y_s = y_scores[mask]

        if len(np.unique(y_t)) < 2:
            # Skip flights with only one class
            continue

        auroc = roc_auc_score(y_t, y_s)
        aurocs.append(auroc)

        flight_metrics[str(flight)] = {
            'auroc': auroc,
            'n_samples': int(mask.sum()),
            'n_positive': int(y_t.sum()),
            'n_negative': int((1 - y_t).sum()),
        }

    aurocs = np.array(aurocs)

    return {
        'per_flight': flight_metrics,
        'summary': {
            'mean_auroc': float(np.mean(aurocs)),
            'std_auroc': float(np.std(aurocs)),
            'min_auroc': float(np.min(aurocs)),
            'max_auroc': float(np.max(aurocs)),
            'n_flights': len(aurocs),
        }
    }


def per_attack_metrics(
    attack_types: np.ndarray,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Compute metrics per attack type with confidence intervals.

    Args:
        attack_types: Attack type for each sample (use 'nominal' for normal)
        y_true: Ground truth labels
        y_scores: Predicted scores
        n_bootstrap: Bootstrap samples for CI
        seed: Random seed

    Returns:
        Dictionary with per-attack metrics
    """
    unique_attacks = [a for a in np.unique(attack_types) if a != 'nominal']

    results = {}

    for attack in unique_attacks:
        # Get attack samples + nominal samples
        mask_attack = attack_types == attack
        mask_nominal = attack_types == 'nominal'
        mask = mask_attack | mask_nominal

        y_t = y_true[mask]
        y_s = y_scores[mask]

        if len(np.unique(y_t)) < 2:
            continue

        # Compute CI
        auroc_ci = bootstrap_auroc_ci(y_t, y_s, n_bootstrap, seed=seed)
        recall_ci = bootstrap_recall_at_fpr(y_t, y_s, 0.05, n_bootstrap, seed=seed)

        results[attack] = {
            'auroc': auroc_ci.point,
            'auroc_ci': (auroc_ci.lower, auroc_ci.upper),
            'recall_at_5fpr': recall_ci.point,
            'recall_ci': (recall_ci.lower, recall_ci.upper),
            'n_attack': int(mask_attack.sum()),
            'n_nominal': int(mask_nominal.sum()),
        }

    return results


def format_results_table(
    results: Dict[str, Dict],
    metric: str = 'auroc'
) -> str:
    """
    Format results as publication-ready table.

    Args:
        results: Output from per_attack_metrics
        metric: 'auroc' or 'recall_at_5fpr'

    Returns:
        Markdown table string
    """
    lines = []
    lines.append(f"| Attack | {metric.upper()} | 95% CI |")
    lines.append("|--------|-------|--------|")

    for attack, metrics in sorted(results.items()):
        val = metrics[metric]
        ci = metrics[f'{metric}_ci'] if metric == 'auroc' else metrics['recall_ci']
        lines.append(f"| {attack} | {val:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")

    return '\n'.join(lines)


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)

    # Simulate predictions
    n = 1000
    y_true = np.random.binomial(1, 0.3, n)
    y_scores = y_true * 0.6 + np.random.randn(n) * 0.2 + 0.3
    y_scores = np.clip(y_scores, 0, 1)

    print("=" * 60)
    print("Statistical Rigor Demo")
    print("=" * 60)

    # Bootstrap AUROC
    auroc_ci = bootstrap_auroc_ci(y_true, y_scores)
    print(f"\nAUROC: {auroc_ci}")

    # Bootstrap Recall@5%FPR
    recall_ci = bootstrap_recall_at_fpr(y_true, y_scores, target_fpr=0.05)
    print(f"Recall@5%FPR: {recall_ci}")

    # Compare two methods
    method_a = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
    method_b = np.array([0.72, 0.75, 0.70, 0.73, 0.71])
    comparison = compare_methods(method_a, method_b)
    print(f"\nMethod comparison: {comparison}")

    print("\n✓ Statistical rigor module ready")
