"""
Temporal Surprise Signal for Intermittent Attack Detection.

Computes KL divergence between rolling residual windows to detect
distribution shifts that indicate attacks.

This addresses the weakness in detecting intermittent attacks (AUROC 0.666).
"""

import numpy as np
from typing import Tuple, Optional
from scipy import stats


class TemporalSurpriseDetector:
    """
    Detects anomalies by measuring temporal surprise in residual distributions.

    Key insight: Intermittent attacks cause sudden distribution shifts that
    may not show up as large individual residuals, but appear as KL divergence
    between adjacent time windows.
    """

    def __init__(
        self,
        window_size: int = 50,  # ~0.25s at 200Hz
        lag: int = 25,  # Compare windows separated by this
        n_bins: int = 20,
        eps: float = 1e-10
    ):
        self.window_size = window_size
        self.lag = lag
        self.n_bins = n_bins
        self.eps = eps
        self.residual_buffer = []

    def reset(self):
        """Reset the residual buffer."""
        self.residual_buffer = []

    def update(self, residual: np.ndarray) -> Optional[float]:
        """
        Update with new residual and compute surprise if enough data.

        Args:
            residual: Current residual vector (any dimension)

        Returns:
            Surprise score if enough data, None otherwise
        """
        # Store residual magnitude
        magnitude = np.linalg.norm(residual)
        self.residual_buffer.append(magnitude)

        # Need enough data for two windows
        required = self.window_size + self.lag + self.window_size
        if len(self.residual_buffer) < required:
            return None

        # Get current and lagged windows
        current_window = np.array(self.residual_buffer[-self.window_size:])
        lagged_start = -(self.window_size + self.lag)
        lagged_end = -self.lag
        lagged_window = np.array(self.residual_buffer[lagged_start:lagged_end])

        # Compute KL divergence
        surprise = self._kl_divergence(lagged_window, current_window)

        # Trim buffer to prevent memory growth
        if len(self.residual_buffer) > required * 2:
            self.residual_buffer = self.residual_buffer[-required:]

        return surprise

    def _kl_divergence(self, p_samples: np.ndarray, q_samples: np.ndarray) -> float:
        """
        Compute KL(P || Q) using histogram approximation.

        Args:
            p_samples: Samples from distribution P (reference)
            q_samples: Samples from distribution Q (current)

        Returns:
            KL divergence estimate
        """
        # Determine bin edges from combined data
        all_samples = np.concatenate([p_samples, q_samples])
        bin_edges = np.linspace(
            all_samples.min() - self.eps,
            all_samples.max() + self.eps,
            self.n_bins + 1
        )

        # Compute histograms
        p_hist, _ = np.histogram(p_samples, bins=bin_edges, density=True)
        q_hist, _ = np.histogram(q_samples, bins=bin_edges, density=True)

        # Add epsilon to avoid division by zero
        p_hist = p_hist + self.eps
        q_hist = q_hist + self.eps

        # Normalize
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()

        # KL divergence
        kl = np.sum(p_hist * np.log(p_hist / q_hist))

        return float(kl)

    def compute_batch(self, residuals: np.ndarray) -> np.ndarray:
        """
        Compute surprise scores for a batch of residuals.

        Args:
            residuals: Array of shape (T, D) or (T,)

        Returns:
            Array of surprise scores, NaN for initial samples
        """
        self.reset()

        if residuals.ndim == 1:
            residuals = residuals.reshape(-1, 1)

        T = len(residuals)
        surprises = np.full(T, np.nan)

        for t in range(T):
            score = self.update(residuals[t])
            if score is not None:
                surprises[t] = score

        return surprises


class HybridAnomalyScorer:
    """
    Combines residual magnitude with temporal surprise for improved detection.

    score = alpha * normalized_residual + (1 - alpha) * normalized_surprise
    """

    def __init__(
        self,
        alpha: float = 0.5,
        window_size: int = 50,
        lag: int = 25
    ):
        self.alpha = alpha
        self.surprise_detector = TemporalSurpriseDetector(
            window_size=window_size,
            lag=lag
        )

    def score_batch(
        self,
        residuals: np.ndarray,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute hybrid anomaly scores.

        Args:
            residuals: Array of residuals (T, D) or (T,)
            normalize: Whether to normalize components before combining

        Returns:
            Tuple of (hybrid_scores, residual_scores, surprise_scores)
        """
        if residuals.ndim == 1:
            residuals = residuals.reshape(-1, 1)

        # Residual magnitudes
        residual_scores = np.linalg.norm(residuals, axis=1)

        # Temporal surprise
        surprise_scores = self.surprise_detector.compute_batch(residuals)

        if normalize:
            # Normalize to [0, 1] using percentiles (robust to outliers)
            r_low, r_high = np.nanpercentile(residual_scores, [5, 95])
            s_low, s_high = np.nanpercentile(surprise_scores, [5, 95])

            residual_norm = np.clip(
                (residual_scores - r_low) / (r_high - r_low + 1e-10),
                0, 1
            )
            surprise_norm = np.clip(
                (surprise_scores - s_low) / (s_high - s_low + 1e-10),
                0, 1
            )
        else:
            residual_norm = residual_scores
            surprise_norm = surprise_scores

        # Combine
        hybrid_scores = (
            self.alpha * residual_norm +
            (1 - self.alpha) * np.nan_to_num(surprise_norm, nan=0.0)
        )

        return hybrid_scores, residual_scores, surprise_scores


def evaluate_temporal_surprise(
    residuals_normal: np.ndarray,
    residuals_attack: np.ndarray,
    attack_type: str = "unknown"
) -> dict:
    """
    Evaluate temporal surprise on normal vs attack data.

    Args:
        residuals_normal: Residuals from normal operation
        residuals_attack: Residuals from attack scenario
        attack_type: Name of attack for logging

    Returns:
        Dictionary with AUROC scores for different methods
    """
    from sklearn.metrics import roc_auc_score

    # Create labels
    y_true = np.concatenate([
        np.zeros(len(residuals_normal)),
        np.ones(len(residuals_attack))
    ])

    all_residuals = np.concatenate([residuals_normal, residuals_attack])

    # Score with different methods
    scorer = HybridAnomalyScorer(alpha=0.5)
    hybrid, residual, surprise = scorer.score_batch(all_residuals)

    # Compute AUROCs
    results = {
        "attack_type": attack_type,
        "auroc_residual": roc_auc_score(y_true, residual),
        "auroc_surprise": roc_auc_score(
            y_true[~np.isnan(surprise)],
            surprise[~np.isnan(surprise)]
        ) if not np.all(np.isnan(surprise)) else np.nan,
        "auroc_hybrid": roc_auc_score(y_true, hybrid),
    }

    return results


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)

    # Normal residuals (low variance Gaussian)
    normal = np.random.randn(1000, 3) * 0.1

    # Intermittent attack (occasional spikes)
    attack = np.random.randn(1000, 3) * 0.1
    spike_indices = np.random.choice(1000, size=50, replace=False)
    attack[spike_indices] *= 10  # Intermittent spikes

    results = evaluate_temporal_surprise(normal, attack, "intermittent")

    print("Temporal Surprise Evaluation")
    print("=" * 40)
    print(f"Attack type: {results['attack_type']}")
    print(f"AUROC (residual only):    {results['auroc_residual']:.3f}")
    print(f"AUROC (surprise only):    {results['auroc_surprise']:.3f}")
    print(f"AUROC (hybrid):           {results['auroc_hybrid']:.3f}")
