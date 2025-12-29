"""
Cross-Dataset Transfer Evaluation

Evaluates model generalization across different:
1. UAV platforms
2. Flight regimes
3. Environmental conditions
4. Sensor configurations

Key insight: True robustness requires good transfer performance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class TransferResult:
    """Container for transfer evaluation results."""
    source_domain: str
    target_domain: str
    n_train: int
    n_test: int

    # Metrics before adaptation
    recall_1pct_fpr: float
    recall_5pct_fpr: float
    auroc: float
    auprc: float

    # Metrics after optional fine-tuning
    adapted_recall_1pct_fpr: Optional[float] = None
    adapted_recall_5pct_fpr: Optional[float] = None
    adapted_auroc: Optional[float] = None

    # Domain shift metrics
    feature_shift: float = 0.0  # MMD between domains
    label_shift: float = 0.0   # KL divergence of attack distributions


@dataclass
class DomainInfo:
    """Information about a data domain."""
    name: str
    description: str
    platform: str  # e.g., "quadrotor", "fixed_wing"
    flight_regime: str  # e.g., "hover", "aggressive", "outdoor"
    sensor_config: str  # e.g., "px4", "ardupilot", "custom"
    sample_rate_hz: float
    n_sequences: int
    data_path: Optional[str] = None


class TransferEvaluator:
    """
    Evaluate cross-domain transfer performance.

    Tests model generalization by:
    1. Training on source domain
    2. Evaluating on target domain (zero-shot)
    3. Optionally fine-tuning on target domain
    """

    def __init__(self, feature_dim: int = 100):
        self.feature_dim = feature_dim
        self.results: List[TransferResult] = []

        # Define standard evaluation domains
        self.domains = self._define_domains()

    def _define_domains(self) -> Dict[str, DomainInfo]:
        """Define standard evaluation domains."""
        return {
            'euroc_easy': DomainInfo(
                name='euroc_easy',
                description='EuRoC MAV easy sequences',
                platform='quadrotor',
                flight_regime='slow_indoor',
                sensor_config='vi_sensor',
                sample_rate_hz=200,
                n_sequences=4
            ),
            'euroc_medium': DomainInfo(
                name='euroc_medium',
                description='EuRoC MAV medium sequences',
                platform='quadrotor',
                flight_regime='medium_indoor',
                sensor_config='vi_sensor',
                sample_rate_hz=200,
                n_sequences=4
            ),
            'euroc_difficult': DomainInfo(
                name='euroc_difficult',
                description='EuRoC MAV difficult sequences',
                platform='quadrotor',
                flight_regime='aggressive_indoor',
                sensor_config='vi_sensor',
                sample_rate_hz=200,
                n_sequences=3
            ),
            'synthetic_hover': DomainInfo(
                name='synthetic_hover',
                description='Simulated hover flight',
                platform='quadrotor',
                flight_regime='hover',
                sensor_config='simulated',
                sample_rate_hz=200,
                n_sequences=10
            ),
            'synthetic_aggressive': DomainInfo(
                name='synthetic_aggressive',
                description='Simulated aggressive maneuvers',
                platform='quadrotor',
                flight_regime='aggressive',
                sensor_config='simulated',
                sample_rate_hz=200,
                n_sequences=10
            ),
            'padre_outdoor': DomainInfo(
                name='padre_outdoor',
                description='PADRE dataset outdoor flights',
                platform='quadrotor',
                flight_regime='outdoor',
                sensor_config='px4',
                sample_rate_hz=50,
                n_sequences=20
            )
        }

    def compute_domain_shift(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        n_samples: int = 1000
    ) -> float:
        """
        Compute domain shift using Maximum Mean Discrepancy (MMD).

        Args:
            source_features: [N, D] features from source domain
            target_features: [M, D] features from target domain
            n_samples: Number of samples for MMD estimation

        Returns:
            MMD value (higher = more shift)
        """
        # Subsample for efficiency
        if len(source_features) > n_samples:
            idx = np.random.choice(len(source_features), n_samples, replace=False)
            source_features = source_features[idx]
        if len(target_features) > n_samples:
            idx = np.random.choice(len(target_features), n_samples, replace=False)
            target_features = target_features[idx]

        # Compute MMD with Gaussian kernel
        def gaussian_kernel(X, Y, sigma=1.0):
            XX = np.sum(X ** 2, axis=1, keepdims=True)
            YY = np.sum(Y ** 2, axis=1, keepdims=True)
            XY = X @ Y.T
            distances = XX + YY.T - 2 * XY
            return np.exp(-distances / (2 * sigma ** 2))

        # Estimate sigma using median heuristic
        combined = np.vstack([source_features, target_features])
        dists = np.sqrt(np.sum((combined[:100, None] - combined[None, :100]) ** 2, axis=2))
        sigma = np.median(dists[dists > 0])

        K_ss = gaussian_kernel(source_features, source_features, sigma)
        K_tt = gaussian_kernel(target_features, target_features, sigma)
        K_st = gaussian_kernel(source_features, target_features, sigma)

        n = len(source_features)
        m = len(target_features)

        mmd = (
            np.sum(K_ss) / (n * n) +
            np.sum(K_tt) / (m * m) -
            2 * np.sum(K_st) / (n * m)
        )

        return float(max(0, mmd))

    def evaluate_transfer(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        source_data: Dict[str, np.ndarray],
        target_data: Dict[str, np.ndarray],
        source_name: str,
        target_name: str
    ) -> TransferResult:
        """
        Evaluate transfer performance.

        Args:
            model_fn: Function that takes features and returns scores
            source_data: Dict with 'features', 'labels' from source
            target_data: Dict with 'features', 'labels' from target
            source_name: Name of source domain
            target_name: Name of target domain

        Returns:
            TransferResult with metrics
        """
        target_features = target_data['features']
        target_labels = target_data['labels']

        # Get predictions on target domain
        scores = model_fn(target_features)

        # Compute metrics
        normal_mask = target_labels == 0
        attack_mask = target_labels == 1

        # AUROC and AUPRC
        if len(np.unique(target_labels)) > 1:
            auroc = roc_auc_score(target_labels, scores)
            auprc = average_precision_score(target_labels, scores)
        else:
            auroc = 0.0
            auprc = 0.0

        # Recall at FPR thresholds
        normal_scores = scores[normal_mask]
        attack_scores = scores[attack_mask]

        if len(normal_scores) > 0 and len(attack_scores) > 0:
            # 1% FPR
            threshold_1pct = np.percentile(normal_scores, 99)
            recall_1pct = np.mean(attack_scores > threshold_1pct)

            # 5% FPR
            threshold_5pct = np.percentile(normal_scores, 95)
            recall_5pct = np.mean(attack_scores > threshold_5pct)
        else:
            recall_1pct = 0.0
            recall_5pct = 0.0

        # Compute domain shift
        feature_shift = self.compute_domain_shift(
            source_data['features'],
            target_features
        )

        result = TransferResult(
            source_domain=source_name,
            target_domain=target_name,
            n_train=len(source_data['features']),
            n_test=len(target_features),
            recall_1pct_fpr=recall_1pct,
            recall_5pct_fpr=recall_5pct,
            auroc=auroc,
            auprc=auprc,
            feature_shift=feature_shift
        )

        self.results.append(result)
        return result

    def run_transfer_matrix(
        self,
        model_train_fn: Callable[[Dict], Callable],
        domain_data: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, TransferResult]]:
        """
        Run full transfer evaluation matrix.

        Trains on each domain and evaluates on all others.

        Args:
            model_train_fn: Function that trains model on data dict, returns scorer
            domain_data: Dict mapping domain name to data dict

        Returns:
            Matrix of transfer results
        """
        results_matrix = {}

        for source_name, source_data in domain_data.items():
            print(f"\nTraining on {source_name}...")
            model_fn = model_train_fn(source_data)

            results_matrix[source_name] = {}

            for target_name, target_data in domain_data.items():
                if source_name == target_name:
                    continue

                result = self.evaluate_transfer(
                    model_fn, source_data, target_data, source_name, target_name
                )
                results_matrix[source_name][target_name] = result

                print(f"  -> {target_name}: AUROC={result.auroc:.3f}, "
                      f"R@1%FPR={result.recall_1pct_fpr:.3f}, "
                      f"shift={result.feature_shift:.4f}")

        return results_matrix

    def summarize_results(self) -> Dict:
        """Summarize all transfer results."""
        if not self.results:
            return {}

        # Aggregate by target domain
        by_target = {}
        for r in self.results:
            if r.target_domain not in by_target:
                by_target[r.target_domain] = []
            by_target[r.target_domain].append(r)

        summary = {
            'overall': {
                'mean_auroc': float(np.mean([r.auroc for r in self.results])),
                'mean_recall_1pct': float(np.mean([r.recall_1pct_fpr for r in self.results])),
                'mean_recall_5pct': float(np.mean([r.recall_5pct_fpr for r in self.results])),
                'mean_feature_shift': float(np.mean([r.feature_shift for r in self.results])),
            },
            'by_target': {}
        }

        for target, results in by_target.items():
            summary['by_target'][target] = {
                'mean_auroc': float(np.mean([r.auroc for r in results])),
                'mean_recall_1pct': float(np.mean([r.recall_1pct_fpr for r in results])),
                'n_sources': len(results)
            }

        return summary

    def save_results(self, path: str):
        """Save results to JSON."""
        data = {
            'results': [
                {
                    'source': r.source_domain,
                    'target': r.target_domain,
                    'n_train': r.n_train,
                    'n_test': r.n_test,
                    'auroc': r.auroc,
                    'auprc': r.auprc,
                    'recall_1pct_fpr': r.recall_1pct_fpr,
                    'recall_5pct_fpr': r.recall_5pct_fpr,
                    'feature_shift': r.feature_shift
                }
                for r in self.results
            ],
            'summary': self.summarize_results()
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class DomainAdaptation:
    """
    Domain adaptation techniques for improving transfer.

    Implements:
    1. Feature normalization
    2. Domain adversarial training (simple version)
    3. Few-shot fine-tuning
    """

    def __init__(self):
        self.source_mean = None
        self.source_std = None

    def fit_normalization(self, source_features: np.ndarray):
        """Fit normalization on source domain."""
        self.source_mean = np.mean(source_features, axis=0)
        self.source_std = np.std(source_features, axis=0) + 1e-8

    def normalize_target(self, target_features: np.ndarray) -> np.ndarray:
        """Normalize target features to match source distribution."""
        if self.source_mean is None:
            return target_features

        # Z-normalize using source statistics
        target_mean = np.mean(target_features, axis=0)
        target_std = np.std(target_features, axis=0) + 1e-8

        normalized = (target_features - target_mean) / target_std
        aligned = normalized * self.source_std + self.source_mean

        return aligned

    def coral_alignment(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray
    ) -> np.ndarray:
        """
        CORAL (CORrelation ALignment) for domain adaptation.

        Aligns second-order statistics (covariance) of target to source.
        """
        # Source covariance
        source_centered = source_features - np.mean(source_features, axis=0)
        C_s = np.cov(source_centered.T) + np.eye(source_features.shape[1]) * 1e-6

        # Target covariance
        target_centered = target_features - np.mean(target_features, axis=0)
        C_t = np.cov(target_centered.T) + np.eye(target_features.shape[1]) * 1e-6

        # Compute transformation
        # target_aligned = target_centered @ C_t^{-1/2} @ C_s^{1/2}

        # SVD for stable inverse square root
        U_t, S_t, _ = np.linalg.svd(C_t)
        C_t_inv_sqrt = U_t @ np.diag(1.0 / np.sqrt(S_t + 1e-6)) @ U_t.T

        U_s, S_s, _ = np.linalg.svd(C_s)
        C_s_sqrt = U_s @ np.diag(np.sqrt(S_s + 1e-6)) @ U_s.T

        # Transform
        aligned = target_centered @ C_t_inv_sqrt @ C_s_sqrt
        aligned += np.mean(source_features, axis=0)

        return aligned


def generate_flight_regime_split(
    data: np.ndarray,
    labels: np.ndarray,
    n_regimes: int = 3
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split data by flight regime using motion magnitude.

    Creates domains based on flight aggressiveness for transfer testing.

    Args:
        data: [N, D] feature data
        labels: [N] labels
        n_regimes: Number of regimes to create

    Returns:
        List of (features, labels) tuples for each regime
    """
    # Estimate flight aggressiveness from feature variance
    # (aggressive flight = high variance in velocity/acceleration features)
    window_size = 50
    n_windows = len(data) // window_size

    aggressiveness = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_var = np.var(data[start:end], axis=0)
        aggressiveness.append(np.mean(window_var))

    aggressiveness = np.array(aggressiveness)

    # Assign windows to regimes
    percentiles = [100 * i / n_regimes for i in range(1, n_regimes)]
    thresholds = np.percentile(aggressiveness, percentiles)

    regimes = []
    for regime_idx in range(n_regimes):
        if regime_idx == 0:
            mask = aggressiveness < thresholds[0]
        elif regime_idx == n_regimes - 1:
            mask = aggressiveness >= thresholds[-1]
        else:
            mask = (aggressiveness >= thresholds[regime_idx-1]) & \
                   (aggressiveness < thresholds[regime_idx])

        # Collect windows
        regime_data = []
        regime_labels = []
        for i, in_regime in enumerate(mask):
            if in_regime:
                start = i * window_size
                end = start + window_size
                regime_data.append(data[start:end])
                regime_labels.append(labels[start:end])

        if regime_data:
            regimes.append((
                np.vstack(regime_data),
                np.concatenate(regime_labels)
            ))

    return regimes


if __name__ == "__main__":
    # Test transfer evaluation
    np.random.seed(42)

    n = 1000
    d = 50

    # Create mock domains with different distributions
    source_data = {
        'features': np.random.randn(n, d),
        'labels': np.random.randint(0, 2, n)
    }

    # Target with shift
    target_data = {
        'features': np.random.randn(n, d) * 1.5 + 0.5,  # Different mean/var
        'labels': np.random.randint(0, 2, n)
    }

    # Simple scorer
    def mock_scorer(features):
        return np.random.rand(len(features))

    # Evaluate transfer
    evaluator = TransferEvaluator(feature_dim=d)
    result = evaluator.evaluate_transfer(
        mock_scorer, source_data, target_data, 'source', 'target'
    )

    print(f"Transfer result:")
    print(f"  AUROC: {result.auroc:.3f}")
    print(f"  Recall@1%FPR: {result.recall_1pct_fpr:.3f}")
    print(f"  Feature shift (MMD): {result.feature_shift:.4f}")

    # Test domain adaptation
    adapter = DomainAdaptation()
    adapter.fit_normalization(source_data['features'])
    aligned_target = adapter.normalize_target(target_data['features'])

    print(f"\nAfter normalization:")
    print(f"  Source mean: {np.mean(source_data['features']):.3f}")
    print(f"  Target mean (original): {np.mean(target_data['features']):.3f}")
    print(f"  Target mean (aligned): {np.mean(aligned_target):.3f}")
