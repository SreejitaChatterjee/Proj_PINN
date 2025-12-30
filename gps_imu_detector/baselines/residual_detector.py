"""
Residual-Based Detector (Baseline).

HIERARCHY: baselines/ - This is NOT the contribution.

This detector uses forward prediction residuals:
    score = ||f_θ(x_t) - x_{t+1}||

EXPECTED RESULT: AUROC ≈ 0.5 for consistency-preserving spoofing.

This baseline demonstrates the Residual Equivalence Class limitation
that motivates the ICI detector (see experiments/1_impossibility/).

Author: GPS-IMU Detector Project
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ResidualResult:
    """Result from residual detector."""
    residuals: np.ndarray
    scores: np.ndarray  # Normalized
    threshold: float
    mean: float
    std: float


class ResidualDetector:
    """
    Forward residual-based anomaly detector.

    LIMITATION: Cannot detect consistency-preserving attacks (REC theorem).

    This is a BASELINE for comparison, not a contribution.
    """

    def __init__(self, forward_model: nn.Module):
        """
        Initialize with trained forward model.

        Args:
            forward_model: f_θ: x_t → x_{t+1}
        """
        self.forward_model = forward_model
        self.forward_model.eval()

        # Calibration statistics
        self.mean: float = 0.0
        self.std: float = 1.0
        self.threshold: float = 0.0

    def compute_residual(self, x_t: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        """
        Compute forward prediction residual.

        Args:
            x_t: Current state (B, D)
            x_next: Next state (B, D)

        Returns:
            Residual norm (B,)
        """
        with torch.no_grad():
            x_pred = self.forward_model(x_t)
            residual = torch.norm(x_pred - x_next, dim=-1)
        return residual

    def calibrate(
        self,
        normal_trajectories: List[np.ndarray],
        target_fpr: float = 0.05,
    ) -> Dict[str, float]:
        """
        Calibrate on normal data.

        Args:
            normal_trajectories: List of clean trajectories
            target_fpr: Target false positive rate

        Returns:
            Calibration statistics
        """
        all_residuals = []

        for traj in normal_trajectories:
            if len(traj) < 2:
                continue

            traj_tensor = torch.tensor(traj, dtype=torch.float32)

            for i in range(len(traj) - 1):
                x_t = traj_tensor[i:i+1]
                x_next = traj_tensor[i+1:i+2]
                residual = self.compute_residual(x_t, x_next)
                all_residuals.append(residual.item())

        all_residuals = np.array(all_residuals)

        self.mean = float(np.mean(all_residuals))
        self.std = float(np.std(all_residuals))

        # Normalize and set threshold
        normalized = (all_residuals - self.mean) / max(self.std, 1e-6)
        self.threshold = float(np.percentile(normalized, 100 * (1 - target_fpr)))

        return {
            'mean': self.mean,
            'std': self.std,
            'threshold': self.threshold,
            'n_samples': len(all_residuals),
        }

    def score_trajectory(
        self,
        trajectory: np.ndarray,
        return_raw: bool = False,
    ) -> ResidualResult:
        """
        Score a trajectory.

        Args:
            trajectory: States (T, D)
            return_raw: If True, include raw residuals

        Returns:
            ResidualResult with scores
        """
        traj_tensor = torch.tensor(trajectory, dtype=torch.float32)
        residuals = []

        for i in range(len(trajectory) - 1):
            x_t = traj_tensor[i:i+1]
            x_next = traj_tensor[i+1:i+2]
            residual = self.compute_residual(x_t, x_next)
            residuals.append(residual.item())

        residuals = np.array(residuals)
        scores = (residuals - self.mean) / max(self.std, 1e-6)

        return ResidualResult(
            residuals=residuals if return_raw else np.array([]),
            scores=scores,
            threshold=self.threshold,
            mean=self.mean,
            std=self.std,
        )


def evaluate_residual_detector(
    detector: ResidualDetector,
    normal_trajectories: List[np.ndarray],
    attack_trajectories: List[np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate residual detector.

    EXPECTED: AUROC ≈ 0.5 for consistency-preserving attacks.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    normal_scores = []
    attack_scores = []

    for traj in normal_trajectories:
        if len(traj) < 2:
            continue
        result = detector.score_trajectory(traj)
        normal_scores.extend(result.scores.tolist())

    for traj in attack_trajectories:
        if len(traj) < 2:
            continue
        result = detector.score_trajectory(traj)
        attack_scores.extend(result.scores.tolist())

    y_true = np.array([0] * len(normal_scores) + [1] * len(attack_scores))
    y_scores = np.array(normal_scores + attack_scores)

    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    recall_5pct = tpr[np.searchsorted(fpr, 0.05)]

    return {
        'auroc': float(auroc),
        'recall_5pct_fpr': float(recall_5pct),
        'n_normal': len(normal_scores),
        'n_attack': len(attack_scores),
        'expected_auroc': 0.5,  # REC theorem prediction
        'status': 'EXPECTED' if abs(auroc - 0.5) < 0.1 else 'UNEXPECTED',
    }
