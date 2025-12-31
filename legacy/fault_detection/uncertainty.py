"""
Uncertainty Quantification Module
=================================

Methods for quantifying prediction uncertainty:
- MC Dropout: Monte Carlo Dropout for epistemic uncertainty
- Temperature Scaling: Calibrated confidence scores
- OOD Detection: Detect out-of-distribution samples
- Conformal Prediction: Guaranteed coverage prediction sets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class UncertaintyOutput:
    """Container for uncertainty-aware predictions."""
    predictions: torch.Tensor      # Class predictions
    probabilities: torch.Tensor    # Class probabilities
    confidence: torch.Tensor       # Confidence scores
    uncertainty: torch.Tensor      # Uncertainty estimates
    is_ood: Optional[torch.Tensor] = None  # OOD flags


# =============================================================================
# MC Dropout
# =============================================================================

class MCDropoutWrapper(nn.Module):
    """
    Monte Carlo Dropout wrapper for uncertainty estimation.

    Performs multiple forward passes with dropout enabled at inference
    to estimate epistemic uncertainty (model uncertainty).

    Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation", 2016
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 30,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            model: Base model (should have dropout layers)
            n_samples: Number of MC samples
            dropout_rate: Dropout rate to apply
        """
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        # Add dropout after each major layer if not present
        self._ensure_dropout()

    def _ensure_dropout(self):
        """Ensure model has dropout layers."""
        # This is a simple check; for production, you'd want to
        # actually add dropout layers if missing
        has_dropout = any(
            isinstance(m, nn.Dropout) for m in self.model.modules()
        )
        if not has_dropout:
            print("Warning: Model has no dropout layers. MC Dropout may not work properly.")

    def _enable_dropout(self):
        """Enable dropout during inference."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(
        self,
        x: torch.Tensor,
        return_samples: bool = False
    ) -> UncertaintyOutput:
        """
        Forward pass with MC Dropout.

        Args:
            x: Input tensor (batch, channels, time)
            return_samples: Whether to return individual samples

        Returns:
            UncertaintyOutput with predictions and uncertainty
        """
        batch_size = x.shape[0]
        n_classes = self.model.n_classes

        # Collect MC samples
        samples = []
        self._enable_dropout()

        for _ in range(self.n_samples):
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs)

        # Stack samples: (n_samples, batch, n_classes)
        samples = torch.stack(samples, dim=0)

        # Mean prediction
        mean_probs = samples.mean(dim=0)

        # Predictions
        predictions = mean_probs.argmax(dim=-1)

        # Confidence: max probability
        confidence = mean_probs.max(dim=-1)[0]

        # Uncertainty metrics
        # 1. Predictive entropy: H[p(y|x)]
        predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)

        # 2. Expected entropy: E[H[p(y|x, w)]] (aleatoric)
        sample_entropy = -(samples * torch.log(samples + 1e-8)).sum(dim=-1)
        expected_entropy = sample_entropy.mean(dim=0)

        # 3. Mutual information: I[y; w|x] = H - E[H] (epistemic)
        mutual_info = predictive_entropy - expected_entropy

        # 4. Variance of predictions
        variance = samples.var(dim=0).mean(dim=-1)

        # Combined uncertainty (normalized)
        uncertainty = (predictive_entropy + mutual_info) / 2

        output = UncertaintyOutput(
            predictions=predictions,
            probabilities=mean_probs,
            confidence=confidence,
            uncertainty=uncertainty
        )

        if return_samples:
            output.samples = samples

        return output

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions with uncertainty-based rejection.

        Args:
            x: Input tensor
            threshold: Uncertainty threshold for rejection

        Returns:
            predictions, uncertainty, reject_mask
        """
        output = self.forward(x)
        reject_mask = output.uncertainty > threshold

        return output.predictions, output.uncertainty, reject_mask


# =============================================================================
# Temperature Scaling
# =============================================================================

class TemperatureScaler(nn.Module):
    """
    Temperature scaling for calibrated confidence scores.

    Learns a single temperature parameter to scale logits,
    improving calibration of confidence scores.

    Reference: Guo et al., "On Calibration of Modern Neural Networks", 2017
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature scaling."""
        logits = self.model(x)
        return logits / self.temperature

    def calibrate(
        self,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """
        Calibrate temperature on validation set.

        Args:
            val_loader: Validation data loader
            device: Device to use
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations

        Returns:
            Final temperature value
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()

        # Collect all logits and labels
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                logits = self.model(data)
                logits_list.append(logits.cpu())
                labels_list.append(labels)

        logits = torch.cat(logits_list, dim=0).to(device)
        labels = torch.cat(labels_list, dim=0).to(device)

        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_nll():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_nll)

        print(f"Calibrated temperature: {self.temperature.item():.4f}")
        return self.temperature.item()

    def get_calibrated_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get calibrated class probabilities.

        Returns:
            predictions, calibrated_probabilities
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1)
        return predictions, probs

    @staticmethod
    def compute_ece(
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            probs: Predicted probabilities (batch, n_classes)
            labels: True labels
            n_bins: Number of confidence bins

        Returns:
            ECE value (lower is better)
        """
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels).float()

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += torch.abs(avg_accuracy - avg_confidence) * prop_in_bin

        return ece.item()


# =============================================================================
# Out-of-Distribution Detection
# =============================================================================

class OODDetector(nn.Module):
    """
    Out-of-Distribution detection module.

    Detects samples that are different from the training distribution.
    Important for safety-critical applications.

    Methods:
    - Maximum Softmax Probability (MSP)
    - Energy-based detection
    - Mahalanobis distance
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = 'energy',  # 'msp', 'energy', 'mahalanobis'
        temperature: float = 1.0
    ):
        super().__init__()
        self.model = model
        self.method = method
        self.temperature = temperature

        # For Mahalanobis distance
        self.class_means = None
        self.precision_matrix = None

    def compute_ood_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute OOD score for each sample.

        Higher score = more likely to be OOD.
        """
        if self.method == 'msp':
            return self._msp_score(x)
        elif self.method == 'energy':
            return self._energy_score(x)
        elif self.method == 'mahalanobis':
            return self._mahalanobis_score(x)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _msp_score(self, x: torch.Tensor) -> torch.Tensor:
        """Maximum Softmax Probability score."""
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits / self.temperature, dim=-1)
            max_prob = probs.max(dim=-1)[0]
        return 1 - max_prob  # Higher = more OOD

    def _energy_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Energy-based OOD score.

        Reference: Liu et al., "Energy-based Out-of-distribution Detection", 2020
        """
        with torch.no_grad():
            logits = self.model(x)
            # Negative log-sum-exp (energy)
            energy = -self.temperature * torch.logsumexp(
                logits / self.temperature, dim=-1
            )
        return energy  # Higher energy = more OOD

    def _mahalanobis_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mahalanobis distance-based OOD score.

        Requires calling fit() first with training data.
        """
        if self.class_means is None:
            raise RuntimeError("Call fit() before using Mahalanobis detection")

        with torch.no_grad():
            features = self.model.get_features(x)

            # Compute distance to nearest class mean
            distances = []
            for mean in self.class_means:
                diff = features - mean
                dist = torch.sqrt(
                    torch.sum(diff @ self.precision_matrix * diff, dim=-1)
                )
                distances.append(dist)

            distances = torch.stack(distances, dim=-1)
            min_dist = distances.min(dim=-1)[0]

        return min_dist

    def fit(self, train_loader: DataLoader, device: torch.device):
        """
        Fit Mahalanobis detector on training data.

        Computes class means and shared covariance matrix.
        """
        self.model.eval()
        n_classes = self.model.n_classes

        # Collect features per class
        class_features = {i: [] for i in range(n_classes)}

        with torch.no_grad():
            for data, labels in train_loader:
                data = data.to(device)
                features = self.model.get_features(data).cpu()

                for i, label in enumerate(labels):
                    class_features[label.item()].append(features[i])

        # Compute class means
        self.class_means = []
        for i in range(n_classes):
            if class_features[i]:
                mean = torch.stack(class_features[i]).mean(dim=0).to(device)
                self.class_means.append(mean)

        # Compute shared covariance matrix
        all_features = []
        all_means = []
        for i in range(n_classes):
            if class_features[i]:
                feats = torch.stack(class_features[i])
                all_features.append(feats)
                all_means.append(self.class_means[i].cpu().expand(len(feats), -1))

        all_features = torch.cat(all_features, dim=0)
        all_means = torch.cat(all_means, dim=0)

        centered = all_features - all_means
        cov = (centered.T @ centered) / len(centered)

        # Add regularization for numerical stability
        cov += 1e-5 * torch.eye(cov.shape[0])

        self.precision_matrix = torch.inverse(cov).to(device)

    def forward(
        self,
        x: torch.Tensor,
        threshold: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect OOD samples.

        Args:
            x: Input tensor
            threshold: OOD threshold (if None, just returns scores)

        Returns:
            ood_scores, is_ood (if threshold provided)
        """
        scores = self.compute_ood_score(x)

        if threshold is not None:
            is_ood = scores > threshold
            return scores, is_ood

        return scores, None


# =============================================================================
# Conformal Prediction
# =============================================================================

class ConformalPredictor(nn.Module):
    """
    Conformal prediction for guaranteed coverage.

    Instead of point predictions, returns prediction sets that
    contain the true label with a guaranteed probability.

    Reference: Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction", 2022
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1  # 1 - coverage level (0.1 = 90% coverage)
    ):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.q_hat = None  # Calibrated threshold

    def calibrate(self, cal_loader: DataLoader, device: torch.device):
        """
        Calibrate conformal predictor on calibration set.

        Args:
            cal_loader: Calibration data loader
            device: Device to use
        """
        self.model.eval()
        scores = []

        with torch.no_grad():
            for data, labels in cal_loader:
                data, labels = data.to(device), labels.to(device)
                logits = self.model(data)
                probs = F.softmax(logits, dim=-1)

                # Non-conformity score: 1 - probability of true class
                true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()
                score = 1 - true_probs
                scores.append(score.cpu())

        scores = torch.cat(scores)
        n = len(scores)

        # Compute quantile with finite-sample correction
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = torch.quantile(scores, min(q_level, 1.0)).item()

        print(f"Calibrated threshold q_hat: {self.q_hat:.4f}")
        return self.q_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        Get prediction sets.

        Args:
            x: Input tensor

        Returns:
            probabilities, prediction_sets
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before making predictions")

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)

        # Prediction set: all classes with score <= q_hat
        # Score = 1 - probability
        prediction_sets = []
        for i in range(len(probs)):
            pred_set = []
            for c in range(probs.shape[1]):
                if 1 - probs[i, c].item() <= self.q_hat:
                    pred_set.append(c)
            # If empty (shouldn't happen with proper calibration), include top class
            if not pred_set:
                pred_set = [probs[i].argmax().item()]
            prediction_sets.append(pred_set)

        return probs, prediction_sets

    def predict_with_coverage(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions with coverage information.

        Returns:
            Dictionary with point predictions, sets, and set sizes
        """
        probs, pred_sets = self.forward(x)

        point_predictions = probs.argmax(dim=-1)
        set_sizes = torch.tensor([len(s) for s in pred_sets])
        confidence = probs.max(dim=-1)[0]

        return {
            'predictions': point_predictions,
            'probabilities': probs,
            'prediction_sets': pred_sets,
            'set_sizes': set_sizes,
            'confidence': confidence
        }

    def compute_coverage(
        self,
        test_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Compute empirical coverage on test set.

        Returns:
            Dictionary with coverage and average set size
        """
        self.model.eval()
        covered = 0
        total = 0
        set_sizes = []

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                _, pred_sets = self.forward(data)

                for i, (label, pred_set) in enumerate(zip(labels, pred_sets)):
                    total += 1
                    if label.item() in pred_set:
                        covered += 1
                    set_sizes.append(len(pred_set))

        coverage = covered / total
        avg_set_size = np.mean(set_sizes)

        return {
            'coverage': coverage,
            'target_coverage': 1 - self.alpha,
            'avg_set_size': avg_set_size,
            'total_samples': total
        }
