"""
Advanced Fault Detection Tasks
==============================

Beyond basic classification:
- SeverityRegressor: Estimate fault severity (0-100%)
- AnomalyDetector: Detect unknown/novel fault types
- PerMotorClassifier: Identify which specific motor(s) are faulty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from .models import ConvBlock, ChannelAttention


# =============================================================================
# Fault Severity Regression
# =============================================================================

class SeverityRegressor(nn.Module):
    """
    Estimate fault severity as a continuous value.

    Predicts severity score per motor (0 = healthy, 1 = severe damage).
    Useful for predictive maintenance - schedule repairs before failure.
    """

    def __init__(
        self,
        n_input_channels: int = 24,
        n_motors: int = 4,
        base_channels: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        self.n_motors = n_motors

        # Shared feature extractor
        self.features = nn.Sequential(
            ConvBlock(n_input_channels, base_channels, kernel_size=15),
            ChannelAttention(base_channels),
            nn.MaxPool1d(2),

            ConvBlock(base_channels, base_channels * 2, kernel_size=11),
            ChannelAttention(base_channels * 2),
            nn.MaxPool1d(2),

            ConvBlock(base_channels * 2, base_channels * 4, kernel_size=7),
            ChannelAttention(base_channels * 4),
            nn.AdaptiveAvgPool1d(1),
        )

        # Per-motor severity heads
        self.motor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_channels * 4, base_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(base_channels, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )
            for _ in range(n_motors)
        ])

        # Overall severity (max/mean of motor severities)
        self.overall_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict fault severity.

        Args:
            x: Input tensor (batch, 24, time)

        Returns:
            Dictionary with:
                - motor_severities: (batch, 4) severity per motor
                - overall_severity: (batch, 1) overall severity
        """
        features = self.features(x)

        # Per-motor severity
        motor_severities = torch.cat([
            head(features) for head in self.motor_heads
        ], dim=-1)

        # Overall severity
        overall_severity = self.overall_head(features)

        return {
            'motor_severities': motor_severities,
            'overall_severity': overall_severity,
            'max_severity': motor_severities.max(dim=-1, keepdim=True)[0]
        }

    def get_maintenance_priority(
        self,
        x: torch.Tensor,
        thresholds: Dict[str, float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get maintenance priority based on severity.

        Args:
            x: Input tensor
            thresholds: Severity thresholds for priority levels

        Returns:
            Dictionary with priority levels and recommendations
        """
        if thresholds is None:
            thresholds = {
                'critical': 0.8,   # Immediate action needed
                'high': 0.6,       # Schedule maintenance soon
                'medium': 0.4,     # Monitor closely
                'low': 0.2         # Normal wear
            }

        output = self.forward(x)
        max_severity = output['max_severity'].squeeze(-1)

        # Determine priority
        priority = torch.zeros_like(max_severity, dtype=torch.long)
        priority[max_severity > thresholds['critical']] = 4
        priority[(max_severity > thresholds['high']) & (max_severity <= thresholds['critical'])] = 3
        priority[(max_severity > thresholds['medium']) & (max_severity <= thresholds['high'])] = 2
        priority[(max_severity > thresholds['low']) & (max_severity <= thresholds['medium'])] = 1

        # Find most affected motor
        worst_motor = output['motor_severities'].argmax(dim=-1)

        return {
            **output,
            'priority': priority,
            'priority_labels': ['normal', 'low', 'medium', 'high', 'critical'],
            'worst_motor': worst_motor
        }


# =============================================================================
# Anomaly Detection (Unknown Faults)
# =============================================================================

class AnomalyDetector(nn.Module):
    """
    Detect unknown/novel fault types not seen during training.

    Uses autoencoder-based anomaly detection:
    - Learns to reconstruct normal and known fault patterns
    - High reconstruction error indicates anomaly

    Also supports deep SVDD (Support Vector Data Description).
    """

    def __init__(
        self,
        n_input_channels: int = 24,
        latent_dim: int = 64,
        base_channels: int = 32,
        method: str = 'autoencoder'  # 'autoencoder' or 'svdd'
    ):
        super().__init__()
        self.method = method
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_input_channels, base_channels, 7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base_channels, base_channels * 2, 5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),

            nn.Flatten(),
            nn.Linear(base_channels * 4 * 8, latent_dim)
        )

        if method == 'autoencoder':
            # Decoder for reconstruction
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, base_channels * 4 * 8),
                nn.GELU(),
                nn.Unflatten(1, (base_channels * 4, 8)),

                nn.ConvTranspose1d(base_channels * 4, base_channels * 2, 4, stride=4),
                nn.BatchNorm1d(base_channels * 2),
                nn.GELU(),

                nn.ConvTranspose1d(base_channels * 2, base_channels, 4, stride=4),
                nn.BatchNorm1d(base_channels),
                nn.GELU(),

                nn.ConvTranspose1d(base_channels, n_input_channels, 4, stride=2, padding=1),
            )
        else:
            # For SVDD: center of hypersphere
            self.register_buffer('center', torch.zeros(latent_dim))
            self.R = nn.Parameter(torch.tensor(0.0))  # Radius

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor, target_len: int = 256) -> torch.Tensor:
        """Decode latent representation."""
        if self.method != 'autoencoder':
            raise ValueError("Decode only available for autoencoder method")
        out = self.decoder(z)
        # Adjust to target length
        if out.shape[-1] != target_len:
            out = F.interpolate(out, size=target_len, mode='linear', align_corners=False)
        return out

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with anomaly scoring.

        Returns:
            Dictionary with latent representation and anomaly score
        """
        z = self.encode(x)

        if self.method == 'autoencoder':
            reconstruction = self.decode(z, x.shape[-1])
            # Reconstruction error as anomaly score
            anomaly_score = F.mse_loss(reconstruction, x, reduction='none').mean(dim=(1, 2))
            return {
                'latent': z,
                'reconstruction': reconstruction,
                'anomaly_score': anomaly_score
            }
        else:
            # SVDD: distance to center as anomaly score
            dist = torch.sum((z - self.center) ** 2, dim=-1)
            anomaly_score = dist - self.R ** 2
            return {
                'latent': z,
                'distance': dist,
                'anomaly_score': anomaly_score
            }

    def fit_threshold(
        self,
        train_loader: DataLoader,
        device: torch.device,
        percentile: float = 95
    ) -> float:
        """
        Fit anomaly threshold on training data.

        Args:
            train_loader: Training data loader
            device: Device to use
            percentile: Percentile for threshold

        Returns:
            Anomaly threshold
        """
        self.eval()
        scores = []

        with torch.no_grad():
            for data, _ in train_loader:
                data = data.to(device)
                output = self.forward(data)
                scores.append(output['anomaly_score'].cpu())

        scores = torch.cat(scores).numpy()
        threshold = np.percentile(scores, percentile)

        print(f"Anomaly threshold (p{percentile}): {threshold:.4f}")
        return threshold

    def detect_anomalies(
        self,
        x: torch.Tensor,
        threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalies.

        Args:
            x: Input tensor
            threshold: Anomaly threshold

        Returns:
            anomaly_scores, is_anomaly
        """
        output = self.forward(x)
        is_anomaly = output['anomaly_score'] > threshold
        return output['anomaly_score'], is_anomaly

    def svdd_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Deep SVDD loss function.

        Minimizes volume of hypersphere containing normal data.
        """
        dist = torch.sum((z - self.center) ** 2, dim=-1)
        # Soft-boundary SVDD
        scores = dist - self.R ** 2
        loss = self.R ** 2 + (1 / len(z)) * torch.sum(F.relu(scores))
        return loss


# =============================================================================
# Per-Motor Classification
# =============================================================================

class PerMotorClassifier(nn.Module):
    """
    Classify fault status for each motor independently.

    Instead of overall fault classification, predicts:
    - Motor A: [Normal, Chipped, Bent]
    - Motor B: [Normal, Chipped, Bent]
    - Motor C: [Normal, Chipped, Bent]
    - Motor D: [Normal, Chipped, Bent]

    This enables pinpointing which motor(s) need maintenance.
    """

    MOTOR_CHANNELS = {
        'A': [0, 1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10, 11],
        'C': [12, 13, 14, 15, 16, 17],
        'D': [18, 19, 20, 21, 22, 23]
    }

    def __init__(
        self,
        n_classes_per_motor: int = 3,  # Normal, Chipped, Bent
        base_channels: int = 32,
        dropout: float = 0.3,
        shared_features: bool = True
    ):
        super().__init__()
        self.n_classes = n_classes_per_motor
        self.n_motors = 4
        self.shared_features = shared_features

        if shared_features:
            # Shared feature extractor
            self.shared_encoder = nn.Sequential(
                ConvBlock(24, base_channels * 2, kernel_size=15),
                nn.MaxPool1d(2),
                ConvBlock(base_channels * 2, base_channels * 4, kernel_size=7),
                nn.MaxPool1d(2),
            )
        else:
            # Per-motor encoders (6 channels each)
            self.motor_encoders = nn.ModuleDict({
                motor: nn.Sequential(
                    ConvBlock(6, base_channels, kernel_size=15),
                    nn.MaxPool1d(2),
                    ConvBlock(base_channels, base_channels * 2, kernel_size=7),
                    nn.MaxPool1d(2),
                )
                for motor in ['A', 'B', 'C', 'D']
            })

        # Per-motor classification heads
        feat_dim = base_channels * 4 if shared_features else base_channels * 2
        self.motor_heads = nn.ModuleDict({
            motor: nn.Sequential(
                ConvBlock(feat_dim, base_channels * 4, kernel_size=5),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(base_channels * 4, base_channels * 2),
                nn.GELU(),
                nn.Linear(base_channels * 2, n_classes_per_motor)
            )
            for motor in ['A', 'B', 'C', 'D']
        })

        # Cross-motor attention (detect asymmetry)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=base_channels * 4,
            num_heads=4,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Classify each motor independently.

        Args:
            x: Input tensor (batch, 24, time)

        Returns:
            Dictionary with per-motor predictions
        """
        batch = x.shape[0]

        if self.shared_features:
            # Shared features
            shared_feat = self.shared_encoder(x)  # (batch, channels, time')

            # Get per-motor logits
            motor_logits = {}
            for motor in ['A', 'B', 'C', 'D']:
                motor_logits[motor] = self.motor_heads[motor](shared_feat)
        else:
            # Per-motor encoding
            motor_feats = {}
            for motor, channels in self.MOTOR_CHANNELS.items():
                motor_x = x[:, channels, :]  # (batch, 6, time)
                motor_feats[motor] = self.motor_encoders[motor](motor_x)

            # Cross-motor attention
            feat_stack = torch.stack(
                [motor_feats[m].mean(dim=-1) for m in ['A', 'B', 'C', 'D']],
                dim=1
            )  # (batch, 4, channels)

            attended, _ = self.cross_attention(feat_stack, feat_stack, feat_stack)

            # Per-motor classification with attention
            motor_logits = {}
            for i, motor in enumerate(['A', 'B', 'C', 'D']):
                # Combine original and attended features
                combined = motor_feats[motor]
                motor_logits[motor] = self.motor_heads[motor](combined)

        # Stack into tensor
        logits_tensor = torch.stack(
            [motor_logits[m] for m in ['A', 'B', 'C', 'D']],
            dim=1
        )  # (batch, 4, n_classes)

        predictions = logits_tensor.argmax(dim=-1)  # (batch, 4)
        probabilities = F.softmax(logits_tensor, dim=-1)

        return {
            'logits': logits_tensor,
            'predictions': predictions,
            'probabilities': probabilities,
            'motor_A': motor_logits['A'],
            'motor_B': motor_logits['B'],
            'motor_C': motor_logits['C'],
            'motor_D': motor_logits['D'],
        }

    def get_fault_report(self, x: torch.Tensor) -> List[Dict]:
        """
        Generate human-readable fault report.

        Returns:
            List of reports, one per sample
        """
        output = self.forward(x)
        predictions = output['predictions']
        probabilities = output['probabilities']

        fault_names = {0: 'Normal', 1: 'Chipped', 2: 'Bent'}
        motor_names = ['A', 'B', 'C', 'D']

        reports = []
        for i in range(len(predictions)):
            report = {
                'motors': {},
                'faulty_motors': [],
                'summary': ''
            }

            for j, motor in enumerate(motor_names):
                pred = predictions[i, j].item()
                conf = probabilities[i, j, pred].item()

                report['motors'][motor] = {
                    'status': fault_names[pred],
                    'confidence': f"{conf:.1%}",
                    'probabilities': {
                        fault_names[k]: f"{probabilities[i, j, k].item():.1%}"
                        for k in range(self.n_classes)
                    }
                }

                if pred != 0:
                    report['faulty_motors'].append(f"{motor} ({fault_names[pred]})")

            # Summary
            if report['faulty_motors']:
                report['summary'] = f"Faults detected: {', '.join(report['faulty_motors'])}"
            else:
                report['summary'] = "All motors operating normally"

            reports.append(report)

        return reports


# =============================================================================
# Remaining Useful Life (RUL) Prediction
# =============================================================================

class RULPredictor(nn.Module):
    """
    Predict Remaining Useful Life of motor.

    Estimates how many flight hours/cycles remain before failure.
    Critical for predictive maintenance scheduling.
    """

    def __init__(
        self,
        n_input_channels: int = 24,
        base_channels: int = 64,
        dropout: float = 0.3,
        max_rul: float = 100.0  # Maximum RUL in hours
    ):
        super().__init__()
        self.max_rul = max_rul

        self.features = nn.Sequential(
            ConvBlock(n_input_channels, base_channels, kernel_size=15),
            ChannelAttention(base_channels),
            nn.MaxPool1d(2),

            ConvBlock(base_channels, base_channels * 2, kernel_size=11),
            ChannelAttention(base_channels * 2),
            nn.MaxPool1d(2),

            ConvBlock(base_channels * 2, base_channels * 4, kernel_size=7),
            ChannelAttention(base_channels * 4),
            nn.AdaptiveAvgPool1d(1),
        )

        # RUL regression head
        self.rul_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, base_channels),
            nn.GELU(),
            nn.Linear(base_channels, 1),
            nn.ReLU()  # RUL must be non-negative
        )

        # Uncertainty head (aleatoric uncertainty)
        self.uncertainty_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels),
            nn.GELU(),
            nn.Linear(base_channels, 1),
            nn.Softplus()  # Positive variance
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict RUL with uncertainty.

        Returns:
            Dictionary with RUL prediction and uncertainty
        """
        features = self.features(x)

        rul = self.rul_head(features)
        rul = torch.clamp(rul, 0, self.max_rul)

        uncertainty = self.uncertainty_head(features)

        return {
            'rul': rul,
            'uncertainty': uncertainty,
            'rul_lower': rul - 2 * uncertainty,  # 95% CI
            'rul_upper': rul + 2 * uncertainty
        }

    def gaussian_nll_loss(
        self,
        predictions: torch.Tensor,
        uncertainty: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Gaussian negative log-likelihood loss.

        Trains both mean and variance.
        """
        variance = uncertainty ** 2 + 1e-6
        loss = 0.5 * (torch.log(variance) + (targets - predictions) ** 2 / variance)
        return loss.mean()
