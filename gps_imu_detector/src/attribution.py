"""
Attribution Head for Attack Type Classification

Multi-task learning: anomaly detection + attack type classification.
Helps understand what kind of attack is occurring when detected.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AttributionResult:
    """Container for attribution results."""
    is_anomaly: bool
    anomaly_score: float
    attack_type: str
    attack_probabilities: Dict[str, float]
    sensor_attribution: Dict[str, float]


class AttackTypeHead(nn.Module):
    """
    Classification head for attack type attribution.

    Predicts which type of attack is occurring (if any).
    """

    ATTACK_TYPES = [
        'normal',
        'bias',
        'drift',
        'noise',
        'coordinated',
        'intermittent',
        'ramp',
        'adversarial'
    ]

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.n_classes = len(self.ATTACK_TYPES)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.n_classes)
        )

        # Class weights for imbalanced data (normal is majority)
        self.register_buffer(
            'class_weights',
            torch.tensor([0.1] + [1.0] * (self.n_classes - 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, D] hidden features

        Returns:
            logits: [B, n_classes] class logits
        """
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict attack type.

        Args:
            x: [B, D] hidden features

        Returns:
            predicted_class: [B] predicted class indices
            probabilities: [B, n_classes] class probabilities
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        predicted = torch.argmax(probs, dim=-1)
        return predicted, probs


class SensorAttributionHead(nn.Module):
    """
    Attribution head for identifying which sensor is compromised.

    Uses attention mechanism to highlight anomalous sensors.
    """

    SENSOR_GROUPS = [
        'gps_position',
        'gps_velocity',
        'imu_acceleration',
        'imu_gyroscope',
        'attitude'
    ]

    def __init__(self, input_dim: int, n_sensors: int = 5):
        super().__init__()

        self.n_sensors = n_sensors

        # Attention for sensor attribution
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, n_sensors)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute sensor attention weights.

        Args:
            x: [B, D] hidden features

        Returns:
            attention_weights: [B, n_sensors] normalized attention
        """
        attn_logits = self.attention(x)
        return F.softmax(attn_logits, dim=-1)

    def get_attribution(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Get sensor attribution scores.

        Args:
            x: [1, D] single sample features

        Returns:
            Dict mapping sensor name to attribution score
        """
        with torch.no_grad():
            weights = self.forward(x).squeeze().cpu().numpy()

        return {name: float(w) for name, w in zip(self.SENSOR_GROUPS, weights)}


class MultiTaskDetector(nn.Module):
    """
    Multi-task detector with anomaly detection + attribution.

    Architecture:
    - Shared encoder (CNN + GRU)
    - Anomaly detection head
    - Attack type classification head
    - Sensor attribution head
    """

    def __init__(
        self,
        input_dim: int,
        cnn_channels: Tuple[int, ...] = (32, 64),
        gru_hidden_size: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = gru_hidden_size

        # Shared CNN encoder
        cnn_layers = []
        in_channels = 1
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)

        # Shared GRU
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # Task-specific heads
        self.anomaly_head = nn.Linear(gru_hidden_size, 1)
        self.attack_type_head = AttackTypeHead(gru_hidden_size)
        self.sensor_head = SensorAttributionHead(gru_hidden_size)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, T, D] input sequence

        Returns:
            anomaly_logits: [B, T, 1] anomaly scores
            attack_logits: [B, T, n_classes] attack type logits
            sensor_attn: [B, T, n_sensors] sensor attribution
            hidden: [num_layers, B, H] final hidden state
        """
        batch_size, seq_len, _ = x.shape

        # CNN expects [B, C, L] format
        # Reshape to [B*T, 1, D] for per-timestep CNN
        x_cnn = x.reshape(batch_size * seq_len, 1, self.input_dim)
        cnn_out = self.cnn(x_cnn)  # [B*T, C, D]

        # Pool over feature dimension
        cnn_pooled = cnn_out.mean(dim=2)  # [B*T, C]
        cnn_pooled = cnn_pooled.reshape(batch_size, seq_len, -1)  # [B, T, C]

        # GRU
        gru_out, hidden = self.gru(cnn_pooled)  # [B, T, H]

        # Task heads (applied to each timestep)
        gru_flat = gru_out.reshape(batch_size * seq_len, -1)

        anomaly_logits = self.anomaly_head(gru_flat)
        anomaly_logits = anomaly_logits.reshape(batch_size, seq_len, 1)

        attack_logits = self.attack_type_head(gru_flat)
        attack_logits = attack_logits.reshape(batch_size, seq_len, -1)

        sensor_attn = self.sensor_head(gru_flat)
        sensor_attn = sensor_attn.reshape(batch_size, seq_len, -1)

        return anomaly_logits, attack_logits, sensor_attn, hidden

    def predict(self, x: torch.Tensor) -> List[AttributionResult]:
        """
        Full prediction with attribution.

        Args:
            x: [B, T, D] input sequence

        Returns:
            List of AttributionResult for each timestep
        """
        self.eval()
        with torch.no_grad():
            anomaly_logits, attack_logits, sensor_attn, _ = self.forward(x)

            # Convert to probabilities
            anomaly_probs = torch.sigmoid(anomaly_logits).squeeze(-1)  # [B, T]
            attack_probs = F.softmax(attack_logits, dim=-1)  # [B, T, C]

        results = []
        batch_size, seq_len = anomaly_probs.shape

        for b in range(batch_size):
            for t in range(seq_len):
                anomaly_score = float(anomaly_probs[b, t].cpu())
                is_anomaly = anomaly_score > 0.5

                # Attack type
                attack_prob_dict = {
                    name: float(attack_probs[b, t, i].cpu())
                    for i, name in enumerate(AttackTypeHead.ATTACK_TYPES)
                }
                attack_type = max(attack_prob_dict, key=attack_prob_dict.get)

                # Sensor attribution
                sensor_dict = {
                    name: float(sensor_attn[b, t, i].cpu())
                    for i, name in enumerate(SensorAttributionHead.SENSOR_GROUPS)
                }

                results.append(AttributionResult(
                    is_anomaly=is_anomaly,
                    anomaly_score=anomaly_score,
                    attack_type=attack_type if is_anomaly else 'normal',
                    attack_probabilities=attack_prob_dict,
                    sensor_attribution=sensor_dict
                ))

        return results


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining anomaly detection and attribution.

    Loss = w1 * BCE(anomaly) + w2 * CE(attack_type) + w3 * Consistency
    """

    def __init__(
        self,
        anomaly_weight: float = 1.0,
        attack_weight: float = 0.5,
        consistency_weight: float = 0.1
    ):
        super().__init__()

        self.anomaly_weight = anomaly_weight
        self.attack_weight = attack_weight
        self.consistency_weight = consistency_weight

        self.anomaly_loss = nn.BCEWithLogitsLoss()
        self.attack_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        anomaly_logits: torch.Tensor,
        attack_logits: torch.Tensor,
        sensor_attn: torch.Tensor,
        anomaly_labels: torch.Tensor,
        attack_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            anomaly_logits: [B, T, 1] anomaly logits
            attack_logits: [B, T, C] attack type logits
            sensor_attn: [B, T, S] sensor attention
            anomaly_labels: [B, T] binary labels
            attack_labels: [B, T] attack type indices (optional)

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Anomaly detection loss
        anomaly_loss = self.anomaly_loss(
            anomaly_logits.squeeze(-1),
            anomaly_labels.float()
        )

        # Attack type loss (only on anomalous samples)
        if attack_labels is not None:
            # Mask for anomalous samples
            attack_mask = anomaly_labels == 1
            if attack_mask.any():
                attack_logits_masked = attack_logits[attack_mask]
                attack_labels_masked = attack_labels[attack_mask]
                attack_loss = self.attack_loss(attack_logits_masked, attack_labels_masked)
            else:
                attack_loss = torch.tensor(0.0, device=anomaly_logits.device)
        else:
            attack_loss = torch.tensor(0.0, device=anomaly_logits.device)

        # Consistency loss: sensor attention should be sparse (one sensor attacked)
        # Use entropy regularization
        sensor_entropy = -(sensor_attn * torch.log(sensor_attn + 1e-8)).sum(dim=-1)
        consistency_loss = sensor_entropy.mean()

        # Total loss
        total_loss = (
            self.anomaly_weight * anomaly_loss +
            self.attack_weight * attack_loss +
            self.consistency_weight * consistency_loss
        )

        loss_dict = {
            'total': float(total_loss.item()),
            'anomaly': float(anomaly_loss.item()),
            'attack_type': float(attack_loss.item()),
            'consistency': float(consistency_loss.item())
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    # Test multi-task detector
    batch_size = 8
    seq_len = 50
    input_dim = 100

    model = MultiTaskDetector(input_dim=input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    anomaly_logits, attack_logits, sensor_attn, hidden = model(x)

    print(f"Anomaly logits shape: {anomaly_logits.shape}")
    print(f"Attack logits shape: {attack_logits.shape}")
    print(f"Sensor attention shape: {sensor_attn.shape}")

    # Test prediction
    results = model.predict(x[:1, :5, :])  # 5 timesteps
    for i, r in enumerate(results):
        print(f"\nTimestep {i}:")
        print(f"  Anomaly: {r.is_anomaly} (score: {r.anomaly_score:.3f})")
        print(f"  Attack type: {r.attack_type}")
        print(f"  Top sensor: {max(r.sensor_attribution, key=r.sensor_attribution.get)}")

    # Test loss
    loss_fn = MultiTaskLoss()
    anomaly_labels = torch.randint(0, 2, (batch_size, seq_len))
    attack_labels = torch.randint(0, 8, (batch_size, seq_len))

    loss, loss_dict = loss_fn(
        anomaly_logits, attack_logits, sensor_attn,
        anomaly_labels, attack_labels
    )
    print(f"\nLoss: {loss_dict}")
