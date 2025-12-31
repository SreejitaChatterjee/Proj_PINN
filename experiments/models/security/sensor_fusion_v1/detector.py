"""
Sensor Fusion Attack Detector

Architecture based on empirical analysis:
- Physics consistency checks detect 60%+ attacks (no learning needed)
- Temporal patterns handle remaining stealthy attacks
- Trained ONLY on normal data (no attack labels required)

Key insight: Attacks break sensor CONSISTENCY, not state prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Detector configuration."""
    dt: float = 0.005  # 200 Hz
    seq_len: int = 100  # Input sequence length
    hidden_dim: int = 64
    dropout: float = 0.1


class PhysicsConsistency(nn.Module):
    """
    Compute physics-based consistency features.

    These features require NO LEARNING - pure physics.
    From analysis: achieves 60%+ detection alone.
    """

    def __init__(self, dt: float = 0.005):
        super().__init__()
        self.dt = dt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, 16] - state(12) + control(4)
               state = [x, y, z, phi, theta, psi, p, q, r, vx, vy, vz]

        Returns:
            features: [batch, seq-1, n_features]
        """
        batch, seq, _ = x.shape

        # Extract components
        pos = x[:, :, 0:3]    # x, y, z
        att = x[:, :, 3:6]    # phi, theta, psi
        rate = x[:, :, 6:9]   # p, q, r
        vel = x[:, :, 9:12]   # vx, vy, vz

        features = []

        # 1. Position-Velocity Consistency (TOP METRIC: 58.6% avg detection)
        # d(pos)/dt should equal velocity
        pos_deriv = (pos[:, 1:] - pos[:, :-1]) / self.dt
        pos_vel_diff = pos_deriv - vel[:, 1:]
        features.append(pos_vel_diff)  # [batch, seq-1, 3]
        features.append(torch.norm(pos_vel_diff, dim=-1, keepdim=True))  # magnitude

        # 2. Attitude-Rate Consistency
        # d(att)/dt should equal angular rates
        att_deriv = (att[:, 1:] - att[:, :-1]) / self.dt
        att_rate_diff = att_deriv - rate[:, 1:]
        features.append(att_rate_diff)
        features.append(torch.norm(att_rate_diff, dim=-1, keepdim=True))

        # 3. Velocity Smoothness (Jerk)
        # d²(vel)/dt² should be bounded
        accel = (vel[:, 1:] - vel[:, :-1]) / self.dt
        jerk = (accel[:, 1:] - accel[:, :-1]) / self.dt
        jerk_padded = torch.cat([torch.zeros(batch, 1, 3, device=x.device), jerk], dim=1)
        features.append(jerk_padded)
        features.append(torch.norm(jerk_padded, dim=-1, keepdim=True))

        # 4. Angular Acceleration
        angular_accel = (rate[:, 1:] - rate[:, :-1]) / self.dt
        features.append(angular_accel)
        features.append(torch.norm(angular_accel, dim=-1, keepdim=True))

        # 5. Kinematic Consistency (TOP METRIC: 60.4% avg detection)
        # integral(vel) should equal position change
        window = 20
        if seq > window + 1:
            # Velocity integral over window
            vel_integral = torch.zeros(batch, seq - 1, 3, device=x.device)
            for i in range(window, seq - 1):
                vel_integral[:, i] = vel[:, i-window+1:i+1].sum(dim=1) * self.dt

            # Position change over window
            pos_change = torch.zeros(batch, seq - 1, 3, device=x.device)
            for i in range(window, seq - 1):
                pos_change[:, i] = pos[:, i+1] - pos[:, i-window+1]

            kinematic_diff = vel_integral - pos_change
            features.append(kinematic_diff)
            features.append(torch.norm(kinematic_diff, dim=-1, keepdim=True))
        else:
            features.append(torch.zeros(batch, seq-1, 3, device=x.device))
            features.append(torch.zeros(batch, seq-1, 1, device=x.device))

        return torch.cat(features, dim=-1)  # [batch, seq-1, 20]


class TemporalBlock(nn.Module):
    """1D CNN block for temporal pattern extraction."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class AttackDetector(nn.Module):
    """
    Main attack detector.

    Architecture:
    1. Physics consistency (no learning) - handles 60%+ attacks
    2. Multi-scale CNN - captures temporal patterns
    3. LSTM - handles long-range dependencies
    4. Reconstruction head - anomaly scoring
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__()
        self.config = config or DetectorConfig()

        # Layer 1: Physics consistency (NO LEARNING)
        self.physics = PhysicsConsistency(dt=self.config.dt)
        physics_dim = 20  # From PhysicsConsistency output

        # Layer 2: Multi-scale temporal CNN
        self.cnn = nn.Sequential(
            TemporalBlock(physics_dim, 32, kernel_size=3),
            TemporalBlock(32, 64, kernel_size=5),
            TemporalBlock(64, self.config.hidden_dim, kernel_size=7),
        )

        # Layer 3: Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.dropout
        )

        # Layer 4: Reconstruction decoder (autoencoder style)
        self.decoder = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, physics_dim),
        )

        # Anomaly score head
        self.score_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim + 1, 32),  # +1 for physics score
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Learnable threshold
        self.register_buffer('threshold', torch.tensor(0.5))
        self.register_buffer('physics_threshold', torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq, 16] - sensor data

        Returns:
            dict with anomaly_score, predictions, physics_score
        """
        # Physics features (no learning)
        physics_feat = self.physics(x)  # [batch, seq-1, 20]

        # Physics-only anomaly score (magnitude of key inconsistencies)
        # Index 3 = pos_vel_norm, Index 7 = att_rate_norm
        physics_score = physics_feat[:, :, 3:4] + physics_feat[:, :, 7:8] * 0.1

        # CNN expects [batch, channels, seq]
        h = physics_feat.transpose(1, 2)
        h = self.cnn(h)
        h = h.transpose(1, 2)  # [batch, seq-1, hidden]

        # LSTM
        h, _ = self.lstm(h)  # [batch, seq-1, hidden]

        # Reconstruction
        reconstructed = self.decoder(h)
        recon_error = torch.norm(physics_feat - reconstructed, dim=-1, keepdim=True)

        # Combined score
        combined = torch.cat([h, physics_score], dim=-1)
        anomaly_score = self.score_head(combined)

        # Predictions (use PHYSICS for high-confidence, learned for edge cases)
        physics_pred = (physics_score > self.physics_threshold).float()
        learned_pred = (anomaly_score > self.threshold).float()

        # Final: OR of physics and learned (physics catches obvious, learned catches subtle)
        predictions = torch.clamp(physics_pred + learned_pred, 0, 1)

        return {
            'anomaly_score': anomaly_score,
            'physics_score': physics_score,
            'recon_error': recon_error,
            'predictions': predictions,
        }

    def calibrate(self, normal_data: torch.Tensor, percentile: float = 99.0):
        """Calibrate thresholds on normal data."""
        self.eval()
        with torch.no_grad():
            out = self.forward(normal_data)

            physics_vals = out['physics_score'].cpu().numpy().flatten()
            self.physics_threshold = torch.tensor(np.percentile(physics_vals, percentile))

            score_vals = out['anomaly_score'].cpu().numpy().flatten()
            self.threshold = torch.tensor(np.percentile(score_vals, percentile))

        return {
            'physics_threshold': self.physics_threshold.item(),
            'learned_threshold': self.threshold.item()
        }


def train_detector(
    model: AttackDetector,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True
):
    """
    Train detector on NORMAL data only.

    Loss: Reconstruction error should be low for normal patterns.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Create sequences
    def make_sequences(data, seq_len=100):
        sequences = []
        for i in range(0, len(data) - seq_len, seq_len // 2):
            sequences.append(data[i:i+seq_len])
        return torch.stack(sequences)

    train_seqs = make_sequences(train_data)
    val_seqs = make_sequences(val_data)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_seqs),
        batch_size=batch_size,
        shuffle=True
    )

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)

            # Loss: reconstruction should be low for normal data
            loss = out['recon_error'].mean() + 0.1 * out['anomaly_score'].mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_seqs_device = val_seqs.to(device)
            val_out = model(val_seqs_device)
            val_loss = val_out['recon_error'].mean().item()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    return history
