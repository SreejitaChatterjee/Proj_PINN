"""
Sensor-PINN for Motor Fault Detection.

Instead of converting sensor data to state vectors (which causes drift),
this PINN learns directly from raw IMU data and uses physics-based losses
that encode expected sensor relationships.

Key insight: Motor faults change the physical relationships between sensors,
which can be detected through physics residuals without explicit state estimation.

Physics constraints:
1. Rigid body: All gyroscopes should measure similar angular rates
2. Symmetry: Opposite motors (A-C, B-D) have related vibration patterns
3. Conservation: Sum of accelerations relates to body motion
4. Consistency: Sensor readings should be smooth over time
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict


class SensorPINN(nn.Module):
    """
    Physics-Informed Neural Network for raw sensor dynamics.

    Learns to predict sensor readings at t+1 given readings at t,
    while enforcing physics-based consistency constraints.

    Input: 24 channels (4 motors × [ax, ay, az, gx, gy, gz])
    Output: 24 channels (predicted next timestep)

    Physics losses encode:
    - Gyroscope consistency across motors
    - Accelerometer symmetry for opposite motors
    - Temporal smoothness constraints
    """

    # Motor configuration (X-frame quadrotor)
    # A: front-right, B: front-left, C: back-left, D: back-right
    MOTOR_PAIRS = [('A', 'C'), ('B', 'D')]  # Diagonal pairs
    MOTOR_INDICES = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = 'tanh',
        use_residual: bool = True
    ):
        """
        Args:
            hidden_size: Width of hidden layers
            num_layers: Number of hidden layers
            dropout: Dropout rate for regularization
            activation: Activation function ('tanh', 'relu', 'gelu')
            use_residual: If True, predict delta (change) instead of absolute
        """
        super().__init__()

        self.input_dim = 24  # 4 motors × 6 sensors
        self.output_dim = 24
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_residual = use_residual

        # Activation
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []

        # Input layer
        layers.append(nn.Linear(self.input_dim, hidden_size))
        layers.append(self.activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_size, self.output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict sensor readings at t+1.

        Args:
            x: Sensor readings at time t, shape (batch, 24)

        Returns:
            Predicted readings at t+1, shape (batch, 24)
        """
        delta = self.network(x)

        if self.use_residual:
            return x + delta
        else:
            return delta

    def _get_motor_data(
        self,
        x: torch.Tensor,
        motor: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract accelerometer and gyroscope data for a motor.

        Args:
            x: Full sensor tensor, shape (batch, 24)
            motor: Motor name ('A', 'B', 'C', 'D')

        Returns:
            accel: (batch, 3) accelerometer [ax, ay, az]
            gyro: (batch, 3) gyroscope [gx, gy, gz]
        """
        idx = self.MOTOR_INDICES[motor]
        base = idx * 6
        accel = x[:, base:base+3]
        gyro = x[:, base+3:base+6]
        return accel, gyro

    def gyro_consistency_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Physics loss: All gyroscopes should measure similar angular rates.

        For a rigid body, all points rotate with the same angular velocity.
        Differences indicate sensor noise, calibration errors, or motor faults.

        Returns:
            Mean squared difference between motor gyroscope readings
        """
        gyros = []
        for motor in ['A', 'B', 'C', 'D']:
            _, gyro = self._get_motor_data(x, motor)
            gyros.append(gyro)

        # Compute pairwise differences
        loss = torch.tensor(0.0, device=x.device)
        n_pairs = 0

        for i in range(4):
            for j in range(i+1, 4):
                diff = gyros[i] - gyros[j]
                loss = loss + diff.pow(2).mean()
                n_pairs += 1

        return loss / n_pairs

    def accel_symmetry_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Physics loss: Diagonal motors should have related acceleration patterns.

        For a symmetric quadrotor in hover, diagonal motors (A-C, B-D) should
        have similar vibration magnitudes. Asymmetry indicates motor faults.

        Returns:
            Symmetry violation loss
        """
        loss = torch.tensor(0.0, device=x.device)

        for m1, m2 in self.MOTOR_PAIRS:
            accel1, _ = self._get_motor_data(x, m1)
            accel2, _ = self._get_motor_data(x, m2)

            # Compare acceleration magnitudes (not directions, due to vibration)
            mag1 = accel1.pow(2).sum(dim=1).sqrt()
            mag2 = accel2.pow(2).sum(dim=1).sqrt()

            # Relative difference
            mean_mag = (mag1 + mag2) / 2 + 1e-6
            rel_diff = (mag1 - mag2).abs() / mean_mag

            loss = loss + rel_diff.mean()

        return loss / len(self.MOTOR_PAIRS)

    def temporal_smoothness_loss(
        self,
        x_t: torch.Tensor,
        x_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Physics loss: Predictions should be temporally smooth.

        Large jumps in sensor readings are physically implausible.

        Args:
            x_t: Current sensor readings
            x_pred: Predicted next readings

        Returns:
            Smoothness violation loss (penalizes large changes)
        """
        delta = x_pred - x_t
        # L2 norm of change, normalized by expected noise level
        return delta.pow(2).mean()

    def physics_loss(
        self,
        x_t: torch.Tensor,
        x_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Combined physics loss for sensor consistency.

        Args:
            x_t: Current sensor readings (batch, 24)
            x_pred: Predicted next readings (batch, 24)

        Returns:
            Combined physics loss
        """
        # Gyroscope consistency on predicted state
        gyro_loss = self.gyro_consistency_loss(x_pred)

        # Accelerometer symmetry on predicted state
        accel_loss = self.accel_symmetry_loss(x_pred)

        # Temporal smoothness
        smooth_loss = self.temporal_smoothness_loss(x_t, x_pred)

        # Weighted combination
        return gyro_loss + 0.5 * accel_loss + 0.1 * smooth_loss

    def compute_residuals(
        self,
        x_t: torch.Tensor,
        x_actual: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute prediction and physics residuals for fault detection.

        Args:
            x_t: Current sensor readings
            x_actual: Actual next sensor readings

        Returns:
            Dictionary of residual metrics
        """
        with torch.no_grad():
            x_pred = self.forward(x_t)

            # Prediction residual (how well did we predict?)
            pred_residual = (x_pred - x_actual).pow(2).sum(dim=1).sqrt()

            # Per-motor prediction residuals
            motor_residuals = {}
            for motor in ['A', 'B', 'C', 'D']:
                idx = self.MOTOR_INDICES[motor]
                base = idx * 6
                motor_res = (x_pred[:, base:base+6] - x_actual[:, base:base+6])
                motor_residuals[motor] = motor_res.pow(2).sum(dim=1).sqrt()

            # Gyroscope consistency residual
            gyro_residual = self.gyro_consistency_loss(x_actual)

            # Symmetry residual
            sym_residual = self.accel_symmetry_loss(x_actual)

        return {
            'prediction': pred_residual,
            'gyro_consistency': gyro_residual,
            'symmetry': sym_residual,
            'motor_residuals': motor_residuals
        }

    def detect_fault(
        self,
        x_t: torch.Tensor,
        x_actual: torch.Tensor,
        threshold: float = 2.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Detect motor faults based on residuals.

        Args:
            x_t: Current sensor readings
            x_actual: Actual next sensor readings
            threshold: Anomaly threshold (in std deviations)

        Returns:
            is_fault: Boolean tensor indicating fault detection
            fault_scores: Per-motor fault scores
        """
        residuals = self.compute_residuals(x_t, x_actual)

        # Total residual
        total_residual = residuals['prediction']

        # Fault detection (simple threshold)
        # Could be improved with learned threshold or statistical methods
        is_fault = total_residual > threshold

        # Per-motor fault scores
        motor_scores = residuals['motor_residuals']

        return is_fault, motor_scores


class SensorPINNTrainer:
    """
    Trainer for Sensor-PINN fault detection.
    """

    def __init__(
        self,
        model: SensorPINN,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        physics_weight: float = 1.0,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.physics_weight = physics_weight

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'physics_loss': []
        }

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        data_loss_sum = 0
        physics_loss_sum = 0
        n_batches = 0

        for x_t, x_next, _ in dataloader:
            x_t = x_t.to(self.device)
            x_next = x_next.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            x_pred = self.model(x_t)

            # Data loss
            data_loss = self.criterion(x_pred, x_next)

            # Physics loss
            physics_loss = self.model.physics_loss(x_t, x_pred)

            # Total loss
            loss = data_loss + self.physics_weight * physics_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            data_loss_sum += data_loss.item()
            physics_loss_sum += physics_loss.item()
            n_batches += 1

        return {
            'total': total_loss / n_batches,
            'data': data_loss_sum / n_batches,
            'physics': physics_loss_sum / n_batches
        }

    def validate(self, dataloader) -> Dict[str, float]:
        """Validate and compute fault detection metrics."""
        self.model.eval()
        total_loss = 0
        n_batches = 0

        all_residuals = []
        all_labels = []

        with torch.no_grad():
            for x_t, x_next, labels in dataloader:
                x_t = x_t.to(self.device)
                x_next = x_next.to(self.device)

                x_pred = self.model(x_t)
                loss = self.criterion(x_pred, x_next)

                # Prediction residuals
                residuals = (x_pred - x_next).pow(2).sum(dim=1).sqrt()
                all_residuals.extend(residuals.cpu().numpy())
                all_labels.extend(labels.numpy())

                total_loss += loss.item()
                n_batches += 1

        all_residuals = np.array(all_residuals)
        all_labels = np.array(all_labels)

        # Fault detection metrics
        normal_res = all_residuals[all_labels == 0]
        faulty_res = all_residuals[all_labels == 1]

        metrics = {
            'loss': total_loss / n_batches,
            'normal_residual': normal_res.mean() if len(normal_res) > 0 else 0,
            'faulty_residual': faulty_res.mean() if len(faulty_res) > 0 else 0,
        }

        if len(normal_res) > 0 and len(faulty_res) > 0:
            metrics['separation'] = faulty_res.mean() / (normal_res.mean() + 1e-8)

            # ROC-like threshold sweep
            thresholds = np.percentile(all_residuals, [90, 95, 99])
            for pct, thresh in zip([90, 95, 99], thresholds):
                tp = (faulty_res > thresh).sum()
                fp = (normal_res > thresh).sum()
                fn = (faulty_res <= thresh).sum()
                tn = (normal_res <= thresh).sum()

                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)

                metrics[f'precision_{pct}'] = prec
                metrics[f'recall_{pct}'] = rec

        return metrics


if __name__ == "__main__":
    # Quick test
    print("SensorPINN Test")
    print("=" * 50)

    model = SensorPINN(hidden_size=128, num_layers=4)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(32, 24)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Test physics loss
    phys_loss = model.physics_loss(x, y)
    print(f"Physics loss: {phys_loss.item():.4f}")

    # Test residuals
    x_next = torch.randn(32, 24)
    residuals = model.compute_residuals(x, x_next)
    print(f"Prediction residual: {residuals['prediction'].mean():.4f}")
