"""
Sequence-based Physics-Informed Neural Network for temporal pattern detection.

Unlike the standard single-step PINN which predicts next_state from current_state,
SequencePINN takes a window of past states to capture temporal patterns.

This enables detection of:
- Replay attacks (repeated patterns)
- Time delays (shifted patterns)
- Gradual drift (accumulated small changes)
- Frozen sensors (constant values over time)
- Temporal inconsistencies

Example:
    model = SequencePINN(sequence_length=20)

    # Input: sequence of past states + controls
    # Shape: [batch, sequence_length, 16]
    sequence = torch.randn(32, 20, 16)

    # Output: predicted next state
    # Shape: [batch, 12]
    next_state = model(sequence)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .base import DynamicsPINN


class SequencePINN(nn.Module):
    """
    Sequence-based PINN for temporal anomaly detection.

    Uses LSTM to capture temporal dependencies in state evolution,
    enabling detection of attacks that exploit temporal patterns.

    Args:
        sequence_length: Number of past timesteps to consider (default: 20)
        state_dim: Dimension of state vector (default: 12 for quadrotor)
        control_dim: Dimension of control vector (default: 4 for quadrotor)
        hidden_size: LSTM hidden size (default: 128)
        num_lstm_layers: Number of LSTM layers (default: 2)
        fc_hidden_size: Fully connected layer size (default: 256)
        dropout: Dropout rate (default: 0.1)
        bidirectional: Use bidirectional LSTM (default: False)
    """

    # Keep same dimensions as QuadrotorPINN for compatibility
    state_dim = 12
    control_dim = 4

    def __init__(
        self,
        sequence_length: int = 20,
        state_dim: int = 12,
        control_dim: int = 4,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
        fc_hidden_size: int = 256,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.input_dim = state_dim + control_dim
        self.output_dim = state_dim
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional

        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Calculate LSTM output size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        # Fully connected layers for prediction
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, fc_hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, fc_hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, self.output_dim),
        )

        # Temporal consistency head - detects anomalies in sequence patterns
        self.consistency_head = nn.Sequential(
            nn.Linear(lstm_output_size, fc_hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Store hidden state for streaming inference
        self._hidden = None

    def forward(
        self,
        x: torch.Tensor,
        return_anomaly_score: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass: predict next state from sequence of past states.

        Args:
            x: Input tensor of shape:
               - [batch, sequence_length, input_dim] for sequence input
               - [batch, input_dim] for single-step (uses stored hidden state)
            return_anomaly_score: If True, also return temporal consistency score

        Returns:
            [batch, state_dim] predicted next state
            If return_anomaly_score: tuple of (prediction, anomaly_score)
        """
        # Handle single-step input (for compatibility with Predictor)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
            use_hidden = True
        else:
            use_hidden = False
            self._hidden = None  # Reset hidden state for new sequence

        # LSTM forward
        if use_hidden and self._hidden is not None:
            lstm_out, self._hidden = self.lstm(x, self._hidden)
        else:
            lstm_out, self._hidden = self.lstm(x)

        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # [batch, lstm_output_size]

        # Predict next state
        prediction = self.fc(last_output)

        if return_anomaly_score:
            # Compute temporal consistency score (higher = more anomalous)
            anomaly_score = self.consistency_head(last_output)
            return prediction, anomaly_score

        return prediction

    def reset_hidden(self):
        """Reset hidden state (call at start of new trajectory)."""
        self._hidden = None

    def detect_anomaly(
        self,
        sequence: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None,
        prediction_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute anomaly score for a sequence.

        Combines:
        1. Temporal consistency score from LSTM (learned patterns)
        2. Prediction error (if ground_truth provided)

        Args:
            sequence: [batch, sequence_length, input_dim] input sequence
            ground_truth: [batch, state_dim] optional true next state
            prediction_weight: Weight for prediction error vs consistency score

        Returns:
            [batch, 1] anomaly scores (higher = more anomalous)
        """
        prediction, consistency_score = self.forward(sequence, return_anomaly_score=True)

        if ground_truth is not None:
            # Combine prediction error with consistency score
            pred_error = torch.norm(prediction - ground_truth, dim=1, keepdim=True)
            # Normalize prediction error
            pred_error = pred_error / (pred_error.mean() + 1e-8)
            anomaly_score = (
                prediction_weight * pred_error +
                (1 - prediction_weight) * consistency_score
            )
        else:
            anomaly_score = consistency_score

        return anomaly_score

    def physics_loss(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        dt: float = 0.001,
    ) -> torch.Tensor:
        """
        Compute physics-informed loss for sequence.

        For SequencePINN, we enforce temporal smoothness and
        consistency across the sequence.

        Args:
            inputs: [batch, sequence_length, input_dim] input sequence
            outputs: [batch, state_dim] predicted next state
            dt: timestep

        Returns:
            Scalar physics loss
        """
        # Extract states from sequence
        states = inputs[:, :, :self.state_dim]  # [batch, seq_len, state_dim]

        # Temporal smoothness: penalize large jumps in state evolution
        state_diff = states[:, 1:, :] - states[:, :-1, :]  # [batch, seq_len-1, state_dim]

        # Velocity should be bounded by physical limits
        # Position: max 5 m/step, Angles: max 0.5 rad/step, Rates: max 2 rad/s/step
        position_limit = 5.0 * dt
        angle_limit = 0.5 * dt
        rate_limit = 2.0 * dt

        # Apply limits per state type
        position_violation = torch.relu(torch.abs(state_diff[:, :, :3]) - position_limit).mean()
        angle_violation = torch.relu(torch.abs(state_diff[:, :, 3:6]) - angle_limit).mean()
        rate_violation = torch.relu(torch.abs(state_diff[:, :, 6:9]) - rate_limit).mean()

        # Velocity consistency: predicted output should be consistent with sequence trend
        if inputs.shape[1] >= 3:
            # Estimate velocity trend from last 3 states
            recent_states = states[:, -3:, :]
            trend = (recent_states[:, -1, :] - recent_states[:, 0, :]) / 2
            expected_next = states[:, -1, :] + trend * dt

            # Penalize predictions that deviate too far from trend
            trend_violation = torch.norm(outputs - expected_next, dim=1).mean()
        else:
            trend_violation = torch.tensor(0.0, device=inputs.device)

        return position_violation + angle_violation + rate_violation + 0.1 * trend_violation

    def temporal_smoothness_loss(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temporal smoothness loss for anomaly detection training.

        Penalizes sequences with unrealistic state changes.

        Args:
            sequence: [batch, sequence_length, input_dim]

        Returns:
            Scalar smoothness loss
        """
        states = sequence[:, :, :self.state_dim]
        state_diff = states[:, 1:, :] - states[:, :-1, :]

        # Second derivative (acceleration) - should be smooth
        state_accel = state_diff[:, 1:, :] - state_diff[:, :-1, :]

        # Penalize high accelerations
        return torch.norm(state_accel, dim=-1).mean()

    def get_state_names(self) -> List[str]:
        """Return names of state variables."""
        return [
            "x", "y", "z",
            "phi", "theta", "psi",
            "p", "q", "r",
            "vx", "vy", "vz",
        ]

    def get_control_names(self) -> List[str]:
        """Return names of control variables."""
        return ["thrust", "torque_x", "torque_y", "torque_z"]

    def summary(self) -> str:
        """Return a summary of the model architecture."""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        lines = [
            f"SequencePINN",
            f"  Sequence length: {self.sequence_length}",
            f"  State dim: {self.state_dim} ({', '.join(self.get_state_names())})",
            f"  Control dim: {self.control_dim} ({', '.join(self.get_control_names())})",
            f"  LSTM hidden: {self.hidden_size} x {self.num_lstm_layers} layers",
            f"  Bidirectional: {self.bidirectional}",
            f"  Parameters: {n_params:,} ({n_trainable:,} trainable)",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SequencePINN(sequence_length={self.sequence_length}, "
            f"state_dim={self.state_dim}, control_dim={self.control_dim})"
        )


class SequenceAnomalyDetector(nn.Module):
    """
    Specialized anomaly detector using SequencePINN.

    Combines:
    1. Prediction error from sequence model
    2. Learned temporal consistency patterns
    3. Statistical deviation from normal behavior

    This detector is specifically designed for UAV attack detection.
    """

    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 128,
        threshold_percentile: float = 95.0,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.threshold_percentile = threshold_percentile

        # Core sequence model
        self.sequence_pinn = SequencePINN(
            sequence_length=sequence_length,
            hidden_size=hidden_size,
        )

        # Running statistics for threshold calibration
        self.register_buffer("error_mean", torch.tensor(0.0))
        self.register_buffer("error_std", torch.tensor(1.0))
        self.register_buffer("threshold", torch.tensor(1.0))
        self.register_buffer("calibrated", torch.tensor(False))

    def forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute prediction and anomaly score.

        Args:
            sequence: [batch, sequence_length, input_dim]

        Returns:
            prediction: [batch, state_dim]
            anomaly_score: [batch, 1]
        """
        return self.sequence_pinn(sequence, return_anomaly_score=True)

    def calibrate(self, normal_sequences: torch.Tensor):
        """
        Calibrate detection threshold on normal (attack-free) data.

        Args:
            normal_sequences: [n_samples, sequence_length, input_dim] normal data
        """
        self.eval()
        with torch.no_grad():
            _, scores = self.forward(normal_sequences)
            scores = scores.squeeze()

            self.error_mean = scores.mean()
            self.error_std = scores.std()
            self.threshold = torch.quantile(scores, self.threshold_percentile / 100)
            self.calibrated = torch.tensor(True)

    def detect(
        self,
        sequence: torch.Tensor,
        return_scores: bool = False,
    ) -> torch.Tensor:
        """
        Detect anomalies in sequence.

        Args:
            sequence: [batch, sequence_length, input_dim]
            return_scores: If True, return raw scores instead of binary

        Returns:
            [batch] binary detection (1=anomaly) or raw scores
        """
        _, scores = self.forward(sequence)
        scores = scores.squeeze(-1)

        if return_scores:
            return scores

        # Binary detection
        return (scores > self.threshold).float()
