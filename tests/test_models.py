"""
Unit tests for PINN models.

Run: pytest tests/ -v
"""

import numpy as np
import pytest
import torch

from pinn_dynamics import QuadrotorPINN
from pinn_dynamics.systems import CartPolePINN, PendulumPINN


class TestDynamicsPINN:
    """Tests for base DynamicsPINN class."""

    def test_pendulum_init(self):
        """Test PendulumPINN initialization."""
        model = PendulumPINN()
        assert model.state_dim == 2
        assert model.control_dim == 1
        assert model.input_dim == 3
        assert model.output_dim == 2

    def test_pendulum_forward(self):
        """Test PendulumPINN forward pass."""
        model = PendulumPINN()
        x = torch.randn(32, 3)  # batch of state + control
        y = model(x)

        assert y.shape == (32, 2)
        assert not torch.isnan(y).any()

    def test_pendulum_physics_loss(self):
        """Test PendulumPINN physics loss computation."""
        model = PendulumPINN()
        x = torch.randn(32, 3)
        y = model(x)
        loss = model.physics_loss(x, y, dt=0.01)

        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0  # non-negative
        assert not torch.isnan(loss)

    def test_cartpole_init(self):
        """Test CartPolePINN initialization."""
        model = CartPolePINN()
        assert model.state_dim == 4
        assert model.control_dim == 1

    def test_cartpole_forward(self):
        """Test CartPolePINN forward pass."""
        model = CartPolePINN()
        x = torch.randn(16, 5)
        y = model(x)

        assert y.shape == (16, 4)

    def test_learnable_params(self):
        """Test learnable physical parameters."""
        model = PendulumPINN()

        assert "g" in model.params
        assert "L" in model.params
        assert "m" in model.params

        # Check they're trainable
        assert model.params["g"].requires_grad
        assert model.params["L"].requires_grad

    def test_param_bounds(self):
        """Test parameter clamping."""
        model = PendulumPINN()

        # Set param outside bounds
        with torch.no_grad():
            model.params["g"].fill_(100.0)

        model.constrain_parameters()

        # Should be clamped to upper bound (10.5)
        assert model.params["g"].item() <= 10.5


class TestQuadrotorPINN:
    """Tests for QuadrotorPINN."""

    def test_init(self):
        """Test QuadrotorPINN initialization."""
        model = QuadrotorPINN()
        assert model.state_dim == 12
        assert model.control_dim == 4

    def test_forward(self):
        """Test QuadrotorPINN forward pass."""
        model = QuadrotorPINN()
        x = torch.randn(8, 16)  # 12 states + 4 controls
        y = model(x)

        assert y.shape == (8, 12)

    def test_physics_loss(self):
        """Test QuadrotorPINN physics loss."""
        model = QuadrotorPINN()
        x = torch.randn(8, 16)
        y = model(x)
        loss = model.physics_loss(x, y, dt=0.001)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_state_names(self):
        """Test state/control name introspection."""
        model = QuadrotorPINN()

        states = model.get_state_names()
        controls = model.get_control_names()

        assert len(states) == 12
        assert len(controls) == 4
        assert "x" in states
        assert "thrust" in controls

    def test_summary(self):
        """Test model summary."""
        model = QuadrotorPINN()
        summary = model.summary()

        assert "QuadrotorPINN" in summary
        assert "State dim: 12" in summary


class TestRollout:
    """Tests for autoregressive rollout."""

    def test_rollout_shape(self):
        """Test rollout output shape."""
        model = PendulumPINN()
        initial_state = torch.randn(2)
        controls = torch.randn(50, 1)

        trajectory = model.rollout(initial_state, controls)

        assert trajectory.shape == (50, 2)

    def test_rollout_batched(self):
        """Test batched rollout."""
        model = PendulumPINN()
        initial_state = torch.randn(4, 2)  # batch of 4
        controls = torch.randn(4, 50, 1)

        trajectory = model.rollout(initial_state, controls)

        assert trajectory.shape == (4, 50, 2)


class TestGradients:
    """Tests for gradient flow."""

    def test_backward_pass(self):
        """Test gradients flow through model."""
        model = PendulumPINN()
        x = torch.randn(32, 3, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check network params have gradients (not learnable physics params)
        for name, param in model.network.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_physics_loss_gradients(self):
        """Test gradients flow through physics loss."""
        model = PendulumPINN()
        x = torch.randn(32, 3)
        y = model(x)
        loss = model.physics_loss(x, y, dt=0.01)
        loss.backward()

        # Check learnable params have gradients
        assert model.params["g"].grad is not None


class TestSequencePINN:
    """Tests for SequencePINN temporal model."""

    def test_init(self):
        """Test SequencePINN initialization."""
        from pinn_dynamics import SequencePINN

        model = SequencePINN(sequence_length=20)
        assert model.state_dim == 12
        assert model.control_dim == 4
        assert model.sequence_length == 20

    def test_forward_sequence(self):
        """Test SequencePINN forward pass with sequence input."""
        from pinn_dynamics import SequencePINN

        model = SequencePINN(sequence_length=20)
        # Input: [batch, sequence_length, state_dim + control_dim]
        x = torch.randn(8, 20, 16)
        y = model(x)

        # Output: [batch, state_dim]
        assert y.shape == (8, 12)
        assert not torch.isnan(y).any()

    def test_forward_single_step(self):
        """Test SequencePINN with single-step input (compatibility mode)."""
        from pinn_dynamics import SequencePINN

        model = SequencePINN(sequence_length=20)
        # Single-step input: [batch, state_dim + control_dim]
        x = torch.randn(8, 16)
        y = model(x)

        # Output: [batch, state_dim]
        assert y.shape == (8, 12)

    def test_anomaly_score(self):
        """Test anomaly score computation."""
        from pinn_dynamics import SequencePINN

        model = SequencePINN(sequence_length=20)
        x = torch.randn(8, 20, 16)
        y, score = model(x, return_anomaly_score=True)

        assert y.shape == (8, 12)
        assert score.shape == (8, 1)
        # Scores should be between 0 and 1 (sigmoid output)
        assert (score >= 0).all() and (score <= 1).all()

    def test_detect_anomaly(self):
        """Test detect_anomaly method."""
        from pinn_dynamics import SequencePINN

        model = SequencePINN(sequence_length=20)
        x = torch.randn(8, 20, 16)
        ground_truth = torch.randn(8, 12)

        score = model.detect_anomaly(x, ground_truth)
        assert score.shape == (8, 1)

    def test_physics_loss(self):
        """Test SequencePINN physics loss."""
        from pinn_dynamics import SequencePINN

        model = SequencePINN(sequence_length=20)
        x = torch.randn(8, 20, 16)
        y = model(x)
        loss = model.physics_loss(x, y, dt=0.001)

        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)

    def test_temporal_smoothness_loss(self):
        """Test temporal smoothness loss."""
        from pinn_dynamics import SequencePINN

        model = SequencePINN(sequence_length=20)
        x = torch.randn(8, 20, 16)
        loss = model.temporal_smoothness_loss(x)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_summary(self):
        """Test model summary."""
        from pinn_dynamics import SequencePINN

        model = SequencePINN(sequence_length=20)
        summary = model.summary()

        assert "SequencePINN" in summary
        assert "Sequence length: 20" in summary
        assert "LSTM" in summary

    def test_hidden_state_reset(self):
        """Test hidden state reset for new trajectories."""
        from pinn_dynamics import SequencePINN

        model = SequencePINN(sequence_length=20)

        # Process a sequence
        x = torch.randn(1, 20, 16)
        _ = model(x)

        # Hidden state should be set
        assert model._hidden is not None

        # Reset
        model.reset_hidden()
        assert model._hidden is None


class TestSequenceAnomalyDetector:
    """Tests for SequenceAnomalyDetector."""

    def test_init(self):
        """Test SequenceAnomalyDetector initialization."""
        from pinn_dynamics import SequenceAnomalyDetector

        detector = SequenceAnomalyDetector(sequence_length=20)
        assert detector.sequence_length == 20

    def test_forward(self):
        """Test forward pass returns prediction and score."""
        from pinn_dynamics import SequenceAnomalyDetector

        detector = SequenceAnomalyDetector(sequence_length=20)
        x = torch.randn(8, 20, 16)
        pred, score = detector(x)

        assert pred.shape == (8, 12)
        assert score.shape == (8, 1)

    def test_calibrate(self):
        """Test threshold calibration on normal data."""
        from pinn_dynamics import SequenceAnomalyDetector

        detector = SequenceAnomalyDetector(sequence_length=20, threshold_percentile=95.0)

        # Generate fake normal data
        normal_data = torch.randn(100, 20, 16)
        detector.calibrate(normal_data)

        assert detector.calibrated.item() == True
        assert detector.threshold.item() > 0

    def test_detect(self):
        """Test anomaly detection."""
        from pinn_dynamics import SequenceAnomalyDetector

        detector = SequenceAnomalyDetector(sequence_length=20)

        # Calibrate first
        normal_data = torch.randn(100, 20, 16)
        detector.calibrate(normal_data)

        # Test detection
        test_data = torch.randn(8, 20, 16)
        detections = detector.detect(test_data)

        assert detections.shape == (8,)
        assert set(detections.unique().tolist()).issubset({0.0, 1.0})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
