"""
Unit tests for PINN models.

Run: pytest tests/ -v
"""

import numpy as np
import pytest
import torch


class TestDynamicsPINN:
    """Tests for base DynamicsPINN class."""

    def test_pendulum_init(self):
        """Test PendulumPINN initialization."""
        from scripts.pinn_base import PendulumPINN

        model = PendulumPINN()
        assert model.state_dim == 2
        assert model.control_dim == 1
        assert model.input_dim == 3
        assert model.output_dim == 2

    def test_pendulum_forward(self):
        """Test PendulumPINN forward pass."""
        from scripts.pinn_base import PendulumPINN

        model = PendulumPINN()
        x = torch.randn(32, 3)  # batch of state + control
        y = model(x)

        assert y.shape == (32, 2)
        assert not torch.isnan(y).any()

    def test_pendulum_physics_loss(self):
        """Test PendulumPINN physics loss computation."""
        from scripts.pinn_base import PendulumPINN

        model = PendulumPINN()
        x = torch.randn(32, 3)
        y = model(x)
        loss = model.physics_loss(x, y, dt=0.01)

        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0  # non-negative
        assert not torch.isnan(loss)

    def test_cartpole_init(self):
        """Test CartPolePINN initialization."""
        from scripts.pinn_base import CartPolePINN

        model = CartPolePINN()
        assert model.state_dim == 4
        assert model.control_dim == 1

    def test_cartpole_forward(self):
        """Test CartPolePINN forward pass."""
        from scripts.pinn_base import CartPolePINN

        model = CartPolePINN()
        x = torch.randn(16, 5)
        y = model(x)

        assert y.shape == (16, 4)

    def test_learnable_params(self):
        """Test learnable physical parameters."""
        from scripts.pinn_base import PendulumPINN

        model = PendulumPINN()

        assert "g" in model.params
        assert "L" in model.params
        assert "m" in model.params

        # Check they're trainable
        assert model.params["g"].requires_grad
        assert model.params["L"].requires_grad

    def test_param_bounds(self):
        """Test parameter clamping."""
        from scripts.pinn_base import PendulumPINN

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
        from scripts.pinn_model import QuadrotorPINN

        model = QuadrotorPINN()
        assert model.state_dim == 12
        assert model.control_dim == 4

    def test_forward(self):
        """Test QuadrotorPINN forward pass."""
        from scripts.pinn_model import QuadrotorPINN

        model = QuadrotorPINN()
        x = torch.randn(8, 16)  # 12 states + 4 controls
        y = model(x)

        assert y.shape == (8, 12)

    def test_physics_loss(self):
        """Test QuadrotorPINN physics loss."""
        from scripts.pinn_model import QuadrotorPINN

        model = QuadrotorPINN()
        x = torch.randn(8, 16)
        y = model(x)
        loss = model.physics_loss(x, y, dt=0.001)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_state_names(self):
        """Test state/control name introspection."""
        from scripts.pinn_model import QuadrotorPINN

        model = QuadrotorPINN()

        states = model.get_state_names()
        controls = model.get_control_names()

        assert len(states) == 12
        assert len(controls) == 4
        assert "x" in states
        assert "thrust" in controls

    def test_summary(self):
        """Test model summary."""
        from scripts.pinn_model import QuadrotorPINN

        model = QuadrotorPINN()
        summary = model.summary()

        assert "QuadrotorPINN" in summary
        assert "State dim: 12" in summary


class TestRollout:
    """Tests for autoregressive rollout."""

    def test_rollout_shape(self):
        """Test rollout output shape."""
        from scripts.pinn_base import PendulumPINN

        model = PendulumPINN()
        initial_state = torch.randn(2)
        controls = torch.randn(50, 1)

        trajectory = model.rollout(initial_state, controls)

        assert trajectory.shape == (50, 2)

    def test_rollout_batched(self):
        """Test batched rollout."""
        from scripts.pinn_base import PendulumPINN

        model = PendulumPINN()
        initial_state = torch.randn(4, 2)  # batch of 4
        controls = torch.randn(4, 50, 1)

        trajectory = model.rollout(initial_state, controls)

        assert trajectory.shape == (4, 50, 2)


class TestGradients:
    """Tests for gradient flow."""

    def test_backward_pass(self):
        """Test gradients flow through model."""
        from scripts.pinn_base import PendulumPINN

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
        from scripts.pinn_base import PendulumPINN

        model = PendulumPINN()
        x = torch.randn(32, 3)
        y = model(x)
        loss = model.physics_loss(x, y, dt=0.01)
        loss.backward()

        # Check learnable params have gradients
        assert model.params["g"].grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
