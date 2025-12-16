"""
Base class for Physics-Informed Neural Networks.

This module provides the abstract base class for building PINNs for
arbitrary dynamical systems. Subclass DynamicsPINN and implement
physics_loss() to define your system's governing equations.

Example:
    class MyRobot(DynamicsPINN):
        def __init__(self):
            super().__init__(
                state_dim=6,
                control_dim=2,
                learnable_params={'mass': 1.0, 'inertia': 0.1}
            )

        def physics_loss(self, inputs, outputs, dt=0.01):
            # Extract states and controls
            x, v = inputs[:, :3], inputs[:, 3:6]
            force = inputs[:, 6:8]

            # Physics prediction
            accel = force / self.params['mass']
            x_next_pred = x + v * dt
            v_next_pred = v + accel * dt

            # Compare with neural network prediction
            x_next, v_next = outputs[:, :3], outputs[:, 3:6]
            return ((x_next - x_next_pred)**2 + (v_next - v_next_pred)**2).mean()
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class DynamicsPINN(nn.Module, ABC):
    """
    Abstract base class for Physics-Informed Neural Networks.

    Subclass this and implement:
        - physics_loss(): Your system's governing equations (required)
        - get_state_names(): Names of state variables (optional)
        - get_control_names(): Names of control variables (optional)

    Args:
        state_dim: Dimension of state vector
        control_dim: Dimension of control vector
        hidden_size: Width of hidden layers (default: 256)
        num_layers: Number of hidden layers (default: 5)
        dropout: Dropout rate (default: 0.1)
        activation: Activation function: 'tanh', 'relu', 'gelu', 'silu', 'elu' (default: 'tanh')
        learnable_params: Dict of learnable physical parameters with initial values
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_size: int = 256,
        num_layers: int = 5,
        dropout: float = 0.1,
        activation: str = "tanh",
        learnable_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.input_dim = state_dim + control_dim
        self.output_dim = state_dim

        # Build network
        activation_fn = self._get_activation(activation)

        layers = [
            nn.Linear(self.input_dim, hidden_size),
            activation_fn(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                activation_fn(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_size, self.output_dim))

        self.network = nn.Sequential(*layers)

        # Learnable physical parameters
        self.params = nn.ParameterDict()
        self._param_bounds = {}
        if learnable_params:
            for name, value in learnable_params.items():
                self.params[name] = nn.Parameter(torch.tensor(float(value)))

    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "elu": nn.ELU,
        }
        return activations.get(name.lower(), nn.Tanh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict next state from current state + control.

        Args:
            x: [batch, state_dim + control_dim] input tensor

        Returns:
            [batch, state_dim] predicted next state
        """
        return self.network(x)

    def set_param_bounds(self, bounds: Dict[str, Tuple[float, float]]):
        """
        Set bounds for learnable parameters.

        Args:
            bounds: Dict mapping parameter name to (min, max) tuple
        """
        self._param_bounds = bounds

    def constrain_parameters(self):
        """Clamp learnable parameters to their bounds (call after optimizer step)."""
        with torch.no_grad():
            for name, (lo, hi) in self._param_bounds.items():
                if name in self.params:
                    self.params[name].clamp_(lo, hi)

    @abstractmethod
    def physics_loss(
        self, inputs: torch.Tensor, outputs: torch.Tensor, dt: float = 0.001
    ) -> torch.Tensor:
        """
        Compute physics-informed loss based on governing equations.

        This is the core method that enforces physics. Implement the dynamics
        equations for your system here.

        Args:
            inputs: [batch, state_dim + control_dim] current state + control
            outputs: [batch, state_dim] predicted next state
            dt: timestep (seconds)

        Returns:
            Scalar loss tensor measuring physics violation
        """
        pass

    def rollout(
        self,
        initial_state: torch.Tensor,
        controls: torch.Tensor,
        return_intermediate: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive rollout: predict multiple steps into the future.

        Args:
            initial_state: [state_dim] or [batch, state_dim] initial state
            controls: [n_steps, control_dim] or [batch, n_steps, control_dim] control sequence
            return_intermediate: If True, return all states; if False, return only final

        Returns:
            [n_steps, state_dim] or [batch, n_steps, state_dim] predicted trajectory
        """
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
            controls = controls.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = initial_state.shape[0]
        n_steps = controls.shape[1]

        predictions = []
        state = initial_state

        with torch.no_grad():
            for i in range(n_steps):
                inp = torch.cat([state, controls[:, i]], dim=-1)
                state = self.forward(inp)
                if return_intermediate:
                    predictions.append(state)

        if return_intermediate:
            result = torch.stack(predictions, dim=1)
        else:
            result = state.unsqueeze(1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def get_state_names(self) -> List[str]:
        """Return names of state variables. Override in subclass."""
        return [f"x{i}" for i in range(self.state_dim)]

    def get_control_names(self) -> List[str]:
        """Return names of control variables. Override in subclass."""
        return [f"u{i}" for i in range(self.control_dim)]

    def summary(self) -> str:
        """Return a summary of the model architecture and parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        n_learnable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        lines = [
            f"{self.__class__.__name__}",
            f"  State dim: {self.state_dim} ({', '.join(self.get_state_names())})",
            f"  Control dim: {self.control_dim} ({', '.join(self.get_control_names())})",
            f"  Parameters: {n_params:,} ({n_learnable:,} trainable)",
        ]

        if self.params:
            lines.append("  Learnable physics params:")
            for name, param in self.params.items():
                bounds_str = ""
                if name in self._param_bounds:
                    lo, hi = self._param_bounds[name]
                    bounds_str = f" [{lo:.2e}, {hi:.2e}]"
                lines.append(f"    {name}: {param.item():.6g}{bounds_str}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(state_dim={self.state_dim}, control_dim={self.control_dim})"
