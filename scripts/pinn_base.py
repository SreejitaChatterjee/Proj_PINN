"""
Base Physics-Informed Neural Network for Dynamics Learning

This module provides the abstract base class for building PINNs for
arbitrary dynamical systems. Subclass DynamicsPINN and implement
physics_loss() to define your system's governing equations.

Example:
    class PendulumPINN(DynamicsPINN):
        def physics_loss(self, inputs, outputs, dt):
            # Implement pendulum dynamics: theta_ddot = -g/L * sin(theta)
            ...
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class DynamicsPINN(nn.Module, ABC):
    """
    Abstract base class for Physics-Informed Neural Networks.

    Subclass this and implement:
    - physics_loss(): Your system's governing equations
    - Optionally: get_state_names(), get_control_names() for introspection

    Args:
        state_dim: Dimension of state vector
        control_dim: Dimension of control vector
        hidden_size: Width of hidden layers (default: 256)
        num_layers: Number of hidden layers (default: 5)
        dropout: Dropout rate (default: 0.1)
        activation: Activation function (default: 'tanh')
        learnable_params: Dict of learnable physical parameters with initial values
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_size: int = 256,
        num_layers: int = 5,
        dropout: float = 0.1,
        activation: str = 'tanh',
        learnable_params: Optional[Dict[str, float]] = None
    ):
        super().__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.input_dim = state_dim + control_dim
        self.output_dim = state_dim

        # Build network
        activation_fn = self._get_activation(activation)

        layers = [nn.Linear(self.input_dim, hidden_size), activation_fn(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), activation_fn(), nn.Dropout(dropout)])
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
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'elu': nn.ELU,
        }
        return activations.get(name.lower(), nn.Tanh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict next state from current state + control."""
        return self.network(x)

    def set_param_bounds(self, bounds: Dict[str, Tuple[float, float]]):
        """Set bounds for learnable parameters."""
        self._param_bounds = bounds

    def constrain_parameters(self):
        """Clamp learnable parameters to their bounds."""
        with torch.no_grad():
            for name, (lo, hi) in self._param_bounds.items():
                if name in self.params:
                    self.params[name].clamp_(lo, hi)

    @abstractmethod
    def physics_loss(self, inputs: torch.Tensor, outputs: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
        """
        Compute physics-informed loss based on governing equations.

        This is the core method that enforces physics. Implement the dynamics
        equations for your system here.

        Args:
            inputs: [batch, state_dim + control_dim] current state + control
            outputs: [batch, state_dim] predicted next state
            dt: timestep

        Returns:
            Scalar loss tensor measuring physics violation
        """
        pass

    def rollout(self, initial_state: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive rollout: predict multiple steps into the future.

        Args:
            initial_state: [state_dim] or [batch, state_dim] initial state
            controls: [n_steps, control_dim] or [batch, n_steps, control_dim] control sequence

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
                predictions.append(state)

        result = torch.stack(predictions, dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def get_state_names(self) -> List[str]:
        """Return names of state variables. Override in subclass."""
        return [f'x{i}' for i in range(self.state_dim)]

    def get_control_names(self) -> List[str]:
        """Return names of control variables. Override in subclass."""
        return [f'u{i}' for i in range(self.control_dim)]

    def summary(self) -> str:
        """Return a summary of the model."""
        n_params = sum(p.numel() for p in self.parameters())
        n_learnable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        lines = [
            f"{self.__class__.__name__}",
            f"  State dim: {self.state_dim}",
            f"  Control dim: {self.control_dim}",
            f"  Parameters: {n_params:,} ({n_learnable:,} trainable)",
        ]

        if self.params:
            lines.append("  Learnable physics params:")
            for name, param in self.params.items():
                lines.append(f"    {name}: {param.item():.6g}")

        return '\n'.join(lines)


# =============================================================================
# Example: Simple Pendulum PINN
# =============================================================================

class PendulumPINN(DynamicsPINN):
    """
    PINN for simple pendulum dynamics.

    State: [theta, theta_dot] (angle, angular velocity)
    Control: [tau] (torque)

    Dynamics: theta_ddot = -g/L * sin(theta) + tau / (m * L^2)
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 3):
        super().__init__(
            state_dim=2,
            control_dim=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learnable_params={
                'g': 9.81,
                'L': 1.0,
                'm': 1.0,
            }
        )

        self.set_param_bounds({
            'g': (9.0, 10.5),
            'L': (0.5, 2.0),
            'm': (0.5, 2.0),
        })

    def physics_loss(self, inputs: torch.Tensor, outputs: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """Enforce pendulum dynamics."""
        # Current state
        theta = inputs[:, 0]
        theta_dot = inputs[:, 1]
        tau = inputs[:, 2]

        # Predicted next state
        theta_next = outputs[:, 0]
        theta_dot_next = outputs[:, 1]

        # Physics prediction
        g, L, m = self.params['g'], self.params['L'], self.params['m']

        # theta_ddot = -g/L * sin(theta) + tau / (m * L^2)
        theta_ddot = -g / L * torch.sin(theta) + tau / (m * L ** 2)

        # Euler integration
        theta_pred = theta + theta_dot * dt
        theta_dot_pred = theta_dot + theta_ddot * dt

        # Loss: predicted vs physics
        loss = (
            (theta_next - theta_pred) ** 2 +
            (theta_dot_next - theta_dot_pred) ** 2
        ).mean()

        return loss

    def get_state_names(self) -> List[str]:
        return ['theta', 'theta_dot']

    def get_control_names(self) -> List[str]:
        return ['tau']


# =============================================================================
# Example: Cart-Pole PINN
# =============================================================================

class CartPolePINN(DynamicsPINN):
    """
    PINN for cart-pole (inverted pendulum) dynamics.

    State: [x, x_dot, theta, theta_dot]
    Control: [F] (force on cart)
    """

    def __init__(self, hidden_size: int = 128, num_layers: int = 4):
        super().__init__(
            state_dim=4,
            control_dim=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learnable_params={
                'g': 9.81,
                'mc': 1.0,    # cart mass
                'mp': 0.1,    # pole mass
                'L': 0.5,     # pole half-length
            }
        )

    def physics_loss(self, inputs: torch.Tensor, outputs: torch.Tensor, dt: float = 0.02) -> torch.Tensor:
        """Enforce cart-pole dynamics."""
        # Current state
        x, x_dot, theta, theta_dot = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
        F = inputs[:, 4]

        # Predicted next state
        x_next, x_dot_next = outputs[:, 0], outputs[:, 1]
        theta_next, theta_dot_next = outputs[:, 2], outputs[:, 3]

        # Physics
        g = self.params['g']
        mc, mp, L = self.params['mc'], self.params['mp'], self.params['L']

        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        total_mass = mc + mp

        # Cart-pole equations of motion
        temp = (F + mp * L * theta_dot ** 2 * sin_t) / total_mass
        theta_ddot = (g * sin_t - cos_t * temp) / (L * (4/3 - mp * cos_t ** 2 / total_mass))
        x_ddot = temp - mp * L * theta_ddot * cos_t / total_mass

        # Euler integration
        x_pred = x + x_dot * dt
        x_dot_pred = x_dot + x_ddot * dt
        theta_pred = theta + theta_dot * dt
        theta_dot_pred = theta_dot + theta_ddot * dt

        # Loss
        loss = (
            (x_next - x_pred) ** 2 +
            (x_dot_next - x_dot_pred) ** 2 +
            (theta_next - theta_pred) ** 2 +
            (theta_dot_next - theta_dot_pred) ** 2
        ).mean()

        return loss

    def get_state_names(self) -> List[str]:
        return ['x', 'x_dot', 'theta', 'theta_dot']

    def get_control_names(self) -> List[str]:
        return ['F']
