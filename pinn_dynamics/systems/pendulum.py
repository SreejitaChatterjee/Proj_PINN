"""
Physics-Informed Neural Network for Simple Pendulum Dynamics.

State vector (2):
    - theta: Angle from vertical (radians)
    - theta_dot: Angular velocity (rad/s)

Control vector (1):
    - tau: Applied torque (N*m)

Dynamics: theta_ddot = -g/L * sin(theta) + tau / (m * L^2)
"""

import torch
from .base import DynamicsPINN


class PendulumPINN(DynamicsPINN):
    """
    PINN for simple pendulum dynamics.

    A simple example system demonstrating how to implement physics constraints
    for a 2-state, 1-control system.

    Args:
        hidden_size: Width of hidden layers (default: 64)
        num_layers: Number of hidden layers (default: 3)

    Example:
        model = PendulumPINN()
        state = torch.tensor([[0.1, 0.0]])  # theta=0.1 rad, theta_dot=0
        control = torch.tensor([[0.0]])  # no torque
        next_state = model(torch.cat([state, control], dim=-1))
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 3):
        super().__init__(
            state_dim=2,
            control_dim=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learnable_params={
                "g": 9.81,
                "L": 1.0,
                "m": 1.0,
            },
        )

        self.set_param_bounds({
            "g": (9.0, 10.5),
            "L": (0.5, 2.0),
            "m": (0.5, 2.0),
        })

    def physics_loss(self, inputs: torch.Tensor, outputs: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Enforce pendulum dynamics: theta_ddot = -g/L * sin(theta) + tau / (m * L^2)
        """
        # Current state
        theta = inputs[:, 0]
        theta_dot = inputs[:, 1]
        tau = inputs[:, 2]

        # Predicted next state
        theta_next = outputs[:, 0]
        theta_dot_next = outputs[:, 1]

        # Physics prediction
        g, L, m = self.params["g"], self.params["L"], self.params["m"]

        theta_ddot = -g / L * torch.sin(theta) + tau / (m * L**2)

        # Euler integration
        theta_pred = theta + theta_dot * dt
        theta_dot_pred = theta_dot + theta_ddot * dt

        # Loss
        loss = (
            (theta_next - theta_pred) ** 2 + (theta_dot_next - theta_dot_pred) ** 2
        ).mean()

        return loss

    def get_state_names(self):
        return ["theta", "theta_dot"]

    def get_control_names(self):
        return ["tau"]
