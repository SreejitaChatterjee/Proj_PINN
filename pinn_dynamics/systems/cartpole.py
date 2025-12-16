"""
Physics-Informed Neural Network for Cart-Pole (Inverted Pendulum) Dynamics.

State vector (4):
    - x: Cart position (meters)
    - x_dot: Cart velocity (m/s)
    - theta: Pole angle from vertical (radians)
    - theta_dot: Pole angular velocity (rad/s)

Control vector (1):
    - F: Force applied to cart (N)

This is a classic nonlinear control benchmark with coupled dynamics.
"""

import torch
from .base import DynamicsPINN


class CartPolePINN(DynamicsPINN):
    """
    PINN for cart-pole (inverted pendulum) dynamics.

    Implements the standard cart-pole equations with coupled nonlinear dynamics
    between cart position and pole angle.

    Args:
        hidden_size: Width of hidden layers (default: 128)
        num_layers: Number of hidden layers (default: 4)

    Example:
        model = CartPolePINN()
        # Cart at origin, pole slightly tilted
        state = torch.tensor([[0.0, 0.0, 0.05, 0.0]])
        control = torch.tensor([[1.0]])  # 1N force
        next_state = model(torch.cat([state, control], dim=-1))
    """

    def __init__(self, hidden_size: int = 128, num_layers: int = 4):
        super().__init__(
            state_dim=4,
            control_dim=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learnable_params={
                "g": 9.81,
                "mc": 1.0,  # cart mass (kg)
                "mp": 0.1,  # pole mass (kg)
                "L": 0.5,   # pole half-length (m)
            },
        )

        self.set_param_bounds({
            "g": (9.0, 10.5),
            "mc": (0.5, 2.0),
            "mp": (0.05, 0.5),
            "L": (0.25, 1.0),
        })

    def physics_loss(self, inputs: torch.Tensor, outputs: torch.Tensor, dt: float = 0.02) -> torch.Tensor:
        """Enforce cart-pole dynamics with coupled equations of motion."""
        # Current state
        x, x_dot, theta, theta_dot = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
        F = inputs[:, 4]

        # Predicted next state
        x_next, x_dot_next = outputs[:, 0], outputs[:, 1]
        theta_next, theta_dot_next = outputs[:, 2], outputs[:, 3]

        # Physics parameters
        g = self.params["g"]
        mc, mp, L = self.params["mc"], self.params["mp"], self.params["L"]

        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        total_mass = mc + mp

        # Cart-pole equations of motion
        temp = (F + mp * L * theta_dot**2 * sin_t) / total_mass
        theta_ddot = (g * sin_t - cos_t * temp) / (L * (4 / 3 - mp * cos_t**2 / total_mass))
        x_ddot = temp - mp * L * theta_ddot * cos_t / total_mass

        # Euler integration
        x_pred = x + x_dot * dt
        x_dot_pred = x_dot + x_ddot * dt
        theta_pred = theta + theta_dot * dt
        theta_dot_pred = theta_dot + theta_ddot * dt

        # Loss
        loss = (
            (x_next - x_pred) ** 2
            + (x_dot_next - x_dot_pred) ** 2
            + (theta_next - theta_pred) ** 2
            + (theta_dot_next - theta_dot_pred) ** 2
        ).mean()

        return loss

    def get_state_names(self):
        return ["x", "x_dot", "theta", "theta_dot"]

    def get_control_names(self):
        return ["F"]
