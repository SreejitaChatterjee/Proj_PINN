"""
Custom System: Define your own dynamical system.

This example shows how to create a PINN for a custom system by
subclassing DynamicsPINN and implementing physics_loss().
"""

import torch
import numpy as np
from pinn_dynamics import DynamicsPINN


class DoublePendulumPINN(DynamicsPINN):
    """
    PINN for double pendulum dynamics.

    State (4): [theta1, theta1_dot, theta2, theta2_dot]
    Control (2): [tau1, tau2] - torques at each joint

    This is a chaotic system - perfect for testing learned dynamics!
    """

    def __init__(self, hidden_size: int = 128, num_layers: int = 4):
        super().__init__(
            state_dim=4,
            control_dim=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learnable_params={
                "m1": 1.0,  # Mass of pendulum 1
                "m2": 1.0,  # Mass of pendulum 2
                "L1": 1.0,  # Length of pendulum 1
                "L2": 1.0,  # Length of pendulum 2
                "g": 9.81,  # Gravity
            },
        )

        # Set reasonable parameter bounds
        self.set_param_bounds({
            "m1": (0.5, 2.0),
            "m2": (0.5, 2.0),
            "L1": (0.5, 2.0),
            "L2": (0.5, 2.0),
            "g": (9.0, 10.5),
        })

    def physics_loss(self, inputs, outputs, dt=0.01):
        """
        Enforce double pendulum dynamics.

        The equations are derived from Lagrangian mechanics.
        """
        # Extract current state
        theta1, theta1_dot, theta2, theta2_dot = inputs[:, :4].T
        tau1, tau2 = inputs[:, 4:6].T

        # Predicted next state
        theta1_next, theta1_dot_next, theta2_next, theta2_dot_next = outputs[:, :4].T

        # Parameters
        m1, m2 = self.params["m1"], self.params["m2"]
        L1, L2 = self.params["L1"], self.params["L2"]
        g = self.params["g"]

        # Intermediate calculations
        delta = theta2 - theta1
        sin_d, cos_d = torch.sin(delta), torch.cos(delta)
        sin1, sin2 = torch.sin(theta1), torch.sin(theta2)

        # Mass matrix elements
        M11 = (m1 + m2) * L1
        M12 = m2 * L2 * cos_d
        M21 = L1 * cos_d
        M22 = L2

        # Right-hand side (Coriolis + gravity + control)
        f1 = m2 * L2 * theta2_dot**2 * sin_d - (m1 + m2) * g * sin1 + tau1 / L1
        f2 = -L1 * theta1_dot**2 * sin_d - g * sin2 + tau2 / (m2 * L2)

        # Solve for accelerations (2x2 system)
        det = M11 * M22 - M12 * M21
        theta1_ddot = (M22 * f1 - M12 * f2) / det
        theta2_ddot = (M11 * f2 - M21 * f1) / det

        # Euler integration for physics prediction
        theta1_pred = theta1 + theta1_dot * dt
        theta1_dot_pred = theta1_dot + theta1_ddot * dt
        theta2_pred = theta2 + theta2_dot * dt
        theta2_dot_pred = theta2_dot + theta2_ddot * dt

        # Physics loss
        loss = (
            (theta1_next - theta1_pred) ** 2
            + (theta1_dot_next - theta1_dot_pred) ** 2
            + (theta2_next - theta2_pred) ** 2
            + (theta2_dot_next - theta2_dot_pred) ** 2
        ).mean()

        return loss

    def get_state_names(self):
        return ["theta1", "theta1_dot", "theta2", "theta2_dot"]

    def get_control_names(self):
        return ["tau1", "tau2"]


def main():
    print("Creating Double Pendulum PINN...")
    model = DoublePendulumPINN()

    print("\nModel Summary:")
    print(model.summary())

    # Test forward pass
    print("\nTest forward pass:")
    batch_size = 32
    state = torch.randn(batch_size, 4)  # Random states
    control = torch.randn(batch_size, 2)  # Random controls
    inputs = torch.cat([state, control], dim=-1)

    with torch.no_grad():
        next_state = model(inputs)

    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {next_state.shape}")

    # Test physics loss
    print("\nTest physics loss:")
    physics_loss = model.physics_loss(inputs, next_state)
    print(f"  Physics loss: {physics_loss.item():.4f}")

    # Test rollout
    print("\nTest rollout (50 steps):")
    initial_state = torch.tensor([0.5, 0.0, 0.5, 0.0])  # Both pendulums at 0.5 rad
    controls = torch.zeros(50, 2)  # No control (free swing)

    trajectory = model.rollout(initial_state, controls)
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Final state: {trajectory[-1].numpy()}")


if __name__ == "__main__":
    main()
