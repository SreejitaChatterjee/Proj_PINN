"""
Minimal improvement to baseline PINN - ONLY add residual connections

Changes from baseline:
1. Add residual connections (proven to help with gradient flow)
2. Keep EVERYTHING else the same (architecture, physics, hyperparameters)

This conservative approach ensures we don't break what works.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Simple residual block"""
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size),
            nn.Tanh(),
        )

    def forward(self, x):
        return x + 0.1 * self.net(x)  # Small residual weight for stability


class QuadrotorPINNMinimal(nn.Module):
    """
    Baseline architecture + residual connections only
    """
    def __init__(self, input_size=12, output_size=8, hidden_size=250):
        super().__init__()

        # Baseline: 5-layer MLP with tanh
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Add 3 residual blocks (same depth as baseline 5 layers)
        self.residual1 = ResidualBlock(hidden_size)
        self.residual2 = ResidualBlock(hidden_size)
        self.residual3 = ResidualBlock(hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_size)

        # Learnable physics parameters (same as baseline)
        self.params = nn.ParameterDict({
            'Jxx': nn.Parameter(torch.tensor(6.86e-5)),
            'Jyy': nn.Parameter(torch.tensor(9.20e-5)),
            'Jzz': nn.Parameter(torch.tensor(1.366e-4)),
            'm': nn.Parameter(torch.tensor(0.068))
        })

        self.g = 9.81

    def forward(self, x):
        """Forward pass through network"""
        h = torch.tanh(self.input_layer(x))
        h = self.residual1(h)
        h = self.residual2(h)
        h = self.residual3(h)
        return self.output_layer(h)

    def physics_loss(self, inputs, outputs, dt=0.001):
        """
        Exact copy of baseline physics loss
        """
        # Extract states and controls
        z, phi, theta, psi, p, q, r, vz = inputs[:, :8].T
        thrust, tx, ty, tz = inputs[:, 8:12].T
        z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vz_next = outputs.T

        # Rotational dynamics
        J = self.params
        t1 = (J['Jyy'] - J['Jzz']) / J['Jxx']
        t2 = (J['Jzz'] - J['Jxx']) / J['Jyy']
        t3 = (J['Jxx'] - J['Jyy']) / J['Jzz']

        pdot = t1 * q * r + tx / J['Jxx']
        qdot = t2 * p * r + ty / J['Jyy']
        rdot = t3 * p * q + tz / J['Jzz']

        # Kinematics (proper Euler)
        phi_dot = p + torch.sin(phi) * torch.tan(theta) * q + torch.cos(phi) * torch.tan(theta) * r
        theta_dot = torch.cos(phi) * q - torch.sin(phi) * r
        psi_dot = torch.sin(phi) * q / torch.cos(theta) + torch.cos(phi) * r / torch.cos(theta)

        # Vertical dynamics
        drag_coeff = 0.05
        wdot = -thrust * torch.cos(theta) * torch.cos(phi) / J['m'] + self.g - drag_coeff * vz * torch.abs(vz)

        # Physics predictions
        p_pred = p + pdot * dt
        q_pred = q + qdot * dt
        r_pred = r + rdot * dt
        phi_pred = phi + phi_dot * dt
        theta_pred = theta + theta_dot * dt
        psi_pred = psi + psi_dot * dt
        vz_pred = vz + wdot * dt
        z_pred = z + vz * dt

        # Normalized loss (same scales as baseline)
        scales = {'ang': 0.2, 'rate': 0.1, 'vz': 5.0, 'z': 5.0}
        return torch.mean(
            ((phi_next - phi_pred) / scales['ang']) ** 2 +
            ((theta_next - theta_pred) / scales['ang']) ** 2 +
            ((psi_next - psi_pred) / scales['ang']) ** 2 +
            ((p_next - p_pred) / scales['rate']) ** 2 +
            ((q_next - q_pred) / scales['rate']) ** 2 +
            ((r_next - r_pred) / scales['rate']) ** 2 +
            ((vz_next - vz_pred) / scales['vz']) ** 2 +
            ((z_next - z_pred) / scales['z']) ** 2
        )


if __name__ == "__main__":
    model = QuadrotorPINNMinimal()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    x = torch.randn(32, 12)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Test physics loss
    phys_loss = model.physics_loss(x, y)
    print(f"Physics loss: {phys_loss.item():.6f}")
