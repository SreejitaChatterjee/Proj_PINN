"""
Optimized PINN v2 - Complete Implementation of Stability Fixes

Architecture:
- Baseline 5-layer MLP with residual connections
- Merged coupling layer (translational + rotational fusion)
- All baseline losses (physics, temporal, stability, regularization)
- Designed for curriculum rollout training

Based on proven fixes for autoregressive stability.
"""

import torch
import torch.nn as nn


class OptimizedPINNv2(nn.Module):
    """
    Optimized PINN with merged coupling for stability

    Key improvements over baseline:
    1. Residual connections for better gradient flow
    2. Merged coupling layer (no hard modularity)
    3. All baseline losses maintained
    """

    def __init__(self, input_size=12, hidden_size=256, output_size=8, num_layers=5, dropout=0.1):
        super().__init__()

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_act = nn.Tanh()
        self.input_drop = nn.Dropout(dropout)

        # Shared trunk (maintains coupling)
        self.shared_layers = nn.ModuleList()
        self.shared_acts = nn.ModuleList()
        self.shared_drops = nn.ModuleList()

        for _ in range(num_layers - 3):  # Leave room for coupling layer
            self.shared_layers.append(nn.Linear(hidden_size, hidden_size))
            self.shared_acts.append(nn.Tanh())
            self.shared_drops.append(nn.Dropout(dropout))

        # Coupling layer: Small specialized branches that merge
        # This preserves physical coupling while allowing task-specific features
        self.translational_branch = nn.Linear(hidden_size, hidden_size // 2)
        self.rotational_branch = nn.Linear(hidden_size, hidden_size // 2)
        self.coupling_merge = nn.Linear(hidden_size, hidden_size)  # Merge back

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Learnable physics parameters (same as baseline)
        self.params = nn.ParameterDict({
            'Jxx': nn.Parameter(torch.tensor(6.86e-5)),
            'Jyy': nn.Parameter(torch.tensor(9.2e-5)),
            'Jzz': nn.Parameter(torch.tensor(1.366e-4)),
            'm': nn.Parameter(torch.tensor(0.068)),
            'kt': nn.Parameter(torch.tensor(0.01)),
            'kq': nn.Parameter(torch.tensor(7.8263e-4))
        })

        self.g = 9.81
        self.true_params = {k: v.item() for k, v in self.params.items()}

    def forward(self, x):
        """
        Forward pass with residual connections and merged coupling

        Args:
            x: [batch, 12] - normalized and clipped inputs

        Returns:
            output: [batch, 8] - predicted next state
        """
        # Input layer
        h = self.input_drop(self.input_act(self.input_layer(x)))

        # Shared trunk with residual connections
        for layer, act, drop in zip(self.shared_layers, self.shared_acts, self.shared_drops):
            # Residual: h = h + 0.1 * f(h)
            h = h + 0.1 * drop(act(layer(h)))

        # Coupling layer: branch, then merge
        # This allows task-specific features while maintaining physical coupling
        h_trans = torch.tanh(self.translational_branch(h))
        h_rot = torch.tanh(self.rotational_branch(h))
        h_coupled = torch.cat([h_trans, h_rot], dim=-1)
        h = torch.tanh(self.coupling_merge(h_coupled))

        # Output
        return self.output_layer(h)

    def physics_loss(self, inputs, outputs, dt=0.001):
        """Exact copy of baseline physics loss"""
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

    def temporal_smoothness_loss(self, inputs, outputs, dt=0.001):
        """Exact copy of baseline temporal loss"""
        z, phi, theta, psi, p, q, r, vz = inputs[:, :8].T
        z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vz_next = outputs[:, :8].T

        # Compute state changes
        dz = (z_next - z) / dt
        dphi = (phi_next - phi) / dt
        dtheta = (theta_next - theta) / dt
        dpsi = (psi_next - psi) / dt
        dp = (p_next - p) / dt
        dq = (q_next - q) / dt
        dr = (r_next - r) / dt
        dvz = (vz_next - vz) / dt

        # Physical limits (balanced for realistic flight)
        max_velocities = {'z': 5.0, 'ang': 3.0, 'rate': 15.0, 'acc': 50.0}

        loss = (
            torch.mean(torch.relu(torch.abs(dz) - max_velocities['z']) ** 2) +
            torch.mean(torch.relu(torch.abs(dphi) - max_velocities['ang']) ** 2) +
            torch.mean(torch.relu(torch.abs(dtheta) - max_velocities['ang']) ** 2) +
            torch.mean(torch.relu(torch.abs(dpsi) - max_velocities['ang']) ** 2) +
            torch.mean(torch.relu(torch.abs(dp) - max_velocities['rate']) ** 2) +
            torch.mean(torch.relu(torch.abs(dq) - max_velocities['rate']) ** 2) +
            torch.mean(torch.relu(torch.abs(dr) - max_velocities['rate']) ** 2) +
            torch.mean(torch.relu(torch.abs(dvz) - max_velocities['acc']) ** 2)
        )

        return loss

    def stability_loss(self, inputs, outputs):
        """Exact copy of baseline stability loss"""
        z, phi, theta, psi, p, q, r, vz = inputs[:, :8].T
        z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vz_next = outputs[:, :8].T

        # Bounded state evolution
        state_bounds = {'z': 10.0, 'ang': 1.0, 'rate': 10.0, 'vz': 10.0}

        loss = (
            torch.mean(torch.relu(torch.abs(z_next) - state_bounds['z']) ** 2) +
            torch.mean(torch.relu(torch.abs(phi_next) - state_bounds['ang']) ** 2) +
            torch.mean(torch.relu(torch.abs(theta_next) - state_bounds['ang']) ** 2) +
            torch.mean(torch.relu(torch.abs(psi_next) - state_bounds['ang']) ** 2) +
            torch.mean(torch.relu(torch.abs(p_next) - state_bounds['rate']) ** 2) +
            torch.mean(torch.relu(torch.abs(q_next) - state_bounds['rate']) ** 2) +
            torch.mean(torch.relu(torch.abs(r_next) - state_bounds['rate']) ** 2) +
            torch.mean(torch.relu(torch.abs(vz_next) - state_bounds['vz']) ** 2)
        )

        return loss

    def energy_consistency_loss(self, inputs, outputs):
        """Energy conservation check"""
        z, phi, theta, psi, p, q, r, vz = inputs[:, :8].T
        z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vz_next = outputs[:, :8].T

        J = self.params

        # Kinetic energy
        KE_current = 0.5 * J['m'] * vz ** 2 + 0.5 * (J['Jxx'] * p ** 2 + J['Jyy'] * q ** 2 + J['Jzz'] * r ** 2)
        KE_next = 0.5 * J['m'] * vz_next ** 2 + 0.5 * (J['Jxx'] * p_next ** 2 + J['Jyy'] * q_next ** 2 + J['Jzz'] * r_next ** 2)

        # Potential energy
        PE_current = J['m'] * self.g * z
        PE_next = J['m'] * self.g * z_next

        # Total energy change (should be small for conservative forces)
        dE = (KE_next + PE_next) - (KE_current + PE_current)

        return torch.mean(dE ** 2)

    def constrain_parameters(self):
        """Apply parameter constraints"""
        with torch.no_grad():
            bounds = {
                'm': (0.0646, 0.0714),
                'Jxx': (5.831e-5, 7.889e-5),
                'Jyy': (7.82e-5, 1.058e-4),
                'Jzz': (1.1611e-4, 1.5709e-4),
                'kt': (0.0095, 0.0105),
                'kq': (7.04e-4, 8.61e-4)
            }
            for param_name, param in self.params.items():
                if param_name in bounds:
                    param.clamp_(*bounds[param_name])


if __name__ == "__main__":
    model = OptimizedPINNv2()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Optimized PINN v2 parameters: {total_params:,}")

    # Test forward pass
    x = torch.randn(32, 12)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Test all losses
    phys_loss = model.physics_loss(x, y)
    temp_loss = model.temporal_smoothness_loss(x, y)
    stab_loss = model.stability_loss(x, y)
    energy_loss = model.energy_consistency_loss(x, y)

    print(f"\nLoss components:")
    print(f"  Physics: {phys_loss.item():.6f}")
    print(f"  Temporal: {temp_loss.item():.6f}")
    print(f"  Stability: {stab_loss.item():.6f}")
    print(f"  Energy: {energy_loss.item():.6f}")
