"""Vanilla Optimized PINN - No Fourier Features (Stable for Autoregressive)

Key Features (No Fourier!):
- Residual MLP layers with Swish activation
- Modular architecture (translational + rotational)
- Energy-based physics constraints
- Multi-step rollout capability
- Raw input features (stable extrapolation)
"""
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual MLP block with Swish activation"""
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.net(x))  # Residual connection

class TranslationalModule(nn.Module):
    """Submodule for vertical dynamics (z, vz)"""
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            nn.Linear(hidden_size, 2)  # Output: z_next, vz_next
        )

    def forward(self, x):
        return self.net(x)

class RotationalModule(nn.Module):
    """Submodule for rotational dynamics (phi, theta, psi, p, q, r)"""
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            nn.Linear(hidden_size, 6)  # Output: phi, theta, psi, p, q, r
        )

    def forward(self, x):
        return self.net(x)

class QuadrotorPINNVanillaOptimized(nn.Module):
    def __init__(self, hidden_size=128, dropout=0.1):
        super().__init__()

        # NO Fourier features - use raw inputs directly
        # Input: [z, phi, theta, psi, p, q, r, vz, thrust, tx, ty, tz] = 12 features

        # Modular architecture
        self.translational = TranslationalModule(12, hidden_size // 2)
        self.rotational = RotationalModule(12, hidden_size // 2)

        # Learnable physical parameters
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
        Args:
            x: (batch, 12) - [z, phi, theta, psi, p, q, r, vz, thrust, tx, ty, tz]
        Returns:
            output: (batch, 8) - [z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vz_next]
        """
        # Use raw features directly (NO Fourier encoding)
        features = x  # All 12 features

        # Modular prediction
        trans_out = self.translational(features)  # (batch, 2): z_next, vz_next
        rot_out = self.rotational(features)  # (batch, 6): phi, theta, psi, p, q, r

        # Combine outputs in correct order: [z, phi, theta, psi, p, q, r, vz]
        return torch.cat([trans_out[:, 0:1], rot_out, trans_out[:, 1:2]], dim=1)

    def constrain_parameters(self):
        with torch.no_grad():
            bounds = {
                'm': (0.0646, 0.0714),
                'Jxx': (5.831e-5, 7.889e-5),
                'Jyy': (7.82e-5, 1.058e-4),
                'Jzz': (1.1611e-4, 1.5709e-4),
                'kt': (0.0095, 0.0105),
                'kq': (7.435e-4, 8.218e-4)
            }
            for k, (lo, hi) in bounds.items():
                self.params[k].clamp_(lo, hi)

    def energy_loss(self, inputs, outputs):
        """Energy-based physics constraint"""
        # Current state
        z, phi, theta, psi, p, q, r, vz = inputs[:, :8].T

        # Next state
        z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vz_next = outputs[:, :8].T

        # Current energy
        kinetic_trans = 0.5 * self.params['m'] * vz**2
        kinetic_rot = 0.5 * (self.params['Jxx']*p**2 + self.params['Jyy']*q**2 + self.params['Jzz']*r**2)
        potential = self.params['m'] * self.g * z
        E_current = kinetic_trans + kinetic_rot + potential

        # Next energy
        kinetic_trans_next = 0.5 * self.params['m'] * vz_next**2
        kinetic_rot_next = 0.5 * (self.params['Jxx']*p_next**2 + self.params['Jyy']*q_next**2 + self.params['Jzz']*r_next**2)
        potential_next = self.params['m'] * self.g * z_next
        E_next = kinetic_trans_next + kinetic_rot_next + potential_next

        # Energy change should match work done by controls
        thrust = inputs[:, 8]
        dE_pred = E_next - E_current
        work = thrust * (z_next - z)

        # Penalize energy violations (soft constraint)
        energy_residual = (dE_pred - work) / (torch.abs(E_current) + 1e-6)
        return energy_residual.pow(2).mean()

    def physics_loss(self, inputs, outputs, dt=0.001):
        """Physics-based residual loss"""
        z, phi, theta, psi, p, q, r, vz = inputs[:, :8].T
        thrust, tx, ty, tz = inputs[:, 8:12].T
        z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vz_next = outputs[:, :8].T

        # Rotational dynamics
        J = self.params
        t1 = (J['Jyy'] - J['Jzz']) / J['Jxx']
        t2 = (J['Jzz'] - J['Jxx']) / J['Jyy']
        t3 = (J['Jxx'] - J['Jyy']) / J['Jzz']

        pdot = t1*q*r + tx/J['Jxx']
        qdot = t2*p*r + ty/J['Jyy']
        rdot = t3*p*q + tz/J['Jzz']

        # Kinematics
        phi_dot = p + torch.sin(phi)*torch.tan(theta)*q + torch.cos(phi)*torch.tan(theta)*r
        theta_dot = torch.cos(phi)*q - torch.sin(phi)*r
        psi_dot = torch.sin(phi)*q/torch.cos(theta) + torch.cos(phi)*r/torch.cos(theta)

        # Vertical dynamics
        drag_coeff = 0.05
        wdot = -thrust*torch.cos(theta)*torch.cos(phi)/J['m'] + self.g - drag_coeff*vz*torch.abs(vz)

        # Physics predictions
        p_pred = p + pdot*dt
        q_pred = q + qdot*dt
        r_pred = r + rdot*dt
        phi_pred = phi + phi_dot*dt
        theta_pred = theta + theta_dot*dt
        psi_pred = psi + psi_dot*dt
        vz_pred = vz + wdot*dt
        z_pred = z + vz*dt

        # Normalized residuals
        scales = {'ang': 0.2, 'rate': 0.1, 'vz': 5.0, 'z': 5.0}
        return sum([
            ((phi_next - phi_pred)/scales['ang'])**2,
            ((theta_next - theta_pred)/scales['ang'])**2,
            ((psi_next - psi_pred)/scales['ang'])**2,
            ((p_next - p_pred)/scales['rate'])**2,
            ((q_next - q_pred)/scales['rate'])**2,
            ((r_next - r_pred)/scales['rate'])**2,
            ((vz_next - vz_pred)/scales['vz'])**2,
            ((z_next - z_pred)/scales['z'])**2
        ]).mean()

    def temporal_smoothness_loss(self, inputs, outputs, dt=0.001):
        """Penalize unrealistic state changes"""
        z, phi, theta, psi, p, q, r, vz = inputs[:, :8].T
        z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vz_next = outputs[:, :8].T

        # Compute accelerations
        dz = (z_next - z) / dt
        dphi = (phi_next - phi) / dt
        dtheta = (theta_next - theta) / dt
        dpsi = (psi_next - psi) / dt
        dp = (p_next - p) / dt
        dq = (q_next - q) / dt
        dr = (r_next - r) / dt
        dvz = (vz_next - vz) / dt

        limits = {
            'dz': 5.0, 'dphi': 3.0, 'dtheta': 3.0, 'dpsi': 2.0,
            'dp': 35.0, 'dq': 35.0, 'dr': 20.0, 'dvz': 15.0
        }

        loss = 0.0
        loss += torch.relu(torch.abs(dz - vz) - limits['dz']).pow(2).mean()
        loss += torch.relu(torch.abs(dphi) - limits['dphi']).pow(2).mean()
        loss += torch.relu(torch.abs(dtheta) - limits['dtheta']).pow(2).mean()
        loss += torch.relu(torch.abs(dpsi) - limits['dpsi']).pow(2).mean()
        loss += torch.relu(torch.abs(dp) - limits['dp']).pow(2).mean()
        loss += torch.relu(torch.abs(dq) - limits['dq']).pow(2).mean()
        loss += torch.relu(torch.abs(dr) - limits['dr']).pow(2).mean()
        loss += torch.relu(torch.abs(dvz) - limits['dvz']).pow(2).mean()

        return loss

    def stability_loss(self, inputs, outputs):
        """Prevent autoregressive divergence"""
        z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vz_next = outputs[:, :8].T

        state_bounds = {
            'z': 25.0, 'phi': 0.5, 'theta': 0.5,
            'p': 5.0, 'q': 5.0, 'r': 3.0, 'vz': 10.0
        }

        loss = 0.0
        loss += torch.relu(torch.abs(z_next) - state_bounds['z']).pow(2).mean()
        loss += torch.relu(torch.abs(phi_next) - state_bounds['phi']).pow(2).mean()
        loss += torch.relu(torch.abs(theta_next) - state_bounds['theta']).pow(2).mean()
        loss += torch.relu(torch.abs(p_next) - state_bounds['p']).pow(2).mean()
        loss += torch.relu(torch.abs(q_next) - state_bounds['q']).pow(2).mean()
        loss += torch.relu(torch.abs(r_next) - state_bounds['r']).pow(2).mean()
        loss += torch.relu(torch.abs(vz_next) - state_bounds['vz']).pow(2).mean()

        return loss

    def regularization_loss(self):
        """Regularize parameters to stay near true values"""
        return 100 * sum((self.params[k] - self.true_params[k])**2 / self.true_params[k]**2
                        for k in self.params)

    def multistep_rollout_loss(self, inputs, num_steps=5):
        """Multi-step autoregressive rollout loss during training"""
        batch_size = inputs.shape[0]
        current_state = inputs[:, :8].clone()
        controls = inputs[:, 8:12]

        total_loss = 0.0

        for step in range(num_steps):
            # Concatenate state + controls
            model_input = torch.cat([current_state, controls], dim=1)

            # Predict next state
            next_state = self.forward(model_input)

            # Accumulate physics + stability losses
            phys_loss = self.physics_loss(model_input, next_state)
            stab_loss = self.stability_loss(model_input, next_state)
            temp_loss = self.temporal_smoothness_loss(model_input, next_state)

            total_loss += phys_loss + 0.2*stab_loss + 0.5*temp_loss

            # Update current state for next iteration
            current_state = next_state.detach()

        return total_loss / num_steps
