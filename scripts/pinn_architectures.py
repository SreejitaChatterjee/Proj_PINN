"""
Multiple PINN Architectures for Comparative Study

This module implements 4 architectures claimed in the papers:
1. Baseline: Monolithic 5-layer MLP (existing QuadrotorPINN)
2. Modular: Separate translation/rotation subnetworks
3. Fourier: Periodic encoding of angular states
4. Curriculum: Monolithic with curriculum training support

All share identical physics constraints; only architecture differs.
"""
import torch
import torch.nn as nn
import numpy as np


class PhysicsLossMixin:
    """Shared physics loss computation for all architectures"""

    def _init_physics_params(self):
        """Initialize learnable physical parameters"""
        self.params = nn.ParameterDict({
            'Jxx': nn.Parameter(torch.tensor(6.86e-5)),
            'Jyy': nn.Parameter(torch.tensor(9.2e-5)),
            'Jzz': nn.Parameter(torch.tensor(1.366e-4)),
            'm': nn.Parameter(torch.tensor(0.068)),
            'kt': nn.Parameter(torch.tensor(0.01)),
            'kq': nn.Parameter(torch.tensor(7.8263e-4))
        })
        self.g = 9.81
        self.drag_coeff = 0.05
        self.true_params = {k: v.item() for k, v in self.params.items()}

    def constrain_parameters(self):
        """Clamp parameters to physical bounds"""
        with torch.no_grad():
            bounds = {
                'm': (0.0408, 0.0952),
                'Jxx': (2.74e-5, 1.10e-4),
                'Jyy': (3.68e-5, 1.47e-4),
                'Jzz': (5.46e-5, 2.19e-4),
                'kt': (0.0095, 0.0105),
                'kq': (7.435e-4, 8.218e-4)
            }
            for k, (lo, hi) in bounds.items():
                self.params[k].clamp_(lo, hi)

    def physics_loss(self, inputs, outputs, dt=0.001):
        """Enforce Newton-Euler equations"""
        x, y, z, phi, theta, psi, p, q, r, vx, vy, vz = inputs[:, :12].T
        thrust, tx, ty, tz = inputs[:, 12:16].T
        x_next, y_next, z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vx_next, vy_next, vz_next = outputs[:, :12].T

        J = self.params
        t1 = (J['Jyy'] - J['Jzz']) / J['Jxx']
        t2 = (J['Jzz'] - J['Jxx']) / J['Jyy']
        t3 = (J['Jxx'] - J['Jyy']) / J['Jzz']

        pdot = t1*q*r + tx/J['Jxx']
        qdot = t2*p*r + ty/J['Jyy']
        rdot = t3*p*q + tz/J['Jzz']

        phi_dot = p + torch.sin(phi)*torch.tan(theta)*q + torch.cos(phi)*torch.tan(theta)*r
        theta_dot = torch.cos(phi)*q - torch.sin(phi)*r
        psi_dot = torch.sin(phi)*q/torch.cos(theta + 1e-8) + torch.cos(phi)*r/torch.cos(theta + 1e-8)

        c_d = self.drag_coeff
        u, v, w = vx, vy, vz
        fx, fy, fz = 0.0, 0.0, -thrust

        udot = r*v - q*w + fx/J['m'] - self.g*torch.sin(theta) - c_d*u*torch.abs(u)
        vdot = p*w - r*u + fy/J['m'] + self.g*torch.cos(theta)*torch.sin(phi) - c_d*v*torch.abs(v)
        wdot = q*u - p*v + fz/J['m'] + self.g*torch.cos(theta)*torch.cos(phi) - c_d*w*torch.abs(w)

        c_phi, s_phi = torch.cos(phi), torch.sin(phi)
        c_theta, s_theta = torch.cos(theta), torch.sin(theta)
        c_psi, s_psi = torch.cos(psi), torch.sin(psi)

        xdot = (c_psi*c_theta)*u + (c_psi*s_theta*s_phi - s_psi*c_phi)*v + (s_psi*s_phi + c_psi*s_theta*c_phi)*w
        ydot = (s_psi*c_theta)*u + (c_psi*c_phi + s_psi*s_theta*s_phi)*v + (s_psi*s_theta*c_phi - c_psi*s_phi)*w
        zdot = -s_theta*u + c_theta*s_phi*v + c_theta*c_phi*w

        p_pred = p + pdot*dt
        q_pred = q + qdot*dt
        r_pred = r + rdot*dt
        phi_pred = phi + phi_dot*dt
        theta_pred = theta + theta_dot*dt
        psi_pred = psi + psi_dot*dt
        vx_pred = vx + udot*dt
        vy_pred = vy + vdot*dt
        vz_pred = vz + wdot*dt
        x_pred = x + xdot*dt
        y_pred = y + ydot*dt
        z_pred = z + zdot*dt

        scales = {'pos': 5.0, 'ang': 0.2, 'rate': 0.1, 'vel': 5.0}

        loss = (
            ((x_next - x_pred)/scales['pos'])**2 +
            ((y_next - y_pred)/scales['pos'])**2 +
            ((z_next - z_pred)/scales['pos'])**2 +
            ((phi_next - phi_pred)/scales['ang'])**2 +
            ((theta_next - theta_pred)/scales['ang'])**2 +
            ((psi_next - psi_pred)/scales['ang'])**2 +
            ((p_next - p_pred)/scales['rate'])**2 +
            ((q_next - q_pred)/scales['rate'])**2 +
            ((r_next - r_pred)/scales['rate'])**2 +
            ((vx_next - vx_pred)/scales['vel'])**2 +
            ((vy_next - vy_pred)/scales['vel'])**2 +
            ((vz_next - vz_pred)/scales['vel'])**2
        )
        return loss.mean()

    def energy_conservation_loss(self, inputs, outputs, dt=0.001):
        """Energy conservation constraint"""
        x, y, z, phi, theta, psi, p, q, r, vx, vy, vz = inputs[:, :12].T
        thrust, tx, ty, tz = inputs[:, 12:16].T
        x_next, y_next, z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vx_next, vy_next, vz_next = outputs[:, :12].T

        E_trans = 0.5 * self.params['m'] * (vx**2 + vy**2 + vz**2)
        E_rot = 0.5 * (self.params['Jxx']*p**2 + self.params['Jyy']*q**2 + self.params['Jzz']*r**2)
        E_pot = self.params['m'] * self.g * z
        E_total = E_trans + E_rot + E_pot

        E_trans_next = 0.5 * self.params['m'] * (vx_next**2 + vy_next**2 + vz_next**2)
        E_rot_next = 0.5 * (self.params['Jxx']*p_next**2 + self.params['Jyy']*q_next**2 + self.params['Jzz']*r_next**2)
        E_pot_next = self.params['m'] * self.g * z_next
        E_total_next = E_trans_next + E_rot_next + E_pot_next

        dE_dt = (E_total_next - E_total) / dt
        P_thrust = thrust * vz
        P_torque = tx*p + ty*q + tz*r
        P_input = P_thrust + P_torque
        c_d = self.drag_coeff
        P_drag = c_d * (vx**2 * torch.abs(vx) + vy**2 * torch.abs(vy) + vz**2 * torch.abs(vz))
        energy_residual = dE_dt - (P_input - P_drag)
        power_scale = self.params['m'] * self.g * 1.0
        normalized_residual = energy_residual / (power_scale + 1e-8)
        return (normalized_residual**2).mean()


class BaselinePINN(nn.Module, PhysicsLossMixin):
    """
    Architecture 1: Baseline Monolithic MLP
    - Standard 5-layer MLP with Tanh activations
    - Single network for all state predictions
    """
    def __init__(self, input_size=16, hidden_size=256, output_size=12, num_layers=5, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)
        self._init_physics_params()
        self.name = "Baseline"

    def forward(self, x):
        return self.network(x)


class ModularPINN(nn.Module, PhysicsLossMixin):
    """
    Architecture 2: Modular PINN with Separate Subnetworks

    FAILURE MODE: This architecture breaks dynamic coupling!
    - Translation module: predicts x, y, z, vx, vy, vz
    - Rotation module: predicts phi, theta, psi, p, q, r

    Problem: During autoregressive rollout, errors in phi, theta (rotation)
    cause thrust projection errors in z'' (translation), but gradients
    don't flow between modules to enable coordinated correction.
    """
    def __init__(self, input_size=16, hidden_size=128, dropout=0.1):
        super().__init__()

        # Translation module: takes full input, predicts [x, y, z, vx, vy, vz]
        self.translation_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 6)  # x, y, z, vx, vy, vz
        )

        # Rotation module: takes full input, predicts [phi, theta, psi, p, q, r]
        self.rotation_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 6)  # phi, theta, psi, p, q, r
        )

        self._init_physics_params()
        self.name = "Modular"

    def forward(self, x):
        # Translation: [x, y, z, vx, vy, vz]
        trans = self.translation_net(x)
        # Rotation: [phi, theta, psi, p, q, r]
        rot = self.rotation_net(x)

        # Combine: [x, y, z, phi, theta, psi, p, q, r, vx, vy, vz]
        # Reorder to match expected output format
        x_out = trans[:, 0:1]
        y_out = trans[:, 1:2]
        z_out = trans[:, 2:3]
        phi_out = rot[:, 0:1]
        theta_out = rot[:, 1:2]
        psi_out = rot[:, 2:3]
        p_out = rot[:, 3:4]
        q_out = rot[:, 4:5]
        r_out = rot[:, 5:6]
        vx_out = trans[:, 3:4]
        vy_out = trans[:, 4:5]
        vz_out = trans[:, 5:6]

        return torch.cat([x_out, y_out, z_out, phi_out, theta_out, psi_out,
                          p_out, q_out, r_out, vx_out, vy_out, vz_out], dim=1)


class FourierFeatureEncoding(nn.Module):
    """
    Fourier feature encoding for angular states

    gamma(theta) = [sin(w_1*theta), cos(w_1*theta), ..., sin(w_K*theta), cos(w_K*theta)]

    FAILURE MODE: For high frequencies w_K, small state perturbations cause
    large feature-space jumps during autoregressive rollout!
    """
    def __init__(self, num_frequencies=64, max_frequency=256):
        super().__init__()
        # Frequencies: 1, 2, 4, 8, ..., max_frequency (logarithmically spaced)
        frequencies = 2 ** torch.linspace(0, np.log2(max_frequency), num_frequencies)
        self.register_buffer('frequencies', frequencies)
        self.output_dim = num_frequencies * 2  # sin + cos for each frequency

    def forward(self, x):
        """Encode angles with Fourier features"""
        # x shape: (batch, 1) - single angle
        # Output: (batch, num_freq * 2)
        proj = x * self.frequencies.unsqueeze(0)  # (batch, num_freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FourierPINN(nn.Module, PhysicsLossMixin):
    """
    Architecture 3: Fourier Feature PINN

    Applies Fourier encoding to angular states (phi, theta, psi).

    FAILURE MODE: High-frequency Fourier features cause catastrophic
    extrapolation during rollout:
    - Small state drift -> large feature-space discontinuity
    - Feature jump -> poor prediction -> larger drift
    - Creates exponential feedback loop

    Paper claim: 5.2M meter error at 100 steps (10^6x worse than baseline)
    """
    def __init__(self, input_size=16, hidden_size=256, num_layers=5, dropout=0.1,
                 num_frequencies=64, max_frequency=256):
        super().__init__()

        # Fourier encoding for 3 angular states
        self.fourier_encoder = FourierFeatureEncoding(num_frequencies, max_frequency)
        fourier_dim = self.fourier_encoder.output_dim * 3  # 3 angles

        # Input: 9 non-angular states + 4 controls + Fourier-encoded angles
        # Non-angular: x, y, z, p, q, r, vx, vy, vz (9)
        # Controls: thrust, tx, ty, tz (4)
        # Fourier: 3 angles * (num_freq * 2)
        total_input = 13 + fourier_dim

        layers = [nn.Linear(total_input, hidden_size), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_size, 12))
        self.network = nn.Sequential(*layers)

        self._init_physics_params()
        self.name = "Fourier"
        self.num_frequencies = num_frequencies
        self.max_frequency = max_frequency

    def forward(self, x):
        # Extract components
        # Input order: x, y, z, phi, theta, psi, p, q, r, vx, vy, vz, thrust, tx, ty, tz
        positions = x[:, 0:3]      # x, y, z
        angles = x[:, 3:6]         # phi, theta, psi
        rates = x[:, 6:9]          # p, q, r
        velocities = x[:, 9:12]    # vx, vy, vz
        controls = x[:, 12:16]     # thrust, tx, ty, tz

        # Apply Fourier encoding to each angle
        phi_enc = self.fourier_encoder(angles[:, 0:1])
        theta_enc = self.fourier_encoder(angles[:, 1:2])
        psi_enc = self.fourier_encoder(angles[:, 2:3])

        # Concatenate all features
        encoded = torch.cat([positions, phi_enc, theta_enc, psi_enc,
                            rates, velocities, controls], dim=1)

        return self.network(encoded)


class CurriculumPINN(nn.Module, PhysicsLossMixin):
    """
    Architecture 4: Curriculum-Trained Monolithic PINN (OURS)

    Same architecture as Baseline, but with:
    1. Curriculum learning over increasing horizons (5->10->25->50 steps)
    2. Scheduled sampling (replace ground truth with predictions during training)
    3. Energy conservation regularization

    This is a TRAINING methodology difference, not architecture difference.
    The architecture is identical to Baseline for fair comparison.
    """
    def __init__(self, input_size=16, hidden_size=256, output_size=12, num_layers=5, dropout=0.3):
        super().__init__()
        # Same architecture as baseline, but with higher dropout for regularization
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)
        self._init_physics_params()
        self.name = "Curriculum"

    def forward(self, x):
        return self.network(x)


def get_model(model_type, **kwargs):
    """Factory function to create models by name"""
    models = {
        'baseline': BaselinePINN,
        'modular': ModularPINN,
        'fourier': FourierPINN,
        'curriculum': CurriculumPINN
    }
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    return models[model_type.lower()](**kwargs)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test all architectures
    print("="*70)
    print("PINN Architecture Comparison")
    print("="*70)

    batch_size = 32
    x = torch.randn(batch_size, 16)  # 12 states + 4 controls

    for model_type in ['baseline', 'modular', 'fourier', 'curriculum']:
        model = get_model(model_type)
        y = model(x)
        n_params = count_parameters(model)
        print(f"\n{model.name} PINN:")
        print(f"  Parameters: {n_params:,}")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")

        # Test physics loss
        physics_loss = model.physics_loss(x, y)
        print(f"  Physics loss: {physics_loss.item():.6f}")
