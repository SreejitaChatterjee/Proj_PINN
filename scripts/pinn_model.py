"""Unified Physics-Informed Neural Network for Quadrotor Dynamics"""
import torch
import torch.nn as nn

class QuadrotorPINN(nn.Module):
    def __init__(self, input_size=12, hidden_size=256, output_size=8, num_layers=5, dropout=0.1):
        super().__init__()
        # Increased capacity: 128->256 neurons, 4->5 layers
        # Added dropout for robustness against compounding errors
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

        # 6 learnable parameters
        self.params = nn.ParameterDict({
            'Jxx': nn.Parameter(torch.tensor(6.86e-5)),
            'Jyy': nn.Parameter(torch.tensor(9.2e-5)),
            'Jzz': nn.Parameter(torch.tensor(1.366e-4)),
            'm': nn.Parameter(torch.tensor(0.068)),
            'kt': nn.Parameter(torch.tensor(0.01)),
            'kq': nn.Parameter(torch.tensor(7.8263e-4))
        })

        self.g = 9.81  # Fixed constant
        self.true_params = {k: v.item() for k, v in self.params.items()}

    def forward(self, x):
        return self.network(x)

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

    def physics_loss(self, inputs, outputs, dt=0.001):
        # Extract states (8) and controls (4) - NO acceleration inputs
        z, phi, theta, psi, p, q, r, vz = inputs[:, :8].T
        thrust, tx, ty, tz = inputs[:, 8:12].T
        z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vz_next = outputs[:, :8].T

        # Rotational dynamics
        J = self.params
        t1, t2, t3 = (J['Jyy'] - J['Jzz'])/J['Jxx'], (J['Jzz'] - J['Jxx'])/J['Jyy'], (J['Jxx'] - J['Jyy'])/J['Jzz']

        # PHYSICS FIX: Removed artificial damping terms (-2*p, -2*q, -2*r)
        # Real Euler rotation equations have no viscous damping
        pdot = t1*q*r + tx/J['Jxx']
        qdot = t2*p*r + ty/J['Jyy']
        rdot = t3*p*q + tz/J['Jzz']

        # Kinematics
        phi_dot = p + torch.sin(phi)*torch.tan(theta)*q + torch.cos(phi)*torch.tan(theta)*r
        theta_dot = torch.cos(phi)*q - torch.sin(phi)*r
        psi_dot = torch.sin(phi)*q/torch.cos(theta) + torch.cos(phi)*r/torch.cos(theta)

        # Vertical dynamics
        # PHYSICS FIX: Changed to quadratic drag (more realistic than linear)
        drag_coeff = 0.05
        wdot = -thrust*torch.cos(theta)*torch.cos(phi)/J['m'] + self.g - drag_coeff*vz*torch.abs(vz)

        # Physics predictions
        p_pred, q_pred, r_pred = p + pdot*dt, q + qdot*dt, r + rdot*dt
        phi_pred = phi + phi_dot*dt
        theta_pred = theta + theta_dot*dt
        psi_pred = psi + psi_dot*dt
        vz_pred = vz + wdot*dt
        z_pred = z + vz*dt

        # Normalized loss
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
        """
        Penalize unrealistic state changes between timesteps.
        Enforces physical limits on acceleration and jerk.
        """
        # Extract current and next states
        z, phi, theta, psi, p, q, r, vz = inputs[:, :8].T
        z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vz_next = outputs[:, :8].T

        # Compute state changes (velocities/accelerations)
        dz = (z_next - z) / dt
        dphi = (phi_next - phi) / dt
        dtheta = (theta_next - theta) / dt
        dpsi = (psi_next - psi) / dt
        dp = (p_next - p) / dt
        dq = (q_next - q) / dt
        dr = (r_next - r) / dt
        dvz = (vz_next - vz) / dt

        # Physical limits for quadrotors (based on realistic dynamic flight)
        # BALANCED: Allow realistic transient dynamics while preventing extreme values
        limits = {
            'dz': 5.0,        # Max vertical velocity 5 m/s
            'dphi': 3.0,      # Max roll rate 3 rad/s (~172 deg/s)
            'dtheta': 3.0,    # Max pitch rate 3 rad/s
            'dpsi': 2.0,      # Max yaw rate 2 rad/s (~115 deg/s)
            'dp': 35.0,       # Max roll angular acceleration 35 rad/s^2 (allows realistic transients)
            'dq': 35.0,       # Max pitch angular acceleration 35 rad/s^2 (allows realistic transients)
            'dr': 20.0,       # Max yaw angular acceleration 20 rad/s^2 (allows realistic transients)
            'dvz': 15.0       # Max vertical acceleration 15 m/s^2
        }

        # Smooth L2 penalty (soft constraints)
        loss = 0.0
        loss += torch.relu(torch.abs(dz - vz) - limits['dz']).pow(2).mean()  # dz should equal vz
        loss += torch.relu(torch.abs(dphi) - limits['dphi']).pow(2).mean()
        loss += torch.relu(torch.abs(dtheta) - limits['dtheta']).pow(2).mean()
        loss += torch.relu(torch.abs(dpsi) - limits['dpsi']).pow(2).mean()
        loss += torch.relu(torch.abs(dp) - limits['dp']).pow(2).mean()
        loss += torch.relu(torch.abs(dq) - limits['dq']).pow(2).mean()
        loss += torch.relu(torch.abs(dr) - limits['dr']).pow(2).mean()
        loss += torch.relu(torch.abs(dvz) - limits['dvz']).pow(2).mean()

        return loss

    def stability_loss(self, inputs, outputs):
        """
        Penalize predictions that would cause instability in autoregressive rollout.
        Encourages predictions to stay close to physically reasonable state space.
        """
        # Extract predicted states
        z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vz_next = outputs[:, :8].T

        # Soft constraints on state magnitudes to prevent divergence
        state_bounds = {
            'z': 25.0,      # Altitude limit (meters)
            'phi': 0.5,     # Roll limit (rad, ~28 deg)
            'theta': 0.5,   # Pitch limit (rad, ~28 deg)
            'p': 5.0,       # Roll rate limit (rad/s)
            'q': 5.0,       # Pitch rate limit (rad/s)
            'r': 3.0,       # Yaw rate limit (rad/s)
            'vz': 10.0      # Vertical velocity limit (m/s)
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
        return 100 * sum((self.params[k] - self.true_params[k])**2 / self.true_params[k]**2
                        for k in self.params)
