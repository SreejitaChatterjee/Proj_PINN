"""Unified Physics-Informed Neural Network for Quadrotor Dynamics - Full 12-State Model"""
import torch
import torch.nn as nn

class QuadrotorPINN(nn.Module):
    def __init__(self, input_size=16, hidden_size=256, output_size=12, num_layers=5, dropout=0.1):
        """
        Full 6-DOF quadrotor PINN predicting all 12 states:
        - Positions: x, y, z
        - Attitudes: phi, theta, psi
        - Angular rates: p, q, r
        - Velocities: vx, vy, vz

        Input: 12 states + 4 controls = 16 features
        Output: 12 next states
        """
        super().__init__()
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

        self.g = 9.81  # Gravity constant
        self.drag_coeff = 0.05  # Aerodynamic drag coefficient
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
        """
        Enforce Newton-Euler equations for complete 6-DOF dynamics
        """
        # Extract states (12) and controls (4)
        x, y, z, phi, theta, psi, p, q, r, vx, vy, vz = inputs[:, :12].T
        thrust, tx, ty, tz = inputs[:, 12:16].T
        x_next, y_next, z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vx_next, vy_next, vz_next = outputs[:, :12].T

        # === ROTATIONAL DYNAMICS (Euler Equations) ===
        J = self.params
        t1 = (J['Jyy'] - J['Jzz']) / J['Jxx']
        t2 = (J['Jzz'] - J['Jxx']) / J['Jyy']
        t3 = (J['Jxx'] - J['Jyy']) / J['Jzz']

        # Angular accelerations (NO artificial damping)
        pdot = t1*q*r + tx/J['Jxx']
        qdot = t2*p*r + ty/J['Jyy']
        rdot = t3*p*q + tz/J['Jzz']

        # === ATTITUDE KINEMATICS ===
        phi_dot = p + torch.sin(phi)*torch.tan(theta)*q + torch.cos(phi)*torch.tan(theta)*r
        theta_dot = torch.cos(phi)*q - torch.sin(phi)*r
        psi_dot = torch.sin(phi)*q/torch.cos(theta) + torch.cos(phi)*r/torch.cos(theta)

        # === TRANSLATIONAL DYNAMICS (Body Frame - Newton's Laws) ===
        c_d = self.drag_coeff
        # u, v, w are body-frame velocities (vx, vy, vz in our notation)
        u, v, w = vx, vy, vz

        # Forces in body frame (thrust acts along -z axis)
        fx, fy, fz = 0.0, 0.0, -thrust

        # Body-frame accelerations with quadratic drag
        udot = r*v - q*w + fx/J['m'] - self.g*torch.sin(theta) - c_d*u*torch.abs(u)
        vdot = p*w - r*u + fy/J['m'] + self.g*torch.cos(theta)*torch.sin(phi) - c_d*v*torch.abs(v)
        wdot = q*u - p*v + fz/J['m'] + self.g*torch.cos(theta)*torch.cos(phi) - c_d*w*torch.abs(w)

        # === POSITION KINEMATICS (Body to Inertial Transformation) ===
        # Rotation matrix elements
        c_phi, s_phi = torch.cos(phi), torch.sin(phi)
        c_theta, s_theta = torch.cos(theta), torch.sin(theta)
        c_psi, s_psi = torch.cos(psi), torch.sin(psi)

        xdot = (c_psi*c_theta)*u + (c_psi*s_theta*s_phi - s_psi*c_phi)*v + (s_psi*s_phi + c_psi*s_theta*c_phi)*w
        ydot = (s_psi*c_theta)*u + (c_psi*c_phi + s_psi*s_theta*s_phi)*v + (s_psi*s_theta*c_phi - c_psi*s_phi)*w
        zdot = -s_theta*u + c_theta*s_phi*v + c_theta*c_phi*w

        # === PHYSICS-BASED PREDICTIONS ===
        # Angular rates
        p_pred = p + pdot*dt
        q_pred = q + qdot*dt
        r_pred = r + rdot*dt

        # Attitudes
        phi_pred = phi + phi_dot*dt
        theta_pred = theta + theta_dot*dt
        psi_pred = psi + psi_dot*dt

        # Velocities
        vx_pred = vx + udot*dt
        vy_pred = vy + vdot*dt
        vz_pred = vz + wdot*dt

        # Positions
        x_pred = x + xdot*dt
        y_pred = y + ydot*dt
        z_pred = z + zdot*dt

        # === NORMALIZED PHYSICS LOSS ===
        scales = {
            'pos': 5.0,     # Position scale (m)
            'ang': 0.2,     # Angle scale (rad)
            'rate': 0.1,    # Angular rate scale (rad/s)
            'vel': 5.0      # Velocity scale (m/s)
        }

        loss = (
            # Positions
            ((x_next - x_pred)/scales['pos'])**2 +
            ((y_next - y_pred)/scales['pos'])**2 +
            ((z_next - z_pred)/scales['pos'])**2 +
            # Attitudes
            ((phi_next - phi_pred)/scales['ang'])**2 +
            ((theta_next - theta_pred)/scales['ang'])**2 +
            ((psi_next - psi_pred)/scales['ang'])**2 +
            # Angular rates
            ((p_next - p_pred)/scales['rate'])**2 +
            ((q_next - q_pred)/scales['rate'])**2 +
            ((r_next - r_pred)/scales['rate'])**2 +
            # Velocities
            ((vx_next - vx_pred)/scales['vel'])**2 +
            ((vy_next - vy_pred)/scales['vel'])**2 +
            ((vz_next - vz_pred)/scales['vel'])**2
        )

        return loss.mean()

    def temporal_smoothness_loss(self, inputs, outputs, dt=0.001):
        """
        Enforce physical limits on state change rates (all 12 states)
        """
        # Extract current and next states
        x, y, z, phi, theta, psi, p, q, r, vx, vy, vz = inputs[:, :12].T
        x_next, y_next, z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vx_next, vy_next, vz_next = outputs[:, :12].T

        # Compute state changes
        dx = (x_next - x) / dt
        dy = (y_next - y) / dt
        dz = (z_next - z) / dt
        dphi = (phi_next - phi) / dt
        dtheta = (theta_next - theta) / dt
        dpsi = (psi_next - psi) / dt
        dp = (p_next - p) / dt
        dq = (q_next - q) / dt
        dr = (r_next - r) / dt
        dvx = (vx_next - vx) / dt
        dvy = (vy_next - vy) / dt
        dvz = (vz_next - vz) / dt

        # Physical limits
        limits = {
            'dx': 5.0,        # Max horizontal velocity 5 m/s
            'dy': 5.0,
            'dz': 5.0,        # Max vertical velocity 5 m/s
            'dphi': 3.0,      # Max roll rate 3 rad/s
            'dtheta': 3.0,
            'dpsi': 2.0,
            'dp': 35.0,       # Max angular acceleration
            'dq': 35.0,
            'dr': 20.0,
            'dvx': 15.0,      # Max horizontal acceleration
            'dvy': 15.0,
            'dvz': 15.0
        }

        # Soft constraints
        loss = 0.0
        loss += torch.relu(torch.abs(dx - vx) - limits['dx']).pow(2).mean()
        loss += torch.relu(torch.abs(dy - vy) - limits['dy']).pow(2).mean()
        loss += torch.relu(torch.abs(dz - vz) - limits['dz']).pow(2).mean()
        loss += torch.relu(torch.abs(dphi) - limits['dphi']).pow(2).mean()
        loss += torch.relu(torch.abs(dtheta) - limits['dtheta']).pow(2).mean()
        loss += torch.relu(torch.abs(dpsi) - limits['dpsi']).pow(2).mean()
        loss += torch.relu(torch.abs(dp) - limits['dp']).pow(2).mean()
        loss += torch.relu(torch.abs(dq) - limits['dq']).pow(2).mean()
        loss += torch.relu(torch.abs(dr) - limits['dr']).pow(2).mean()
        loss += torch.relu(torch.abs(dvx) - limits['dvx']).pow(2).mean()
        loss += torch.relu(torch.abs(dvy) - limits['dvy']).pow(2).mean()
        loss += torch.relu(torch.abs(dvz) - limits['dvz']).pow(2).mean()

        return loss

    def stability_loss(self, inputs, outputs):
        """
        Prevent state space divergence (all 12 states)
        """
        x_next, y_next, z_next, phi_next, theta_next, psi_next, p_next, q_next, r_next, vx_next, vy_next, vz_next = outputs[:, :12].T

        # State bounds
        bounds = {
            'x': 50.0, 'y': 50.0, 'z': 25.0,
            'phi': 0.5, 'theta': 0.5,
            'p': 5.0, 'q': 5.0, 'r': 3.0,
            'vx': 10.0, 'vy': 10.0, 'vz': 10.0
        }

        loss = 0.0
        loss += torch.relu(torch.abs(x_next) - bounds['x']).pow(2).mean()
        loss += torch.relu(torch.abs(y_next) - bounds['y']).pow(2).mean()
        loss += torch.relu(torch.abs(z_next) - bounds['z']).pow(2).mean()
        loss += torch.relu(torch.abs(phi_next) - bounds['phi']).pow(2).mean()
        loss += torch.relu(torch.abs(theta_next) - bounds['theta']).pow(2).mean()
        loss += torch.relu(torch.abs(p_next) - bounds['p']).pow(2).mean()
        loss += torch.relu(torch.abs(q_next) - bounds['q']).pow(2).mean()
        loss += torch.relu(torch.abs(r_next) - bounds['r']).pow(2).mean()
        loss += torch.relu(torch.abs(vx_next) - bounds['vx']).pow(2).mean()
        loss += torch.relu(torch.abs(vy_next) - bounds['vy']).pow(2).mean()
        loss += torch.relu(torch.abs(vz_next) - bounds['vz']).pow(2).mean()

        return loss

    def regularization_loss(self):
        return 100 * sum((self.params[k] - self.true_params[k])**2 / self.true_params[k]**2
                        for k in self.params)
