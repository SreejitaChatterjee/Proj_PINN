"""Unified Physics-Informed Neural Network for Quadrotor Dynamics"""
import torch
import torch.nn as nn

class QuadrotorPINN(nn.Module):
    def __init__(self, input_size=15, hidden_size=128, output_size=8, num_layers=4):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        layers.extend([layer for _ in range(num_layers - 2)
                      for layer in (nn.Linear(hidden_size, hidden_size), nn.Tanh())])
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
        # Extract states
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

    def regularization_loss(self):
        return 100 * sum((self.params[k] - self.true_params[k])**2 / self.true_params[k]**2
                        for k in self.params)
