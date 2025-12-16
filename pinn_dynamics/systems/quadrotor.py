"""
Physics-Informed Neural Network for Quadrotor Dynamics.

This module provides QuadrotorPINN, a concrete implementation of DynamicsPINN
for 6-DOF quadrotor dynamics with 12 states and 4 controls.

State vector (12):
    - Position: x, y, z (meters)
    - Attitude: phi, theta, psi (roll, pitch, yaw in radians)
    - Angular rates: p, q, r (rad/s)
    - Velocity: vx, vy, vz (m/s in body frame)

Control vector (4):
    - thrust: Total thrust force (N)
    - torque_x, torque_y, torque_z: Body torques (N*m)

Learnable parameters:
    - m: Mass (kg)
    - Jxx, Jyy, Jzz: Principal moments of inertia (kg*m^2)
    - kt, kq: Motor thrust and torque coefficients
"""

import torch
from .base import DynamicsPINN


class QuadrotorPINN(DynamicsPINN):
    """
    PINN for 6-DOF quadrotor dynamics.

    Implements the full Newton-Euler equations including:
        - Rotational dynamics (Euler equations)
        - Attitude kinematics
        - Translational dynamics with quadratic drag
        - Position kinematics (body-to-inertial transformation)

    Args:
        hidden_size: Width of hidden layers (default: 256)
        num_layers: Number of hidden layers (default: 5)
        dropout: Dropout rate (default: 0.1)

    Example:
        model = QuadrotorPINN()
        state = torch.randn(32, 12)  # batch of states
        control = torch.randn(32, 4)  # batch of controls
        next_state = model(torch.cat([state, control], dim=-1))
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__(
            state_dim=12,
            control_dim=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learnable_params={
                "Jxx": 6.86e-5,
                "Jyy": 9.2e-5,
                "Jzz": 1.366e-4,
                "m": 0.068,
                "kt": 0.01,
                "kq": 7.8263e-4,
            },
        )

        self.g = 9.81  # Gravity constant
        self.drag_coeff = 0.05  # Aerodynamic drag coefficient
        self.true_params = {k: v.item() for k, v in self.params.items()}

        # Set parameter bounds (physical constraints)
        self.set_param_bounds({
            "m": (0.0408, 0.0952),  # +/-40%
            "Jxx": (2.74e-5, 1.10e-4),  # +/-60%
            "Jyy": (3.68e-5, 1.47e-4),  # +/-60%
            "Jzz": (5.46e-5, 2.19e-4),  # +/-60%
            "kt": (0.0095, 0.0105),  # +/-5%
            "kq": (7.435e-4, 8.218e-4),  # +/-5%
        })

    def get_state_names(self):
        return ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]

    def get_control_names(self):
        return ["thrust", "torque_x", "torque_y", "torque_z"]

    def physics_loss(self, inputs, outputs, dt=0.001):
        """
        Enforce Newton-Euler equations for complete 6-DOF dynamics.

        Returns normalized physics violation loss.
        """
        # Extract states (12) and controls (4)
        x, y, z, phi, theta, psi, p, q, r, vx, vy, vz = inputs[:, :12].T
        thrust, tx, ty, tz = inputs[:, 12:16].T
        (
            x_next, y_next, z_next,
            phi_next, theta_next, psi_next,
            p_next, q_next, r_next,
            vx_next, vy_next, vz_next,
        ) = outputs[:, :12].T

        # === ROTATIONAL DYNAMICS (Euler Equations) ===
        J = self.params
        t1 = (J["Jyy"] - J["Jzz"]) / J["Jxx"]
        t2 = (J["Jzz"] - J["Jxx"]) / J["Jyy"]
        t3 = (J["Jxx"] - J["Jyy"]) / J["Jzz"]

        pdot = t1 * q * r + tx / J["Jxx"]
        qdot = t2 * p * r + ty / J["Jyy"]
        rdot = t3 * p * q + tz / J["Jzz"]

        # === ATTITUDE KINEMATICS ===
        phi_dot = p + torch.sin(phi) * torch.tan(theta) * q + torch.cos(phi) * torch.tan(theta) * r
        theta_dot = torch.cos(phi) * q - torch.sin(phi) * r
        psi_dot = torch.sin(phi) * q / torch.cos(theta) + torch.cos(phi) * r / torch.cos(theta)

        # === TRANSLATIONAL DYNAMICS (Body Frame) ===
        c_d = self.drag_coeff
        u, v, w = vx, vy, vz

        # Forces in body frame (thrust acts along -z axis)
        fx, fy, fz = 0.0, 0.0, -thrust

        # Body-frame accelerations with quadratic drag
        udot = r * v - q * w + fx / J["m"] - self.g * torch.sin(theta) - c_d * u * torch.abs(u)
        vdot = p * w - r * u + fy / J["m"] + self.g * torch.cos(theta) * torch.sin(phi) - c_d * v * torch.abs(v)
        wdot = q * u - p * v + fz / J["m"] + self.g * torch.cos(theta) * torch.cos(phi) - c_d * w * torch.abs(w)

        # === POSITION KINEMATICS (Body to Inertial) ===
        c_phi, s_phi = torch.cos(phi), torch.sin(phi)
        c_theta, s_theta = torch.cos(theta), torch.sin(theta)
        c_psi, s_psi = torch.cos(psi), torch.sin(psi)

        xdot = (c_psi * c_theta) * u + (c_psi * s_theta * s_phi - s_psi * c_phi) * v + (s_psi * s_phi + c_psi * s_theta * c_phi) * w
        ydot = (s_psi * c_theta) * u + (c_psi * c_phi + s_psi * s_theta * s_phi) * v + (s_psi * s_theta * c_phi - c_psi * s_phi) * w
        zdot = -s_theta * u + c_theta * s_phi * v + c_theta * c_phi * w

        # === EULER INTEGRATION ===
        p_pred = p + pdot * dt
        q_pred = q + qdot * dt
        r_pred = r + rdot * dt
        phi_pred = phi + phi_dot * dt
        theta_pred = theta + theta_dot * dt
        psi_pred = psi + psi_dot * dt
        vx_pred = vx + udot * dt
        vy_pred = vy + vdot * dt
        vz_pred = vz + wdot * dt
        x_pred = x + xdot * dt
        y_pred = y + ydot * dt
        z_pred = z + zdot * dt

        # === NORMALIZED PHYSICS LOSS ===
        scales = {"pos": 5.0, "ang": 0.2, "rate": 0.1, "vel": 5.0}

        loss = (
            ((x_next - x_pred) / scales["pos"]) ** 2
            + ((y_next - y_pred) / scales["pos"]) ** 2
            + ((z_next - z_pred) / scales["pos"]) ** 2
            + ((phi_next - phi_pred) / scales["ang"]) ** 2
            + ((theta_next - theta_pred) / scales["ang"]) ** 2
            + ((psi_next - psi_pred) / scales["ang"]) ** 2
            + ((p_next - p_pred) / scales["rate"]) ** 2
            + ((q_next - q_pred) / scales["rate"]) ** 2
            + ((r_next - r_pred) / scales["rate"]) ** 2
            + ((vx_next - vx_pred) / scales["vel"]) ** 2
            + ((vy_next - vy_pred) / scales["vel"]) ** 2
            + ((vz_next - vz_pred) / scales["vel"]) ** 2
        )

        return loss.mean()

    def temporal_smoothness_loss(self, inputs, outputs, dt=0.001):
        """Enforce physical limits on state change rates."""
        x, y, z, phi, theta, psi, p, q, r, vx, vy, vz = inputs[:, :12].T
        (
            x_next, y_next, z_next,
            phi_next, theta_next, psi_next,
            p_next, q_next, r_next,
            vx_next, vy_next, vz_next,
        ) = outputs[:, :12].T

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
            "dx": 5.0, "dy": 5.0, "dz": 5.0,
            "dphi": 3.0, "dtheta": 3.0, "dpsi": 2.0,
            "dp": 35.0, "dq": 35.0, "dr": 20.0,
            "dvx": 15.0, "dvy": 15.0, "dvz": 15.0,
        }

        # Soft constraints
        loss = (
            torch.relu(torch.abs(dx - vx) - limits["dx"]).pow(2).mean()
            + torch.relu(torch.abs(dy - vy) - limits["dy"]).pow(2).mean()
            + torch.relu(torch.abs(dz - vz) - limits["dz"]).pow(2).mean()
            + torch.relu(torch.abs(dphi) - limits["dphi"]).pow(2).mean()
            + torch.relu(torch.abs(dtheta) - limits["dtheta"]).pow(2).mean()
            + torch.relu(torch.abs(dpsi) - limits["dpsi"]).pow(2).mean()
            + torch.relu(torch.abs(dp) - limits["dp"]).pow(2).mean()
            + torch.relu(torch.abs(dq) - limits["dq"]).pow(2).mean()
            + torch.relu(torch.abs(dr) - limits["dr"]).pow(2).mean()
            + torch.relu(torch.abs(dvx) - limits["dvx"]).pow(2).mean()
            + torch.relu(torch.abs(dvy) - limits["dvy"]).pow(2).mean()
            + torch.relu(torch.abs(dvz) - limits["dvz"]).pow(2).mean()
        )

        return loss

    def stability_loss(self, inputs, outputs):
        """Prevent state space divergence."""
        (
            x_next, y_next, z_next,
            phi_next, theta_next, psi_next,
            p_next, q_next, r_next,
            vx_next, vy_next, vz_next,
        ) = outputs[:, :12].T

        bounds = {
            "x": 50.0, "y": 50.0, "z": 25.0,
            "phi": 0.5, "theta": 0.5,
            "p": 5.0, "q": 5.0, "r": 3.0,
            "vx": 10.0, "vy": 10.0, "vz": 10.0,
        }

        loss = (
            torch.relu(torch.abs(x_next) - bounds["x"]).pow(2).mean()
            + torch.relu(torch.abs(y_next) - bounds["y"]).pow(2).mean()
            + torch.relu(torch.abs(z_next) - bounds["z"]).pow(2).mean()
            + torch.relu(torch.abs(phi_next) - bounds["phi"]).pow(2).mean()
            + torch.relu(torch.abs(theta_next) - bounds["theta"]).pow(2).mean()
            + torch.relu(torch.abs(p_next) - bounds["p"]).pow(2).mean()
            + torch.relu(torch.abs(q_next) - bounds["q"]).pow(2).mean()
            + torch.relu(torch.abs(r_next) - bounds["r"]).pow(2).mean()
            + torch.relu(torch.abs(vx_next) - bounds["vx"]).pow(2).mean()
            + torch.relu(torch.abs(vy_next) - bounds["vy"]).pow(2).mean()
            + torch.relu(torch.abs(vz_next) - bounds["vz"]).pow(2).mean()
        )

        return loss

    def regularization_loss(self):
        """Penalize deviation from initial parameter estimates."""
        return 100 * sum(
            (self.params[k] - self.true_params[k]) ** 2 / self.true_params[k] ** 2
            for k in self.params
        )

    def energy_conservation_loss(self, inputs, outputs, dt=0.001):
        """
        Enforce energy conservation for parameter identification.

        Total energy: E = KE_trans + KE_rot + PE
        Power balance: dE/dt = P_input - P_drag
        """
        x, y, z, phi, theta, psi, p, q, r, vx, vy, vz = inputs[:, :12].T
        thrust, tx, ty, tz = inputs[:, 12:16].T
        (
            x_next, y_next, z_next,
            phi_next, theta_next, psi_next,
            p_next, q_next, r_next,
            vx_next, vy_next, vz_next,
        ) = outputs[:, :12].T

        # Current energy
        E_trans = 0.5 * self.params["m"] * (vx**2 + vy**2 + vz**2)
        E_rot = 0.5 * (self.params["Jxx"] * p**2 + self.params["Jyy"] * q**2 + self.params["Jzz"] * r**2)
        E_pot = self.params["m"] * self.g * z
        E_total = E_trans + E_rot + E_pot

        # Next step energy
        E_trans_next = 0.5 * self.params["m"] * (vx_next**2 + vy_next**2 + vz_next**2)
        E_rot_next = 0.5 * (self.params["Jxx"] * p_next**2 + self.params["Jyy"] * q_next**2 + self.params["Jzz"] * r_next**2)
        E_pot_next = self.params["m"] * self.g * z_next
        E_total_next = E_trans_next + E_rot_next + E_pot_next

        # Energy change
        dE_dt = (E_total_next - E_total) / dt

        # Input power
        P_thrust = thrust * vz
        P_torque = tx * p + ty * q + tz * r
        P_input = P_thrust + P_torque

        # Drag dissipation
        c_d = self.drag_coeff
        P_drag = c_d * (vx**2 * torch.abs(vx) + vy**2 * torch.abs(vy) + vz**2 * torch.abs(vz))

        # Energy balance residual
        energy_residual = dE_dt - (P_input - P_drag)

        # Normalize
        power_scale = self.params["m"] * self.g * 1.0
        normalized_residual = energy_residual / power_scale

        return (normalized_residual**2).mean()
