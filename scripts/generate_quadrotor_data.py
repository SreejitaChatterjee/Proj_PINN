#!/usr/bin/env python3
"""
Generate realistic quadrotor training data using nonlinear model with PID controllers
Based on nonlinearmodel.m
"""

from pathlib import Path

import numpy as np
import pandas as pd


class QuadrotorSimulator:
    """Nonlinear quadrotor simulator with PID controllers"""

    def __init__(self):
        # Physical parameters (from MATLAB model)
        self.Jxx = 6.86e-5
        self.Jyy = 9.2e-5
        self.Jzz = 1.366e-4
        self.m = 0.068
        self.kt = 0.01
        self.kq = 7.8263e-4
        self.b = 0.062 / np.sqrt(2)
        self.g = 9.81

        # Derived parameters
        self.t1 = (self.Jyy - self.Jzz) / self.Jxx
        self.t2 = (self.Jzz - self.Jxx) / self.Jyy
        self.t3 = (self.Jxx - self.Jyy) / self.Jzz

        # Limits
        Tmax = 2.0 * self.m * self.g
        nmax = np.sqrt(Tmax / (4 * self.kt))
        self.txymax = (Tmax / 4) * 2 * self.b
        self.tzmax = 2 * self.kq * nmax * nmax
        self.Tmax = Tmax
        self.Tmin = 0.1 * self.m * self.g

        # Controller gains (TUNED for smoother, more realistic response)
        # Anomaly #2 fix: Reduced gains to prevent spiky torque behavior
        self.k2 = 0.05  # Roll rate controller (reduced from 0.1)
        self.k1 = 0.8  # Roll angle controller (reduced from 1.0)
        self.ki = 0.2 * 0.01  # Roll integral gain (reduced from 0.4*0.01)

        self.k21 = 0.05  # Pitch rate controller (reduced from 0.1)
        self.k11 = 0.8  # Pitch angle controller (reduced from 1.0)
        self.ki1 = 0.2 * 0.01  # Pitch integral gain (reduced from 0.4*0.01)

        self.k22 = 0.05  # Yaw rate controller (reduced from 0.1)
        self.k12 = 0.8  # Yaw angle controller (reduced from 1.0)
        self.ki2 = 0.2 * 0.01  # Yaw integral gain (reduced from 0.4*0.01)

        # Anomaly #6 fix: Significantly reduced vertical velocity controller gain
        self.kv = -0.25  # Vertical velocity controller (reduced from -0.4)
        self.kz1 = 1.5  # Altitude P gain (reduced from 2.0)
        self.kz2 = 0.15  # Altitude I gain (reduced from 0.22)

        self.th = 1e-7  # Threshold for zero torque

        # Anomaly #2 & #3 fix: Motor dynamics - time constants for realistic actuator response
        # Typical quadrotor motor time constant is 50-100ms
        self.motor_time_constant = 0.08  # seconds (80ms motor spin-up time)

        # Anomaly #2 & #3 fix: Slew rate limits (maximum rate of change)
        # Prevent instantaneous jumps in thrust and torques
        self.thrust_slew_rate = 15.0  # N/s (thrust can change max 15 N per second)
        self.torque_slew_rate = 0.5  # N·m/s (torque can change max 0.5 N·m per second)

        # Anomaly #1 & #4 fix: Reference trajectory filter
        # Low-pass filter time constant for smooth setpoint transitions
        # Tuned to 400ms: balance between smoothness and preserving dynamic transients
        self.ref_filter_time_constant = 0.4  # seconds (400ms rise time)

    def square_wave(self, t, period, amplitude_low, amplitude_high):
        """Generate square wave signal"""
        cycle_position = (t % period) / period
        if cycle_position < 0.5:
            return amplitude_low
        else:
            return amplitude_high

    def low_pass_filter(self, current_value, target_value, time_constant, dt):
        """
        First-order low-pass filter for smooth transitions
        Anomaly #1 & #4 fix: Prevents discontinuous reference changes

        Args:
            current_value: Current filtered value
            target_value: Target (desired) value
            time_constant: Filter time constant (tau)
            dt: Time step

        Returns:
            Updated filtered value
        """
        alpha = dt / (time_constant + dt)
        return current_value + alpha * (target_value - current_value)

    def apply_slew_rate_limit(self, current_value, target_value, max_rate, dt):
        """
        Apply slew rate limiting to prevent instantaneous changes
        Anomaly #2 & #3 fix: Limits rate of change for thrust and torques

        Args:
            current_value: Current value
            target_value: Desired value
            max_rate: Maximum rate of change (units/second)
            dt: Time step

        Returns:
            Rate-limited value
        """
        max_change = max_rate * dt
        change = target_value - current_value
        if abs(change) > max_change:
            return current_value + np.sign(change) * max_change
        return target_value

    def motor_dynamics(self, current_actuator, command_actuator, time_constant, dt):
        """
        First-order motor dynamics model
        Anomaly #3 fix: Realistic motor spin-up/spin-down behavior

        Args:
            current_actuator: Current actuator output
            command_actuator: Commanded actuator value
            time_constant: Motor time constant
            dt: Time step

        Returns:
            Updated actuator output with motor lag
        """
        alpha = dt / (time_constant + dt)
        return current_actuator + alpha * (command_actuator - current_actuator)

    def simulate_trajectory(
        self, phi_config, theta_config, psi_config, z_config, dt=0.001, tend=5.0
    ):
        """
        Simulate a single trajectory with square wave reference setpoints

        Args:
            phi_config: tuple (period, low_val, high_val) for roll reference (radians)
            theta_config: tuple (period, low_val, high_val) for pitch reference (radians)
            psi_config: tuple (period, low_val, high_val) for yaw reference (radians)
            z_config: tuple (period, low_val, high_val) for altitude reference (meters)
            dt: Time step (seconds)
            tend: End time (seconds)

        Returns:
            DataFrame with trajectory data
        """
        # Initialize states
        x, y, z = 0.0, 0.0, 0.0
        u, v, w = 0.0, 0.0, 0.0
        p, q, r = 0.0, 0.0, 0.0
        phi, theta, psi = 0.0, 0.0, 0.0

        # Integral terms
        sump, sumt, sumpsi, sumz = 0.0, 0.0, 0.0, 0.0

        # Anomaly #1 & #4 fix: Initialize filtered reference values
        # Start with initial setpoint values
        phi_ref_filtered = phi_config[1]
        theta_ref_filtered = theta_config[1]
        psi_ref_filtered = psi_config[1]
        z_ref_filtered = z_config[1]

        # Anomaly #2 & #3 fix: Initialize actual motor outputs (with dynamics)
        # Start with hovering thrust to prevent immediate crash
        T_actual = self.m * self.g  # Hovering thrust
        tx_actual = 0.0
        ty_actual = 0.0
        tz_actual = 0.0

        # Storage
        data = []

        num_steps = int(tend / dt)
        for i in range(num_steps):
            t = i * dt

            # Generate raw square wave references
            phi_ref_raw = self.square_wave(t, phi_config[0], phi_config[1], phi_config[2])
            theta_ref_raw = self.square_wave(t, theta_config[0], theta_config[1], theta_config[2])
            psi_ref_raw = self.square_wave(t, psi_config[0], psi_config[1], psi_config[2])
            z_ref_raw = self.square_wave(t, z_config[0], z_config[1], z_config[2])

            # Anomaly #1 & #4 fix: Apply low-pass filter for smooth reference transitions
            phi_ref_filtered = self.low_pass_filter(
                phi_ref_filtered, phi_ref_raw, self.ref_filter_time_constant, dt
            )
            theta_ref_filtered = self.low_pass_filter(
                theta_ref_filtered, theta_ref_raw, self.ref_filter_time_constant, dt
            )
            psi_ref_filtered = self.low_pass_filter(
                psi_ref_filtered, psi_ref_raw, self.ref_filter_time_constant, dt
            )
            z_ref_filtered = self.low_pass_filter(
                z_ref_filtered, z_ref_raw, self.ref_filter_time_constant, dt
            )

            # ===== ROLL CONTROLLER =====
            sump += phi_ref_filtered - phi
            pr = self.k1 * (phi_ref_filtered - phi) + self.ki * sump * dt
            tx_cmd = self.k2 * (pr - p)
            tx_cmd = np.clip(tx_cmd, -self.txymax, self.txymax)
            if abs(tx_cmd) < self.th:
                tx_cmd = 0.0

            # ===== PITCH CONTROLLER =====
            sumt += theta_ref_filtered - theta
            qr = self.k11 * (theta_ref_filtered - theta) + self.ki1 * sumt * dt
            ty_cmd = self.k21 * (qr - q)
            ty_cmd = np.clip(ty_cmd, -self.txymax, self.txymax)
            if abs(ty_cmd) < self.th:
                ty_cmd = 0.0

            # ===== YAW CONTROLLER =====
            sumpsi += psi_ref_filtered - psi
            rref = self.k12 * (psi_ref_filtered - psi) + self.ki2 * sumpsi * dt
            tz_cmd = self.k22 * (rref - r)
            tz_cmd = np.clip(tz_cmd, -self.tzmax, self.tzmax)
            if abs(tz_cmd) < self.th:
                tz_cmd = 0.0

            # ===== ALTITUDE CONTROLLER =====
            sumz += z_ref_filtered - z
            vzr = self.kz1 * (z_ref_filtered - z) + self.kz2 * sumz * dt
            T_cmd = self.kv * (vzr - w)  # Note: w is vertical velocity in body frame
            T_cmd = np.clip(T_cmd, self.Tmin, self.Tmax)

            # Anomaly #2 & #3 fix: Apply motor dynamics and slew rate limits
            # First apply slew rate limits to prevent instantaneous changes
            T_slew = self.apply_slew_rate_limit(T_actual, T_cmd, self.thrust_slew_rate, dt)
            tx_slew = self.apply_slew_rate_limit(tx_actual, tx_cmd, self.torque_slew_rate, dt)
            ty_slew = self.apply_slew_rate_limit(ty_actual, ty_cmd, self.torque_slew_rate, dt)
            tz_slew = self.apply_slew_rate_limit(tz_actual, tz_cmd, self.torque_slew_rate, dt)

            # Then apply motor time constant (first-order lag)
            T_actual = self.motor_dynamics(T_actual, T_slew, self.motor_time_constant, dt)
            tx_actual = self.motor_dynamics(tx_actual, tx_slew, self.motor_time_constant, dt)
            ty_actual = self.motor_dynamics(ty_actual, ty_slew, self.motor_time_constant, dt)
            tz_actual = self.motor_dynamics(tz_actual, tz_slew, self.motor_time_constant, dt)

            # ===== ROTATIONAL DYNAMICS =====
            # Use actual motor outputs (with dynamics) instead of commanded values
            # PHYSICS FIX: Removed artificial damping terms (-2*p, -2*q, -2*r)
            # Real Euler rotation equations have no viscous damping
            pdot = self.t1 * q * r + tx_actual / self.Jxx
            qdot = self.t2 * p * r + ty_actual / self.Jyy
            rdot = self.t3 * p * q + tz_actual / self.Jzz

            p += pdot * dt
            q += qdot * dt
            r += rdot * dt

            phidot = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
            thetadot = np.cos(phi) * q - np.sin(phi) * r
            psidot = np.sin(phi) * q / np.cos(theta) + np.cos(phi) * r / np.cos(theta)

            phi += phidot * dt
            theta += thetadot * dt
            psi += psidot * dt

            # Wrap angles to [-pi, pi]
            phi = np.arctan2(np.sin(phi), np.cos(phi))
            theta = np.arctan2(np.sin(theta), np.cos(theta))
            psi = np.arctan2(np.sin(psi), np.cos(psi))

            # ===== TRANSLATIONAL DYNAMICS =====
            # Use actual thrust (with motor dynamics) instead of commanded thrust
            fz = -T_actual
            fx, fy = 0.0, 0.0

            # PHYSICS FIX: Changed to quadratic drag (more realistic than linear)
            # Drag coefficient: 0.05 kg/m (tuned for small quadrotor)
            drag_coeff = 0.05
            udot = r * v - q * w + fx / self.m - self.g * np.sin(theta) - drag_coeff * u * np.abs(u)
            vdot = (
                p * w
                - r * u
                + fy / self.m
                + self.g * np.cos(theta) * np.sin(phi)
                - drag_coeff * v * np.abs(v)
            )
            wdot = (
                q * u
                - p * v
                + fz / self.m
                + self.g * np.cos(theta) * np.cos(phi)
                - drag_coeff * w * np.abs(w)
            )

            u += udot * dt
            v += vdot * dt
            w += wdot * dt

            # Position update
            xdot = (
                (np.cos(psi) * np.cos(theta)) * u
                + (np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi)) * v
                + (np.sin(psi) * np.sin(phi) + np.cos(psi) * np.sin(theta) * np.cos(phi)) * w
            )

            ydot = (
                (np.sin(psi) * np.cos(theta)) * u
                + (np.cos(psi) * np.cos(phi) + np.sin(psi) * np.sin(theta) * np.sin(phi)) * v
                + (np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)) * w
            )

            zdot = -1 * (
                np.sin(theta) * u
                - np.cos(theta) * np.sin(phi) * v
                - np.cos(theta) * np.cos(phi) * w
            )

            x += xdot * dt
            y += ydot * dt
            z += zdot * dt

            # Stop if above ground
            if z > 0:
                break

            # Store data (including kt and kq for PINN learning)
            # Store ACTUAL motor outputs (with dynamics), not commanded values
            # Option 1 implementation: Include angular accelerations for improved inertia identification
            # Complete state: 10 predicted states (x, y, z, roll, pitch, yaw, p, q, r, vx, vy, vz)
            data.append(
                {
                    "timestamp": i * dt,
                    "x": x,  # Horizontal position (m)
                    "y": y,  # Horizontal position (m)
                    "z": z,  # Vertical position (m)
                    "thrust": T_actual,
                    "torque_x": tx_actual,
                    "torque_y": ty_actual,
                    "torque_z": tz_actual,
                    "roll": phi,
                    "pitch": theta,
                    "yaw": psi,
                    "p": p,
                    "q": q,
                    "r": r,
                    "p_dot": pdot,  # Angular acceleration in roll (rad/s²)
                    "q_dot": qdot,  # Angular acceleration in pitch (rad/s²)
                    "r_dot": rdot,  # Angular acceleration in yaw (rad/s²)
                    "vx": xdot,
                    "vy": ydot,
                    "vz": zdot,
                    "mass": self.m,
                    "inertia_xx": self.Jxx,
                    "inertia_yy": self.Jyy,
                    "inertia_zz": self.Jzz,
                    "kt": self.kt,  # Thrust coefficient
                    "kq": self.kq,  # Torque coefficient
                }
            )

        return pd.DataFrame(data)


def generate_diverse_trajectories():
    """Generate 10 diverse flight trajectories with different setpoints"""

    sim = QuadrotorSimulator()

    # Define 10 diverse trajectory configurations with square wave inputs
    # Format: dict with (period, low_value, high_value) for each axis
    # Issue #6 fix: Reduced altitude setpoints to limit vz to realistic range (±7 m/s)
    trajectories = [
        {
            "phi": (2.0, -10 * np.pi / 180, 10 * np.pi / 180),
            "theta": (2.5, -5 * np.pi / 180, 5 * np.pi / 180),
            "psi": (3.0, -5 * np.pi / 180, 5 * np.pi / 180),
            "z": (2.0, -5.0, -3.0),
            "desc": "Standard square wave maneuver",
        },
        {
            "phi": (1.5, -15 * np.pi / 180, 15 * np.pi / 180),
            "theta": (2.0, -8 * np.pi / 180, 8 * np.pi / 180),
            "psi": (2.5, -10 * np.pi / 180, 10 * np.pi / 180),
            "z": (1.5, -6.0, -4.0),
            "desc": "Fast aggressive square waves",
        },
        {
            "phi": (3.0, -5 * np.pi / 180, 5 * np.pi / 180),
            "theta": (3.5, -3 * np.pi / 180, 3 * np.pi / 180),
            "psi": (4.0, -5 * np.pi / 180, 5 * np.pi / 180),
            "z": (3.0, -3.0, -2.0),
            "desc": "Slow gentle square waves",
        },
        {
            "phi": (2.0, -12 * np.pi / 180, 8 * np.pi / 180),
            "theta": (2.0, -6 * np.pi / 180, 4 * np.pi / 180),
            "psi": (2.5, -12 * np.pi / 180, 12 * np.pi / 180),
            "z": (2.0, -7.0, -5.0),
            "desc": "Asymmetric square wave maneuvers",
        },
        {
            "phi": (1.8, -20 * np.pi / 180, 20 * np.pi / 180),
            "theta": (2.2, -10 * np.pi / 180, 10 * np.pi / 180),
            "psi": (2.0, -8 * np.pi / 180, 8 * np.pi / 180),
            "z": (1.8, -6.0, -4.0),
            "desc": "High amplitude square waves",
        },
        {
            "phi": (2.5, -8 * np.pi / 180, 8 * np.pi / 180),
            "theta": (3.0, -4 * np.pi / 180, 4 * np.pi / 180),
            "psi": (3.5, -10 * np.pi / 180, 10 * np.pi / 180),
            "z": (2.5, -4.0, -3.0),
            "desc": "Medium frequency square waves",
        },
        {
            "phi": (3.5, -6 * np.pi / 180, 12 * np.pi / 180),
            "theta": (3.0, -7 * np.pi / 180, 5 * np.pi / 180),
            "psi": (4.0, -8 * np.pi / 180, 16 * np.pi / 180),
            "z": (3.5, -8.0, -6.0),
            "desc": "Large asymmetric square waves",
        },
        {
            "phi": (1.2, -18 * np.pi / 180, 18 * np.pi / 180),
            "theta": (1.5, -12 * np.pi / 180, 12 * np.pi / 180),
            "psi": (1.8, -15 * np.pi / 180, 15 * np.pi / 180),
            "z": (1.2, -5.0, -3.0),
            "desc": "Very fast high amplitude square waves",
        },
        {
            "phi": (4.0, -6 * np.pi / 180, 6 * np.pi / 180),
            "theta": (4.5, -4 * np.pi / 180, 4 * np.pi / 180),
            "psi": (5.0, -8 * np.pi / 180, 8 * np.pi / 180),
            "z": (4.0, -5.0, -4.0),
            "desc": "Very slow square waves",
        },
        {
            "phi": (2.2, -10 * np.pi / 180, 10 * np.pi / 180),
            "theta": (2.6, -8 * np.pi / 180, 8 * np.pi / 180),
            "psi": (3.0, -14 * np.pi / 180, 14 * np.pi / 180),
            "z": (2.2, -7.0, -5.0),
            "desc": "Mixed frequency square waves",
        },
    ]

    all_data = []

    print("Generating diverse quadrotor flight trajectories with SQUARE WAVE inputs...")
    print("=" * 60)

    for traj_id, traj_config in enumerate(trajectories):
        print(f"\nTrajectory {traj_id}: {traj_config['desc']}")
        print(
            f"  Roll config: period={traj_config['phi'][0]}s, range=[{traj_config['phi'][1]*180/np.pi:.1f}, {traj_config['phi'][2]*180/np.pi:.1f}]deg"
        )
        print(
            f"  Pitch config: period={traj_config['theta'][0]}s, range=[{traj_config['theta'][1]*180/np.pi:.1f}, {traj_config['theta'][2]*180/np.pi:.1f}]deg"
        )
        print(
            f"  Yaw config: period={traj_config['psi'][0]}s, range=[{traj_config['psi'][1]*180/np.pi:.1f}, {traj_config['psi'][2]*180/np.pi:.1f}]deg"
        )
        print(
            f"  Altitude config: period={traj_config['z'][0]}s, range=[{traj_config['z'][1]:.1f}, {traj_config['z'][2]:.1f}]m"
        )

        # Simulate trajectory
        traj_data = sim.simulate_trajectory(
            traj_config["phi"],
            traj_config["theta"],
            traj_config["psi"],
            traj_config["z"],
        )

        # Add trajectory ID
        traj_data["trajectory_id"] = traj_id

        print(f"  Generated {len(traj_data)} samples")
        print(
            f"  Thrust range: [{traj_data['thrust'].min():.3f}, {traj_data['thrust'].max():.3f}] N"
        )
        print(f"  Altitude range: [{traj_data['z'].min():.3f}, {traj_data['z'].max():.3f}] m")

        all_data.append(traj_data)

    # Combine all trajectories
    combined_data = pd.concat(all_data, ignore_index=True)

    print("\n" + "=" * 60)
    print(
        f"SUCCESS: Generated {len(combined_data)} total samples across {len(trajectories)} trajectories"
    )
    print(
        f"  Overall thrust range: [{combined_data['thrust'].min():.3f}, {combined_data['thrust'].max():.3f}] N"
    )
    print(
        f"  Overall altitude range: [{combined_data['z'].min():.3f}, {combined_data['z'].max():.3f}] m"
    )

    return combined_data


def main():
    """Main function to generate and save training data"""

    # Generate data
    data = generate_diverse_trajectories()

    # Save to CSV - Use absolute path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_path = project_root / "data" / "quadrotor_training_data.csv"
    output_path.parent.mkdir(exist_ok=True, parents=True)

    data.to_csv(output_path, index=False)

    print(f"\nSUCCESS: Data saved to: {output_path}")
    print(f"  Total samples: {len(data)}")
    print(f"  Trajectories: {data['trajectory_id'].nunique()}")
    print(f"  Columns: {list(data.columns)}")


if __name__ == "__main__":
    main()
