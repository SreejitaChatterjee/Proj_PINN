#!/usr/bin/env python3
"""
Generate realistic quadrotor training data using nonlinear model with PID controllers
Based on nonlinearmodel.m
"""

import numpy as np
import pandas as pd
from pathlib import Path

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

        # Controller gains
        self.k2 = 0.1    # Roll rate controller
        self.k1 = 1.0    # Roll angle controller
        self.ki = 0.4 * 0.01  # Roll integral gain

        self.k21 = 0.1   # Pitch rate controller
        self.k11 = 1.0   # Pitch angle controller
        self.ki1 = 0.4 * 0.01  # Pitch integral gain

        self.k22 = 0.1   # Yaw rate controller
        self.k12 = 1.0   # Yaw angle controller
        self.ki2 = 0.4 * 0.01  # Yaw integral gain

        self.kv = -0.4   # Vertical velocity controller (Issue #7 fix: reduced from -1.0 for more realistic response)
        self.kz1 = 2.0   # Altitude P gain
        self.kz2 = 0.22  # Altitude I gain (increased from 0.15 to eliminate steady-state error)

        self.th = 1e-7   # Threshold for zero torque

    def simulate_trajectory(self, phi_ref, theta_ref, psi_ref, z_ref, dt=0.001, tend=5.0):
        """
        Simulate a single trajectory with given reference setpoints

        Args:
            phi_ref: Roll angle reference (radians)
            theta_ref: Pitch angle reference (radians)
            psi_ref: Yaw angle reference (radians)
            z_ref: Altitude reference (meters, negative down)
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

        # Storage
        data = []

        num_steps = int(tend / dt)
        for i in range(num_steps):
            # ===== ROLL CONTROLLER =====
            sump += (phi_ref - phi)
            pr = self.k1 * (phi_ref - phi) + self.ki * sump * dt
            tx = self.k2 * (pr - p)
            tx = np.clip(tx, -self.txymax, self.txymax)
            if abs(tx) < self.th:
                tx = 0.0

            # ===== PITCH CONTROLLER =====
            sumt += (theta_ref - theta)
            qr = self.k11 * (theta_ref - theta) + self.ki1 * sumt * dt
            ty = self.k21 * (qr - q)
            ty = np.clip(ty, -self.txymax, self.txymax)
            if abs(ty) < self.th:
                ty = 0.0

            # ===== YAW CONTROLLER =====
            sumpsi += (psi_ref - psi)
            rref = self.k12 * (psi_ref - psi) + self.ki2 * sumpsi * dt
            tz = self.k22 * (rref - r)
            tz = np.clip(tz, -self.tzmax, self.tzmax)
            if abs(tz) < self.th:
                tz = 0.0

            # ===== ALTITUDE CONTROLLER =====
            sumz += (z_ref - z)
            vzr = self.kz1 * (z_ref - z) + self.kz2 * sumz * dt
            T = self.kv * (vzr - w)  # Note: w is vertical velocity in body frame
            T = np.clip(T, self.Tmin, self.Tmax)

            # ===== ROTATIONAL DYNAMICS =====
            pdot = self.t1 * q * r + tx / self.Jxx - 2 * p
            qdot = self.t2 * p * r + ty / self.Jyy - 2 * q
            rdot = self.t3 * p * q + tz / self.Jzz - 2 * r

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
            fz = -T
            fx, fy = 0.0, 0.0

            udot = r * v - q * w + fx / self.m - self.g * np.sin(theta) - 0.1 * u
            vdot = p * w - r * u + fy / self.m + self.g * np.cos(theta) * np.sin(phi) - 0.1 * v
            wdot = q * u - p * v + fz / self.m + self.g * np.cos(theta) * np.cos(phi) - 0.1 * w

            u += udot * dt
            v += vdot * dt
            w += wdot * dt

            # Position update
            xdot = (np.cos(psi) * np.cos(theta)) * u + \
                   (np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi)) * v + \
                   (np.sin(psi) * np.sin(phi) + np.cos(psi) * np.sin(theta) * np.cos(phi)) * w

            ydot = (np.sin(psi) * np.cos(theta)) * u + \
                   (np.cos(psi) * np.cos(phi) + np.sin(psi) * np.sin(theta) * np.sin(phi)) * v + \
                   (np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)) * w

            zdot = -1 * (np.sin(theta) * u - np.cos(theta) * np.sin(phi) * v - np.cos(theta) * np.cos(phi) * w)

            x += xdot * dt
            y += ydot * dt
            z += zdot * dt

            # Stop if above ground
            if z > 0:
                break

            # Store data (including kt and kq for PINN learning)
            data.append({
                'timestamp': i * dt,
                'thrust': T,
                'z': z,
                'torque_x': tx,
                'torque_y': ty,
                'torque_z': tz,
                'roll': phi,
                'pitch': theta,
                'yaw': psi,
                'p': p,
                'q': q,
                'r': r,
                'vx': xdot,
                'vy': ydot,
                'vz': zdot,
                'mass': self.m,
                'inertia_xx': self.Jxx,
                'inertia_yy': self.Jyy,
                'inertia_zz': self.Jzz,
                'kt': self.kt,  # Thrust coefficient
                'kq': self.kq   # Torque coefficient
            })

        return pd.DataFrame(data)


def generate_diverse_trajectories():
    """Generate 10 diverse flight trajectories with different setpoints"""

    sim = QuadrotorSimulator()

    # Define 10 diverse trajectory configurations
    # Format: (phi_ref_deg, theta_ref_deg, psi_ref_deg, z_ref_m, description)
    # Issue #6 fix: Reduced altitude setpoints to limit vz to realistic range (±7 m/s)
    trajectories = [
        (10, -5, 5, -5.0, "Standard maneuver"),
        (15, -8, 10, -6.0, "Aggressive roll and descent"),        # Changed from -8.0
        (5, -3, -5, -3.0, "Gentle maneuver shallow altitude"),
        (-10, 5, 15, -7.0, "Negative roll descent"),              # Changed from -10.0
        (20, -10, 8, -6.0, "High roll angle"),
        (8, -2, -10, -4.0, "Moderate roll low altitude"),
        (-15, 8, 12, -8.0, "Negative roll high altitude"),        # Changed from -12.0
        (12, -6, 20, -7.0, "High yaw angle"),
        (6, -4, -8, -5.0, "Balanced moderate maneuver"),
        (-8, 3, -15, -7.0, "Negative roll and yaw")               # Changed from -9.0
    ]

    all_data = []

    print("Generating diverse quadrotor flight trajectories...")
    print("=" * 60)

    for traj_id, (phi_deg, theta_deg, psi_deg, z_ref, desc) in enumerate(trajectories):
        # Convert degrees to radians
        phi_ref = phi_deg * np.pi / 180
        theta_ref = theta_deg * np.pi / 180
        psi_ref = psi_deg * np.pi / 180

        print(f"\nTrajectory {traj_id}: {desc}")
        print(f"  Setpoints: phi={phi_deg}deg, theta={theta_deg}deg, psi={psi_deg}deg, z={z_ref}m")

        # Simulate trajectory
        traj_data = sim.simulate_trajectory(phi_ref, theta_ref, psi_ref, z_ref)

        # Add trajectory ID
        traj_data['trajectory_id'] = traj_id

        print(f"  Generated {len(traj_data)} samples")
        print(f"  Thrust range: [{traj_data['thrust'].min():.3f}, {traj_data['thrust'].max():.3f}] N")
        print(f"  Altitude range: [{traj_data['z'].min():.3f}, {traj_data['z'].max():.3f}] m")

        all_data.append(traj_data)

    # Combine all trajectories
    combined_data = pd.concat(all_data, ignore_index=True)

    print("\n" + "=" * 60)
    print(f"SUCCESS: Generated {len(combined_data)} total samples across {len(trajectories)} trajectories")
    print(f"  Overall thrust range: [{combined_data['thrust'].min():.3f}, {combined_data['thrust'].max():.3f}] N")
    print(f"  Overall altitude range: [{combined_data['z'].min():.3f}, {combined_data['z'].max():.3f}] m")

    return combined_data


def main():
    """Main function to generate and save training data"""

    # Generate data
    data = generate_diverse_trajectories()

    # Save to CSV - Use absolute path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_path = project_root / 'data' / 'quadrotor_training_data.csv'
    output_path.parent.mkdir(exist_ok=True, parents=True)

    data.to_csv(output_path, index=False)

    print(f"\nSUCCESS: Data saved to: {output_path}")
    print(f"  Total samples: {len(data)}")
    print(f"  Trajectories: {data['trajectory_id'].nunique()}")
    print(f"  Columns: {list(data.columns)}")


if __name__ == "__main__":
    main()
