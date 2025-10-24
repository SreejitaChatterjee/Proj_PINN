#!/usr/bin/env python3
"""
Generate realistic quadrotor training data using nonlinear model with PID controllers
Enhanced version with square wave reference inputs and kt/kq as learnable parameters
Based on nonlinearmodel.m
"""

import numpy as np
import pandas as pd
from pathlib import Path

class QuadrotorSimulator:
    """Nonlinear quadrotor simulator with PID controllers and square wave references"""

    def __init__(self):
        # Physical parameters (from MATLAB model)
        self.Jxx = 6.86e-5
        self.Jyy = 9.2e-5
        self.Jzz = 1.366e-4
        self.m = 0.068
        self.kt = 0.01  # Thrust coefficient (NOW LEARNABLE)
        self.kq = 7.8263e-4  # Torque coefficient (NOW LEARNABLE)
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

    def square_wave(self, t, period, amplitude_low, amplitude_high):
        """Generate square wave signal"""
        cycle_position = (t % period) / period
        if cycle_position < 0.5:
            return amplitude_low
        else:
            return amplitude_high

    def simulate_trajectory(self, phi_config, theta_config, psi_config, z_config,
                          dt=0.001, tend=5.0):
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

        # Storage
        data = []

        num_steps = int(tend / dt)
        for i in range(num_steps):
            t = i * dt

            # Generate square wave references
            phi_ref = self.square_wave(t, phi_config[0], phi_config[1], phi_config[2])
            theta_ref = self.square_wave(t, theta_config[0], theta_config[1], theta_config[2])
            psi_ref = self.square_wave(t, psi_config[0], psi_config[1], psi_config[2])
            z_ref = self.square_wave(t, z_config[0], z_config[1], z_config[2])

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
            T = self.kv * (vzr - w)
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

            # Store data (NOW INCLUDING kt and kq)
            data.append({
                'timestamp': t,
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
                'kt': self.kt,  # NEW: Thrust coefficient
                'kq': self.kq   # NEW: Torque coefficient
            })

        return pd.DataFrame(data)


def generate_diverse_trajectories():
    """Generate 10 diverse flight trajectories with SQUARE WAVE setpoints"""

    sim = QuadrotorSimulator()

    # Define 10 diverse trajectory configurations with square wave inputs
    # Format: (period_s, low_value, high_value)
    # Issue #6 fix: Reduced altitude setpoints to limit vz to realistic range (±7 m/s)
    trajectories = [
        {
            'phi': (2.0, -10*np.pi/180, 10*np.pi/180),
            'theta': (2.5, -5*np.pi/180, 5*np.pi/180),
            'psi': (3.0, -8*np.pi/180, 8*np.pi/180),
            'z': (2.0, -5.0, -3.0),
            'desc': "Moderate square wave maneuvers"
        },
        {
            'phi': (1.5, -15*np.pi/180, 15*np.pi/180),
            'theta': (2.0, -8*np.pi/180, 8*np.pi/180),
            'psi': (2.5, -10*np.pi/180, 10*np.pi/180),
            'z': (1.5, -6.0, -4.0),  # Changed from -8.0, -5.0
            'desc': "Fast aggressive square waves"
        },
        {
            'phi': (3.0, -5*np.pi/180, 5*np.pi/180),
            'theta': (3.5, -3*np.pi/180, 3*np.pi/180),
            'psi': (4.0, -5*np.pi/180, 5*np.pi/180),
            'z': (3.0, -3.0, -2.0),
            'desc': "Slow gentle square waves"
        },
        {
            'phi': (2.0, -12*np.pi/180, 8*np.pi/180),
            'theta': (2.0, -6*np.pi/180, 4*np.pi/180),
            'psi': (2.5, -12*np.pi/180, 12*np.pi/180),
            'z': (2.0, -7.0, -5.0),  # Changed from -10.0, -7.0
            'desc': "Asymmetric square wave maneuvers"
        },
        {
            'phi': (1.8, -18*np.pi/180, 18*np.pi/180),
            'theta': (2.2, -9*np.pi/180, 9*np.pi/180),
            'psi': (2.0, -15*np.pi/180, 15*np.pi/180),
            'z': (1.8, -6.0, -4.0),
            'desc': "High amplitude square waves"
        },
        {
            'phi': (2.5, -8*np.pi/180, 8*np.pi/180),
            'theta': (3.0, -4*np.pi/180, 4*np.pi/180),
            'psi': (2.5, -10*np.pi/180, 10*np.pi/180),
            'z': (2.5, -4.0, -3.0),
            'desc': "Medium frequency square waves"
        },
        {
            'phi': (3.5, -6*np.pi/180, 12*np.pi/180),
            'theta': (3.0, -7*np.pi/180, 5*np.pi/180),
            'psi': (4.0, -8*np.pi/180, 16*np.pi/180),
            'z': (3.5, -8.0, -6.0),  # Changed from -12.0, -9.0
            'desc': "Large asymmetric square waves"
        },
        {
            'phi': (1.6, -14*np.pi/180, 14*np.pi/180),
            'theta': (1.8, -7*np.pi/180, 7*np.pi/180),
            'psi': (2.2, -12*np.pi/180, 12*np.pi/180),
            'z': (1.6, -7.0, -5.0),
            'desc': "Fast balanced square waves"
        },
        {
            'phi': (2.8, -7*np.pi/180, 9*np.pi/180),
            'theta': (3.2, -5*np.pi/180, 6*np.pi/180),
            'psi': (3.5, -9*np.pi/180, 11*np.pi/180),
            'z': (2.8, -5.0, -4.0),
            'desc': "Moderate asymmetric square waves"
        },
        {
            'phi': (2.2, -10*np.pi/180, 10*np.pi/180),
            'theta': (2.6, -8*np.pi/180, 8*np.pi/180),
            'psi': (3.0, -14*np.pi/180, 14*np.pi/180),
            'z': (2.2, -7.0, -5.0),  # Changed from -9.0, -6.0
            'desc': "Mixed frequency square waves"
        }
    ]

    all_data = []

    print("Generating diverse quadrotor flight trajectories with SQUARE WAVE inputs...")
    print("=" * 70)

    for traj_id, config in enumerate(trajectories):
        print(f"\nTrajectory {traj_id}: {config['desc']}")
        print(f"  Square wave periods:")
        print(f"    Roll: {config['phi'][0]}s, range: [{config['phi'][1]*180/np.pi:.1f}, {config['phi'][2]*180/np.pi:.1f}] deg")
        print(f"    Pitch: {config['theta'][0]}s, range: [{config['theta'][1]*180/np.pi:.1f}, {config['theta'][2]*180/np.pi:.1f}] deg")
        print(f"    Yaw: {config['psi'][0]}s, range: [{config['psi'][1]*180/np.pi:.1f}, {config['psi'][2]*180/np.pi:.1f}] deg")
        print(f"    Altitude: {config['z'][0]}s, range: [{config['z'][1]:.1f}, {config['z'][2]:.1f}] m")

        # Simulate trajectory
        traj_data = sim.simulate_trajectory(
            config['phi'], config['theta'], config['psi'], config['z']
        )

        # Add trajectory ID
        traj_data['trajectory_id'] = traj_id

        print(f"  Generated {len(traj_data)} samples")
        print(f"  Thrust range: [{traj_data['thrust'].min():.3f}, {traj_data['thrust'].max():.3f}] N")
        print(f"  Altitude range: [{traj_data['z'].min():.3f}, {traj_data['z'].max():.3f}] m")

        all_data.append(traj_data)

    # Combine all trajectories
    combined_data = pd.concat(all_data, ignore_index=True)

    print("\n" + "=" * 70)
    print(f"SUCCESS: Generated {len(combined_data)} total samples across {len(trajectories)} trajectories")
    print(f"  Overall thrust range: [{combined_data['thrust'].min():.3f}, {combined_data['thrust'].max():.3f}] N")
    print(f"  Overall altitude range: [{combined_data['z'].min():.3f}, {combined_data['z'].max():.3f}] m")
    print(f"  NEW: kt = {sim.kt} (learnable)")
    print(f"  NEW: kq = {sim.kq} (learnable)")

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
    print(f"  Total learnable parameters: 6 (mass, Jxx, Jyy, Jzz, kt, kq)")


if __name__ == "__main__":
    main()
