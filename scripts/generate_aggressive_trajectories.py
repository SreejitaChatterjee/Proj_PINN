"""
Generate aggressive quadrotor trajectories with large attitude angles (+/-45 deg)
to test the physics fix under extreme conditions.

This script creates challenging flight maneuvers that would expose the physics
error in the old equation.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

class AggressiveQuadrotorSimulator:
    """Quadrotor simulator with aggressive maneuver capability"""

    def __init__(self):
        # Physical parameters (same as original)
        self.m = 0.068  # kg
        self.g = 9.81   # m/s^2
        self.Jxx = 6.86e-5
        self.Jyy = 9.20e-5
        self.Jzz = 1.366e-4

        # Cross-coupling terms
        self.t1 = (self.Jyy - self.Jzz) / self.Jxx
        self.t2 = (self.Jzz - self.Jxx) / self.Jyy
        self.t3 = (self.Jxx - self.Jyy) / self.Jzz

        # Control limits (more aggressive)
        self.Tmin = 0.0
        self.Tmax = 2.0
        self.txymax = 0.05  # Increased from 0.02
        self.tzmax = 0.02   # Increased from 0.01
        self.th = 1e-6

        # Controller gains (more aggressive for faster response)
        self.k1 = 2.0  # Roll outer loop (increased from 1.0)
        self.ki = 0.008  # Roll integral (increased from 0.004)
        self.k2 = 0.2  # Roll inner loop (increased from 0.1)

        self.k11 = 2.0  # Pitch outer loop
        self.ki1 = 0.008  # Pitch integral
        self.k21 = 0.2  # Pitch inner loop

        self.k12 = 2.0  # Yaw outer loop
        self.ki2 = 0.008  # Yaw integral
        self.k22 = 0.2  # Yaw inner loop

        self.kz1 = 3.0  # Altitude position gain (increased from 2.0)
        self.kz2 = 0.2  # Altitude integral (increased from 0.15)
        self.kv = -1.5  # Velocity feedback (increased from -1.0)

    def simulate_trajectory(self, z_ref, phi_ref, theta_ref, psi_ref, duration=5.0, dt=0.001):
        """
        Simulate one aggressive trajectory

        Args:
            z_ref: Target altitude (m, negative for NED)
            phi_ref: Target roll angle (radians)
            theta_ref: Target pitch angle (radians)
            psi_ref: Target yaw angle (radians)
            duration: Flight time (seconds)
            dt: Time step (seconds)
        """
        print(f"  Simulating: z={-z_ref:.1f}m, phi={np.degrees(phi_ref):.1f}deg, " +
              f"theta={np.degrees(theta_ref):.1f}deg, psi={np.degrees(psi_ref):.1f}deg")

        steps = int(duration / dt)
        data = []

        # Initial state (hovering at ground level)
        z, x, y = 0.0, 0.0, 0.0
        phi, theta, psi = 0.0, 0.0, 0.0
        p, q, r = 0.0, 0.0, 0.0
        u, v, w = 0.0, 0.0, 0.0

        # Integral terms
        sumphi, sumt, sumpsi, sumz = 0.0, 0.0, 0.0, 0.0

        for i in range(steps):
            # Roll controller
            sumphi += (phi_ref - phi)
            pref = self.k1 * (phi_ref - phi) + self.ki * sumphi * dt
            tx = self.k2 * (pref - p)
            tx = np.clip(tx, -self.txymax, self.txymax)
            if abs(tx) < self.th:
                tx = 0.0

            # Pitch controller
            sumt += (theta_ref - theta)
            qr = self.k11 * (theta_ref - theta) + self.ki1 * sumt * dt
            ty = self.k21 * (qr - q)
            ty = np.clip(ty, -self.txymax, self.txymax)
            if abs(ty) < self.th:
                ty = 0.0

            # Yaw controller
            sumpsi += (psi_ref - psi)
            rref = self.k12 * (psi_ref - psi) + self.ki2 * sumpsi * dt
            tz = self.k22 * (rref - r)
            tz = np.clip(tz, -self.tzmax, self.tzmax)
            if abs(tz) < self.th:
                tz = 0.0

            # Altitude controller
            sumz += (z_ref - z)
            vzr = self.kz1 * (z_ref - z) + self.kz2 * sumz * dt
            T = self.kv * (vzr - w)
            T = np.clip(T, self.Tmin, self.Tmax)

            # Rotational dynamics (Euler equations)
            pdot = self.t1 * q * r + tx / self.Jxx - 2 * p
            qdot = self.t2 * p * r + ty / self.Jyy - 2 * q
            rdot = self.t3 * p * q + tz / self.Jzz - 2 * r

            p += pdot * dt
            q += qdot * dt
            r += rdot * dt

            # Attitude kinematics
            phidot = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
            thetadot = np.cos(phi) * q - np.sin(phi) * r
            psidot = np.sin(phi) * q / np.cos(theta) + np.cos(phi) * r / np.cos(theta)

            phi += phidot * dt
            theta += thetadot * dt
            psi += psidot * dt

            # Wrap angles
            phi = np.arctan2(np.sin(phi), np.cos(phi))
            theta = np.arctan2(np.sin(theta), np.cos(theta))
            psi = np.arctan2(np.sin(psi), np.cos(psi))

            # Translational dynamics (BODY FRAME - correct for data generation)
            fz = -T
            fx, fy = 0.0, 0.0

            udot = r * v - q * w + fx / self.m - self.g * np.sin(theta) - 0.1 * u
            vdot = p * w - r * u + fy / self.m + self.g * np.cos(theta) * np.sin(phi) - 0.1 * v
            wdot = q * u - p * v + fz / self.m + self.g * np.cos(theta) * np.cos(phi) - 0.1 * w

            u += udot * dt
            v += vdot * dt
            w += wdot * dt

            # Position update (rotation from body to inertial frame)
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

            # Store data (vz is the INERTIAL FRAME vertical velocity = zdot)
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
                'vz': zdot,  # INERTIAL FRAME vertical velocity
                'mass': self.m,
                'inertia_xx': self.Jxx,
                'inertia_yy': self.Jyy,
                'inertia_zz': self.Jzz,
                'kt': 0.01,
                'kq': 7.8263e-4
            })

        return data

def main():
    print("=" * 80)
    print("GENERATING AGGRESSIVE TEST TRAJECTORIES")
    print("=" * 80)
    print()

    simulator = AggressiveQuadrotorSimulator()

    # Define aggressive trajectory setpoints
    aggressive_configs = [
        # (z_ref, phi_ref_deg, theta_ref_deg, psi_ref_deg, name)
        (-5.0, 45, 0, 0, "Max Roll (+45deg)"),
        (-5.0, -45, 0, 0, "Max Roll (-45deg)"),
        (-5.0, 0, 45, 0, "Max Pitch (+45deg)"),
        (-5.0, 0, -45, 0, "Max Pitch (-45deg)"),
        (-5.0, 30, 30, 0, "Combined Roll+Pitch (+30deg)"),
        (-5.0, -30, -30, 0, "Combined Roll+Pitch (-30deg)"),
        (-5.0, 20, -20, 45, "Complex 3-axis maneuver"),
        (-3.0, 40, 10, 15, "Aggressive climb with roll"),
        (-7.0, -35, -15, -20, "Deep descent with attitude"),
        (-10.0, 25, 25, 30, "High altitude complex maneuver"),
    ]

    all_trajectories = []

    print(f"Generating {len(aggressive_configs)} aggressive trajectories...")
    print()

    for idx, (z_ref, phi_deg, theta_deg, psi_deg, name) in enumerate(aggressive_configs):
        print(f"Trajectory {idx}: {name}")

        # Convert degrees to radians
        phi_ref = np.radians(phi_deg)
        theta_ref = np.radians(theta_deg)
        psi_ref = np.radians(psi_deg)

        # Simulate
        traj_data = simulator.simulate_trajectory(z_ref, phi_ref, theta_ref, psi_ref)

        all_trajectories.append({
            'name': name,
            'setpoints': {'z': z_ref, 'phi': phi_deg, 'theta': theta_deg, 'psi': psi_deg},
            'data': traj_data
        })

        print(f"  Generated {len(traj_data)} samples")
        print()

    # Save data
    output_file = 'aggressive_test_trajectories.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_trajectories, f)

    print("=" * 80)
    print(f"SAVED: {output_file}")
    print(f"Total trajectories: {len(all_trajectories)}")
    print(f"Total samples: {sum(len(t['data']) for t in all_trajectories)}")
    print("=" * 80)
    print()

    # Generate summary plot
    print("Generating summary visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for traj in all_trajectories:
        data = traj['data']
        time = [d['timestamp'] for d in data]
        roll = [np.degrees(d['roll']) for d in data]
        pitch = [np.degrees(d['pitch']) for d in data]
        z = [-d['z'] for d in data]  # Convert to altitude
        thrust = [d['thrust'] for d in data]

        axes[0, 0].plot(time, roll, alpha=0.7, linewidth=1, label=traj['name'])
        axes[0, 1].plot(time, pitch, alpha=0.7, linewidth=1)
        axes[1, 0].plot(time, z, alpha=0.7, linewidth=1)
        axes[1, 1].plot(time, thrust, alpha=0.7, linewidth=1)

    axes[0, 0].set_title('Roll Angle (degrees)', fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Roll (deg)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=45, color='r', linestyle='--', alpha=0.3, label='±45° limit')
    axes[0, 0].axhline(y=-45, color='r', linestyle='--', alpha=0.3)

    axes[0, 1].set_title('Pitch Angle (degrees)', fontweight='bold')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Pitch (deg)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=45, color='r', linestyle='--', alpha=0.3, label='±45° limit')
    axes[0, 1].axhline(y=-45, color='r', linestyle='--', alpha=0.3)

    axes[1, 0].set_title('Altitude (meters)', fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Altitude (m)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Thrust Force (Newtons)', fontweight='bold')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Thrust (N)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Aggressive Test Trajectories Overview\n(±45° attitude angles)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('aggressive_trajectories_overview.png', dpi=300, bbox_inches='tight')
    print("[SAVED] Plot: aggressive_trajectories_overview.png")
    print()

    # Statistics
    print("=" * 80)
    print("TRAJECTORY STATISTICS")
    print("=" * 80)
    for traj in all_trajectories:
        data = traj['data']
        roll_vals = [np.degrees(d['roll']) for d in data]
        pitch_vals = [np.degrees(d['pitch']) for d in data]

        print(f"{traj['name']}:")
        print(f"  Roll range: [{min(roll_vals):.1f}, {max(roll_vals):.1f}] deg")
        print(f"  Pitch range: [{min(pitch_vals):.1f}, {max(pitch_vals):.1f}] deg")
        print(f"  Samples: {len(data)}")
        print()

    print("=" * 80)
    print("NEXT STEPS:")
    print("1. Load these trajectories to test the corrected PINN models")
    print("2. Compare prediction accuracy with original (small-angle) data")
    print("3. Verify physics loss is lower with corrected equation")
    print("=" * 80)

if __name__ == "__main__":
    main()
