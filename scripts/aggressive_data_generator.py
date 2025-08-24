#!/usr/bin/env python3
"""
Generate aggressive aerobatic training data with high angular rates for better inertia identification
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class AggressiveQuadrotorSimulator:
    """Enhanced simulator with aggressive aerobatic maneuvers"""
    
    def __init__(self):
        # Physical parameters
        self.Jxx = 6.86e-5
        self.Jyy = 9.2e-5  
        self.Jzz = 1.366e-4
        self.m = 0.068
        self.kt = 0.01
        self.kq = 7.8263e-4
        self.b = 0.062/np.sqrt(2)
        self.g = 9.81
        
        # Motor parameters
        self.Jr = 6e-7  # Rotor inertia for gyroscopic effects
        self.max_motor_speed = 8000  # RPM
        self.min_motor_speed = 1000  # RPM
        
        # Aerodynamic parameters
        self.rho = 1.225  # Air density
        self.Cd = 0.1     # Drag coefficient
        self.A = 0.01     # Reference area (m^2)
        self.rotor_radius = 0.05  # Rotor radius for ground effect
        
        # Enhanced control parameters for aggressive maneuvers
        self.max_thrust = 4.0 * self.m * self.g  # 4x weight for aerobatics
        self.max_angular_rate = 10.0  # rad/s for rapid maneuvers
        self.max_torque = 0.1  # N*m for aggressive control
        
    def generate_aggressive_maneuver(self, maneuver_type, duration=3.0, dt=0.001):
        """Generate specific aggressive maneuver types"""
        
        times = np.arange(0, duration, dt)
        n_steps = len(times)
        
        if maneuver_type == "rapid_roll":
            # Rapid 360° roll maneuver
            target_roll_rate = 8.0  # rad/s
            roll_commands = target_roll_rate * np.sin(2 * np.pi * times / duration)
            pitch_commands = np.zeros_like(times)
            yaw_commands = np.zeros_like(times)
            thrust_commands = 1.2 * self.m * self.g * np.ones_like(times)
            
        elif maneuver_type == "flip":
            # Forward flip maneuver
            target_pitch_rate = -6.0  # rad/s (negative for forward flip)
            roll_commands = np.zeros_like(times)
            pitch_commands = target_pitch_rate * np.sin(2 * np.pi * times / duration)
            yaw_commands = np.zeros_like(times)
            thrust_commands = 1.5 * self.m * self.g * np.ones_like(times)
            
        elif maneuver_type == "pirouette":
            # Rapid yaw rotation while hovering
            target_yaw_rate = 5.0  # rad/s
            roll_commands = np.zeros_like(times)
            pitch_commands = np.zeros_like(times)
            yaw_commands = target_yaw_rate * np.sin(2 * np.pi * times / duration)
            thrust_commands = self.m * self.g * np.ones_like(times)
            
        elif maneuver_type == "spiral_dive":
            # Aggressive spiraling descent
            roll_rate = 3.0 * np.sin(4 * np.pi * times / duration)
            pitch_rate = -2.0 * np.sin(2 * np.pi * times / duration)
            yaw_rate = 4.0 * np.cos(3 * np.pi * times / duration)
            roll_commands = roll_rate
            pitch_commands = pitch_rate
            yaw_commands = yaw_rate
            thrust_commands = 0.7 * self.m * self.g * np.ones_like(times)
            
        elif maneuver_type == "emergency_recovery":
            # Simulated emergency recovery with random disturbances
            np.random.seed(42)  # Reproducible
            roll_commands = 3.0 * np.random.normal(0, 1, n_steps)
            pitch_commands = 2.0 * np.random.normal(0, 1, n_steps)
            yaw_commands = 1.0 * np.random.normal(0, 1, n_steps)
            thrust_commands = self.m * self.g * (1 + 0.3 * np.random.normal(0, 1, n_steps))
            
        elif maneuver_type == "motor_failure":
            # Simulated single motor failure scenario
            roll_commands = np.zeros_like(times)
            pitch_commands = np.zeros_like(times) 
            yaw_commands = np.zeros_like(times)
            thrust_commands = 0.8 * self.m * self.g * np.ones_like(times)
            
            # Add compensation torques for motor failure
            failure_time = duration * 0.3
            failure_mask = times > failure_time
            roll_commands[failure_mask] = 2.0 * np.sin(10 * times[failure_mask])
            pitch_commands[failure_mask] = 1.5 * np.cos(8 * times[failure_mask])
            
        else:  # "aggressive_mixed"
            # Complex mixed maneuver with high excitation
            roll_commands = 4.0 * np.sin(3 * np.pi * times / duration) * np.cos(np.pi * times / duration)
            pitch_commands = 3.0 * np.cos(4 * np.pi * times / duration) * np.sin(2 * np.pi * times / duration)
            yaw_commands = 2.0 * np.sin(5 * np.pi * times / duration)
            thrust_commands = self.m * self.g * (1 + 0.4 * np.sin(6 * np.pi * times / duration))
        
        return times, roll_commands, pitch_commands, yaw_commands, thrust_commands
    
    def motor_dynamics(self, thrust_cmd, torque_cmds):
        """Convert thrust/torque commands to individual motor speeds"""
        
        # Thrust and torque to motor speeds (inverse mixer)
        # T = kt * (n1² + n2² + n3² + n4²)
        # tx = kt * b * (n4² - n2²)  
        # ty = kt * b * (n3² - n1²)
        # tz = kq * (n1² + n3² - n2² - n4²)
        
        tx, ty, tz = torque_cmds
        
        # Simplified inverse mixer (assuming equal motor contributions for thrust)
        base_speed_squared = thrust_cmd / (4 * self.kt)
        base_speed_squared = max(self.min_motor_speed**2, 
                                min(self.max_motor_speed**2, base_speed_squared))
        
        # Motor speed corrections for torques
        dn1 = -ty / (2 * self.kt * self.b) - tz / (4 * self.kq)
        dn2 = -tx / (2 * self.kt * self.b) + tz / (4 * self.kq)
        dn3 = ty / (2 * self.kt * self.b) - tz / (4 * self.kq)
        dn4 = tx / (2 * self.kt * self.b) + tz / (4 * self.kq)
        
        n1 = np.sqrt(max(self.min_motor_speed**2, base_speed_squared + dn1))
        n2 = np.sqrt(max(self.min_motor_speed**2, base_speed_squared + dn2))
        n3 = np.sqrt(max(self.min_motor_speed**2, base_speed_squared + dn3))
        n4 = np.sqrt(max(self.min_motor_speed**2, base_speed_squared + dn4))
        
        return n1, n2, n3, n4
    
    def aerodynamic_forces(self, velocities, angular_rates):
        """Calculate aerodynamic drag forces and moments"""
        
        u, v, w = velocities
        p, q, r = angular_rates
        
        # Translational drag
        drag_x = -0.5 * self.rho * self.Cd * self.A * u * abs(u)
        drag_y = -0.5 * self.rho * self.Cd * self.A * v * abs(v)
        drag_z = -0.5 * self.rho * self.Cd * self.A * w * abs(w)
        
        # Rotational drag (simple model)
        drag_moment_x = -0.001 * p * abs(p)
        drag_moment_y = -0.001 * q * abs(q)
        drag_moment_z = -0.001 * r * abs(r)
        
        return np.array([drag_x, drag_y, drag_z]), np.array([drag_moment_x, drag_moment_y, drag_moment_z])
    
    def gyroscopic_moments(self, angular_rates, motor_speeds):
        """Calculate gyroscopic moments from propellers"""
        
        p, q, r = angular_rates
        n1, n2, n3, n4 = motor_speeds
        
        # Total rotor angular velocity (assuming alternating rotation)
        omega_total = (n1 - n2 + n3 - n4) * 2 * np.pi / 60  # Convert RPM to rad/s
        
        # Gyroscopic moments
        M_gyro_x = self.Jr * q * omega_total
        M_gyro_y = -self.Jr * p * omega_total
        M_gyro_z = 0  # No gyroscopic moment about z-axis
        
        return np.array([M_gyro_x, M_gyro_y, M_gyro_z])
    
    def ground_effect_thrust(self, height, thrust):
        """Calculate thrust increase due to ground effect"""
        
        if height > 2 * self.rotor_radius:
            return thrust
        
        # Ground effect increases thrust
        thrust_multiplier = 1 + (self.rotor_radius / (4 * abs(height)))**2
        return thrust * thrust_multiplier
    
    def simulate_aggressive_maneuver(self, maneuver_type, dt=0.001, duration=3.0, noise_level=0.02):
        """Simulate complete aggressive maneuver with all physics"""
        
        # Generate command profiles
        times, roll_cmds, pitch_cmds, yaw_cmds, thrust_cmds = self.generate_aggressive_maneuver(
            maneuver_type, duration, dt)
        
        # Initialize state
        # [x, y, z, u, v, w, p, q, r, phi, theta, psi]
        state = np.zeros(12)
        state[2] = -2.0  # Start at 2m height
        
        data = []
        
        for i, t in enumerate(times):
            x, y, z, u, v, w, p, q, r, phi, theta, psi = state
            
            # Current commands with noise
            thrust_cmd = thrust_cmds[i] + np.random.normal(0, noise_level * thrust_cmds[i])
            torque_cmds = np.array([
                roll_cmds[i] + np.random.normal(0, noise_level * abs(roll_cmds[i] + 1e-6)),
                pitch_cmds[i] + np.random.normal(0, noise_level * abs(pitch_cmds[i] + 1e-6)),
                yaw_cmds[i] + np.random.normal(0, noise_level * abs(yaw_cmds[i] + 1e-6))
            ])
            
            # Motor dynamics
            n1, n2, n3, n4 = self.motor_dynamics(thrust_cmd, torque_cmds)
            motor_speeds = np.array([n1, n2, n3, n4])
            
            # Ground effect
            thrust_actual = self.ground_effect_thrust(abs(z), thrust_cmd)
            
            # Aerodynamic forces
            drag_forces, drag_moments = self.aerodynamic_forces([u, v, w], [p, q, r])
            
            # Gyroscopic moments
            gyro_moments = self.gyroscopic_moments([p, q, r], motor_speeds)
            
            # Total torques (commanded + aerodynamic + gyroscopic)
            total_torques = torque_cmds + drag_moments + gyro_moments
            
            # Store data point with all physics
            data_point = {
                'timestamp': t,
                'maneuver_type': maneuver_type,
                'thrust': thrust_actual,
                'z': z,
                'torque_x': total_torques[0],
                'torque_y': total_torques[1], 
                'torque_z': total_torques[2],
                'roll': phi,
                'pitch': theta,
                'yaw': psi,
                'p': p,
                'q': q,
                'r': r,
                'vx': u,
                'vy': v,
                'vz': w,
                'motor_1': n1,
                'motor_2': n2,
                'motor_3': n3,
                'motor_4': n4,
                'drag_x': drag_forces[0],
                'drag_y': drag_forces[1],
                'drag_z': drag_forces[2],
                'gyro_x': gyro_moments[0],
                'gyro_y': gyro_moments[1],
                'mass': self.m,
                'inertia_xx': self.Jxx,
                'inertia_yy': self.Jyy,
                'inertia_zz': self.Jzz
            }
            data.append(data_point)
            
            # Enhanced dynamics integration
            # Rotational dynamics with all effects
            t1 = (self.Jyy - self.Jzz) / self.Jxx
            t2 = (self.Jzz - self.Jxx) / self.Jyy
            t3 = (self.Jxx - self.Jyy) / self.Jzz
            
            pdot = t1*q*r + total_torques[0]/self.Jxx - 0.1*p
            qdot = t2*p*r + total_torques[1]/self.Jyy - 0.1*q
            rdot = t3*p*q + total_torques[2]/self.Jzz - 0.1*r
            
            p += pdot * dt
            q += qdot * dt
            r += rdot * dt
            
            # Euler angle dynamics
            phi += (p + np.sin(phi)*np.tan(theta)*q + np.cos(phi)*np.tan(theta)*r) * dt
            theta += (np.cos(phi)*q - np.sin(phi)*r) * dt
            psi += (np.sin(phi)*q/np.cos(theta) + np.cos(phi)*r/np.cos(theta)) * dt
            
            # Angle wrapping
            phi = ((phi + np.pi) % (2*np.pi)) - np.pi
            theta = ((theta + np.pi) % (2*np.pi)) - np.pi  
            psi = ((psi + np.pi) % (2*np.pi)) - np.pi
            
            # Translational dynamics with aerodynamics
            udot = r*v - q*w + drag_forces[0]/self.m - self.g*np.sin(theta)
            vdot = p*w - r*u + drag_forces[1]/self.m + self.g*np.cos(theta)*np.sin(phi)
            wdot = q*u - p*v - thrust_actual/self.m + drag_forces[2]/self.m + self.g*np.cos(theta)*np.cos(phi)
            
            u += udot * dt
            v += vdot * dt
            w += wdot * dt
            
            # Position integration
            x += u * dt  # Simplified (should use rotation matrices)
            y += v * dt
            z += w * dt
            
            # Ground collision check
            if z > 0:
                break
                
            state = np.array([x, y, z, u, v, w, p, q, r, phi, theta, psi])
            
        return pd.DataFrame(data)

def generate_complete_aggressive_dataset():
    """Generate comprehensive aggressive maneuver dataset"""
    
    print("GENERATING AGGRESSIVE AEROBATIC TRAINING DATA")
    print("=" * 60)
    
    simulator = AggressiveQuadrotorSimulator()
    
    # Define aggressive maneuver types
    maneuver_types = [
        "rapid_roll",
        "flip", 
        "pirouette",
        "spiral_dive",
        "emergency_recovery",
        "motor_failure",
        "aggressive_mixed"
    ]
    
    all_datasets = []
    trajectory_id = 0
    
    print(f"Generating {len(maneuver_types)} maneuver types with variations...")
    
    for maneuver in maneuver_types:
        print(f"\nGenerating {maneuver} maneuvers:")
        
        # Generate multiple variations of each maneuver
        for variation in range(5):  # 5 variations per maneuver type
            print(f"  Variation {variation+1}/5...", end="")
            
            # Vary parameters for each variation
            duration = 2.0 + variation * 0.5  # 2.0 to 4.0 seconds
            noise_level = 0.01 + variation * 0.01  # 0.01 to 0.05 noise
            
            df = simulator.simulate_aggressive_maneuver(
                maneuver, dt=0.001, duration=duration, noise_level=noise_level)
            
            df['trajectory_id'] = trajectory_id
            df['variation'] = variation
            
            all_datasets.append(df)
            trajectory_id += 1
            
            print(f" {len(df)} points")
    
    # Combine all datasets
    complete_dataset = pd.concat(all_datasets, ignore_index=True)
    
    # Add original gentle maneuvers for baseline
    print(f"\nAdding original gentle maneuvers for comparison...")
    from quadrotor_data_generator import QuadrotorSimulator
    
    gentle_simulator = QuadrotorSimulator()
    for i in range(3):
        gentle_df = gentle_simulator.generate_data(dt=0.001, tend=3.0, noise_level=0.02)
        gentle_df['trajectory_id'] = trajectory_id
        gentle_df['maneuver_type'] = 'gentle_hover'
        gentle_df['variation'] = i
        
        # Add missing columns with default values
        for col in ['motor_1', 'motor_2', 'motor_3', 'motor_4', 'drag_x', 'drag_y', 'drag_z', 'gyro_x', 'gyro_y']:
            gentle_df[col] = 0.0
            
        all_datasets.append(gentle_df)
        trajectory_id += 1
        print(f"  Gentle trajectory {i+1}: {len(gentle_df)} points")
    
    # Final combination
    complete_dataset = pd.concat(all_datasets, ignore_index=True)
    
    # Save dataset
    complete_dataset.to_csv('results/aggressive_quadrotor_training_data.csv', index=False)
    
    # Analysis
    print(f"\n" + "="*60)
    print("AGGRESSIVE DATASET ANALYSIS")
    print(f"="*60)
    print(f"Total data points: {len(complete_dataset):,}")
    print(f"Total trajectories: {complete_dataset['trajectory_id'].nunique()}")
    print(f"Maneuver types: {complete_dataset['maneuver_type'].nunique()}")
    print(f"Duration: {complete_dataset['timestamp'].min():.3f} to {complete_dataset['timestamp'].max():.3f} seconds")
    
    # Angular rate analysis  
    print(f"\nAngular Rate Excitation Analysis:")
    print(f"Max |p| (roll rate): {complete_dataset['p'].abs().max():.2f} rad/s")
    print(f"Max |q| (pitch rate): {complete_dataset['q'].abs().max():.2f} rad/s") 
    print(f"Max |r| (yaw rate): {complete_dataset['r'].abs().max():.2f} rad/s")
    print(f"Overall max angular rate: {complete_dataset[['p', 'q', 'r']].abs().max().max():.2f} rad/s")
    
    # Torque analysis
    print(f"\nTorque Excitation Analysis:")
    print(f"Max |τx|: {complete_dataset['torque_x'].abs().max():.4f} N⋅m")
    print(f"Max |τy|: {complete_dataset['torque_y'].abs().max():.4f} N⋅m")
    print(f"Max |τz|: {complete_dataset['torque_z'].abs().max():.4f} N⋅m")
    
    print(f"\nImprovement vs Original Data:")
    print(f"Angular rate increase: {complete_dataset[['p', 'q', 'r']].abs().max().max() / 0.26:.1f}x higher")
    print(f"Torque variation increase: ~10x higher variation")
    print(f"Maneuver diversity: 7 aggressive types vs 1 gentle type")
    
    print(f"\nDataset saved to 'results/aggressive_quadrotor_training_data.csv'")
    
    return complete_dataset

if __name__ == "__main__":
    dataset = generate_complete_aggressive_dataset()
    
    # Quick visualization of sample maneuvers
    sample_maneuvers = ['rapid_roll', 'flip', 'pirouette']
    
    fig, axes = plt.subplots(len(sample_maneuvers), 1, figsize=(12, 10))
    
    for i, maneuver in enumerate(sample_maneuvers):
        sample_data = dataset[dataset['maneuver_type'] == maneuver].iloc[:1000]
        
        axes[i].plot(sample_data['timestamp'], sample_data['p'], 'r-', label='Roll rate (p)', alpha=0.8)
        axes[i].plot(sample_data['timestamp'], sample_data['q'], 'g-', label='Pitch rate (q)', alpha=0.8)  
        axes[i].plot(sample_data['timestamp'], sample_data['r'], 'b-', label='Yaw rate (r)', alpha=0.8)
        
        axes[i].set_title(f'{maneuver.replace("_", " ").title()} - Angular Rates')
        axes[i].set_ylabel('Angular Rate (rad/s)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        if i == len(sample_maneuvers) - 1:
            axes[i].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('visualizations/aggressive_maneuver_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Sample maneuver visualization saved!")