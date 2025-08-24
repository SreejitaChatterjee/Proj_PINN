#!/usr/bin/env python3
"""
Simplified aggressive training data generator
"""

import numpy as np
import pandas as pd

def generate_aggressive_training_data():
    """Generate aggressive maneuver data with high angular rates"""
    
    print("GENERATING AGGRESSIVE TRAINING DATA")
    print("=" * 50)
    
    # Physical parameters
    Jxx, Jyy, Jzz = 6.86e-5, 9.2e-5, 1.366e-4
    m, g = 0.068, 9.81
    
    all_data = []
    trajectory_id = 0
    
    # Maneuver types with high angular excitation
    maneuvers = [
        {"name": "rapid_roll", "max_rate": 8.0, "axis": "p"},
        {"name": "aggressive_pitch", "max_rate": 6.0, "axis": "q"}, 
        {"name": "fast_yaw", "max_rate": 5.0, "axis": "r"},
        {"name": "mixed_aggressive", "max_rate": 4.0, "axis": "all"}
    ]
    
    for maneuver in maneuvers:
        print(f"Generating {maneuver['name']} maneuvers...")
        
        for variation in range(8):  # 8 variations per type
            dt = 0.001
            duration = 2.0 + variation * 0.3
            times = np.arange(0, duration, dt)
            
            # Initialize states
            p = q = r = 0.0
            phi = theta = psi = 0.0
            z = -2.0  # Start at 2m height
            vz = 0.0
            
            trajectory_data = []
            
            for i, t in enumerate(times):
                # Generate aggressive commands
                if maneuver["axis"] == "p":
                    target_p = maneuver["max_rate"] * np.sin(4 * np.pi * t / duration)
                    target_q = 0.5 * np.sin(2 * np.pi * t / duration)
                    target_r = 0.3 * np.cos(3 * np.pi * t / duration)
                elif maneuver["axis"] == "q":
                    target_p = 0.3 * np.cos(2 * np.pi * t / duration)
                    target_q = maneuver["max_rate"] * np.sin(3 * np.pi * t / duration)
                    target_r = 0.2 * np.sin(4 * np.pi * t / duration)
                elif maneuver["axis"] == "r":
                    target_p = 0.2 * np.sin(3 * np.pi * t / duration)
                    target_q = 0.4 * np.cos(2 * np.pi * t / duration)
                    target_r = maneuver["max_rate"] * np.sin(2 * np.pi * t / duration)
                else:  # "all"
                    target_p = maneuver["max_rate"] * 0.6 * np.sin(3 * np.pi * t / duration)
                    target_q = maneuver["max_rate"] * 0.8 * np.cos(2 * np.pi * t / duration)  
                    target_r = maneuver["max_rate"] * 0.4 * np.sin(4 * np.pi * t / duration)
                
                # Calculate required torques for target rates
                # Simplified control: torque proportional to rate error
                kp = 0.1  # Proportional gain
                
                torque_x = kp * (target_p - p) + 0.01 * np.random.normal()
                torque_y = kp * (target_q - q) + 0.01 * np.random.normal()
                torque_z = kp * (target_r - r) + 0.01 * np.random.normal()
                
                # Thrust for altitude maintenance with variation
                thrust = m * g * (1 + 0.2 * np.sin(np.pi * t / duration)) + 0.1 * np.random.normal()
                
                # Store data point
                data_point = {
                    'timestamp': t,
                    'maneuver_type': maneuver['name'],
                    'thrust': thrust,
                    'z': z,
                    'torque_x': torque_x,
                    'torque_y': torque_y,
                    'torque_z': torque_z,
                    'roll': phi,
                    'pitch': theta,
                    'yaw': psi,
                    'p': p,
                    'q': q,
                    'r': r,
                    'vx': 0.1 * np.random.normal(),
                    'vy': 0.1 * np.random.normal(),
                    'vz': vz,
                    'mass': m,
                    'inertia_xx': Jxx,
                    'inertia_yy': Jyy,
                    'inertia_zz': Jzz,
                    'trajectory_id': trajectory_id
                }
                trajectory_data.append(data_point)
                
                # Simple dynamics integration
                # Cross-coupling terms
                t1 = (Jyy - Jzz) / Jxx
                t2 = (Jzz - Jxx) / Jyy
                t3 = (Jxx - Jyy) / Jzz
                
                # Angular acceleration
                pdot = t1*q*r + torque_x/Jxx - 0.5*p  # Higher damping
                qdot = t2*p*r + torque_y/Jyy - 0.5*q
                rdot = t3*p*q + torque_z/Jzz - 0.5*r
                
                # Update angular rates
                p += pdot * dt
                q += qdot * dt
                r += rdot * dt
                
                # Update angles (simplified)
                phi += p * dt
                theta += q * dt
                psi += r * dt
                
                # Limit angles
                phi = np.clip(phi, -np.pi/3, np.pi/3)
                theta = np.clip(theta, -np.pi/3, np.pi/3)
                psi = ((psi + np.pi) % (2*np.pi)) - np.pi
                
                # Vertical dynamics
                wdot = -thrust/m + g*np.cos(theta)*np.cos(phi) - 0.2*vz
                vz += wdot * dt
                z += vz * dt
                
                # Stop if too low
                if z > -0.1:
                    break
            
            all_data.extend(trajectory_data)
            trajectory_id += 1
            print(f"  Variation {variation+1}: {len(trajectory_data)} points")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save dataset
    df.to_csv('results/aggressive_quadrotor_training_data.csv', index=False)
    
    # Analysis
    print(f"\nAGGRESSIVE DATASET ANALYSIS")
    print(f"=" * 40)
    print(f"Total points: {len(df):,}")
    print(f"Trajectories: {df['trajectory_id'].nunique()}")
    print(f"Duration: {df['timestamp'].min():.1f} to {df['timestamp'].max():.1f}s")
    
    print(f"\nAngular Rate Excitation:")
    print(f"Max |p|: {df['p'].abs().max():.2f} rad/s")
    print(f"Max |q|: {df['q'].abs().max():.2f} rad/s")
    print(f"Max |r|: {df['r'].abs().max():.2f} rad/s")
    print(f"Peak rate: {df[['p', 'q', 'r']].abs().max().max():.2f} rad/s")
    
    improvement = df[['p', 'q', 'r']].abs().max().max() / 0.26
    print(f"\nImprovement: {improvement:.1f}x higher angular excitation!")
    
    return df

if __name__ == "__main__":
    dataset = generate_aggressive_training_data()