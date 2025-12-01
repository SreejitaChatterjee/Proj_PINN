"""
Generate 50-100 highly diverse training trajectories for improved PINN training
Focus on exciting all dynamic modes, especially for inertia parameter identification
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from generate_quadrotor_data import QuadrotorSimulator

def generate_diverse_trajectory_configs(n_trajectories=100):
    """
    Generate highly diverse trajectory configurations

    Diversity strategies:
    1. Wide range of amplitudes (gentle to aggressive)
    2. Various frequency combinations (slow to fast)
    3. Asymmetric maneuvers (excite cross-coupling)
    4. Different phase relationships between axes
    5. Mixed maneuver types
    """

    configs = []
    np.random.seed(42)  # Reproducibility

    print(f"Generating {n_trajectories} diverse trajectory configurations...")

    for i in range(n_trajectories):
        # Random periods for each axis (0.8s to 4.0s)
        phi_period = np.random.uniform(0.8, 4.0)
        theta_period = np.random.uniform(0.8, 4.0)
        psi_period = np.random.uniform(0.8, 4.0)
        z_period = np.random.uniform(0.8, 4.0)

        # Random amplitudes with strategic distribution
        if i < 20:  # First 20: Aggressive maneuvers for inertia excitation
            phi_amp = np.random.uniform(15, 30) * np.pi/180  # 15-30 degrees
            theta_amp = np.random.uniform(10, 20) * np.pi/180  # 10-20 degrees
            psi_amp = np.random.uniform(10, 25) * np.pi/180  # 10-25 degrees
            z_amp = np.random.uniform(3.0, 5.0)  # 3-5 meters altitude change

        elif i < 50:  # Next 30: Moderate maneuvers
            phi_amp = np.random.uniform(5, 15) * np.pi/180  # 5-15 degrees
            theta_amp = np.random.uniform(3, 12) * np.pi/180  # 3-12 degrees
            psi_amp = np.random.uniform(5, 15) * np.pi/180  # 5-15 degrees
            z_amp = np.random.uniform(2.0, 4.0)  # 2-4 meters

        else:  # Remaining: Gentle maneuvers
            phi_amp = np.random.uniform(2, 10) * np.pi/180  # 2-10 degrees
            theta_amp = np.random.uniform(2, 8) * np.pi/180  # 2-8 degrees
            psi_amp = np.random.uniform(3, 12) * np.pi/180  # 3-12 degrees
            z_amp = np.random.uniform(1.0, 3.0)  # 1-3 meters

        # Random asymmetry (excites cross-coupling)
        asymmetry_factor = np.random.uniform(0.3, 1.0)

        # Altitude setpoint (between -8m and -2m)
        z_low = np.random.uniform(-8.0, -4.0)
        z_high = z_low + z_amp

        # Create asymmetric or symmetric patterns
        if np.random.rand() < 0.3:  # 30% asymmetric
            phi_low = -phi_amp * asymmetry_factor
            phi_high = phi_amp
            theta_low = -theta_amp
            theta_high = theta_amp * asymmetry_factor
            psi_low = -psi_amp * asymmetry_factor
            psi_high = psi_amp
        else:  # 70% symmetric
            phi_low = -phi_amp
            phi_high = phi_amp
            theta_low = -theta_amp
            theta_high = theta_amp
            psi_low = -psi_amp
            psi_high = psi_amp

        config = {
            'phi': (phi_period, phi_low, phi_high),
            'theta': (theta_period, theta_low, theta_high),
            'psi': (psi_period, psi_low, psi_high),
            'z': (z_period, z_low, z_high),
            'desc': f"Trajectory {i}: phi_amp={phi_amp*180/np.pi:.1f}deg, periods=({phi_period:.1f},{theta_period:.1f},{psi_period:.1f},{z_period:.1f}s)"
        }
        configs.append(config)

    return configs

def main():
    print("="*80)
    print("GENERATING DIVERSE TRAINING DATA FOR IMPROVED PINN")
    print("="*80)

    # Configuration
    n_trajectories = 100
    tend = 5.0  # 5 seconds per trajectory
    dt = 0.001  # 1ms timestep

    output_file = Path(__file__).parent.parent / 'data' / 'quadrotor_training_data_diverse.csv'

    # Generate configurations
    configs = generate_diverse_trajectory_configs(n_trajectories)

    # Simulate trajectories
    sim = QuadrotorSimulator()
    all_data = []

    print(f"\nSimulating {n_trajectories} trajectories...")
    print(f"  Duration: {tend}s each")
    print(f"  Timestep: {dt}s")
    print(f"  Expected samples per trajectory: {int(tend/dt)}")

    for i, config in enumerate(configs):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_trajectories} trajectories ({(i+1)/n_trajectories*100:.0f}%)")

        # Simulate trajectory
        df_traj = sim.simulate_trajectory(
            phi_config=config['phi'],
            theta_config=config['theta'],
            psi_config=config['psi'],
            z_config=config['z'],
            dt=dt,
            tend=tend
        )

        # Add trajectory ID
        df_traj['trajectory_id'] = i
        all_data.append(df_traj)

    # Combine all trajectories
    df_combined = pd.concat(all_data, ignore_index=True)

    # Statistics
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total trajectories: {n_trajectories}")
    print(f"Total samples: {len(df_combined):,}")
    print(f"Avg samples per trajectory: {len(df_combined)/n_trajectories:.0f}")
    print(f"\nState ranges:")

    state_cols = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    for col in state_cols:
        if col in df_combined.columns:
            print(f"  {col:8s}: [{df_combined[col].min():8.3f}, {df_combined[col].max():8.3f}]")

    # Save to CSV
    print(f"\nSaving to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_file, index=False)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")

    # Analyze diversity
    print(f"\n{'='*80}")
    print("DIVERSITY ANALYSIS")
    print(f"{'='*80}")

    # Amplitude diversity
    roll_range = df_combined.groupby('trajectory_id')['roll'].apply(lambda x: x.max() - x.min())
    pitch_range = df_combined.groupby('trajectory_id')['pitch'].apply(lambda x: x.max() - x.min())
    yaw_range = df_combined.groupby('trajectory_id')['yaw'].apply(lambda x: x.max() - x.min())

    print(f"Roll amplitude range: {roll_range.min()*180/np.pi:.1f}° to {roll_range.max()*180/np.pi:.1f}°")
    print(f"Pitch amplitude range: {pitch_range.min()*180/np.pi:.1f}° to {pitch_range.max()*180/np.pi:.1f}°")
    print(f"Yaw amplitude range: {yaw_range.min()*180/np.pi:.1f}° to {yaw_range.max()*180/np.pi:.1f}°")

    # Velocity diversity
    vx_max = df_combined.groupby('trajectory_id')['vx'].apply(lambda x: x.abs().max())
    vy_max = df_combined.groupby('trajectory_id')['vy'].apply(lambda x: x.abs().max())
    vz_max = df_combined.groupby('trajectory_id')['vz'].apply(lambda x: x.abs().max())

    print(f"\nMax velocity ranges:")
    print(f"  Vx: 0 to {vx_max.max():.2f} m/s")
    print(f"  Vy: 0 to {vy_max.max():.2f} m/s")
    print(f"  Vz: 0 to {vz_max.max():.2f} m/s")

    # Angular rate diversity
    p_max = df_combined.groupby('trajectory_id')['p'].apply(lambda x: x.abs().max())
    q_max = df_combined.groupby('trajectory_id')['q'].apply(lambda x: x.abs().max())
    r_max = df_combined.groupby('trajectory_id')['r'].apply(lambda x: x.abs().max())

    print(f"\nMax angular rate ranges:")
    print(f"  p: 0 to {p_max.max():.3f} rad/s ({p_max.max()*180/np.pi:.1f}°/s)")
    print(f"  q: 0 to {q_max.max():.3f} rad/s ({q_max.max()*180/np.pi:.1f}°/s)")
    print(f"  r: 0 to {r_max.max():.3f} rad/s ({r_max.max()*180/np.pi:.1f}°/s)")

    print(f"\n{'='*80}")
    print("SUCCESS!")
    print(f"{'='*80}")
    print(f"Generated {n_trajectories} diverse trajectories")
    print(f"Dataset: {output_file}")
    print(f"\nKey improvements over original dataset:")
    print(f"  1. 10× more trajectories (10 → 100)")
    print(f"  2. Wide amplitude range (gentle to aggressive)")
    print(f"  3. 30% asymmetric maneuvers (excites cross-coupling)")
    print(f"  4. Varied frequency combinations")
    print(f"  5. Better coverage of dynamic modes")
    print(f"\nNext step: Create train/val/test split and retrain model")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
