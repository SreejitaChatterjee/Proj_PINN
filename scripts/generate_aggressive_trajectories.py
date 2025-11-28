#!/usr/bin/env python3
"""
Generate aggressive multi-axis trajectories for improved inertia parameter identification.

This script creates trajectories with:
- Large angles (±45-60°) to excite cross-coupling dynamics
- Simultaneous multi-axis rotations (p, q, r all active)
- Fast maneuvers to increase angular accelerations
- Higher frequency content for better observability

These trajectories complement the standard training data and provide
stronger gradient signals for learning inertia parameters (Jxx, Jyy, Jzz).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import the existing simulator (reuse MATLAB parameters - NO changes to physical constants)
sys.path.append(str(Path(__file__).parent))
from generate_quadrotor_data import QuadrotorSimulator

def generate_aggressive_inertia_trajectories():
    """
    Generate 5 aggressive trajectories specifically designed for inertia identification.

    Uses the existing QuadrotorSimulator with MATLAB parameters unchanged.
    Only modifies reference trajectories to be more aggressive.

    Returns:
        pd.DataFrame: Combined trajectory data with trajectory_id column
    """

    sim = QuadrotorSimulator()

    # Define 5 aggressive trajectory configurations
    # These have larger angles and faster dynamics than standard trajectories
    aggressive_trajectories = [
        {
            'phi': (0.8, -45*np.pi/180, 45*np.pi/180),      # ±45° roll, fast (0.8s period)
            'theta': (0.9, -40*np.pi/180, 40*np.pi/180),    # ±40° pitch
            'psi': (1.0, -45*np.pi/180, 45*np.pi/180),      # ±45° yaw
            'z': (1.5, -6.0, -4.0),
            'desc': "Aggressive multi-axis ±45° maneuvers"
        },
        {
            'phi': (1.0, -50*np.pi/180, 50*np.pi/180),      # ±50° roll
            'theta': (1.1, -45*np.pi/180, 45*np.pi/180),    # ±45° pitch
            'psi': (1.2, -50*np.pi/180, 50*np.pi/180),      # ±50° yaw
            'z': (1.8, -7.0, -5.0),
            'desc': "Very aggressive ±50° simultaneous rotations"
        },
        {
            'phi': (0.6, -60*np.pi/180, 60*np.pi/180),      # ±60° roll, very fast
            'theta': (0.7, -50*np.pi/180, 50*np.pi/180),    # ±50° pitch
            'psi': (0.8, -55*np.pi/180, 55*np.pi/180),      # ±55° yaw
            'z': (1.2, -5.0, -3.0),
            'desc': "Extreme ±60° fast multi-axis flips"
        },
        {
            'phi': (1.5, -55*np.pi/180, 55*np.pi/180),      # ±55° asymmetric timing
            'theta': (1.2, -60*np.pi/180, 60*np.pi/180),    # ±60° pitch
            'psi': (1.8, -45*np.pi/180, 45*np.pi/180),      # ±45° yaw
            'z': (2.0, -6.0, -4.0),
            'desc': "Mixed-rate ±55-60° excitation"
        },
        {
            'phi': (0.9, -50*np.pi/180, 50*np.pi/180),      # ±50° coordinated
            'theta': (0.9, -50*np.pi/180, 50*np.pi/180),    # ±50° synchronized
            'psi': (0.9, -50*np.pi/180, 50*np.pi/180),      # ±50° all axes
            'z': (1.5, -6.0, -4.0),
            'desc': "Synchronized ±50° all-axis rotation"
        },
    ]

    all_data = []

    print("="*80)
    print("GENERATING AGGRESSIVE TRAJECTORIES FOR INERTIA IDENTIFICATION")
    print("="*80)
    print(f"\nThese trajectories use ±45-60° angles (vs ±20° in standard training)")
    print(f"This excites cross-coupling dynamics for better Jxx, Jyy, Jzz identification")
    print(f"\nPhysical parameters from MATLAB (unchanged):")
    print(f"  Jxx = {sim.Jxx:.2e} kg·m²")
    print(f"  Jyy = {sim.Jyy:.2e} kg·m²")
    print(f"  Jzz = {sim.Jzz:.2e} kg·m²")
    print(f"  mass = {sim.m} kg")
    print(f"  kt = {sim.kt}, kq = {sim.kq:.2e}")
    print()

    for traj_id, traj_config in enumerate(aggressive_trajectories):
        print(f"\n{'='*80}")
        print(f"Trajectory {traj_id}: {traj_config['desc']}")
        print(f"  Roll:  period={traj_config['phi'][0]:.1f}s, range=[{traj_config['phi'][1]*180/np.pi:.1f}°, {traj_config['phi'][2]*180/np.pi:.1f}°]")
        print(f"  Pitch: period={traj_config['theta'][0]:.1f}s, range=[{traj_config['theta'][1]*180/np.pi:.1f}°, {traj_config['theta'][2]*180/np.pi:.1f}°]")
        print(f"  Yaw:   period={traj_config['psi'][0]:.1f}s, range=[{traj_config['psi'][1]*180/np.pi:.1f}°, {traj_config['psi'][2]*180/np.pi:.1f}°]")
        print(f"  Alt:   period={traj_config['z'][0]:.1f}s, range=[{traj_config['z'][1]:.1f}m, {traj_config['z'][2]:.1f}m]")

        # Simulate trajectory using existing simulator (MATLAB parameters unchanged)
        traj_data = sim.simulate_trajectory(
            phi_config=traj_config['phi'],
            theta_config=traj_config['theta'],
            psi_config=traj_config['psi'],
            z_config=traj_config['z'],
            dt=0.001,
            tend=5.0
        )

        # Add trajectory ID
        traj_data['trajectory_id'] = traj_id + 100  # Start from 100 to distinguish from standard trajectories

        all_data.append(traj_data)

        print(f"  Generated {len(traj_data)} samples over {traj_data['timestamp'].max():.2f}s")

        # Calculate maximum angular rates achieved
        max_p = traj_data['p'].abs().max()
        max_q = traj_data['q'].abs().max()
        max_r = traj_data['r'].abs().max()
        print(f"  Max angular rates: p={max_p:.2f}, q={max_q:.2f}, r={max_r:.2f} rad/s")

        # Calculate cross-coupling term magnitude (for inertia observability)
        cross_coupling_pqr = np.abs(traj_data['p'] * traj_data['q'] * traj_data['r'])
        max_coupling = cross_coupling_pqr.max()
        mean_coupling = cross_coupling_pqr.mean()
        print(f"  Cross-coupling |p·q·r|: max={max_coupling:.3f}, mean={mean_coupling:.3f}")
        print(f"  => Higher coupling = stronger inertia gradient signal")

    # Combine all trajectories
    df_combined = pd.concat(all_data, ignore_index=True)

    print(f"\n{'='*80}")
    print(f"COMPLETE: Generated {len(df_combined)} total samples across 5 aggressive trajectories")
    print(f"Columns: {list(df_combined.columns)}")

    return df_combined


def main():
    """Generate and save aggressive trajectories for inertia identification."""

    # Generate trajectories
    df_aggressive = generate_aggressive_inertia_trajectories()

    # Save to data directory
    PROJECT_ROOT = Path(__file__).parent.parent
    output_path = PROJECT_ROOT / 'data' / 'aggressive_inertia_trajectories.csv'

    df_aggressive.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"SUCCESS: Saved aggressive trajectories to:")
    print(f"  {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Use these trajectories for fine-tuning inertia parameters")
    print(f"  2. Compare inertia identification before/after aggressive data")
    print(f"  3. Expected: Jxx/Jyy/Jzz errors should decrease from current 1300-6700%")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
