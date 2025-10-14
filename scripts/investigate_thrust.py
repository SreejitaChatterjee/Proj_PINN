import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/quadrotor_training_data.csv')

# Check thrust values over time for each trajectory
print("Investigating thrust behavior:\n")

for tid in range(10):
    traj = df[df['trajectory_id'] == tid].sort_values('timestamp')

    # Find where thrust drops below 0.2
    low_thrust = traj[traj['thrust'] < 0.2]
    if len(low_thrust) > 0:
        drop_time = low_thrust['timestamp'].min()
        print(f"Trajectory {tid}: Thrust drops below 0.2N at t={drop_time:.3f}s")
        print(f"  - Mean thrust before drop: {traj[traj['timestamp'] < drop_time]['thrust'].mean():.3f}N")
        print(f"  - Mean thrust after drop: {traj[traj['timestamp'] >= drop_time]['thrust'].mean():.3f}N")
    else:
        print(f"Trajectory {tid}: Thrust stays above 0.2N throughout")

print("\n" + "="*60)
print("Sample thrust values over time (trajectory 0):")
traj0 = df[df['trajectory_id'] == 0].sort_values('timestamp')
for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    row = traj0[traj0['timestamp'] == t]
    if len(row) > 0:
        print(f"  t={t:.1f}s: thrust={row['thrust'].values[0]:.4f}N")
