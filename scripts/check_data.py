import pandas as pd
import numpy as np

df = pd.read_csv('../data/quadrotor_training_data.csv')

print('Thrust statistics by trajectory:')
for tid in range(10):
    traj = df[df['trajectory_id']==tid]
    print(f'  Traj {tid}: mean={traj["thrust"].mean():.3f}, std={traj["thrust"].std():.3f}, range=[{traj["thrust"].min():.3f}, {traj["thrust"].max():.3f}]')

print('\nAltitude (z) statistics by trajectory:')
for tid in range(10):
    traj = df[df['trajectory_id']==tid]
    print(f'  Traj {tid}: mean={traj["z"].mean():.3f}, std={traj["z"].std():.3f}, range=[{traj["z"].min():.3f}, {traj["z"].max():.3f}]')
