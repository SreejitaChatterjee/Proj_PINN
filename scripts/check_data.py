import pandas as pd
import numpy as np
from pathlib import Path

# Get the script directory and construct absolute path to data
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_path = project_root / 'data' / 'quadrotor_training_data.csv'
df = pd.read_csv(data_path)

print('Thrust statistics by trajectory:')
for tid in range(10):
    traj = df[df['trajectory_id']==tid]
    print(f'  Traj {tid}: mean={traj["thrust"].mean():.3f}, std={traj["thrust"].std():.3f}, range=[{traj["thrust"].min():.3f}, {traj["thrust"].max():.3f}]')

print('\nAltitude (z) statistics by trajectory:')
for tid in range(10):
    traj = df[df['trajectory_id']==tid]
    print(f'  Traj {tid}: mean={traj["z"].mean():.3f}, std={traj["z"].std():.3f}, range=[{traj["z"].min():.3f}, {traj["z"].max():.3f}]')
