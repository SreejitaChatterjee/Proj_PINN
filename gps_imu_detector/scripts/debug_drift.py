"""Debug GPS drift detection."""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targeted_improvements_v3 import generate_trajectories, GPSDriftRateDetector

# Generate one GPS drift trajectory
np.random.seed(42)
attacks = generate_trajectories(1, 200, seed=250, is_attack=True)
nominal = generate_trajectories(1, 200, seed=200, is_attack=False)

traj = attacks[0]  # First trajectory is GPS_DRIFT (atype=0)
attack_start = 50  # T // 4

detector = GPSDriftRateDetector(rate_cusum_threshold=4.0, rate_allowance=0.001)

print("GPS DRIFT TRAJECTORY ANALYSIS")
print("="*60)
print(f"Attack starts at t={attack_start}")
print()

print(f"{'t':<6} {'error':<12} {'error_rate':<12} {'cusum':<12} {'detected'}")
print("-"*60)

prev_state = None
for t in range(len(traj) - 1):
    result = detector.update(traj[t], prev_state)
    prev_state = traj[t]

    if t % 20 == 0 or t == attack_start or result['detected']:
        marker = " <-- ATTACK STARTS" if t == attack_start else ""
        marker = " <-- DETECTED!" if result['detected'] else marker
        print(f"{t:<6} {result.get('cumulative_error', 0):<12.4f} {result.get('error_rate', 0):<12.6f} {result['rate_cusum']:<12.4f} {result['detected']}{marker}")

print()
print("ANALYSIS:")
print(f"  Final CUSUM: {result['rate_cusum']:.4f}")
print(f"  Threshold: 4.0")
print(f"  Detected: {result['detected']}")
print()

# Calculate expected signal
print("EXPECTED SIGNAL:")
print(f"  Drift rate per step: 0.5 * 0.005 = 0.0025 per axis")
print(f"  L2 drift rate: 0.0025 * sqrt(3) = {0.0025 * np.sqrt(3):.6f}")
print(f"  Allowance: 0.001")
print(f"  Expected CUSUM increment: {0.0025 * np.sqrt(3) - 0.001:.6f}")
print(f"  After 150 steps: {150 * (0.0025 * np.sqrt(3) - 0.001):.4f}")
print()
print("PROBLEM: Threshold 4.0 is WAY too high for the signal!")
