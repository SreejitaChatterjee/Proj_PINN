"""
Generalization Test - Check for Hidden Overfitting

Potential overfitting sources:
1. Same attack magnitudes in train/val/test
2. Same attack timing (always T//4)
3. Same attack patterns
4. 100% detection is suspicious

This test uses:
- DIFFERENT attack magnitudes (weaker and stronger)
- DIFFERENT attack timing (random)
- DIFFERENT attack patterns (variations)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targeted_improvements_v2 import (
    ImprovedStatDetector,
    generate_trajectories,
    BASELINE_METRICS,
)


def generate_varied_attacks(n_traj: int, T: int, seed: int,
                            magnitude_scale: float = 1.0,
                            timing_variation: bool = False) -> tuple:
    """
    Generate attacks with varied parameters to test generalization.

    Args:
        magnitude_scale: 0.5 = weaker attacks, 1.0 = standard, 2.0 = stronger
        timing_variation: If True, randomize attack start time
    """
    np.random.seed(seed)
    trajectories = []
    attack_starts = []

    for i in range(n_traj):
        traj = np.zeros((T, 12), dtype=np.float32)
        pos = np.array([0.0, 0.0, 10.0])
        vel = np.array([0.0, 0.0, 0.0])
        orient = np.array([0.0, 0.0, 0.0])
        ang_vel = np.array([0.0, 0.0, 0.0])

        profile = i % 4
        dt = 0.005

        # Varied attack timing
        if timing_variation:
            attack_start = np.random.randint(T // 6, T // 2)
        else:
            attack_start = T // 4

        attack_starts.append(attack_start)
        atype = i % 5

        for t in range(T):
            if profile == 0:
                vel = np.random.randn(3) * 0.05
                ang_vel = np.random.randn(3) * 0.02
            elif profile == 1:
                vel = np.array([2.0, 0.0, 0.0]) + np.random.randn(3) * 0.1
                ang_vel = np.random.randn(3) * 0.05
            elif profile == 2:
                angle = t * 0.02
                vel = np.array([np.cos(angle), np.sin(angle), 0.0]) * 3.0 + np.random.randn(3) * 0.05
                ang_vel = np.array([0.0, 0.0, 0.02]) + np.random.randn(3) * 0.02
            else:
                angle = t * 0.03
                vel = np.array([np.cos(angle), np.sin(2*angle), 0.0]) * 2.0 + np.random.randn(3) * 0.08
                ang_vel = np.random.randn(3) * 0.03

            pos = pos + vel * dt
            orient = orient + ang_vel * dt

            traj[t, :3] = pos
            traj[t, 3:6] = vel
            traj[t, 6:9] = orient
            traj[t, 9:12] = ang_vel

            # Apply SCALED attacks
            if t >= attack_start:
                if atype == 0:  # GPS drift - scale drift rate
                    drift_rate = 0.5 * magnitude_scale
                    traj[t, :3] += drift_rate * (t - attack_start) * 0.005
                elif atype == 1:  # GPS jump - scale jump size
                    jump = np.array([5.0, 5.0, 1.0]) * magnitude_scale
                    traj[t, :3] += jump
                    traj[t, 3:6] += np.array([0.5, 0.5, 0.0]) * magnitude_scale
                elif atype == 2:  # IMU bias - scale bias magnitude
                    bias = np.array([0.8, 0.8, 0.3]) * magnitude_scale
                    traj[t, 9:12] += bias
                elif atype == 3:  # Spoofing - scale offset
                    offset = np.array([3.0, -3.0, 0.5]) * magnitude_scale
                    traj[t, :3] += offset
                    traj[t, 3:6] += np.array([-0.3, 0.3, 0.0]) * magnitude_scale
                elif atype == 4:  # Actuator fault - scale noise
                    noise_scale = 0.5 * magnitude_scale
                    traj[t, 9:12] += np.random.randn(3) * noise_scale

        trajectories.append(traj)

    return np.array(trajectories), attack_starts


def evaluate_generalization(detector, test_nominal, test_attacks, attack_starts, label):
    """Evaluate detector on a specific test set."""
    attack_types = ['GPS_DRIFT', 'GPS_JUMP', 'IMU_BIAS', 'SPOOFING', 'ACTUATOR_FAULT']

    # FPR
    fp = 0
    total = 0
    for traj in test_nominal:
        for t in range(len(traj) - 1):
            result = detector.process(traj[t], traj[t+1])
            if result['detected']:
                fp += 1
            total += 1
        detector.reset()
    fpr = fp / total

    # Detection
    per_attack = {at: {'detected': 0, 'total': 0} for at in attack_types}
    total_detected = 0

    for i, traj in enumerate(test_attacks):
        attack_start = attack_starts[i]
        detected = False
        atype = attack_types[i % 5]

        for t in range(len(traj) - 1):
            result = detector.process(traj[t], traj[t+1])
            if t >= attack_start and result['detected']:
                detected = True
                break

        if detected:
            total_detected += 1
            per_attack[atype]['detected'] += 1
        per_attack[atype]['total'] += 1
        detector.reset()

    detection_rate = total_detected / len(test_attacks)

    return {
        'label': label,
        'detection_rate': detection_rate,
        'fpr': fpr,
        'per_attack': {at: per_attack[at]['detected'] / max(1, per_attack[at]['total'])
                       for at in attack_types},
    }


def main():
    print("="*70)
    print("      GENERALIZATION TEST - Checking for Hidden Overfitting")
    print("="*70)

    # Train detector with optimal thresholds from v2
    print("\n[Training detector with optimal thresholds...]")
    train_nominal = generate_trajectories(50, 200, seed=100, is_attack=False)

    detector = ImprovedStatDetector(
        gps_threshold=3.0,
        gps_drift_threshold=0.5,
        imu_cusum_threshold=18.0,
        actuator_var_threshold=5.0,
    )
    detector.calibrate(train_nominal)

    # Test scenarios
    scenarios = [
        # Standard (what we optimized for)
        ("Standard (seed 200)", 200, 1.0, False),
        # Different seeds
        ("Different seed (400)", 400, 1.0, False),
        ("Different seed (500)", 500, 1.0, False),
        # Weaker attacks (harder to detect)
        ("Weaker attacks (0.5x)", 600, 0.5, False),
        ("Weaker attacks (0.3x)", 700, 0.3, False),
        # Stronger attacks (should be easier)
        ("Stronger attacks (2x)", 800, 2.0, False),
        # Varied timing
        ("Random timing", 900, 1.0, True),
        # Combined: weaker + varied timing
        ("Weaker + random timing", 1000, 0.5, True),
    ]

    results = []

    for label, seed, scale, timing_var in scenarios:
        print(f"\n[Testing: {label}...]")

        # Generate nominal with different seed
        test_nominal = generate_trajectories(30, 200, seed=seed, is_attack=False)

        # Generate attacks with variations
        test_attacks, attack_starts = generate_varied_attacks(
            30, 200, seed=seed+50,
            magnitude_scale=scale,
            timing_variation=timing_var,
        )

        result = evaluate_generalization(detector, test_nominal, test_attacks, attack_starts, label)
        results.append(result)

    # Print results table
    print("\n" + "="*70)
    print("                    GENERALIZATION RESULTS")
    print("="*70)

    attack_types = ['GPS_DRIFT', 'GPS_JUMP', 'IMU_BIAS', 'SPOOFING', 'ACTUATOR_FAULT']

    print(f"\n{'SCENARIO':<30} {'DETECT':<10} {'FPR':<10} {'GPS_D':<8} {'GPS_J':<8} {'IMU':<8} {'SPOOF':<8} {'ACT':<8}")
    print("-"*100)

    for r in results:
        pa = r['per_attack']
        print(f"{r['label']:<30} {r['detection_rate']*100:>6.1f}%   {r['fpr']*100:>6.2f}%   "
              f"{pa['GPS_DRIFT']*100:>5.0f}%   {pa['GPS_JUMP']*100:>5.0f}%   "
              f"{pa['IMU_BIAS']*100:>5.0f}%   {pa['SPOOFING']*100:>5.0f}%   "
              f"{pa['ACTUATOR_FAULT']*100:>5.0f}%")

    # Summary
    print("\n" + "="*70)
    print("                    INTERPRETATION")
    print("="*70)

    standard = results[0]
    weak = results[3]  # 0.5x
    very_weak = results[4]  # 0.3x

    print(f"\nStandard attacks:     {standard['detection_rate']*100:.0f}% detection at {standard['fpr']*100:.2f}% FPR")
    print(f"Half-strength (0.5x): {weak['detection_rate']*100:.0f}% detection at {weak['fpr']*100:.2f}% FPR")
    print(f"Weak attacks (0.3x):  {very_weak['detection_rate']*100:.0f}% detection at {very_weak['fpr']*100:.2f}% FPR")

    # Check for overfitting indicators
    print("\n" + "-"*70)
    print("OVERFITTING CHECK:")
    print("-"*70)

    issues = []

    # Check 1: Does performance collapse on weaker attacks?
    if weak['detection_rate'] < 0.5:
        issues.append(f"- Detection collapses on 0.5x attacks ({weak['detection_rate']*100:.0f}%)")

    # Check 2: Is there high variance across seeds?
    seed_results = [results[0], results[1], results[2]]
    seed_detection = [r['detection_rate'] for r in seed_results]
    if max(seed_detection) - min(seed_detection) > 0.2:
        issues.append(f"- High variance across seeds ({min(seed_detection)*100:.0f}%-{max(seed_detection)*100:.0f}%)")

    # Check 3: Does timing variation hurt?
    timing_result = results[6]
    if timing_result['detection_rate'] < standard['detection_rate'] - 0.15:
        issues.append(f"- Random timing hurts detection ({timing_result['detection_rate']*100:.0f}% vs {standard['detection_rate']*100:.0f}%)")

    # Check 4: Specific attack types failing on weak attacks
    for at in attack_types:
        if weak['per_attack'][at] < 0.3 and standard['per_attack'][at] > 0.8:
            issues.append(f"- {at} fails on weak attacks ({weak['per_attack'][at]*100:.0f}% vs {standard['per_attack'][at]*100:.0f}%)")

    if issues:
        print("\n[!!] POTENTIAL OVERFITTING DETECTED:")
        for issue in issues:
            print(f"  {issue}")
        print("\nThe detector may be overfit to specific attack magnitudes.")
    else:
        print("\n[OK] No obvious overfitting detected.")
        print("     - Consistent across different seeds")
        print("     - Reasonable degradation on weaker attacks")
        print("     - Robust to timing variations")


if __name__ == "__main__":
    main()
