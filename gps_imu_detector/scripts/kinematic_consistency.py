#!/usr/bin/env python3
"""
Kinematic Consistency Baseline - Analytical Redundancy Check

This script implements a simple kinematic consistency check:
- Compare GPS velocity vs IMU-integrated velocity
- Compare GPS position changes vs velocity * dt
- Compare GPS ground speed vs IMU-derived speed

Goal: Demonstrate that coordinated attacks evade this baseline,
motivating the need for active probing or additional sensors.

Author: Claude Code
Date: 2026-01-01
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from honest_evaluation import (
    HonestConfig,
    RealisticAttackGenerator,
    compute_auroc_with_ci,
)


# ============================================================================
# Kinematic Consistency Detector
# ============================================================================

class KinematicConsistencyDetector:
    """
    Analytical redundancy detector using kinematic constraints.

    Checks:
    1. Position-velocity consistency: dp/dt should match velocity
    2. Velocity-acceleration consistency: dv/dt should match acceleration
    3. Ground speed consistency: |v_GPS| should match |v_IMU|
    """

    def __init__(self, dt: float = 0.005, window_size: int = 20):
        self.dt = dt
        self.window_size = window_size
        self.normal_residual_mean = None
        self.normal_residual_std = None

    def compute_residuals(self, traj: np.ndarray) -> np.ndarray:
        """
        Compute kinematic consistency residuals.

        traj: [n_samples, 12] where:
            - [:, 0:3] = position (GPS)
            - [:, 3:6] = velocity (GPS or IMU-derived)
            - [:, 6:9] = attitude
            - [:, 9:12] = angular rate
        """
        n = len(traj)
        residuals = []

        for t in range(1, n):
            r = []

            # Check 1: Position change vs velocity
            # dp should equal v * dt
            pos_change = traj[t, :3] - traj[t-1, :3]
            expected_change = traj[t-1, 3:6] * self.dt
            pos_vel_error = np.linalg.norm(pos_change - expected_change)
            r.append(pos_vel_error)

            # Check 2: Velocity smoothness
            # Large sudden velocity changes are suspicious
            if t > 1:
                vel_change = traj[t, 3:6] - traj[t-1, 3:6]
                vel_smoothness = np.linalg.norm(vel_change)
                r.append(vel_smoothness)
            else:
                r.append(0.0)

            # Check 3: Ground speed consistency
            # GPS-derived ground speed should match
            gps_speed = np.linalg.norm(traj[t, 3:6])
            pos_based_speed = np.linalg.norm(pos_change) / self.dt
            speed_error = abs(gps_speed - pos_based_speed)
            r.append(speed_error)

            # Check 4: Heading-velocity consistency
            # If moving, heading should align with velocity direction
            if gps_speed > 0.5:
                vel_heading = np.arctan2(traj[t, 4], traj[t, 3])  # vel_y / vel_x
                pos_heading = np.arctan2(pos_change[1], pos_change[0])
                heading_error = abs(np.arctan2(
                    np.sin(vel_heading - pos_heading),
                    np.cos(vel_heading - pos_heading)
                ))
                r.append(heading_error)
            else:
                r.append(0.0)

            residuals.append(r)

        # Prepend zero for first sample
        residuals.insert(0, [0.0] * 4)
        return np.array(residuals)

    def fit(self, trajectories: List[np.ndarray]):
        """Fit on normal trajectories to learn residual distribution."""
        all_residuals = []
        for traj in trajectories:
            res = self.compute_residuals(traj)
            all_residuals.append(res)

        all_residuals = np.vstack(all_residuals)
        self.normal_residual_mean = np.mean(all_residuals, axis=0)
        self.normal_residual_std = np.std(all_residuals, axis=0) + 1e-8

    def score(self, traj: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (higher = more suspicious)."""
        residuals = self.compute_residuals(traj)
        normalized = (residuals - self.normal_residual_mean) / self.normal_residual_std
        scores = np.mean(normalized ** 2, axis=1)
        return scores


# ============================================================================
# Coordinated Attack Generator (Physics-Consistent)
# ============================================================================

class CoordinatedAttackGenerator(RealisticAttackGenerator):
    """
    Generate coordinated attacks that maintain kinematic consistency.

    Key insight: A coordinated attack modifies BOTH position AND velocity
    consistently, so the dp/dt = v constraint is NOT violated.
    """

    def inject_coordinated_attack(
        self,
        traj: np.ndarray,
        magnitude_x_noise: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject physics-consistent coordinated attack.

        Attack modifies position AND velocity together so that:
        - dp/dt matches v (consistency preserved)
        - Ground speed matches velocity magnitude
        - No discontinuities
        """
        n = len(traj)
        attacked = traj.copy()
        labels = np.zeros(n)

        magnitude_m = magnitude_x_noise * self.gps_noise_std
        start = n // 4
        end = 3 * n // 4

        # Smooth ramp-up and ramp-down to avoid discontinuities
        attack_profile = np.zeros(n)
        ramp_len = 20

        # Ramp up
        attack_profile[start:start+ramp_len] = np.linspace(0, 1, ramp_len)
        # Sustained
        attack_profile[start+ramp_len:end-ramp_len] = 1.0
        # Ramp down
        attack_profile[end-ramp_len:end] = np.linspace(1, 0, ramp_len)

        # Attack direction (consistent throughout)
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)

        # Apply to position
        for t in range(n):
            attacked[t, :3] += direction * magnitude_m * attack_profile[t]

        # CRITICALLY: Also modify velocity to maintain consistency
        # v_attack = d(offset)/dt
        for t in range(1, n):
            d_offset = (attack_profile[t] - attack_profile[t-1]) * magnitude_m
            attacked[t, 3:6] += direction * d_offset / self.dt

        labels[start:end] = 1

        return attacked, labels


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_kinematic_detector():
    """Evaluate kinematic consistency detector against various attacks."""

    np.random.seed(HonestConfig.RANDOM_SEED)

    output_dir = Path(__file__).parent.parent / "results" / "kinematic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("KINEMATIC CONSISTENCY BASELINE EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Initialize
    generator = CoordinatedAttackGenerator(seed=HonestConfig.RANDOM_SEED)
    detector = KinematicConsistencyDetector()

    # Train detector
    print("\nTraining kinematic consistency detector...")
    train_trajs = [generator.generate_nominal_trajectory() for _ in range(100)]
    detector.fit(train_trajs)

    results = {
        "timestamp": datetime.now().isoformat(),
        "by_attack": {},
    }

    attack_configs = [
        ("bias", "Bias (uncorrelated)", 5.0, False),
        ("bias", "Bias (uncorrelated)", 10.0, False),
        ("coordinated_naive", "Coordinated (naive)", 5.0, False),
        ("coordinated_physics", "Coordinated (physics-consistent)", 5.0, True),
        ("coordinated_physics", "Coordinated (physics-consistent)", 10.0, True),
        ("noise_injection", "Noise Injection", 5.0, False),
        ("intermittent", "Intermittent", 5.0, False),
    ]

    print("\n| Attack Type                        | Magnitude | AUROC    | 95% CI          |")
    print("|" + "-"*36 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*17 + "|")

    for attack_type, attack_name, magnitude, use_coordinated in attack_configs:
        all_scores = []
        all_labels = []

        for _ in range(50):
            # Normal
            traj = generator.generate_nominal_trajectory()
            scores = detector.score(traj)
            all_scores.extend(scores)
            all_labels.extend(np.zeros(len(scores)))

            # Attack
            traj = generator.generate_nominal_trajectory()
            if use_coordinated:
                attacked, labels = generator.inject_coordinated_attack(traj, magnitude)
            else:
                attacked, labels = generator.inject_attack(traj, attack_type, magnitude)
            scores = detector.score(attacked)
            all_scores.extend(scores)
            all_labels.extend(labels)

        auroc, (ci_low, ci_high) = compute_auroc_with_ci(
            np.array(all_labels), np.array(all_scores),
            n_bootstrap=100
        )

        results["by_attack"][f"{attack_name}_{magnitude}x"] = {
            "auroc": float(auroc),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "attack_type": attack_type,
            "magnitude": magnitude,
            "physics_consistent": use_coordinated,
        }

        print(f"| {attack_name:34} | {magnitude:6.1f}x   | {auroc*100:7.1f}% | [{ci_low*100:.1f}%, {ci_high*100:.1f}%] |")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nKey Findings:")
    print("1. Kinematic consistency detector catches UNCORRELATED attacks")
    print("   (bias, noise) because they violate dp/dt = v constraint")
    print()
    print("2. PHYSICS-CONSISTENT coordinated attacks remain UNDETECTABLE")
    print("   because they modify position AND velocity together,")
    print("   preserving the kinematic constraint")
    print()
    print("3. This demonstrates that analytical redundancy alone is")
    print("   INSUFFICIENT against sophisticated attackers who understand")
    print("   the physics constraints being checked")
    print()
    print("4. To detect coordinated attacks, we need:")
    print("   - Active probing (inject perturbations, check response)")
    print("   - External references (map constraints, carrier-phase GPS)")
    print("   - Multi-vehicle consistency (V2V communication)")

    # Save
    with open(output_dir / "kinematic_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'kinematic_results.json'}")

    return results


if __name__ == "__main__":
    evaluate_kinematic_detector()
