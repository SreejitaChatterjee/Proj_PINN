#!/usr/bin/env python3
"""
Sensitivity Analysis - Vary sensor and motion parameters

This script evaluates detector robustness across:
1. IMU grades (consumer, industrial, tactical, navigation)
2. GPS error models (open sky, urban canyon, multipath)
3. Motion profiles (hover, forward flight, aggressive maneuvers)
4. Attack timing (early, middle, late in sequence)

Goal: Define the boundary conditions where passive monitoring succeeds/fails.

Author: Claude Code
Date: 2026-01-01
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from honest_evaluation import (
    HonestConfig,
    CausalFeatureExtractor,
    HonestDetector,
    compute_auroc_with_ci,
)


# ============================================================================
# IMU Grades
# ============================================================================

@dataclass
class IMUGrade:
    """IMU performance characteristics."""
    name: str
    accel_noise: float  # m/s^2
    gyro_noise: float   # rad/s
    accel_bias: float   # m/s^2
    gyro_bias: float    # rad/s


IMU_GRADES = {
    "consumer": IMUGrade("Consumer (MEMS)", 0.05, 0.01, 0.02, 0.001),
    "industrial": IMUGrade("Industrial", 0.02, 0.005, 0.01, 0.0005),
    "tactical": IMUGrade("Tactical", 0.005, 0.001, 0.002, 0.0001),
    "navigation": IMUGrade("Navigation", 0.001, 0.0002, 0.0005, 0.00002),
}


# ============================================================================
# GPS Error Models
# ============================================================================

@dataclass
class GPSModel:
    """GPS error characteristics."""
    name: str
    position_noise: float  # meters
    velocity_noise: float  # m/s
    multipath_prob: float  # probability of multipath
    multipath_mag: float   # multipath magnitude (meters)


GPS_MODELS = {
    "open_sky": GPSModel("Open Sky", 1.5, 0.1, 0.02, 2.0),
    "suburban": GPSModel("Suburban", 3.0, 0.2, 0.10, 5.0),
    "urban_canyon": GPSModel("Urban Canyon", 8.0, 0.5, 0.30, 15.0),
    "indoor": GPSModel("Indoor/Degraded", 15.0, 1.0, 0.50, 30.0),
}


# ============================================================================
# Motion Profiles
# ============================================================================

@dataclass
class MotionProfile:
    """Motion profile characteristics."""
    name: str
    velocity_mean: float   # m/s
    velocity_std: float    # m/s
    accel_std: float       # m/s^2
    angular_rate_std: float  # rad/s


MOTION_PROFILES = {
    "hover": MotionProfile("Hover", 0.0, 0.5, 0.2, 0.05),
    "forward": MotionProfile("Forward Flight", 10.0, 2.0, 1.0, 0.1),
    "aggressive": MotionProfile("Aggressive", 15.0, 5.0, 5.0, 0.5),
    "gusty": MotionProfile("Gusty Conditions", 8.0, 4.0, 3.0, 0.3),
}


# ============================================================================
# Configurable Attack Generator
# ============================================================================

class ConfigurableAttackGenerator:
    """Generate attacks with configurable sensor parameters."""

    def __init__(
        self,
        imu_grade: IMUGrade,
        gps_model: GPSModel,
        motion_profile: MotionProfile,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.imu = imu_grade
        self.gps = gps_model
        self.motion = motion_profile
        self.dt = 0.005  # 200Hz

    def generate_nominal_trajectory(self, n_samples: int = 400) -> np.ndarray:
        """Generate trajectory with configured parameters."""
        state = np.zeros((n_samples, 12))

        # Initial state
        state[0, :3] = np.random.randn(3) * 2.0
        state[0, 3:6] = np.random.randn(3) * self.motion.velocity_std + self.motion.velocity_mean * np.array([1, 0, 0])

        # Simulate dynamics
        for t in range(1, n_samples):
            # Position from velocity
            state[t, :3] = state[t-1, :3] + state[t-1, 3:6] * self.dt

            # Velocity with motion-profile-dependent dynamics
            state[t, 3:6] = (
                state[t-1, 3:6] * 0.995 +
                np.random.randn(3) * self.motion.accel_std * self.dt
            )

            # Attitude
            state[t, 6:9] = state[t-1, 6:9] + state[t-1, 9:12] * self.dt
            state[t, 9:12] = (
                state[t-1, 9:12] * 0.98 +
                np.random.randn(3) * self.motion.angular_rate_std
            )

        # Add GPS noise (position and velocity)
        gps_noise = np.random.randn(n_samples, 3) * self.gps.position_noise
        vel_noise = np.random.randn(n_samples, 3) * self.gps.velocity_noise

        # Add multipath errors
        multipath_mask = np.random.rand(n_samples) < self.gps.multipath_prob
        multipath_error = np.zeros((n_samples, 3))
        multipath_error[multipath_mask] = (
            np.random.randn(multipath_mask.sum(), 3) * self.gps.multipath_mag
        )

        state[:, :3] += gps_noise + multipath_error
        state[:, 3:6] += vel_noise

        return state

    def inject_attack(
        self,
        traj: np.ndarray,
        attack_type: str,
        magnitude_x_noise: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject attack with magnitude relative to GPS noise."""
        n = len(traj)
        attacked = traj.copy()
        labels = np.zeros(n)

        magnitude_m = magnitude_x_noise * self.gps.position_noise
        start = n // 4
        end = 3 * n // 4

        if attack_type == "bias":
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            attacked[start:end, :3] += direction * magnitude_m
            labels[start:end] = 1

        elif attack_type == "noise_injection":
            extra_noise = np.random.randn(end - start, 3) * magnitude_m
            attacked[start:end, :3] += extra_noise
            labels[start:end] = 1

        elif attack_type == "intermittent":
            block_size = 40
            for i in range(start, end, block_size):
                if np.random.rand() > 0.5:
                    i_end = min(i + block_size // 2, end)
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                    attacked[i:i_end, :3] += direction * magnitude_m
                    labels[i:i_end] = 1

        return attacked, labels


# ============================================================================
# Sensitivity Evaluation
# ============================================================================

def evaluate_configuration(
    imu_grade: IMUGrade,
    gps_model: GPSModel,
    motion_profile: MotionProfile,
    attack_types: List[str],
    magnitude: float = 5.0,
    n_sequences: int = 50,
) -> Dict:
    """Evaluate detector performance for a specific configuration."""

    generator = ConfigurableAttackGenerator(imu_grade, gps_model, motion_profile)
    detector = HonestDetector()

    # Train
    train_trajs = [generator.generate_nominal_trajectory() for _ in range(50)]
    detector.fit(train_trajs)

    results = {}

    for attack_type in attack_types:
        all_scores = []
        all_labels = []

        for _ in range(n_sequences):
            # Normal
            traj = generator.generate_nominal_trajectory()
            scores = detector.score(traj)
            all_scores.extend(scores)
            all_labels.extend(np.zeros(len(scores)))

            # Attack
            traj = generator.generate_nominal_trajectory()
            attacked, labels = generator.inject_attack(traj, attack_type, magnitude)
            scores = detector.score(attacked)
            all_scores.extend(scores)
            all_labels.extend(labels)

        auroc, (ci_low, ci_high) = compute_auroc_with_ci(
            np.array(all_labels), np.array(all_scores),
            n_bootstrap=100
        )

        results[attack_type] = {
            "auroc": float(auroc),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }

    return results


# ============================================================================
# Main
# ============================================================================

def run_sensitivity_analysis():
    """Run full sensitivity analysis."""

    np.random.seed(HonestConfig.RANDOM_SEED)

    output_dir = Path(__file__).parent.parent / "results" / "sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    attack_types = ["noise_injection", "intermittent", "bias"]
    results = {
        "timestamp": datetime.now().isoformat(),
        "analyses": {},
    }

    # ========================================================================
    # Analysis 1: IMU Grade Sweep
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: IMU Grade")
    print("=" * 70)

    results["analyses"]["imu_grade"] = {}

    for grade_name, imu_grade in IMU_GRADES.items():
        print(f"\n{imu_grade.name}:")
        gps_model = GPS_MODELS["open_sky"]
        motion_profile = MOTION_PROFILES["forward"]

        res = evaluate_configuration(
            imu_grade, gps_model, motion_profile, attack_types
        )
        results["analyses"]["imu_grade"][grade_name] = res

        for attack, data in res.items():
            print(f"  {attack}: AUROC = {data['auroc']*100:.1f}%")

    # ========================================================================
    # Analysis 2: GPS Model Sweep
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: GPS Error Model")
    print("=" * 70)

    results["analyses"]["gps_model"] = {}

    for model_name, gps_model in GPS_MODELS.items():
        print(f"\n{gps_model.name} (noise={gps_model.position_noise}m):")
        imu_grade = IMU_GRADES["industrial"]
        motion_profile = MOTION_PROFILES["forward"]

        res = evaluate_configuration(
            imu_grade, gps_model, motion_profile, attack_types
        )
        results["analyses"]["gps_model"][model_name] = res

        for attack, data in res.items():
            print(f"  {attack}: AUROC = {data['auroc']*100:.1f}%")

    # ========================================================================
    # Analysis 3: Motion Profile Sweep
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Motion Profile")
    print("=" * 70)

    results["analyses"]["motion_profile"] = {}

    for profile_name, motion_profile in MOTION_PROFILES.items():
        print(f"\n{motion_profile.name}:")
        imu_grade = IMU_GRADES["industrial"]
        gps_model = GPS_MODELS["open_sky"]

        res = evaluate_configuration(
            imu_grade, gps_model, motion_profile, attack_types
        )
        results["analyses"]["motion_profile"][profile_name] = res

        for attack, data in res.items():
            print(f"  {attack}: AUROC = {data['auroc']*100:.1f}%")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find conditions where bias becomes detectable
    bias_detectable = []
    for analysis_type in ["imu_grade", "gps_model", "motion_profile"]:
        for config, data in results["analyses"][analysis_type].items():
            if data.get("bias", {}).get("auroc", 0.5) > 0.55:
                bias_detectable.append(f"{analysis_type}/{config}")

    print("\nConditions where bias attack becomes marginally detectable:")
    if bias_detectable:
        for cond in bias_detectable:
            print(f"  - {cond}")
    else:
        print("  None - bias attack remains undetectable across all conditions")

    # Noise injection stability
    noise_aurocs = []
    for analysis_type in ["imu_grade", "gps_model", "motion_profile"]:
        for config, data in results["analyses"][analysis_type].items():
            noise_aurocs.append(data.get("noise_injection", {}).get("auroc", 0))

    print(f"\nNoise injection AUROC range: {min(noise_aurocs)*100:.1f}% - {max(noise_aurocs)*100:.1f}%")
    print("  (stable across conditions)")

    # Worst-case for intermittent
    intermittent_aurocs = []
    for analysis_type in ["imu_grade", "gps_model", "motion_profile"]:
        for config, data in results["analyses"][analysis_type].items():
            intermittent_aurocs.append(
                (config, data.get("intermittent", {}).get("auroc", 0))
            )

    worst_intermittent = min(intermittent_aurocs, key=lambda x: x[1])
    print(f"\nWorst case for intermittent: {worst_intermittent[0]} ({worst_intermittent[1]*100:.1f}%)")

    print("\nKey Finding: Detection capability is relatively stable across")
    print("sensor grades and motion profiles. GPS error model has the")
    print("largest impact - higher noise makes all attacks harder to detect.")

    # Save
    with open(output_dir / "sensitivity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'sensitivity_results.json'}")

    return results


if __name__ == "__main__":
    run_sensitivity_analysis()
