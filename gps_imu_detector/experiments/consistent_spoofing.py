"""
Residual Equivalence Class (REC) Experiment.

Demonstrates the impossibility result: consistent GPS spoofing constructs
trajectories that remain within the same Residual Equivalence Class as
nominal flight, rendering residual-based detection ill-posed.

Definition (REC):
    Given a learned dynamics model f_θ, two trajectories belong to the
    same Residual Equivalence Class (REC) if they induce statistically
    indistinguishable prediction residual distributions under f_θ.
    Any detector operating solely on residual statistics cannot
    distinguish trajectories within the same REC.

Key finding:
    Consistent spoofing ∈ [τ_nominal] REC → undetectable (impossibility)
    Inconsistent spoofing ∈ [τ_other] REC → detectable (different REC)

Implication:
    Residual-based detectors test inconsistency, not truth. GPS spoofing
    succeeds by preserving residual equivalence, indicating that spoofing
    defense is fundamentally an identification problem requiring external
    anchors.
"""

import numpy as np
import torch
from typing import Tuple, Dict
import json
from pathlib import Path


def generate_consistent_drift(
    trajectory: np.ndarray,
    drift_rate: float = 0.001,  # meters per timestep
    direction: np.ndarray = None,
    seed: int = 42,
    perfect_consistency: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a perfectly consistent slow drift attack.

    The attack adds a slowly growing offset that follows
    a smooth trajectory - indistinguishable from legitimate motion.

    When perfect_consistency=True, the attack modifies BOTH position
    and velocity consistently, so dynamics checks pass.

    Args:
        trajectory: Original trajectory (T, D) with position/velocity
        drift_rate: How fast the offset grows (m/timestep)
        direction: Drift direction (normalized), random if None
        seed: Random seed
        perfect_consistency: If True, ensure dynamics are consistent

    Returns:
        spoofed_trajectory: Trajectory with consistent drift
        drift_offset: The applied offset at each timestep
    """
    np.random.seed(seed)

    T, D = trajectory.shape
    dt = 0.005  # 200 Hz

    if direction is None:
        # Random but consistent drift direction
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)

    # Create offset that is PERFECTLY consistent with dynamics
    drift_offset = np.zeros_like(trajectory)

    if perfect_consistency:
        # Constant position offset (no velocity change needed)
        # This is the "perfect" spoofing - just shift everything
        constant_offset = drift_rate * 1000 * direction  # Large constant offset
        drift_offset[:, :3] = constant_offset
        # Velocity unchanged - dynamics still match
    else:
        # Growing offset (detectable because velocity doesn't match)
        t = np.arange(T)
        drift_magnitude = drift_rate * t
        drift_offset[:, :3] = np.outer(drift_magnitude, direction)
        if D >= 6:
            drift_offset[:, 3:6] = drift_rate * direction

    spoofed_trajectory = trajectory + drift_offset

    return spoofed_trajectory, drift_offset


def compute_residuals(
    trajectory: np.ndarray,
    model_predict: callable = None
) -> np.ndarray:
    """
    Compute prediction residuals.

    If no model provided, uses simple finite difference dynamics.

    Args:
        trajectory: Trajectory (T, D)
        model_predict: Function that predicts next state from current

    Returns:
        residuals: (T-1, D) array of residuals
    """
    if model_predict is None:
        # Simple dynamics: x_{t+1} = x_t + v_t * dt
        # Residual = actual change - predicted change
        dt = 0.005  # 200 Hz
        predicted_change = trajectory[:-1, 3:6] * dt  # velocity * dt
        actual_change = trajectory[1:, :3] - trajectory[:-1, :3]
        residuals = actual_change - predicted_change
    else:
        predictions = np.array([model_predict(trajectory[t]) for t in range(len(trajectory) - 1)])
        residuals = trajectory[1:] - predictions

    return residuals


def run_impossibility_experiment(
    n_samples: int = 10000,
    offset_magnitudes: list = [1, 5, 10, 50, 100],  # meters
    seed: int = 42
) -> Dict:
    """
    Run the impossibility experiment.

    Shows that CONSISTENT spoofing (constant offset) is undetectable,
    while INCONSISTENT spoofing (growing drift) is detectable.

    Args:
        n_samples: Number of trajectory samples
        offset_magnitudes: Different offset sizes to test (meters)
        seed: Random seed

    Returns:
        Results dictionary with residual statistics
    """
    np.random.seed(seed)

    # Generate "normal" trajectory (random walk with smooth dynamics)
    dt = 0.005
    trajectory = np.zeros((n_samples, 6))  # [x, y, z, vx, vy, vz]

    # Initialize with random velocity
    trajectory[0, 3:6] = np.random.randn(3) * 0.5

    for t in range(1, n_samples):
        # Smooth random acceleration
        accel = np.random.randn(3) * 0.1
        trajectory[t, 3:6] = trajectory[t-1, 3:6] + accel * dt
        trajectory[t, :3] = trajectory[t-1, :3] + trajectory[t, 3:6] * dt

    # Compute normal residuals
    normal_residuals = compute_residuals(trajectory)
    normal_magnitude = np.linalg.norm(normal_residuals, axis=1)

    results = {
        "normal": {
            "mean_residual": float(np.mean(normal_magnitude)),
            "std_residual": float(np.std(normal_magnitude)),
            "max_residual": float(np.max(normal_magnitude)),
            "p95_residual": float(np.percentile(normal_magnitude, 95)),
        },
        "consistent_attacks": {},
        "inconsistent_attacks": {},
        "conclusion": None
    }

    # Test CONSISTENT spoofing (constant offset - UNDETECTABLE)
    for offset_m in offset_magnitudes:
        direction = np.array([1, 0, 0])  # X direction
        spoofed = trajectory.copy()
        spoofed[:, :3] += offset_m * direction  # Constant offset

        attack_residuals = compute_residuals(spoofed)
        attack_magnitude = np.linalg.norm(attack_residuals, axis=1)

        results["consistent_attacks"][f"offset_{offset_m}m"] = {
            "offset_m": offset_m,
            "mean_residual": float(np.mean(attack_magnitude)),
            "residual_diff": float(np.mean(attack_magnitude) - np.mean(normal_magnitude)),
            "detectable": bool(np.mean(attack_magnitude) > results["normal"]["p95_residual"]),
            "note": "Constant offset - dynamics unchanged"
        }

    # Test INCONSISTENT spoofing (growing drift - DETECTABLE)
    drift_rates = [0.0001, 0.001, 0.01]
    for drift_rate in drift_rates:
        spoofed, offset = generate_consistent_drift(
            trajectory, drift_rate, seed=seed, perfect_consistency=False
        )
        attack_residuals = compute_residuals(spoofed)
        attack_magnitude = np.linalg.norm(attack_residuals, axis=1)

        total_drift = np.linalg.norm(offset[-1, :3])

        results["inconsistent_attacks"][f"drift_{drift_rate}"] = {
            "drift_rate": drift_rate,
            "total_drift_m": float(total_drift),
            "mean_residual": float(np.mean(attack_magnitude)),
            "residual_increase_pct": float(
                (np.mean(attack_magnitude) - np.mean(normal_magnitude))
                / np.mean(normal_magnitude) * 100
            ),
            "detectable": bool(np.mean(attack_magnitude) > results["normal"]["p95_residual"]),
            "note": "Growing drift - dynamics violated"
        }

    # Conclusion
    consistent_detectable = sum(
        1 for v in results["consistent_attacks"].values()
        if v.get("detectable", False)
    )
    inconsistent_detectable = sum(
        1 for v in results["inconsistent_attacks"].values()
        if v.get("detectable", False)
    )

    results["conclusion"] = (
        f"CONSISTENT spoofing: {consistent_detectable}/{len(offset_magnitudes)} in nominal REC "
        f"(even {max(offset_magnitudes)}m offset remains in SAME REC!)\n"
        f"INCONSISTENT spoofing: {inconsistent_detectable}/{len(drift_rates)} in different REC\n\n"
        "CONCLUSION: Consistent GPS spoofing lies within the nominal Residual Equivalence Class.\n"
        "Residual-based detection is fundamentally ill-posed for this attack class.\n"
        "Spoofing defense is an IDENTIFICATION problem requiring external anchors."
    )

    return results


def print_results(results: Dict):
    """Pretty print the experiment results."""
    print("\n" + "=" * 70)
    print("RESIDUAL EQUIVALENCE CLASS (REC) IMPOSSIBILITY EXPERIMENT")
    print("=" * 70)

    print("\n### Normal Operation ###")
    n = results["normal"]
    print(f"  Mean residual:  {n['mean_residual']:.6f}")
    print(f"  Std residual:   {n['std_residual']:.6f}")
    print(f"  P95 threshold:  {n['p95_residual']:.6f}")

    print("\n### CONSISTENT Attacks (Constant Offset) ###")
    print("-" * 70)
    print(f"{'Offset':<12} {'Mean Residual':<15} {'Diff from Normal':<18} {'Detectable'}")
    print("-" * 70)

    for name, attack in results["consistent_attacks"].items():
        print(
            f"{attack['offset_m']:<12}m "
            f"{attack['mean_residual']:<15.6f} "
            f"{attack['residual_diff']:<+18.6f} "
            f"{'YES' if attack['detectable'] else 'NO <<<'}"
        )

    print("\n### INCONSISTENT Attacks (Growing Drift) ###")
    print("-" * 70)
    print(f"{'Drift Rate':<12} {'Total Drift':<12} {'Mean Res':<12} {'Increase %':<15} {'Detectable'}")
    print("-" * 70)

    for name, attack in results["inconsistent_attacks"].items():
        print(
            f"{attack['drift_rate']:<12.4f} "
            f"{attack['total_drift_m']:<12.1f}m "
            f"{attack['mean_residual']:<12.6f} "
            f"{attack['residual_increase_pct']:<+15.0f}% "
            f"{'YES' if attack['detectable'] else 'NO'}"
        )

    print("-" * 70)
    print(f"\n### CONCLUSION ###")
    print(results["conclusion"])
    print()


if __name__ == "__main__":
    results = run_impossibility_experiment()
    print_results(results)

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "impossibility_experiment.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir / 'impossibility_experiment.json'}")
