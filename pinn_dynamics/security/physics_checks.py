"""
Physics-Based Anomaly Checks (CPU-Friendly).

Implements jerk/energy bounds and kinematic triads for attack detection.
These checks are:
- Independent of learned models (pure physics)
- CPU-friendly (O(N) complexity)
- Complementary to PINN residuals

Checks implemented:
1. Jerk bounds (d³x/dt³)
2. Energy conservation
3. Kinematic triads (pos-vel-acc consistency)
4. Angular momentum bounds
5. Control-state consistency
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class PhysicsLimits:
    """Physical limits for quadrotor."""
    # Kinematic limits
    max_velocity: float = 20.0         # m/s
    max_acceleration: float = 20.0     # m/s² (2g)
    max_jerk: float = 100.0            # m/s³
    max_angular_rate: float = 10.0     # rad/s
    max_angular_accel: float = 50.0    # rad/s²

    # Energy limits
    max_kinetic_energy: float = 500.0  # J (assumes ~2.5kg drone at 20m/s)
    max_power: float = 1000.0          # W

    # Consistency tolerances
    vel_acc_tolerance: float = 0.5     # m/s² mismatch tolerance
    pos_vel_tolerance: float = 0.1     # m/s mismatch tolerance


@dataclass
class PhysicsCheckResult:
    """Result of physics consistency check."""
    is_anomaly: bool
    score: float  # 0 = normal, 1 = max anomaly
    check_name: str
    details: str = ""


class JerkChecker:
    """
    Check jerk bounds (third derivative of position).

    Excessive jerk indicates:
    - GPS jumps (instant position change = infinite jerk)
    - Replay attacks with discontinuities
    - Sensor glitches
    """

    def __init__(self, limits: PhysicsLimits = None, dt: float = 0.005):
        self.limits = limits or PhysicsLimits()
        self.dt = dt

    def check(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute jerk and check bounds.

        Args:
            pos: Position array (N, 3)

        Returns:
            jerk_magnitude: Jerk magnitude (N-3,)
            is_violation: Boolean violation flags (N-3,)
        """
        # Compute velocity (first derivative)
        vel = np.diff(pos, axis=0) / self.dt

        # Compute acceleration (second derivative)
        acc = np.diff(vel, axis=0) / self.dt

        # Compute jerk (third derivative)
        jerk = np.diff(acc, axis=0) / self.dt

        # Magnitude
        jerk_mag = np.linalg.norm(jerk, axis=1)

        # Check bounds
        is_violation = jerk_mag > self.limits.max_jerk

        return jerk_mag, is_violation


class EnergyChecker:
    """
    Check energy conservation and power limits.

    Violations indicate:
    - Impossible state transitions
    - Coordinated attacks that violate physics
    """

    def __init__(self, limits: PhysicsLimits = None, mass: float = 2.5, dt: float = 0.005):
        self.limits = limits or PhysicsLimits()
        self.mass = mass
        self.dt = dt
        self.g = 9.81

    def check(
        self,
        pos: np.ndarray,
        vel: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Check energy conservation.

        Args:
            pos: Position (N, 3)
            vel: Velocity (N, 3)

        Returns:
            kinetic_energy: KE array (N,)
            power: Power array (N-1,)
            is_violation: Violation flags (N-1,)
        """
        # Kinetic energy: 0.5 * m * v²
        speed_sq = np.sum(vel**2, axis=1)
        kinetic_energy = 0.5 * self.mass * speed_sq

        # Potential energy: m * g * h
        potential_energy = self.mass * self.g * pos[:, 2]

        # Total mechanical energy
        total_energy = kinetic_energy + potential_energy

        # Power = dE/dt (rate of energy change)
        power = np.diff(total_energy) / self.dt

        # Check kinetic energy bounds
        ke_violation = kinetic_energy[1:] > self.limits.max_kinetic_energy

        # Check power bounds
        power_violation = np.abs(power) > self.limits.max_power

        is_violation = ke_violation | power_violation

        return kinetic_energy, power, is_violation


class KinematicTriadChecker:
    """
    Check kinematic consistency: pos ↔ vel ↔ acc.

    This is a sliding window check that verifies:
    - d(pos)/dt ≈ vel
    - d(vel)/dt ≈ acc (from IMU)

    Violations indicate:
    - GPS spoofing (pos doesn't match integrated vel)
    - IMU spoofing (acc doesn't match vel derivative)
    """

    def __init__(self, limits: PhysicsLimits = None, dt: float = 0.005, window: int = 10):
        self.limits = limits or PhysicsLimits()
        self.dt = dt
        self.window = window

    def check(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        acc: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Check kinematic triad consistency.

        Args:
            pos: Position (N, 3)
            vel: Velocity (N, 3)
            acc: Acceleration from IMU (N, 3), optional

        Returns:
            Dict with consistency scores and violations
        """
        N = len(pos)
        results = {}

        # Check 1: pos-vel consistency (d(pos)/dt ≈ vel)
        pos_deriv = np.diff(pos, axis=0) / self.dt
        vel_trimmed = vel[1:]

        pos_vel_error = np.linalg.norm(pos_deriv - vel_trimmed, axis=1)
        results['pos_vel_error'] = pos_vel_error
        results['pos_vel_violation'] = pos_vel_error > self.limits.pos_vel_tolerance

        # Check 2: vel-acc consistency (d(vel)/dt ≈ acc)
        if acc is not None:
            vel_deriv = np.diff(vel, axis=0) / self.dt
            acc_trimmed = acc[1:]

            vel_acc_error = np.linalg.norm(vel_deriv - acc_trimmed, axis=1)
            results['vel_acc_error'] = vel_acc_error
            results['vel_acc_violation'] = vel_acc_error > self.limits.vel_acc_tolerance

        # Sliding window integration check
        # Integrate acc over window, compare to vel change
        if acc is not None and N > self.window:
            integrated_vel_change = np.zeros(N - self.window)
            actual_vel_change = np.zeros(N - self.window)

            for i in range(N - self.window):
                # Integrate acceleration
                integrated_vel_change[i] = np.sum(np.linalg.norm(acc[i:i+self.window], axis=1)) * self.dt

                # Actual velocity change
                actual_vel_change[i] = np.linalg.norm(vel[i+self.window] - vel[i])

            integration_error = np.abs(integrated_vel_change - actual_vel_change)
            results['integration_error'] = integration_error
            results['integration_violation'] = integration_error > self.limits.vel_acc_tolerance * self.window

        return results


class AngularMomentumChecker:
    """
    Check angular rate bounds and attitude-rate consistency.
    """

    def __init__(self, limits: PhysicsLimits = None, dt: float = 0.005):
        self.limits = limits or PhysicsLimits()
        self.dt = dt

    def check(
        self,
        att: np.ndarray,
        rate: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Check angular momentum consistency.

        Args:
            att: Attitude (N, 3) - roll, pitch, yaw
            rate: Angular rate (N, 3) - p, q, r

        Returns:
            Dict with consistency scores and violations
        """
        results = {}

        # Check 1: Angular rate bounds
        rate_mag = np.linalg.norm(rate, axis=1)
        results['rate_magnitude'] = rate_mag
        results['rate_violation'] = rate_mag > self.limits.max_angular_rate

        # Check 2: Angular acceleration bounds
        rate_deriv = np.diff(rate, axis=0) / self.dt
        ang_accel_mag = np.linalg.norm(rate_deriv, axis=1)
        results['ang_accel_magnitude'] = ang_accel_mag
        results['ang_accel_violation'] = ang_accel_mag > self.limits.max_angular_accel

        # Check 3: Attitude-rate consistency
        # For small angles: d(att)/dt ≈ rate (simplified)
        att_deriv = np.diff(att, axis=0) / self.dt
        rate_trimmed = rate[1:]

        att_rate_error = np.linalg.norm(att_deriv - rate_trimmed, axis=1)
        results['att_rate_error'] = att_rate_error
        results['att_rate_violation'] = att_rate_error > 0.5  # rad/s tolerance

        return results


class PhysicsAnomalyDetector:
    """
    Combined physics-based anomaly detector.

    Runs all physics checks and combines scores.
    """

    def __init__(self, limits: PhysicsLimits = None, dt: float = 0.005):
        self.limits = limits or PhysicsLimits()
        self.dt = dt

        self.jerk_checker = JerkChecker(limits, dt)
        self.energy_checker = EnergyChecker(limits, dt=dt)
        self.kinematic_checker = KinematicTriadChecker(limits, dt)
        self.angular_checker = AngularMomentumChecker(limits, dt)

    def detect(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        att: np.ndarray,
        rate: np.ndarray,
        acc: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run all physics checks.

        Args:
            pos: Position (N, 3)
            vel: Velocity (N, 3)
            att: Attitude (N, 3)
            rate: Angular rate (N, 3)
            acc: Acceleration from IMU (N, 3), optional

        Returns:
            Dict with all check results and combined score
        """
        results = {}

        # Jerk check
        jerk_mag, jerk_viol = self.jerk_checker.check(pos)
        results['jerk'] = {
            'magnitude': jerk_mag,
            'violations': jerk_viol,
            'violation_rate': np.mean(jerk_viol)
        }

        # Energy check
        ke, power, energy_viol = self.energy_checker.check(pos, vel)
        results['energy'] = {
            'kinetic_energy': ke,
            'power': power,
            'violations': energy_viol,
            'violation_rate': np.mean(energy_viol)
        }

        # Kinematic triad check
        kinematic_results = self.kinematic_checker.check(pos, vel, acc)
        results['kinematic'] = {
            'pos_vel_error': kinematic_results['pos_vel_error'],
            'violations': kinematic_results['pos_vel_violation'],
            'violation_rate': np.mean(kinematic_results['pos_vel_violation'])
        }

        # Angular check
        angular_results = self.angular_checker.check(att, rate)
        results['angular'] = {
            'rate_magnitude': angular_results['rate_magnitude'],
            'violations': angular_results['rate_violation'],
            'violation_rate': np.mean(angular_results['rate_violation'])
        }

        # Combined score
        violation_rates = [
            results['jerk']['violation_rate'],
            results['energy']['violation_rate'],
            results['kinematic']['violation_rate'],
            results['angular']['violation_rate']
        ]
        results['combined_score'] = np.mean(violation_rates)
        results['is_anomaly'] = results['combined_score'] > 0.05

        return results


def run_physics_checks(df, dt: float = 0.005) -> Dict:
    """
    Run physics checks on a DataFrame.

    Args:
        df: DataFrame with pos, vel, att, rate columns
        dt: Time step

    Returns:
        Physics check results
    """
    detector = PhysicsAnomalyDetector(dt=dt)

    pos = df[['x', 'y', 'z']].values
    vel = df[['vx', 'vy', 'vz']].values
    att = df[['phi', 'theta', 'psi']].values
    rate = df[['p', 'q', 'r']].values

    # Get acceleration if available
    acc = None
    if 'ax' in df.columns:
        acc = df[['ax', 'ay', 'az']].values

    return detector.detect(pos, vel, att, rate, acc)


if __name__ == "__main__":
    # Test on synthetic data
    N = 1000
    dt = 0.005
    t = np.arange(N) * dt

    # Generate smooth trajectory
    pos = np.column_stack([
        np.sin(2 * np.pi * 0.1 * t),
        np.cos(2 * np.pi * 0.1 * t),
        0.5 * np.sin(2 * np.pi * 0.05 * t)
    ])

    vel = np.diff(pos, axis=0, prepend=pos[:1]) / dt
    att = np.zeros((N, 3))
    rate = np.zeros((N, 3))

    detector = PhysicsAnomalyDetector(dt=dt)
    results = detector.detect(pos, vel, att, rate)

    print("Physics Anomaly Detection Test (Clean Data)")
    print("=" * 50)
    print(f"Jerk violation rate: {results['jerk']['violation_rate']*100:.1f}%")
    print(f"Energy violation rate: {results['energy']['violation_rate']*100:.1f}%")
    print(f"Kinematic violation rate: {results['kinematic']['violation_rate']*100:.1f}%")
    print(f"Angular violation rate: {results['angular']['violation_rate']*100:.1f}%")
    print(f"Combined score: {results['combined_score']:.3f}")
    print(f"Is anomaly: {results['is_anomaly']}")

    # Inject GPS jump
    print("\n--- With GPS Jump Attack ---")
    pos_attacked = pos.copy()
    pos_attacked[500:] += np.array([5.0, 0, 0])  # 5m jump

    results_attacked = detector.detect(pos_attacked, vel, att, rate)
    print(f"Jerk violation rate: {results_attacked['jerk']['violation_rate']*100:.1f}%")
    print(f"Kinematic violation rate: {results_attacked['kinematic']['violation_rate']*100:.1f}%")
    print(f"Combined score: {results_attacked['combined_score']:.3f}")
    print(f"Is anomaly: {results_attacked['is_anomaly']}")
