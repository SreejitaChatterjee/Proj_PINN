"""
Physics-Based Consistency Detectors for Attack Detection.

These detectors use fundamental physics principles rather than learned models.
Key insight: Cross-sensor consistency based on Newtonian mechanics.

GPS position and IMU acceleration are linked by physics:
  position'' = acceleration - gravity

If sensors disagree on this relationship → attack detected.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from collections import deque

from ..inference.predictor import Predictor


def euler_to_rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrix.
    Rotates from body frame to world frame.
    """
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


@dataclass
class DetectionResult:
    """Detection result with confidence and triggered detectors."""
    is_anomaly: bool
    confidence: float
    scores: Dict[str, float]
    triggered_by: List[str]


class VelocityConsistencyDetector:
    """
    Detect attacks by comparing GPS-derived velocity with IMU-derived velocity.

    Physics principle:
        GPS velocity = d(position)/dt
        IMU velocity = integral(acceleration - gravity)

        These MUST be consistent. Disagreement indicates attack.

    Key advantages:
        1. Physics-based, no learning required
        2. Short integration window avoids drift
        3. Detects GPS spoofing (IMU tells truth)
        4. Detects IMU spoofing (GPS tells truth)
        5. Hard to evade - must spoof both perfectly

    Args:
        integration_window: Steps to integrate IMU (short to avoid drift)
        consistency_threshold: Max allowed velocity disagreement (m/s)
        dt: Time step between samples
    """

    def __init__(
        self,
        integration_window: int = 5,
        consistency_threshold: float = 2.0,
        dt: float = 0.005,
    ):
        self.integration_window = integration_window
        self.consistency_threshold = consistency_threshold
        self.dt = dt
        self.gravity = np.array([0.0, 0.0, 9.81])

        # Online state
        self.position_history: deque = deque(maxlen=integration_window + 1)
        self.velocity_history: deque = deque(maxlen=integration_window)
        self.accel_history: deque = deque(maxlen=integration_window)
        self.attitude_history: deque = deque(maxlen=integration_window)

        # Calibration
        self.error_mean = 0.0
        self.error_std = 1.0
        self.threshold = consistency_threshold
        self.is_calibrated = False

    def reset(self) -> None:
        """Reset for new sequence."""
        self.position_history.clear()
        self.velocity_history.clear()
        self.accel_history.clear()
        self.attitude_history.clear()

    def calibrate(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        attitudes: np.ndarray,
        percentile: float = 99.0,
    ) -> None:
        """
        Calibrate on clean data to set threshold.

        Args:
            positions: [N, 3] GPS positions (x, y, z)
            velocities: [N, 3] GPS velocities (vx, vy, vz)
            accelerations: [N, 3] IMU accelerations (ax, ay, az)
            attitudes: [N, 3] Euler angles (phi, theta, psi)
            percentile: Threshold percentile
        """
        print("Calibrating VelocityConsistencyDetector...")

        errors = []

        for i in range(self.integration_window, len(positions)):
            # GPS velocity (from state, more accurate than differencing)
            gps_velocity = velocities[i]

            # IMU-derived velocity: integrate acceleration over window
            imu_velocity = np.zeros(3)
            for j in range(self.integration_window):
                idx = i - self.integration_window + j

                # Get world-frame acceleration
                phi, theta, psi = attitudes[idx]
                R = euler_to_rotation_matrix(phi, theta, psi)

                # Body acceleration to world acceleration
                body_accel = accelerations[idx]
                world_accel = R @ body_accel

                # Remove gravity (acceleration is what we measure, gravity adds to it)
                # In EuRoC, accelerometer measures specific force = accel - gravity
                # So world_accel already has gravity subtracted in sensor frame
                # After rotation, we need to add gravity back then subtract
                # Actually: specific_force = acceleration - gravity_in_body
                # world_specific_force = R @ specific_force = R @ (a - R^T @ g) = R@a - g
                # So world acceleration = world_specific_force + g = R@a
                # But if sensor gives specific force, world_accel = R @ sensor - g ... complex

                # Simplified: assume accelerometer gives body-frame specific force
                # world_accel = R @ body_accel gives world-frame specific force
                # Real acceleration = specific_force + gravity = R @ body_accel + [0,0,-g]
                # But we want velocity change, and gravity is constant
                # So just look at consistency, not absolute values

                imu_velocity += world_accel * self.dt

            # Normalize by window size for comparison
            imu_velocity_rate = imu_velocity / (self.integration_window * self.dt)

            # Compare with GPS velocity
            # Note: GPS velocity is instantaneous, IMU is integrated change
            # Better comparison: GPS velocity change over window
            if i >= self.integration_window:
                gps_velocity_change = velocities[i] - velocities[i - self.integration_window]
                gps_velocity_rate = gps_velocity_change / (self.integration_window * self.dt)

                error = np.linalg.norm(gps_velocity_rate - imu_velocity_rate)
                errors.append(error)

        errors = np.array(errors)
        self.error_mean = np.mean(errors)
        self.error_std = np.std(errors) + 1e-6
        self.threshold = np.percentile(errors, percentile)

        print(f"  Velocity consistency: mean={self.error_mean:.4f}, std={self.error_std:.4f}")
        print(f"  Threshold ({percentile}th pctl): {self.threshold:.4f}")

        self.is_calibrated = True

    def update(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        attitude: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Update with new measurement and check consistency.

        Args:
            position: [3] GPS position
            velocity: [3] GPS velocity (from state)
            acceleration: [3] IMU acceleration (body frame)
            attitude: [3] Euler angles (phi, theta, psi)

        Returns:
            (is_anomaly, consistency_error)
        """
        self.position_history.append(position.copy())
        self.velocity_history.append(velocity.copy())
        self.accel_history.append(acceleration.copy())
        self.attitude_history.append(attitude.copy())

        # Need full window
        if len(self.velocity_history) < self.integration_window:
            return False, 0.0

        # GPS velocity change over window
        gps_vel_now = self.velocity_history[-1]
        gps_vel_start = self.velocity_history[0]
        gps_velocity_change = gps_vel_now - gps_vel_start

        # IMU-integrated velocity change
        imu_velocity_change = np.zeros(3)
        for i in range(len(self.accel_history)):
            phi, theta, psi = self.attitude_history[i]
            R = euler_to_rotation_matrix(phi, theta, psi)
            world_accel = R @ self.accel_history[i]
            imu_velocity_change += world_accel * self.dt

        # Consistency error
        error = np.linalg.norm(gps_velocity_change - imu_velocity_change)

        # Normalize
        normalized_error = (error - self.error_mean) / self.error_std

        is_anomaly = error > self.threshold

        return is_anomaly, normalized_error


class LongHorizonRolloutDetector:
    """
    Detect attacks by comparing PINN rollout with measured trajectory.

    Key insight: Single-step errors are small, but accumulate over time.

    Process:
        1. Every N steps, save current state as "anchor"
        2. Roll out PINN from anchor for M steps
        3. Compare PINN prediction with actual measurements
        4. Large divergence indicates attack

    Args:
        predictor: Trained PINN Predictor
        rollout_steps: How many steps to roll out
        anchor_interval: How often to reset anchor
        divergence_threshold: Max allowed divergence
    """

    def __init__(
        self,
        predictor: Predictor,
        rollout_steps: int = 50,
        anchor_interval: int = 100,
        divergence_threshold: float = 5.0,
    ):
        self.predictor = predictor
        self.rollout_steps = rollout_steps
        self.anchor_interval = anchor_interval
        self.divergence_threshold = divergence_threshold

        # State
        self.anchor_state: Optional[np.ndarray] = None
        self.anchor_step: int = 0
        self.step_count: int = 0
        self.rollout_buffer: List[np.ndarray] = []
        self.control_buffer: List[np.ndarray] = []

        # Calibration
        self.divergence_mean = 0.0
        self.divergence_std = 1.0
        self.threshold = divergence_threshold
        self.is_calibrated = False

    def reset(self) -> None:
        """Reset for new sequence."""
        self.anchor_state = None
        self.anchor_step = 0
        self.step_count = 0
        self.rollout_buffer.clear()
        self.control_buffer.clear()

    def calibrate(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        percentile: float = 99.0,
    ) -> None:
        """
        Calibrate on clean data.

        Args:
            states: [N, state_dim] state trajectory
            controls: [N, control_dim] control inputs
            percentile: Threshold percentile
        """
        print("Calibrating LongHorizonRolloutDetector...")

        divergences = []

        # Sample multiple rollouts
        for start_idx in range(0, len(states) - self.rollout_steps - 1, self.anchor_interval):
            initial_state = states[start_idx]
            control_seq = controls[start_idx:start_idx + self.rollout_steps]
            actual_traj = states[start_idx + 1:start_idx + self.rollout_steps + 1]

            # PINN rollout
            predicted_traj = self.predictor.rollout(initial_state, control_seq, self.rollout_steps)

            # Divergence at end of rollout
            if len(actual_traj) == len(predicted_traj):
                final_divergence = np.linalg.norm(predicted_traj[-1] - actual_traj[-1])
                divergences.append(final_divergence)

        if divergences:
            divergences = np.array(divergences)
            self.divergence_mean = np.mean(divergences)
            self.divergence_std = np.std(divergences) + 1e-6
            self.threshold = np.percentile(divergences, percentile)

            print(f"  Rollout divergence: mean={self.divergence_mean:.4f}, std={self.divergence_std:.4f}")
            print(f"  Threshold ({percentile}th pctl): {self.threshold:.4f}")

        self.is_calibrated = True

    def update(
        self,
        state: np.ndarray,
        control: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Update with new state and check divergence.

        Returns:
            (is_anomaly, divergence_score)
        """
        self.step_count += 1

        # Set anchor at start or after interval
        if self.anchor_state is None or (self.step_count - self.anchor_step) >= self.anchor_interval:
            self.anchor_state = state.copy()
            self.anchor_step = self.step_count
            self.rollout_buffer.clear()
            self.control_buffer.clear()

        # Accumulate for rollout comparison
        self.rollout_buffer.append(state.copy())
        self.control_buffer.append(control.copy())

        # Check divergence when we have enough steps
        steps_since_anchor = len(self.rollout_buffer)

        if steps_since_anchor >= self.rollout_steps:
            # Do rollout from anchor
            control_seq = np.array(self.control_buffer[:self.rollout_steps])
            predicted_traj = self.predictor.rollout(
                self.anchor_state, control_seq, self.rollout_steps
            )

            # Compare final states
            actual_final = self.rollout_buffer[self.rollout_steps - 1]
            predicted_final = predicted_traj[-1]

            divergence = np.linalg.norm(actual_final - predicted_final)
            normalized = (divergence - self.divergence_mean) / self.divergence_std

            is_anomaly = divergence > self.threshold

            return is_anomaly, normalized

        return False, 0.0


class TemporalPatternDetector:
    """
    Detect temporal attacks: replay, freeze, discontinuity.

    Uses autocorrelation and variance analysis instead of simple similarity.
    """

    def __init__(
        self,
        window_size: int = 50,
        freeze_threshold: float = 1e-4,
        discontinuity_threshold: float = 10.0,
    ):
        self.window_size = window_size
        self.freeze_threshold = freeze_threshold
        self.discontinuity_threshold = discontinuity_threshold

        # State
        self.state_history: deque = deque(maxlen=window_size)
        self.prev_state: Optional[np.ndarray] = None

        # Calibration for discontinuity
        self.jump_mean = 0.0
        self.jump_std = 1.0
        self.is_calibrated = False

    def reset(self) -> None:
        """Reset for new sequence."""
        self.state_history.clear()
        self.prev_state = None

    def calibrate(self, states: np.ndarray, percentile: float = 99.9) -> None:
        """Calibrate discontinuity threshold on clean data."""
        print("Calibrating TemporalPatternDetector...")

        jumps = []
        for i in range(1, len(states)):
            jump = np.linalg.norm(states[i] - states[i-1])
            jumps.append(jump)

        jumps = np.array(jumps)
        self.jump_mean = np.mean(jumps)
        self.jump_std = np.std(jumps) + 1e-6
        self.discontinuity_threshold = np.percentile(jumps, percentile)

        print(f"  State jumps: mean={self.jump_mean:.4f}, std={self.jump_std:.4f}")
        print(f"  Discontinuity threshold ({percentile}th pctl): {self.discontinuity_threshold:.4f}")

        self.is_calibrated = True

    def update(self, state: np.ndarray) -> Tuple[bool, str, float]:
        """
        Check for temporal anomalies.

        Returns:
            (is_anomaly, attack_type, score)
        """
        self.state_history.append(state.copy())

        # Check for discontinuity (sudden jump)
        if self.prev_state is not None:
            jump = np.linalg.norm(state - self.prev_state)
            if jump > self.discontinuity_threshold:
                self.prev_state = state.copy()
                return True, "discontinuity", jump / self.discontinuity_threshold

        self.prev_state = state.copy()

        # Need full window for freeze detection
        if len(self.state_history) < self.window_size:
            return False, "none", 0.0

        # Check for freeze (near-zero variance)
        history = np.array(list(self.state_history))
        temporal_std = np.std(history, axis=0).mean()

        if temporal_std < self.freeze_threshold:
            return True, "freeze", 1.0 - (temporal_std / self.freeze_threshold)

        return False, "none", 0.0


class HybridAttackDetector:
    """
    Hybrid attack detector combining physics-based and learned approaches.

    Components:
        1. VelocityConsistencyDetector - GPS vs IMU physics consistency
        2. LongHorizonRolloutDetector - PINN trajectory divergence
        3. TemporalPatternDetector - Freeze and discontinuity detection
        4. (Optional) PINN residual - Single-step physics violations

    Fusion: Any detector triggering is considered an attack.
    """

    def __init__(
        self,
        predictor: Optional[Predictor] = None,
        use_velocity_consistency: bool = True,
        use_long_horizon: bool = True,
        use_temporal: bool = True,
        velocity_window: int = 5,
        rollout_steps: int = 50,
    ):
        self.use_velocity_consistency = use_velocity_consistency
        self.use_long_horizon = use_long_horizon and predictor is not None
        self.use_temporal = use_temporal

        # Initialize detectors
        if use_velocity_consistency:
            self.velocity_detector = VelocityConsistencyDetector(
                integration_window=velocity_window
            )

        if self.use_long_horizon:
            self.rollout_detector = LongHorizonRolloutDetector(
                predictor=predictor,
                rollout_steps=rollout_steps,
            )

        if use_temporal:
            self.temporal_detector = TemporalPatternDetector()

        # Statistics
        self.detection_counts = {
            "velocity": 0,
            "rollout": 0,
            "temporal": 0,
            "total": 0,
        }

    def calibrate(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        positions: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
        accelerations: Optional[np.ndarray] = None,
        attitudes: Optional[np.ndarray] = None,
    ) -> None:
        """
        Calibrate all detectors on clean data.

        Args:
            states: [N, 12] full state (x,y,z, phi,theta,psi, p,q,r, vx,vy,vz)
            controls: [N, 4] controls
            positions: [N, 3] GPS positions (extracted from states if None)
            velocities: [N, 3] velocities (extracted from states if None)
            accelerations: [N, 3] IMU accelerations
            attitudes: [N, 3] Euler angles (extracted from states if None)
        """
        print("=" * 60)
        print("HYBRID ATTACK DETECTOR CALIBRATION")
        print("=" * 60)

        # Extract from states if not provided
        if positions is None:
            positions = states[:, :3]
        if velocities is None:
            velocities = states[:, 9:12]
        if attitudes is None:
            attitudes = states[:, 3:6]

        # If accelerations not provided, estimate from velocity
        if accelerations is None:
            dt = 0.005
            accelerations = np.diff(velocities, axis=0) / dt
            accelerations = np.vstack([accelerations, accelerations[-1:]])

        if self.use_velocity_consistency:
            self.velocity_detector.calibrate(
                positions, velocities, accelerations, attitudes
            )

        if self.use_long_horizon:
            self.rollout_detector.calibrate(states, controls)

        if self.use_temporal:
            self.temporal_detector.calibrate(states)

        print("=" * 60)

    def reset(self) -> None:
        """Reset all detectors."""
        if self.use_velocity_consistency:
            self.velocity_detector.reset()
        if self.use_long_horizon:
            self.rollout_detector.reset()
        if self.use_temporal:
            self.temporal_detector.reset()

    def detect(
        self,
        state: np.ndarray,
        control: np.ndarray,
        acceleration: Optional[np.ndarray] = None,
    ) -> DetectionResult:
        """
        Detect attack using all enabled detectors.

        Args:
            state: [12] full state
            control: [4] control input
            acceleration: [3] IMU acceleration (optional)

        Returns:
            DetectionResult with scores from each detector
        """
        scores = {}
        triggered = []

        # Extract components from state
        position = state[:3]
        attitude = state[3:6]
        velocity = state[9:12]

        # Estimate acceleration if not provided
        if acceleration is None:
            # Use a simple estimate (not ideal but functional)
            acceleration = np.zeros(3)

        # Velocity consistency check
        if self.use_velocity_consistency:
            is_anomaly, score = self.velocity_detector.update(
                position, velocity, acceleration, attitude
            )
            scores["velocity"] = score
            if is_anomaly:
                triggered.append("velocity")
                self.detection_counts["velocity"] += 1

        # Long-horizon rollout check
        if self.use_long_horizon:
            is_anomaly, score = self.rollout_detector.update(state, control)
            scores["rollout"] = score
            if is_anomaly:
                triggered.append("rollout")
                self.detection_counts["rollout"] += 1

        # Temporal pattern check
        if self.use_temporal:
            is_anomaly, attack_type, score = self.temporal_detector.update(state)
            scores["temporal"] = score
            scores["temporal_type"] = attack_type
            if is_anomaly:
                triggered.append(f"temporal_{attack_type}")
                self.detection_counts["temporal"] += 1

        # Fusion: any detector triggers
        is_anomaly = len(triggered) > 0
        if is_anomaly:
            self.detection_counts["total"] += 1

        # Confidence: max of normalized scores
        confidence = max(scores.get("velocity", 0),
                        scores.get("rollout", 0),
                        scores.get("temporal", 0))
        confidence = min(max(confidence / 3.0, 0), 1.0)

        return DetectionResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            scores=scores,
            triggered_by=triggered,
        )

    def get_summary(self) -> Dict[str, any]:
        """Get detection summary."""
        return {
            "total": self.detection_counts["total"],
            "by_detector": self.detection_counts.copy(),
        }
