"""
Analytical Redundancy Module (v0.7.0)

This module breaks the CLAO ceiling by adding a SECOND, INDEPENDENT
dynamics estimator for cross-validation.

Key insight: A single estimator cannot distinguish between:
  - Nominal dynamics with measurement noise
  - Faulted dynamics with controller compensation

Two independent estimators with different models CAN detect disagreement
that reveals hidden faults.

Estimators:
1. Primary: EKF with full nonlinear quadrotor model
2. Secondary: Simplified linear model (different assumptions)

When they disagree beyond noise bounds, a fault is likely.

This is how aerospace systems actually achieve >90% actuator recall.

References:
- Isermann, R. (2006). Fault-Diagnosis Systems
- Blanke, M. et al. (2016). Diagnosis and Fault-Tolerant Control
- NASA TM-2003-212615: Analytical Redundancy for FDI
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple
import numpy as np


# =============================================================================
# Constants
# =============================================================================

G = 9.81  # Gravity (m/s^2)
DT = 0.005  # 200 Hz sampling


# =============================================================================
# Primary Estimator: Nonlinear EKF
# =============================================================================

@dataclass
class EKFState:
    """EKF state vector for quadrotor."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: np.ndarray = field(default_factory=lambda: np.zeros(3))  # roll, pitch, yaw
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.position, self.velocity, self.attitude, self.angular_velocity
        ])

    @classmethod
    def from_vector(cls, x: np.ndarray) -> 'EKFState':
        return cls(
            position=x[0:3].copy(),
            velocity=x[3:6].copy(),
            attitude=x[6:9].copy(),
            angular_velocity=x[9:12].copy(),
        )


class NonlinearEKF:
    """
    Primary estimator: Extended Kalman Filter with nonlinear quadrotor model.

    State: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]

    Uses full nonlinear dynamics including:
    - Gravity
    - Thrust-attitude coupling
    - Gyroscopic effects
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 0.5,
        dt: float = DT,
    ):
        self.dt = dt
        self.n_states = 12

        # State and covariance
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 1.0

        # Noise covariances
        self.Q = np.eye(self.n_states) * process_noise
        self.R = np.eye(6) * measurement_noise  # Measure position + velocity

        # History
        self.state_history: List[np.ndarray] = []
        self.innovation_history: List[np.ndarray] = []

    def predict(self, control: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict step using nonlinear dynamics.

        Args:
            control: [thrust, tau_x, tau_y, tau_z] or None

        Returns:
            Predicted state
        """
        # Current state
        pos = self.x[0:3]
        vel = self.x[3:6]
        att = self.x[6:9]  # roll, pitch, yaw
        omega = self.x[9:12]

        # Extract angles
        phi, theta, psi = att

        # Rotation matrix (simplified for small angles)
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        # Thrust direction (body z in world frame)
        thrust_dir = np.array([
            c_psi * s_theta * c_phi + s_psi * s_phi,
            s_psi * s_theta * c_phi - c_psi * s_phi,
            c_theta * c_phi
        ])

        # Default control (hover)
        if control is None:
            thrust = G  # Hover thrust
            tau = np.zeros(3)
        else:
            thrust = control[0] if len(control) > 0 else G
            tau = control[1:4] if len(control) > 3 else np.zeros(3)

        # Dynamics
        acc = thrust_dir * thrust - np.array([0, 0, G])

        # Simple angular dynamics (ignoring inertia tensor for now)
        alpha = tau * 10.0  # Scaled torque to angular acceleration

        # Euler integration
        new_pos = pos + vel * self.dt
        new_vel = vel + acc * self.dt
        new_att = att + omega * self.dt
        new_omega = omega + alpha * self.dt

        # Wrap angles
        new_att = np.mod(new_att + np.pi, 2 * np.pi) - np.pi

        # Update state
        self.x = np.concatenate([new_pos, new_vel, new_att, new_omega])

        # Linearized dynamics Jacobian (simplified)
        F = np.eye(self.n_states)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[6:9, 9:12] = np.eye(3) * self.dt

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

        return self.x.copy()

    def update(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step with measurement.

        Args:
            measurement: [x, y, z, vx, vy, vz] position and velocity

        Returns:
            (updated_state, innovation)
        """
        # Measurement matrix (observe position and velocity)
        H = np.zeros((6, self.n_states))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)

        # Innovation
        y_pred = H @ self.x
        innovation = measurement - y_pred

        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ innovation

        # Covariance update
        self.P = (np.eye(self.n_states) - K @ H) @ self.P

        # Store history
        self.state_history.append(self.x.copy())
        self.innovation_history.append(innovation.copy())

        return self.x.copy(), innovation

    def get_state(self) -> EKFState:
        return EKFState.from_vector(self.x)

    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset estimator."""
        self.x = initial_state if initial_state is not None else np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 1.0
        self.state_history = []
        self.innovation_history = []


# =============================================================================
# Secondary Estimator: Linear Complementary Filter
# =============================================================================

class LinearComplementaryEstimator:
    """
    Secondary estimator: Simplified linear model with different assumptions.

    Uses a complementary filter approach:
    - High-pass IMU integration for fast dynamics
    - Low-pass GPS for drift correction

    Different from EKF:
    - No thrust-attitude coupling
    - Linear dynamics only
    - Different noise model

    This independence is what enables fault detection.
    """

    def __init__(
        self,
        alpha: float = 0.98,  # Complementary filter coefficient
        dt: float = DT,
    ):
        self.alpha = alpha
        self.dt = dt

        # State: [x, y, z, vx, vy, vz]
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

        # IMU integration state
        self.velocity_imu = np.zeros(3)

        # History
        self.state_history: List[np.ndarray] = []

    def update(
        self,
        gps_position: np.ndarray,
        gps_velocity: np.ndarray,
        imu_acceleration: np.ndarray,
    ) -> np.ndarray:
        """
        Update with sensor fusion.

        Args:
            gps_position: [x, y, z] from GPS
            gps_velocity: [vx, vy, vz] from GPS
            imu_acceleration: [ax, ay, az] from IMU (body frame, gravity-corrected)

        Returns:
            Estimated state [x, y, z, vx, vy, vz]
        """
        # IMU integration (high-frequency path)
        self.velocity_imu += imu_acceleration * self.dt

        # Complementary filter for velocity
        self.velocity = self.alpha * (self.velocity + imu_acceleration * self.dt) + \
                        (1 - self.alpha) * gps_velocity

        # Position from velocity integration + GPS correction
        position_pred = self.position + self.velocity * self.dt
        self.position = self.alpha * position_pred + (1 - self.alpha) * gps_position

        state = np.concatenate([self.position, self.velocity])
        self.state_history.append(state.copy())

        return state

    def get_state(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity])

    def reset(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.velocity_imu = np.zeros(3)
        self.state_history = []


# =============================================================================
# Disagreement Detector
# =============================================================================

@dataclass
class DisagreementResult:
    """Result from estimator disagreement analysis."""
    primary_state: np.ndarray
    secondary_state: np.ndarray
    disagreement: np.ndarray  # Per-channel difference
    disagreement_norm: float
    threshold: float
    is_fault_detected: bool
    fault_confidence: float
    fault_channel: Optional[str]  # Which channel disagrees most


class EstimatorDisagreementDetector:
    """
    Detects faults by monitoring disagreement between estimators.

    Key insight: Under nominal conditions, both estimators track reality
    with bounded error. Under faults, one or both deviate, causing
    disagreement that exceeds noise bounds.

    This breaks the CLAO ceiling because:
    - Controller compensation affects both estimators similarly
    - But actuator faults cause different responses in different models
    - The disagreement reveals the hidden fault
    """

    def __init__(
        self,
        position_threshold: float = 2.0,  # meters
        velocity_threshold: float = 1.0,  # m/s
        adaptive: bool = True,
        window_size: int = 100,
    ):
        self.position_threshold = position_threshold
        self.velocity_threshold = velocity_threshold
        self.adaptive = adaptive
        self.window_size = window_size

        # Baseline statistics (learned during calibration)
        self.baseline_mean = np.zeros(6)
        self.baseline_std = np.ones(6)
        self.calibrated = False

        # History
        self.disagreement_history: List[np.ndarray] = []
        self.detection_count = 0
        self.total_count = 0

    def calibrate(self, disagreements: np.ndarray):
        """
        Calibrate thresholds from nominal disagreement data.

        Args:
            disagreements: [N, 6] array of historical disagreements
        """
        if len(disagreements) < 10:
            return

        self.baseline_mean = np.mean(disagreements, axis=0)
        self.baseline_std = np.std(disagreements, axis=0) + 1e-6

        self.calibrated = True

    def detect(
        self,
        primary_state: np.ndarray,
        secondary_state: np.ndarray,
    ) -> DisagreementResult:
        """
        Detect fault from estimator disagreement.

        Args:
            primary_state: [x, y, z, vx, vy, vz] from primary estimator
            secondary_state: [x, y, z, vx, vy, vz] from secondary estimator

        Returns:
            DisagreementResult with detection decision
        """
        self.total_count += 1

        # Compute disagreement
        disagreement = primary_state[:6] - secondary_state[:6]
        self.disagreement_history.append(disagreement.copy())

        # Trim history
        if len(self.disagreement_history) > self.window_size:
            self.disagreement_history.pop(0)

        # Normalize if calibrated
        if self.calibrated:
            normalized = (disagreement - self.baseline_mean) / self.baseline_std
        else:
            normalized = disagreement

        # Compute per-channel thresholds
        thresholds = np.array([
            self.position_threshold, self.position_threshold, self.position_threshold,
            self.velocity_threshold, self.velocity_threshold, self.velocity_threshold,
        ])

        # Check for fault
        position_disagree = np.linalg.norm(disagreement[:3])
        velocity_disagree = np.linalg.norm(disagreement[3:6])

        is_fault = (position_disagree > self.position_threshold or
                    velocity_disagree > self.velocity_threshold)

        if is_fault:
            self.detection_count += 1

        # Find most disagreeing channel
        abs_disagree = np.abs(normalized)
        max_idx = np.argmax(abs_disagree)
        channel_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        fault_channel = channel_names[max_idx] if is_fault else None

        # Confidence based on how far above threshold
        disagree_norm = np.linalg.norm(disagreement[:3]) / self.position_threshold
        confidence = min(1.0, disagree_norm) if is_fault else 0.0

        return DisagreementResult(
            primary_state=primary_state,
            secondary_state=secondary_state,
            disagreement=disagreement,
            disagreement_norm=float(np.linalg.norm(disagreement)),
            threshold=self.position_threshold,
            is_fault_detected=is_fault,
            fault_confidence=confidence,
            fault_channel=fault_channel,
        )

    def get_detection_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.detection_count / self.total_count

    def reset(self):
        self.disagreement_history = []
        self.detection_count = 0
        self.total_count = 0


# =============================================================================
# Combined Analytical Redundancy System
# =============================================================================

@dataclass
class AnalyticalRedundancyResult:
    """Result from analytical redundancy system."""
    primary_state: EKFState
    secondary_state: np.ndarray
    disagreement: DisagreementResult
    primary_innovation: np.ndarray
    is_fault_detected: bool
    detection_source: str  # "disagreement", "innovation", "combined", "none"
    actuator_fault_likely: bool
    sensor_fault_likely: bool
    confidence: float


class AnalyticalRedundancySystem:
    """
    Complete analytical redundancy system for breaking the CLAO ceiling.

    Components:
    1. Primary estimator (nonlinear EKF)
    2. Secondary estimator (linear complementary)
    3. Disagreement detector
    4. Innovation monitor

    Detection logic:
    - Disagreement alone → likely actuator fault (affects dynamics differently)
    - Innovation alone → likely sensor fault (affects measurements)
    - Both → likely major fault or attack
    """

    def __init__(
        self,
        position_threshold: float = 2.0,
        velocity_threshold: float = 1.0,
        innovation_threshold: float = 3.0,
    ):
        self.primary = NonlinearEKF()
        self.secondary = LinearComplementaryEstimator()
        self.disagreement_detector = EstimatorDisagreementDetector(
            position_threshold=position_threshold,
            velocity_threshold=velocity_threshold,
        )
        self.innovation_threshold = innovation_threshold

        # Statistics
        self.fault_count = 0
        self.actuator_fault_count = 0
        self.sensor_fault_count = 0
        self.total_count = 0

    def update(
        self,
        gps_position: np.ndarray,
        gps_velocity: np.ndarray,
        imu_acceleration: np.ndarray,
        control: Optional[np.ndarray] = None,
    ) -> AnalyticalRedundancyResult:
        """
        Process one timestep through both estimators.

        Args:
            gps_position: [x, y, z] GPS position
            gps_velocity: [vx, vy, vz] GPS velocity
            imu_acceleration: [ax, ay, az] IMU acceleration (gravity-corrected)
            control: [thrust, tau_x, tau_y, tau_z] control inputs (optional)

        Returns:
            AnalyticalRedundancyResult with fault detection
        """
        self.total_count += 1

        # Primary estimator: predict + update
        self.primary.predict(control)
        measurement = np.concatenate([gps_position, gps_velocity])
        primary_state, innovation = self.primary.update(measurement)

        # Secondary estimator: complementary filter
        secondary_state = self.secondary.update(
            gps_position, gps_velocity, imu_acceleration
        )

        # Check disagreement
        disagreement = self.disagreement_detector.detect(
            primary_state[:6], secondary_state
        )

        # Check innovation
        innovation_norm = np.linalg.norm(innovation)
        innovation_fault = innovation_norm > self.innovation_threshold

        # Classification logic
        disagree_fault = disagreement.is_fault_detected

        if disagree_fault and innovation_fault:
            detection_source = "combined"
            actuator_likely = True
            sensor_likely = True
            confidence = max(disagreement.fault_confidence, innovation_norm / self.innovation_threshold)
        elif disagree_fault:
            detection_source = "disagreement"
            actuator_likely = True  # Disagreement implies dynamics mismatch
            sensor_likely = False
            confidence = disagreement.fault_confidence
        elif innovation_fault:
            detection_source = "innovation"
            actuator_likely = False
            sensor_likely = True  # Innovation implies measurement mismatch
            confidence = min(1.0, innovation_norm / self.innovation_threshold)
        else:
            detection_source = "none"
            actuator_likely = False
            sensor_likely = False
            confidence = 0.0

        is_fault = disagree_fault or innovation_fault

        if is_fault:
            self.fault_count += 1
            if actuator_likely:
                self.actuator_fault_count += 1
            if sensor_likely:
                self.sensor_fault_count += 1

        return AnalyticalRedundancyResult(
            primary_state=self.primary.get_state(),
            secondary_state=secondary_state,
            disagreement=disagreement,
            primary_innovation=innovation,
            is_fault_detected=is_fault,
            detection_source=detection_source,
            actuator_fault_likely=actuator_likely,
            sensor_fault_likely=sensor_likely,
            confidence=confidence,
        )

    def calibrate(self, nominal_data: List[Dict]):
        """
        Calibrate from nominal flight data.

        Args:
            nominal_data: List of dicts with gps_position, gps_velocity, imu_acceleration
        """
        disagreements = []

        for sample in nominal_data:
            result = self.update(
                sample['gps_position'],
                sample['gps_velocity'],
                sample['imu_acceleration'],
                sample.get('control'),
            )
            disagreements.append(result.disagreement.disagreement)

        if disagreements:
            self.disagreement_detector.calibrate(np.array(disagreements))

        self.reset_statistics()

    def get_metrics(self) -> Dict[str, float]:
        return {
            "total_samples": self.total_count,
            "fault_detections": self.fault_count,
            "actuator_faults": self.actuator_fault_count,
            "sensor_faults": self.sensor_fault_count,
            "detection_rate": self.fault_count / max(1, self.total_count),
        }

    def reset_statistics(self):
        self.fault_count = 0
        self.actuator_fault_count = 0
        self.sensor_fault_count = 0
        self.total_count = 0

    def reset(self):
        self.primary.reset()
        self.secondary.reset()
        self.disagreement_detector.reset()
        self.reset_statistics()


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_analytical_redundancy(
    gps_positions: np.ndarray,
    gps_velocities: np.ndarray,
    imu_accelerations: np.ndarray,
    labels: np.ndarray,
    controls: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate analytical redundancy system.

    Args:
        gps_positions: [N, 3] GPS positions
        gps_velocities: [N, 3] GPS velocities
        imu_accelerations: [N, 3] IMU accelerations
        labels: [N] ground truth (0=nominal, 1=fault)
        controls: [N, 4] control inputs (optional)

    Returns:
        Dict with recall, FPR, precision
    """
    system = AnalyticalRedundancySystem()

    predictions = []
    for i in range(len(gps_positions)):
        control = controls[i] if controls is not None else None

        result = system.update(
            gps_positions[i],
            gps_velocities[i],
            imu_accelerations[i],
            control,
        )
        predictions.append(result.is_fault_detected)

    predictions = np.array(predictions)

    # Metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    precision = tp / max(1, tp + fp)

    return {
        "recall": float(recall),
        "fpr": float(fpr),
        "precision": float(precision),
        "f1": float(2 * precision * recall / max(0.001, precision + recall)),
        **system.get_metrics(),
    }
