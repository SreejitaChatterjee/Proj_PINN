"""
CPU-Friendly EKF with Integrity Metrics.

Implements a 15-state error-state EKF for IMU/GPS/Baro/Mag fusion with
integrity monitoring that serves as a RAIM proxy without requiring
multi-constellation GNSS.

Integrity Metrics:
- NIS (Normalized Innovation Squared) - chi-squared distributed under H0
- Innovation magnitude - raw measurement residual
- Adaptive covariance scaling - detect measurement degradation
- Consistency ratio - predicted vs actual innovation variance

CPU Impact: O(n^3) for n=15 states, but small constant; runs real-time on CPU.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum


class IntegrityLevel(Enum):
    """Integrity assessment levels."""
    NORMAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class EKFConfig:
    """EKF configuration parameters."""
    dt: float = 0.005  # Time step (200 Hz)

    # Process noise (tuned for quadrotor)
    sigma_acc: float = 0.5      # Accelerometer noise (m/s^2)
    sigma_gyro: float = 0.01    # Gyroscope noise (rad/s)
    sigma_acc_bias: float = 1e-4   # Acc bias random walk
    sigma_gyro_bias: float = 1e-5  # Gyro bias random walk

    # Measurement noise
    sigma_pos: float = 0.1      # Position measurement (m)
    sigma_vel: float = 0.05     # Velocity measurement (m/s)
    sigma_baro: float = 0.5     # Barometer altitude (m)
    sigma_mag: float = 0.1      # Magnetometer heading (rad)

    # Integrity thresholds
    nis_threshold_95: float = 7.815   # Chi2(3) 95th percentile
    nis_threshold_99: float = 11.345  # Chi2(3) 99th percentile
    innovation_threshold: float = 3.0  # Sigma multiplier
    consistency_window: int = 50       # Samples for consistency check


@dataclass
class EKFState:
    """EKF state vector and covariance."""
    # Error state: [dp(3), dv(3), dtheta(3), dba(3), dbg(3)] = 15 states
    x: np.ndarray = field(default_factory=lambda: np.zeros(15))
    P: np.ndarray = field(default_factory=lambda: np.eye(15) * 0.01)

    # Nominal state
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    att: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Euler angles
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class IntegrityMetrics:
    """Integrity monitoring metrics."""
    nis_pos: float = 0.0          # NIS for position update
    nis_baro: float = 0.0         # NIS for baro update
    nis_mag: float = 0.0          # NIS for mag update
    innovation_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    innovation_baro: float = 0.0
    innovation_mag: float = 0.0
    consistency_ratio: float = 1.0  # Actual/predicted innovation variance
    level: IntegrityLevel = IntegrityLevel.NORMAL


class IntegrityEKF:
    """
    Error-state EKF with integrity monitoring.

    State vector (error-state formulation):
    - dp: Position error (3)
    - dv: Velocity error (3)
    - dtheta: Attitude error (3)
    - dba: Accelerometer bias error (3)
    - dbg: Gyroscope bias error (3)
    Total: 15 states

    Measurements:
    - Position (GPS/MoCap): 3D
    - Velocity: 3D
    - Barometer: 1D altitude
    - Magnetometer: 1D heading
    """

    def __init__(self, config: Optional[EKFConfig] = None):
        self.config = config or EKFConfig()
        self.state = EKFState()
        self.metrics = IntegrityMetrics()

        # Build process noise matrix
        self._build_Q()

        # History for consistency monitoring
        self.innovation_history = []
        self.predicted_cov_history = []

    def reset(self):
        """Reset EKF to initial state."""
        self.state = EKFState()
        self.metrics = IntegrityMetrics()
        self.innovation_history = []
        self.predicted_cov_history = []

    def _build_Q(self):
        """Build process noise covariance matrix."""
        cfg = self.config
        dt = cfg.dt

        self.Q = np.zeros((15, 15))
        # Position (from velocity noise integration)
        self.Q[0:3, 0:3] = np.eye(3) * (cfg.sigma_acc * dt**2)**2
        # Velocity (from accelerometer noise)
        self.Q[3:6, 3:6] = np.eye(3) * (cfg.sigma_acc * dt)**2
        # Attitude (from gyroscope noise)
        self.Q[6:9, 6:9] = np.eye(3) * (cfg.sigma_gyro * dt)**2
        # Accelerometer bias random walk
        self.Q[9:12, 9:12] = np.eye(3) * (cfg.sigma_acc_bias * dt)**2
        # Gyroscope bias random walk
        self.Q[12:15, 12:15] = np.eye(3) * (cfg.sigma_gyro_bias * dt)**2

    def initialize(self, pos: np.ndarray, vel: np.ndarray, att: np.ndarray):
        """Initialize EKF with known state."""
        self.state = EKFState()
        self.state.pos = pos.copy()
        self.state.vel = vel.copy()
        self.state.att = att.copy()

        # Initial covariance
        self.state.P = np.diag([
            0.1, 0.1, 0.1,     # Position
            0.05, 0.05, 0.05,  # Velocity
            0.01, 0.01, 0.01,  # Attitude
            0.01, 0.01, 0.01,  # Acc bias
            0.001, 0.001, 0.001  # Gyro bias
        ])

        self.innovation_history = []
        self.predicted_cov_history = []

    def _rotation_matrix(self, att: np.ndarray) -> np.ndarray:
        """Compute rotation matrix from Euler angles (ZYX)."""
        phi, theta, psi = att
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        R = np.array([
            [cth*cpsi, sphi*sth*cpsi - cphi*spsi, cphi*sth*cpsi + sphi*spsi],
            [cth*spsi, sphi*sth*spsi + cphi*cpsi, cphi*sth*spsi - sphi*cpsi],
            [-sth, sphi*cth, cphi*cth]
        ])
        return R

    def predict(self, acc: np.ndarray, gyro: np.ndarray):
        """
        Prediction step using IMU measurements.

        Args:
            acc: Accelerometer reading in body frame (3,)
            gyro: Gyroscope reading in body frame (3,)
        """
        dt = self.config.dt
        state = self.state

        # Remove estimated biases
        acc_corrected = acc - state.acc_bias
        gyro_corrected = gyro - state.gyro_bias

        # Rotation matrix
        R = self._rotation_matrix(state.att)

        # Propagate nominal state
        # Velocity update (gravity in NED: [0, 0, 9.81])
        gravity = np.array([0, 0, 9.81])
        state.vel = state.vel + (R @ acc_corrected - gravity) * dt

        # Position update
        state.pos = state.pos + state.vel * dt

        # Attitude update (small angle approximation)
        state.att = state.att + gyro_corrected * dt

        # Normalize angles
        state.att = np.mod(state.att + np.pi, 2*np.pi) - np.pi

        # Build state transition matrix F
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt  # dp/dv
        F[3:6, 6:9] = -R @ self._skew(acc_corrected) * dt  # dv/dtheta
        F[3:6, 9:12] = -R * dt  # dv/dba
        F[6:9, 12:15] = -np.eye(3) * dt  # dtheta/dbg

        # Propagate covariance
        state.P = F @ state.P @ F.T + self.Q

        # Reset error state
        state.x = np.zeros(15)

    def _skew(self, v: np.ndarray) -> np.ndarray:
        """Skew-symmetric matrix from vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def update_position(self, pos_meas: np.ndarray, R_scale: float = 1.0) -> IntegrityMetrics:
        """
        Update with position measurement (GPS/MoCap).

        Args:
            pos_meas: Measured position (3,)
            R_scale: Measurement noise scaling (increase during suspected spoofing)

        Returns:
            Integrity metrics for this update
        """
        state = self.state
        cfg = self.config

        # Measurement matrix
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)

        # Measurement noise
        R = np.eye(3) * (cfg.sigma_pos * R_scale)**2

        # Innovation
        y = pos_meas - state.pos

        # Innovation covariance
        S = H @ state.P @ H.T + R

        # NIS (Normalized Innovation Squared)
        S_inv = np.linalg.inv(S)
        nis = y.T @ S_inv @ y

        # Kalman gain
        K = state.P @ H.T @ S_inv

        # Update error state
        state.x = state.x + K @ y

        # Update covariance (Joseph form for stability)
        I_KH = np.eye(15) - K @ H
        state.P = I_KH @ state.P @ I_KH.T + K @ R @ K.T

        # Inject error into nominal state
        state.pos = state.pos + state.x[0:3]
        state.vel = state.vel + state.x[3:6]
        state.att = state.att + state.x[6:9]
        state.acc_bias = state.acc_bias + state.x[9:12]
        state.gyro_bias = state.gyro_bias + state.x[12:15]
        state.x = np.zeros(15)

        # Store for consistency check
        self.innovation_history.append(y)
        self.predicted_cov_history.append(np.diag(S))

        # Compute consistency ratio
        if len(self.innovation_history) >= cfg.consistency_window:
            recent_inn = np.array(self.innovation_history[-cfg.consistency_window:])
            recent_cov = np.array(self.predicted_cov_history[-cfg.consistency_window:])
            actual_var = np.var(recent_inn, axis=0)
            predicted_var = np.mean(recent_cov, axis=0)
            consistency = np.mean(actual_var / (predicted_var + 1e-10))
        else:
            consistency = 1.0

        # Update integrity metrics
        self.metrics.nis_pos = nis
        self.metrics.innovation_pos = y
        self.metrics.consistency_ratio = consistency
        self.metrics.level = self._assess_integrity(nis, y, S)

        return self.metrics

    def update_baro(self, baro_z: float, R_scale: float = 1.0) -> IntegrityMetrics:
        """
        Update with barometer altitude measurement.

        Args:
            baro_z: Measured altitude (m)
            R_scale: Measurement noise scaling

        Returns:
            Integrity metrics
        """
        state = self.state
        cfg = self.config

        # Measurement matrix (z component only)
        H = np.zeros((1, 15))
        H[0, 2] = 1.0

        # Measurement noise
        R = np.array([[cfg.sigma_baro * R_scale]])**2

        # Innovation
        y = baro_z - state.pos[2]

        # Innovation covariance
        S = H @ state.P @ H.T + R

        # NIS
        nis = y**2 / S[0, 0]

        # Kalman gain
        K = state.P @ H.T / S[0, 0]

        # Update
        state.x = state.x + K.flatten() * y
        I_KH = np.eye(15) - np.outer(K.flatten(), H)
        state.P = I_KH @ state.P

        # Inject error
        state.pos = state.pos + state.x[0:3]
        state.x = np.zeros(15)

        self.metrics.nis_baro = nis
        self.metrics.innovation_baro = y

        return self.metrics

    def update_mag(self, mag_heading: float, R_scale: float = 1.0) -> IntegrityMetrics:
        """
        Update with magnetometer heading measurement.

        Args:
            mag_heading: Measured magnetic heading (rad)
            R_scale: Measurement noise scaling

        Returns:
            Integrity metrics
        """
        state = self.state
        cfg = self.config

        # Measurement matrix (yaw component only)
        H = np.zeros((1, 15))
        H[0, 8] = 1.0  # psi is index 2 of attitude (index 8 of full state)

        # Measurement noise
        R = np.array([[cfg.sigma_mag * R_scale]])**2

        # Innovation (handle angle wrapping)
        y = mag_heading - state.att[2]
        y = np.mod(y + np.pi, 2*np.pi) - np.pi

        # Innovation covariance
        S = H @ state.P @ H.T + R

        # NIS
        nis = y**2 / S[0, 0]

        # Kalman gain
        K = state.P @ H.T / S[0, 0]

        # Update
        state.x = state.x + K.flatten() * y
        I_KH = np.eye(15) - np.outer(K.flatten(), H)
        state.P = I_KH @ state.P

        # Inject error
        state.att = state.att + state.x[6:9]
        state.x = np.zeros(15)

        self.metrics.nis_mag = nis
        self.metrics.innovation_mag = y

        return self.metrics

    def _assess_integrity(
        self,
        nis: float,
        innovation: np.ndarray,
        S: np.ndarray
    ) -> IntegrityLevel:
        """Assess integrity level from metrics."""
        cfg = self.config

        # Check NIS against chi-squared thresholds
        if nis > cfg.nis_threshold_99:
            return IntegrityLevel.CRITICAL
        elif nis > cfg.nis_threshold_95:
            return IntegrityLevel.WARNING

        # Check individual innovations against sigma bounds
        sigma_bounds = np.sqrt(np.diag(S))
        normalized_inn = np.abs(innovation) / sigma_bounds
        if np.any(normalized_inn > cfg.innovation_threshold):
            return IntegrityLevel.WARNING

        # Check consistency ratio
        if self.metrics.consistency_ratio > 2.0 or self.metrics.consistency_ratio < 0.5:
            return IntegrityLevel.CAUTION

        return IntegrityLevel.NORMAL

    def get_integrity_score(self) -> float:
        """
        Get scalar integrity score for fusion with other detectors.

        Returns:
            Score in [0, 1] where 1 = anomalous
        """
        cfg = self.config
        m = self.metrics

        # Combine NIS scores (normalized by thresholds)
        nis_score = max(
            m.nis_pos / cfg.nis_threshold_99,
            m.nis_baro / cfg.nis_threshold_99 if m.nis_baro > 0 else 0,
            m.nis_mag / cfg.nis_threshold_99 if m.nis_mag > 0 else 0
        )

        # Consistency score
        consistency_score = abs(m.consistency_ratio - 1.0)

        # Combined score (clamped to [0, 1])
        score = min(1.0, 0.7 * nis_score + 0.3 * consistency_score)

        return score

    def get_integrity_level(self) -> IntegrityLevel:
        """
        Get integrity level based on current score.

        Returns:
            IntegrityLevel enum
        """
        score = self.get_integrity_score()
        if score < 0.3:
            return IntegrityLevel.NORMAL
        elif score < 0.5:
            return IntegrityLevel.CAUTION
        elif score < 0.8:
            return IntegrityLevel.WARNING
        else:
            return IntegrityLevel.CRITICAL

    def get_nis(self) -> float:
        """Get current position NIS value."""
        return self.metrics.nis_pos


def run_ekf_detector(
    df,
    config: Optional[EKFConfig] = None,
    emulated_sensors: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Run EKF-based integrity detector on dataframe.

    Args:
        df: DataFrame with state and IMU columns
        config: EKF configuration
        emulated_sensors: Dict with baro_z, mag_heading from emulator

    Returns:
        scores: Integrity scores (N,)
        levels: Integrity levels (N,)
        metrics_history: List of IntegrityMetrics
    """
    ekf = IntegrityEKF(config)

    # Initialize
    ekf.initialize(
        pos=df[['x', 'y', 'z']].values[0],
        vel=df[['vx', 'vy', 'vz']].values[0],
        att=df[['phi', 'theta', 'psi']].values[0]
    )

    N = len(df)
    scores = np.zeros(N)
    levels = np.zeros(N, dtype=int)
    metrics_history = []

    for i in range(1, N):
        # Get IMU (using state derivatives as proxy if not available)
        if 'ax' in df.columns:
            acc = df[['ax', 'ay', 'az']].values[i]
        else:
            # Approximate from velocity change
            acc = (df[['vx', 'vy', 'vz']].values[i] - df[['vx', 'vy', 'vz']].values[i-1]) / ekf.config.dt

        gyro = df[['p', 'q', 'r']].values[i]

        # Predict
        ekf.predict(acc, gyro)

        # Update with position (GPS/MoCap)
        pos_meas = df[['x', 'y', 'z']].values[i]
        metrics = ekf.update_position(pos_meas)

        # Update with emulated sensors if available
        if emulated_sensors is not None:
            ekf.update_baro(emulated_sensors['baro_z'][i])
            ekf.update_mag(emulated_sensors['mag_heading'][i])

        scores[i] = ekf.get_integrity_score()
        levels[i] = metrics.level.value
        metrics_history.append(metrics)

    return scores, levels, metrics_history
