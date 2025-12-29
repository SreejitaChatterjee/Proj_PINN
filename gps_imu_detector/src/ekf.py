"""
Simple Extended Kalman Filter with NIS (Normalized Innovation Squared)

State: [pos(3), vel(3), attitude(3), gyro_bias(3), accel_bias(3)] = 15 dims

NIS is used as an integrity proxy:
- High NIS indicates measurement inconsistency (potential attack)
- Chi-squared test for anomaly detection
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy.stats import chi2


@dataclass
class EKFState:
    """EKF state container."""
    x: np.ndarray        # [15] state estimate
    P: np.ndarray        # [15, 15] state covariance
    timestamp: float


class SimpleEKF:
    """
    Simple Extended Kalman Filter for GPS-IMU fusion.

    State vector [15]:
        [0:3]   - position (x, y, z)
        [3:6]   - velocity (vx, vy, vz)
        [6:9]   - attitude (roll, pitch, yaw)
        [9:12]  - gyro bias (bwx, bwy, bwz)
        [12:15] - accel bias (bax, bay, baz)

    Measurements:
        GPS: position (3), velocity (3)
        IMU: angular rates (3), acceleration (3)
    """

    def __init__(
        self,
        dt: float = 0.005,
        process_noise_pos: float = 0.01,
        process_noise_vel: float = 0.1,
        process_noise_att: float = 0.01,
        process_noise_bias: float = 0.001,
        meas_noise_pos: float = 0.1,
        meas_noise_vel: float = 0.05,
        nis_alpha: float = 0.05,  # Chi-squared significance level
    ):
        self.dt = dt

        # State dimension
        self.n = 15

        # Process noise
        self.Q = np.diag([
            process_noise_pos, process_noise_pos, process_noise_pos,  # pos
            process_noise_vel, process_noise_vel, process_noise_vel,  # vel
            process_noise_att, process_noise_att, process_noise_att,  # att
            process_noise_bias, process_noise_bias, process_noise_bias,  # gyro bias
            process_noise_bias, process_noise_bias, process_noise_bias,  # accel bias
        ]) ** 2

        # Measurement noise (GPS: pos + vel)
        self.R_gps = np.diag([
            meas_noise_pos, meas_noise_pos, meas_noise_pos,
            meas_noise_vel, meas_noise_vel, meas_noise_vel,
        ]) ** 2

        # NIS threshold (chi-squared with 6 DOF for GPS measurement)
        self.nis_threshold = chi2.ppf(1 - nis_alpha, df=6)

        # Initialize state
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 1.0

        # History for NIS
        self.nis_history = []

    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset filter to initial state."""
        if initial_state is not None:
            self.x = initial_state.copy()
        else:
            self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 1.0
        self.nis_history = []

    def predict(self, imu_gyro: np.ndarray, imu_accel: np.ndarray):
        """
        Prediction step using IMU measurements.

        Args:
            imu_gyro: [3] angular rates (rad/s)
            imu_accel: [3] acceleration (m/s^2)
        """
        # Extract state
        pos = self.x[0:3]
        vel = self.x[3:6]
        att = self.x[6:9]
        gyro_bias = self.x[9:12]
        accel_bias = self.x[12:15]

        # Bias-corrected measurements
        omega = imu_gyro - gyro_bias
        accel = imu_accel - accel_bias

        # Attitude update (Euler integration)
        att_new = att + omega * self.dt

        # Rotation matrix (simplified for small angles)
        roll, pitch, yaw = att
        R = self._euler_to_rotation(roll, pitch, yaw)

        # Velocity update (rotate accel to world frame, subtract gravity)
        g = np.array([0, 0, 9.81])
        accel_world = R @ accel - g
        vel_new = vel + accel_world * self.dt

        # Position update
        pos_new = pos + vel * self.dt + 0.5 * accel_world * self.dt**2

        # Update state
        self.x[0:3] = pos_new
        self.x[3:6] = vel_new
        self.x[6:9] = att_new
        # Biases stay constant (random walk)

        # State transition Jacobian (simplified)
        F = np.eye(self.n)
        F[0:3, 3:6] = np.eye(3) * self.dt
        # More complex terms for attitude/bias coupling omitted for simplicity

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(
        self,
        gps_pos: np.ndarray,
        gps_vel: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Update step using GPS measurements.

        Args:
            gps_pos: [3] GPS position
            gps_vel: [3] GPS velocity

        Returns:
            nis: Normalized Innovation Squared
            is_consistent: True if NIS within bounds
        """
        # Measurement
        z = np.concatenate([gps_pos, gps_vel])

        # Predicted measurement
        h = np.concatenate([self.x[0:3], self.x[3:6]])

        # Innovation
        y = z - h

        # Measurement Jacobian
        H = np.zeros((6, self.n))
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:6, 3:6] = np.eye(3)  # Velocity

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_gps

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for stability)
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_gps @ K.T

        # Compute NIS
        nis = y @ np.linalg.inv(S) @ y
        is_consistent = nis < self.nis_threshold

        self.nis_history.append(nis)

        return nis, is_consistent

    def _euler_to_rotation(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])

        return R

    def get_state(self) -> EKFState:
        """Get current state estimate."""
        return EKFState(
            x=self.x.copy(),
            P=self.P.copy(),
            timestamp=0.0
        )


class EKFAnomalyDetector:
    """
    Anomaly detector based on EKF NIS.

    Uses statistical test on NIS to detect attacks.
    """

    def __init__(
        self,
        dt: float = 0.005,
        window_size: int = 50,
        alpha: float = 0.01  # Detection threshold
    ):
        self.ekf = SimpleEKF(dt=dt, nis_alpha=alpha)
        self.window_size = window_size
        self.alpha = alpha

        # NIS statistics
        self.nis_window = []

    def reset(self):
        """Reset detector."""
        self.ekf.reset()
        self.nis_window = []

    def process_step(
        self,
        imu_gyro: np.ndarray,
        imu_accel: np.ndarray,
        gps_pos: np.ndarray,
        gps_vel: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Process one timestep.

        Args:
            imu_gyro: [3] gyroscope
            imu_accel: [3] accelerometer
            gps_pos: [3] GPS position
            gps_vel: [3] GPS velocity

        Returns:
            nis: Current NIS value
            nis_avg: Windowed average NIS
            is_anomaly: True if anomaly detected
        """
        # Predict
        self.ekf.predict(imu_gyro, imu_accel)

        # Update
        nis, is_consistent = self.ekf.update_gps(gps_pos, gps_vel)

        # Update window
        self.nis_window.append(nis)
        if len(self.nis_window) > self.window_size:
            self.nis_window.pop(0)

        # Compute average NIS
        nis_avg = np.mean(self.nis_window)

        # Anomaly detection based on sustained high NIS
        is_anomaly = nis_avg > self.ekf.nis_threshold

        return nis, nis_avg, is_anomaly

    def process_sequence(
        self,
        data: np.ndarray  # [N, 15] - all state data
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process entire sequence.

        Args:
            data: [N, 15] state data

        Returns:
            nis_values: [N] NIS per timestep
            nis_avg: [N] windowed average NIS
            anomaly_flags: [N] boolean anomaly flags
        """
        self.reset()
        n = len(data)

        nis_values = np.zeros(n)
        nis_avg = np.zeros(n)
        anomaly_flags = np.zeros(n, dtype=bool)

        for i in range(n):
            # Extract measurements
            gps_pos = data[i, 0:3]
            gps_vel = data[i, 3:6]
            imu_gyro = data[i, 9:12]  # Angular rates
            imu_accel = data[i, 12:15]  # Acceleration

            nis, avg, is_anomaly = self.process_step(
                imu_gyro, imu_accel, gps_pos, gps_vel
            )

            nis_values[i] = nis
            nis_avg[i] = avg
            anomaly_flags[i] = is_anomaly

        return nis_values, nis_avg, anomaly_flags


if __name__ == "__main__":
    # Test EKF
    dt = 0.005
    n = 1000

    # Generate test data
    t = np.arange(n) * dt
    data = np.zeros((n, 15))

    # Position: circular motion
    data[:, 0] = np.sin(t)
    data[:, 1] = np.cos(t)
    data[:, 2] = 0.1 * t

    # Velocity
    data[:, 3] = np.cos(t)
    data[:, 4] = -np.sin(t)
    data[:, 5] = 0.1

    # Attitude
    data[:, 6:9] = 0.1 * np.random.randn(n, 3)

    # Angular rates
    data[:, 9:12] = 0.05 * np.random.randn(n, 3)

    # Acceleration
    data[:, 12] = -np.sin(t)
    data[:, 13] = -np.cos(t)
    data[:, 14] = 9.81

    # Test detector on clean data
    detector = EKFAnomalyDetector(dt=dt)
    nis_values, nis_avg, anomaly_flags = detector.process_sequence(data)

    print("Clean Data:")
    print(f"  Mean NIS: {np.mean(nis_values):.2f}")
    print(f"  Anomaly rate: {np.mean(anomaly_flags)*100:.1f}%")

    # Test with attack
    attacked_data = data.copy()
    attacked_data[500:, 0] += 0.5  # Position bias

    detector.reset()
    nis_attack, nis_avg_attack, anomaly_attack = detector.process_sequence(attacked_data)

    print("\nWith Bias Attack (t>2.5s):")
    print(f"  Mean NIS: {np.mean(nis_attack):.2f}")
    print(f"  Mean NIS (attack period): {np.mean(nis_attack[500:]):.2f}")
    print(f"  Anomaly rate (attack period): {np.mean(anomaly_attack[500:])*100:.1f}%")
