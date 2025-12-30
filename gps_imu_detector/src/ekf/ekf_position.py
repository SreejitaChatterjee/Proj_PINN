"""
Extended Kalman Filter for Position Tracking with NIS-based Anomaly Detection.

This module implements a minimal 6-state EKF (position + velocity) that uses
GPS measurements for updates. The Normalized Innovation Squared (NIS) metric
serves as the anomaly detection score.

NIS Definition:
    NIS_t = r_t^T @ S_t^{-1} @ r_t

    where:
    - r_t = z_t - H @ x_t^-  (innovation/residual)
    - S_t = H @ P_t^- @ H^T + R  (innovation covariance)

Under nominal conditions, NIS follows a chi-squared distribution with
degrees of freedom equal to measurement dimension.

For GPS spoofing detection:
- High NIS indicates measurement inconsistent with dynamics model
- Normalized NIS (z-score) provides a calibrated anomaly score

Author: GPS-IMU Detector Project
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class EKFState:
    """EKF state container."""
    x: np.ndarray  # State estimate [x, y, z, vx, vy, vz]
    P: np.ndarray  # Covariance matrix (6x6)

    def copy(self) -> 'EKFState':
        return EKFState(x=self.x.copy(), P=self.P.copy())


class EKFPositionTracker:
    """
    Extended Kalman Filter for 3D position tracking.

    State: [x, y, z, vx, vy, vz] (6-dimensional)
    Model: Constant velocity with process noise
    Measurement: GPS position [x, y, z] (3-dimensional)

    Attributes:
        dt: Time step (seconds)
        process_noise_pos: Position process noise variance
        process_noise_vel: Velocity process noise variance
        measurement_noise: GPS measurement noise variance
    """

    def __init__(
        self,
        dt: float = 0.01,  # 100 Hz default
        process_noise_pos: float = 0.01,
        process_noise_vel: float = 0.1,
        measurement_noise: float = 1.0,
    ):
        """
        Initialize EKF.

        Args:
            dt: Time step in seconds
            process_noise_pos: Process noise for position (m^2)
            process_noise_vel: Process noise for velocity (m^2/s^2)
            measurement_noise: GPS measurement noise variance (m^2)
        """
        self.dt = dt
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.measurement_noise = measurement_noise

        # State transition matrix (constant velocity model)
        # x_{k+1} = F @ x_k
        self.F = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ],
        ])

        # Measurement matrix (observe position only)
        # z_k = H @ x_k
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        # Process noise covariance
        # Discrete-time approximation for constant velocity model
        self.Q = self._compute_process_noise()

        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise

        # Current state
        self._state: Optional[EKFState] = None

    def _compute_process_noise(self) -> np.ndarray:
        """Compute process noise covariance matrix."""
        dt = self.dt
        q_pos = self.process_noise_pos
        q_vel = self.process_noise_vel

        # Standard discrete-time process noise for constant velocity
        Q = np.zeros((6, 6))

        # Position variance: q_pos * dt + q_vel * dt^3/3
        pos_var = q_pos * dt + q_vel * (dt**3) / 3
        # Position-velocity covariance: q_vel * dt^2/2
        pos_vel_cov = q_vel * (dt**2) / 2
        # Velocity variance: q_vel * dt
        vel_var = q_vel * dt

        # Fill matrix (block diagonal for x, y, z)
        for i in range(3):
            Q[i, i] = pos_var
            Q[i, i+3] = pos_vel_cov
            Q[i+3, i] = pos_vel_cov
            Q[i+3, i+3] = vel_var

        return Q

    def initialize(
        self,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        pos_uncertainty: float = 1.0,
        vel_uncertainty: float = 1.0,
    ) -> None:
        """
        Initialize filter state.

        Args:
            position: Initial position [x, y, z]
            velocity: Initial velocity [vx, vy, vz] (default: zeros)
            pos_uncertainty: Initial position uncertainty (m)
            vel_uncertainty: Initial velocity uncertainty (m/s)
        """
        if velocity is None:
            velocity = np.zeros(3)

        x = np.concatenate([position, velocity])
        P = np.diag([
            pos_uncertainty**2, pos_uncertainty**2, pos_uncertainty**2,
            vel_uncertainty**2, vel_uncertainty**2, vel_uncertainty**2,
        ])

        self._state = EKFState(x=x, P=P)

    def predict(self) -> EKFState:
        """
        Prediction step (time update).

        Returns:
            Predicted state (before measurement update)
        """
        if self._state is None:
            raise RuntimeError("Filter not initialized. Call initialize() first.")

        # State prediction: x_k^- = F @ x_{k-1}
        x_pred = self.F @ self._state.x

        # Covariance prediction: P_k^- = F @ P_{k-1} @ F^T + Q
        P_pred = self.F @ self._state.P @ self.F.T + self.Q

        self._state = EKFState(x=x_pred, P=P_pred)
        return self._state.copy()

    def update(self, gps_position: np.ndarray) -> Tuple[EKFState, float]:
        """
        Measurement update step with NIS computation.

        Args:
            gps_position: GPS measurement [x, y, z]

        Returns:
            (updated_state, NIS): Updated state and Normalized Innovation Squared
        """
        if self._state is None:
            raise RuntimeError("Filter not initialized. Call initialize() first.")

        z = gps_position

        # Innovation (measurement residual)
        # r = z - H @ x^-
        z_pred = self.H @ self._state.x
        r = z - z_pred

        # Innovation covariance
        # S = H @ P^- @ H^T + R
        S = self.H @ self._state.P @ self.H.T + self.R

        # Normalized Innovation Squared (NIS)
        # NIS = r^T @ S^{-1} @ r
        try:
            S_inv = np.linalg.inv(S)
            NIS = float(r.T @ S_inv @ r)
        except np.linalg.LinAlgError:
            # Fallback if S is singular
            NIS = float(np.sum(r**2) / self.measurement_noise)
            S_inv = np.eye(3) / self.measurement_noise

        # Kalman gain
        # K = P^- @ H^T @ S^{-1}
        K = self._state.P @ self.H.T @ S_inv

        # State update
        # x^+ = x^- + K @ r
        x_upd = self._state.x + K @ r

        # Covariance update (Joseph form for numerical stability)
        # P^+ = (I - K @ H) @ P^- @ (I - K @ H)^T + K @ R @ K^T
        I_KH = np.eye(6) - K @ self.H
        P_upd = I_KH @ self._state.P @ I_KH.T + K @ self.R @ K.T

        self._state = EKFState(x=x_upd, P=P_upd)
        return self._state.copy(), NIS

    def step(self, gps_position: np.ndarray) -> Tuple[EKFState, float]:
        """
        Combined predict + update step.

        Args:
            gps_position: GPS measurement [x, y, z]

        Returns:
            (state, NIS): Updated state and NIS score
        """
        self.predict()
        return self.update(gps_position)

    def get_state(self) -> Optional[EKFState]:
        """Get current state estimate."""
        return self._state.copy() if self._state else None

    def get_position(self) -> Optional[np.ndarray]:
        """Get current position estimate."""
        return self._state.x[:3].copy() if self._state else None

    def get_velocity(self) -> Optional[np.ndarray]:
        """Get current velocity estimate."""
        return self._state.x[3:].copy() if self._state else None


class NISAnomalyDetector:
    """
    NIS-based anomaly detector for GPS spoofing.

    Uses EKF's Normalized Innovation Squared as the anomaly score.
    NIS is normalized to a z-score based on training statistics.

    Detection Logic:
        score = (NIS - mean_NIS) / std_NIS
        anomaly if score > threshold

    Attributes:
        ekf: Underlying EKF tracker
        mean_nis: Mean NIS from training (calibration)
        std_nis: Std NIS from training (calibration)
        threshold: Detection threshold (z-score)
    """

    def __init__(
        self,
        dt: float = 0.01,
        process_noise_pos: float = 0.01,
        process_noise_vel: float = 0.1,
        measurement_noise: float = 1.0,
    ):
        """
        Initialize NIS anomaly detector.

        Args:
            dt: EKF time step
            process_noise_pos: Position process noise
            process_noise_vel: Velocity process noise
            measurement_noise: GPS measurement noise
        """
        self.ekf = EKFPositionTracker(
            dt=dt,
            process_noise_pos=process_noise_pos,
            process_noise_vel=process_noise_vel,
            measurement_noise=measurement_noise,
        )

        # Calibration statistics (set during calibrate())
        self.mean_nis: float = 3.0  # Chi-squared(3) mean
        self.std_nis: float = 2.45  # Chi-squared(3) std ~ sqrt(2*3)
        self.threshold: Optional[float] = None

        # Detection history
        self._nis_history: List[float] = []
        self._score_history: List[float] = []

    def reset(self) -> None:
        """Reset detector state (keeps calibration)."""
        self._nis_history = []
        self._score_history = []

    def initialize(self, position: np.ndarray, velocity: Optional[np.ndarray] = None) -> None:
        """Initialize EKF with starting state."""
        self.ekf.initialize(position, velocity)
        self.reset()

    def step(self, gps_position: np.ndarray) -> Tuple[float, float, bool]:
        """
        Process one GPS measurement and return anomaly score.

        Args:
            gps_position: GPS measurement [x, y, z]

        Returns:
            (nis, score, is_anomaly): Raw NIS, normalized score, anomaly flag
        """
        _, nis = self.ekf.step(gps_position)

        # Normalize to z-score
        score = (nis - self.mean_nis) / max(self.std_nis, 1e-6)

        # Store history
        self._nis_history.append(nis)
        self._score_history.append(score)

        # Anomaly detection
        is_anomaly = score > self.threshold if self.threshold else False

        return nis, score, is_anomaly

    def process_trajectory(
        self,
        positions: np.ndarray,
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """
        Process entire trajectory and compute detection metrics.

        Args:
            positions: GPS positions (T, 3)
            return_all: If True, return all NIS/score values

        Returns:
            Dictionary with:
                - nis_values: Raw NIS array
                - scores: Normalized scores
                - max_score: Maximum anomaly score
                - mean_score: Mean anomaly score
        """
        # Initialize with first position
        self.initialize(positions[0])

        nis_values = []
        scores = []

        for pos in positions[1:]:
            nis, score, _ = self.step(pos)
            nis_values.append(nis)
            scores.append(score)

        nis_values = np.array(nis_values)
        scores = np.array(scores)

        result = {
            'max_score': float(np.max(scores)) if len(scores) > 0 else 0.0,
            'mean_score': float(np.mean(scores)) if len(scores) > 0 else 0.0,
            'max_nis': float(np.max(nis_values)) if len(nis_values) > 0 else 0.0,
            'mean_nis': float(np.mean(nis_values)) if len(nis_values) > 0 else 0.0,
        }

        if return_all:
            result['nis_values'] = nis_values
            result['scores'] = scores

        return result

    def calibrate(
        self,
        normal_trajectories: List[np.ndarray],
        target_fpr: float = 0.01,
    ) -> Dict[str, float]:
        """
        Calibrate detector on normal (clean) data.

        Sets mean_nis, std_nis, and threshold.

        Args:
            normal_trajectories: List of clean GPS trajectories
            target_fpr: Target false positive rate for threshold

        Returns:
            Calibration statistics
        """
        all_nis = []

        for traj in normal_trajectories:
            if len(traj) < 2:
                continue

            # Initialize and process
            self.initialize(traj[0])

            for pos in traj[1:]:
                nis, _, _ = self.step(pos)
                all_nis.append(nis)

        all_nis = np.array(all_nis)

        # Compute statistics
        self.mean_nis = float(np.mean(all_nis))
        self.std_nis = float(np.std(all_nis))

        # Set threshold for target FPR
        # Threshold is the (1 - target_fpr) percentile of normalized scores
        normalized = (all_nis - self.mean_nis) / max(self.std_nis, 1e-6)
        self.threshold = float(np.percentile(normalized, 100 * (1 - target_fpr)))

        return {
            'mean_nis': self.mean_nis,
            'std_nis': self.std_nis,
            'threshold': self.threshold,
            'n_samples': len(all_nis),
        }

    def get_scores(self) -> np.ndarray:
        """Get all normalized anomaly scores from current trajectory."""
        return np.array(self._score_history)

    def get_nis_values(self) -> np.ndarray:
        """Get all raw NIS values from current trajectory."""
        return np.array(self._nis_history)


def evaluate_ekf_detector(
    detector: NISAnomalyDetector,
    normal_trajectories: List[np.ndarray],
    attack_trajectories: List[np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate EKF-NIS detector on normal and attack data.

    Args:
        detector: Calibrated NISAnomalyDetector
        normal_trajectories: List of clean trajectories
        attack_trajectories: List of spoofed trajectories

    Returns:
        Dictionary with AUROC, recall@FPR metrics
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    # Collect scores
    normal_scores = []
    attack_scores = []

    for traj in normal_trajectories:
        if len(traj) < 2:
            continue
        result = detector.process_trajectory(traj, return_all=True)
        normal_scores.extend(result['scores'].tolist())

    for traj in attack_trajectories:
        if len(traj) < 2:
            continue
        result = detector.process_trajectory(traj, return_all=True)
        attack_scores.extend(result['scores'].tolist())

    # Create labels
    y_true = np.array([0] * len(normal_scores) + [1] * len(attack_scores))
    y_scores = np.array(normal_scores + attack_scores)

    # Compute metrics
    auroc = roc_auc_score(y_true, y_scores)

    # Recall at fixed FPR
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    def recall_at_fpr(target_fpr):
        idx = np.searchsorted(fpr, target_fpr)
        if idx >= len(tpr):
            return tpr[-1]
        return tpr[idx]

    return {
        'auroc': float(auroc),
        'recall_1pct_fpr': float(recall_at_fpr(0.01)),
        'recall_5pct_fpr': float(recall_at_fpr(0.05)),
        'n_normal': len(normal_scores),
        'n_attack': len(attack_scores),
    }
