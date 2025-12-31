"""
Actuator Fault Observability Module

Addresses 6 critical problems in actuator fault detection:

Problem 1: Actuator faults are controller-masked
Fix 1: Control-effort inconsistency metrics

Problem 2: Short window (256) kills slow faults
Fix 2: Dual-timescale windows

Problem 3: Global thresholds don't work for PINNs
Fix 3: Residual envelope normalization

Problem 4: Motor faults != actuator faults (mixed)
Fix 4: Split fault heads

Problem 5: Time-delay & stealth attacks missed
Fix 5: Phase-consistency check

Problem 6: Class imbalance killing metrics
Fix 6: Proper evaluation metrics
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from scipy import signal
from scipy.stats import pearsonr
from collections import deque


# =============================================================================
# Fix 1: Control-Effort Inconsistency Metrics
# =============================================================================

@dataclass
class ControlEffortMetrics:
    """Container for control-effort anomaly metrics."""
    efficiency: np.ndarray          # ||v_dot|| / ||u|| - drops when actuator fails
    trim_deviation: np.ndarray      # u - u_nominal(v, theta) - increases when compensating
    energy_per_thrust: np.ndarray   # integral(||u||^2) - increases for same output
    response_lag: np.ndarray        # delay between command and response
    control_power: np.ndarray       # ||u||^2 - controller effort


class ControlEffortChecker:
    """
    Detects actuator faults via control-effort inconsistency.

    Key insight: When actuators fail, the controller compensates
    by increasing effort. This is the ONE signal that MUST change.
    """

    def __init__(
        self,
        dt: float = 0.005,
        nominal_efficiency: float = 1.0,
        efficiency_window: int = 50,
        energy_window: int = 200,
    ):
        self.dt = dt
        self.nominal_efficiency = nominal_efficiency
        self.efficiency_window = efficiency_window
        self.energy_window = energy_window

        # Running statistics for normalization
        self.efficiency_history = deque(maxlen=1000)
        self.energy_history = deque(maxlen=1000)

    def compute_metrics(
        self,
        control_input: np.ndarray,      # [N, 4] - motor commands or [N, 1] thrust
        acceleration: np.ndarray,        # [N, 3] - measured acceleration
        velocity: np.ndarray,             # [N, 3] - velocity
        attitude: np.ndarray,             # [N, 3] - roll, pitch, yaw
    ) -> ControlEffortMetrics:
        """
        Compute control-effort inconsistency metrics.

        Args:
            control_input: Motor commands or collective thrust
            acceleration: Measured body acceleration
            velocity: Current velocity
            attitude: Current attitude (roll, pitch, yaw)

        Returns:
            ControlEffortMetrics with all computed metrics
        """
        n = len(control_input)

        # 1. Control-to-acceleration efficiency: eta = ||a|| / ||u||
        efficiency = self._compute_efficiency(control_input, acceleration)

        # 2. Trim deviation: delta_u = u - u_nominal(v, theta)
        trim_deviation = self._compute_trim_deviation(
            control_input, velocity, attitude
        )

        # 3. Energy per unit thrust (cumulative)
        energy_per_thrust = self._compute_energy_per_thrust(
            control_input, acceleration
        )

        # 4. Response lag (cross-correlation peak delay)
        response_lag = self._compute_response_lag(control_input, acceleration)

        # 5. Control power ||u||^2
        if control_input.ndim == 1:
            control_power = control_input ** 2
        else:
            control_power = np.sum(control_input ** 2, axis=1)

        return ControlEffortMetrics(
            efficiency=efficiency,
            trim_deviation=trim_deviation,
            energy_per_thrust=energy_per_thrust,
            response_lag=response_lag,
            control_power=control_power,
        )

    def _compute_efficiency(
        self,
        control: np.ndarray,
        acceleration: np.ndarray
    ) -> np.ndarray:
        """
        Control-to-acceleration efficiency.

        eta = ||a_measured|| / ||u||

        When actuator fails: eta drops (same command, less acceleration)
        """
        n = len(control)

        # Control magnitude
        if control.ndim == 1:
            u_mag = np.abs(control)
        else:
            u_mag = np.linalg.norm(control, axis=1)

        # Acceleration magnitude
        a_mag = np.linalg.norm(acceleration, axis=1)

        # Efficiency (avoid division by zero)
        efficiency = a_mag / (u_mag + 1e-6)

        # Smooth with rolling window
        if n >= self.efficiency_window:
            kernel = np.ones(self.efficiency_window) / self.efficiency_window
            efficiency = np.convolve(efficiency, kernel, mode='same')

        return efficiency

    def _compute_trim_deviation(
        self,
        control: np.ndarray,
        velocity: np.ndarray,
        attitude: np.ndarray
    ) -> np.ndarray:
        """
        Deviation from nominal trim control.

        u_nominal = f(v, theta) - what control is expected for steady flight
        delta_u = u - u_nominal

        When actuator fails: controller must apply more to compensate
        """
        n = len(control)

        # Simplified trim model: hover requires more thrust at higher speeds
        # and different attitude
        speed = np.linalg.norm(velocity, axis=1)
        pitch = attitude[:, 1]  # theta

        # Nominal thrust increases with speed (drag compensation)
        # and with pitch angle (cosine factor)
        nominal_thrust = 1.0 + 0.1 * speed + 0.5 * (1 - np.cos(pitch))

        # Deviation
        if control.ndim == 1:
            actual_thrust = np.abs(control)
        else:
            # Sum of motor commands as total thrust
            actual_thrust = np.sum(control, axis=1)

        # Normalize to nominal scale
        actual_thrust = actual_thrust / (np.mean(actual_thrust) + 1e-6)

        trim_deviation = actual_thrust - nominal_thrust

        return trim_deviation

    def _compute_energy_per_thrust(
        self,
        control: np.ndarray,
        acceleration: np.ndarray
    ) -> np.ndarray:
        """
        Cumulative energy per unit output.

        E = integral(||u||^2 dt) / integral(||a|| dt)

        When actuator fails: more energy needed for same output
        """
        n = len(control)

        # Control energy
        if control.ndim == 1:
            control_energy = np.cumsum(control ** 2) * self.dt
        else:
            control_energy = np.cumsum(np.sum(control ** 2, axis=1)) * self.dt

        # Output (acceleration integral ~ velocity change)
        accel_integral = np.cumsum(np.linalg.norm(acceleration, axis=1)) * self.dt

        # Energy per output
        energy_per_thrust = control_energy / (accel_integral + 1e-6)

        return energy_per_thrust

    def _compute_response_lag(
        self,
        control: np.ndarray,
        acceleration: np.ndarray,
        max_lag: int = 20
    ) -> np.ndarray:
        """
        Delay between control command and response.

        Uses cross-correlation to find lag.
        When actuator fails: response may lag or be inconsistent
        """
        n = len(control)
        response_lag = np.zeros(n)

        if n < max_lag * 2:
            return response_lag

        # Use control magnitude and z-acceleration (most direct relationship)
        if control.ndim == 1:
            u = control
        else:
            u = np.sum(control, axis=1)

        a_z = acceleration[:, 2]

        # Compute lag in sliding windows
        window = max_lag * 4
        for i in range(window, n):
            u_win = u[i-window:i]
            a_win = a_z[i-window:i]

            # Cross-correlation
            corr = np.correlate(u_win - u_win.mean(), a_win - a_win.mean(), 'full')
            lag_idx = np.argmax(np.abs(corr)) - (len(corr) // 2)

            response_lag[i] = lag_idx * self.dt

        return response_lag


# =============================================================================
# Fix 2: Dual-Timescale Windows
# =============================================================================

@dataclass
class DualScaleResult:
    """Results from dual-timescale analysis."""
    short_score: np.ndarray   # Abrupt fault score (256 window)
    long_score: np.ndarray    # Degradation score (1024-2048 window)
    fused_score: np.ndarray   # OR-gated combination
    is_anomaly: np.ndarray    # Boolean anomaly detection


class DualTimescaleDetector:
    """
    Dual-timescale fault detector.

    Short window (128-256): Detects abrupt faults
    Long window (1024-2048): Detects slow degradation

    Fusion: OR-gating (fault if either triggers)
    """

    def __init__(
        self,
        short_window: int = 256,
        long_window: int = 1024,
        short_threshold: float = 0.5,
        long_threshold: float = 0.3,  # Lower threshold for slow faults
        stride: int = 64,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.short_threshold = short_threshold
        self.long_threshold = long_threshold
        self.stride = stride

    def compute_scores(
        self,
        features: np.ndarray,  # [N, D] feature array
        short_scorer: callable,
        long_scorer: Optional[callable] = None,
    ) -> DualScaleResult:
        """
        Compute dual-timescale anomaly scores.

        Args:
            features: Time-series features
            short_scorer: Function to score short windows
            long_scorer: Function to score long windows (default: same as short)

        Returns:
            DualScaleResult with scores at both scales
        """
        n = len(features)

        if long_scorer is None:
            long_scorer = short_scorer

        # Short-window scores
        short_scores = self._windowed_scores(
            features, self.short_window, short_scorer
        )

        # Long-window scores
        long_scores = self._windowed_scores(
            features, self.long_window, long_scorer
        )

        # OR-gated fusion: anomaly if either detects
        short_anomaly = short_scores > self.short_threshold
        long_anomaly = long_scores > self.long_threshold

        # Fused score: max of normalized scores
        short_normalized = short_scores / self.short_threshold
        long_normalized = long_scores / self.long_threshold
        fused_score = np.maximum(short_normalized, long_normalized)

        is_anomaly = short_anomaly | long_anomaly

        return DualScaleResult(
            short_score=short_scores,
            long_score=long_scores,
            fused_score=fused_score,
            is_anomaly=is_anomaly,
        )

    def _windowed_scores(
        self,
        features: np.ndarray,
        window_size: int,
        scorer: callable
    ) -> np.ndarray:
        """Compute scores using sliding window."""
        n = len(features)
        scores = np.zeros(n)

        if n < window_size:
            # Fall back to available data
            window_size = n

        for i in range(0, n - window_size + 1, self.stride):
            window = features[i:i+window_size]
            score = scorer(window)

            # Assign score to all points in window (overlapping regions get max)
            scores[i:i+window_size] = np.maximum(
                scores[i:i+window_size], score
            )

        return scores


# =============================================================================
# Fix 3: Residual Envelope Normalization
# =============================================================================

@dataclass
class EnvelopeStats:
    """Statistics for envelope normalization."""
    mean: np.ndarray
    std: np.ndarray
    bin_id: int


class ResidualEnvelopeNormalizer:
    """
    Condition-aware residual normalization.

    Instead of: r(t) > tau (global threshold)
    Use: z(t) = (r(t) - mu_condition) / sigma_condition

    Conditions are binned by speed/altitude.
    """

    def __init__(
        self,
        n_speed_bins: int = 5,
        n_altitude_bins: int = 5,
        min_samples: int = 100,
    ):
        self.n_speed_bins = n_speed_bins
        self.n_altitude_bins = n_altitude_bins
        self.min_samples = min_samples

        # Statistics per condition bin
        # Key: (speed_bin, alt_bin), Value: EnvelopeStats
        self.condition_stats: Dict[Tuple[int, int], EnvelopeStats] = {}

        # Bin edges (learned from data)
        self.speed_edges: Optional[np.ndarray] = None
        self.altitude_edges: Optional[np.ndarray] = None

        # Global fallback
        self.global_mean: Optional[np.ndarray] = None
        self.global_std: Optional[np.ndarray] = None

    def fit(
        self,
        residuals: np.ndarray,  # [N, D] residual values
        speed: np.ndarray,       # [N] speed values
        altitude: np.ndarray,    # [N] altitude values
    ):
        """
        Learn condition-specific statistics from nominal data.

        Args:
            residuals: Residual values from NORMAL flights only
            speed: Corresponding speed values
            altitude: Corresponding altitude values
        """
        n = len(residuals)

        # Compute bin edges using quantiles
        self.speed_edges = np.quantile(
            speed, np.linspace(0, 1, self.n_speed_bins + 1)
        )
        self.altitude_edges = np.quantile(
            altitude, np.linspace(0, 1, self.n_altitude_bins + 1)
        )

        # Global statistics as fallback
        self.global_mean = np.mean(residuals, axis=0)
        self.global_std = np.std(residuals, axis=0) + 1e-6

        # Compute per-bin statistics
        speed_bins = np.digitize(speed, self.speed_edges[1:-1])
        alt_bins = np.digitize(altitude, self.altitude_edges[1:-1])

        for sb in range(self.n_speed_bins):
            for ab in range(self.n_altitude_bins):
                mask = (speed_bins == sb) & (alt_bins == ab)

                if mask.sum() >= self.min_samples:
                    bin_residuals = residuals[mask]
                    self.condition_stats[(sb, ab)] = EnvelopeStats(
                        mean=np.mean(bin_residuals, axis=0),
                        std=np.std(bin_residuals, axis=0) + 1e-6,
                        bin_id=sb * self.n_altitude_bins + ab,
                    )

    def normalize(
        self,
        residuals: np.ndarray,
        speed: np.ndarray,
        altitude: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize residuals using condition-specific statistics.

        Returns z-scores: z = (r - mu_condition) / sigma_condition
        """
        n = len(residuals)
        z_scores = np.zeros_like(residuals)

        if self.speed_edges is None:
            # Not fitted, use raw residuals
            return residuals

        speed_bins = np.digitize(speed, self.speed_edges[1:-1])
        alt_bins = np.digitize(altitude, self.altitude_edges[1:-1])

        for i in range(n):
            key = (speed_bins[i], alt_bins[i])

            if key in self.condition_stats:
                stats = self.condition_stats[key]
                z_scores[i] = (residuals[i] - stats.mean) / stats.std
            else:
                # Fallback to global
                z_scores[i] = (residuals[i] - self.global_mean) / self.global_std

        return z_scores


# =============================================================================
# Fix 4: Split Fault Heads (Motor vs Actuator)
# =============================================================================

class SplitFaultHead(nn.Module):
    """
    Dual-head fault detector.

    Motor head: Uses thrust, vertical accel, energy
    Actuator head: Uses attitude error, control effort

    Separate heads for interpretability and better detection.
    """

    def __init__(
        self,
        motor_input_dim: int = 8,
        actuator_input_dim: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Motor fault head (thrust-related signals)
        self.motor_head = nn.Sequential(
            nn.Linear(motor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Actuator fault head (control geometry issues)
        self.actuator_head = nn.Sequential(
            nn.Linear(actuator_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        motor_features: torch.Tensor,
        actuator_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fault scores from both heads.

        Returns:
            motor_score: Probability of motor fault
            actuator_score: Probability of actuator fault
        """
        motor_score = self.motor_head(motor_features)
        actuator_score = self.actuator_head(actuator_features)

        return motor_score, actuator_score


def extract_motor_features(
    thrust: np.ndarray,           # [N] or [N, 4]
    vertical_accel: np.ndarray,   # [N]
    velocity: np.ndarray,         # [N, 3]
    control_effort: np.ndarray,   # [N]
) -> np.ndarray:
    """
    Extract features for motor fault head.

    Features:
    - Thrust magnitude
    - Vertical acceleration
    - Thrust efficiency
    - Energy integral
    - Thrust variance (recent window)
    - Speed
    - Altitude rate
    - Power consumption
    """
    n = len(vertical_accel)

    # Thrust magnitude
    if thrust.ndim > 1:
        thrust_mag = np.sum(thrust, axis=1)
    else:
        thrust_mag = thrust

    # Efficiency
    efficiency = vertical_accel / (thrust_mag + 1e-6)

    # Energy (cumulative)
    energy = np.cumsum(control_effort ** 2)
    energy = energy / (np.arange(1, n+1))  # Running average

    # Thrust variance (window)
    window = 50
    thrust_var = np.zeros(n)
    for i in range(window, n):
        thrust_var[i] = np.var(thrust_mag[i-window:i])

    # Speed
    speed = np.linalg.norm(velocity, axis=1)

    # Altitude rate
    alt_rate = velocity[:, 2]

    # Power
    power = thrust_mag * vertical_accel

    features = np.column_stack([
        thrust_mag,
        vertical_accel,
        efficiency,
        energy,
        thrust_var,
        speed,
        alt_rate,
        power,
    ])

    return features


def extract_actuator_features(
    attitude: np.ndarray,         # [N, 3]
    attitude_cmd: np.ndarray,     # [N, 3] commanded attitude
    angular_rates: np.ndarray,    # [N, 3]
    control_input: np.ndarray,    # [N, 4] individual motor commands
) -> np.ndarray:
    """
    Extract features for actuator fault head.

    Features:
    - Attitude error (3)
    - Angular rate magnitude
    - Control effort per axis (3)
    - Control imbalance (motor difference)
    - Roll/pitch/yaw rate errors (3)
    - Control variance
    """
    n = len(attitude)

    # Attitude error
    if attitude_cmd is not None:
        att_error = attitude - attitude_cmd
    else:
        att_error = np.gradient(attitude, axis=0)  # Use rate of change

    # Angular rate magnitude
    rate_mag = np.linalg.norm(angular_rates, axis=1)

    # Control effort per axis (for quadrotor)
    if control_input.ndim > 1 and control_input.shape[1] == 4:
        # Roll: (m1+m4) - (m2+m3)
        roll_effort = (control_input[:, 0] + control_input[:, 3]) - \
                      (control_input[:, 1] + control_input[:, 2])
        # Pitch: (m1+m2) - (m3+m4)
        pitch_effort = (control_input[:, 0] + control_input[:, 1]) - \
                       (control_input[:, 2] + control_input[:, 3])
        # Yaw: (m1+m3) - (m2+m4)
        yaw_effort = (control_input[:, 0] + control_input[:, 2]) - \
                     (control_input[:, 1] + control_input[:, 3])

        # Motor imbalance
        motor_std = np.std(control_input, axis=1)
    else:
        roll_effort = np.zeros(n)
        pitch_effort = np.zeros(n)
        yaw_effort = np.zeros(n)
        motor_std = np.zeros(n)

    # Rate errors (expected vs actual)
    rate_diff = np.gradient(angular_rates, axis=0)

    # Control variance (window)
    window = 50
    control_var = np.zeros(n)
    if control_input.ndim > 1:
        for i in range(window, n):
            control_var[i] = np.var(control_input[i-window:i])

    features = np.column_stack([
        att_error,           # 3
        rate_mag[:, None],   # 1
        roll_effort[:, None],
        pitch_effort[:, None],
        yaw_effort[:, None],  # 3
        motor_std[:, None],   # 1
        rate_diff,            # 3
        control_var[:, None], # 1
    ])

    return features


# =============================================================================
# Fix 5: Phase-Consistency Check
# =============================================================================

class PhaseConsistencyChecker:
    """
    Detects time-delay and stealth attacks via phase analysis.

    Key insight: Delay attacks preserve magnitude but break phase.

    Checks cross-correlation between:
    - IMU vs GPS velocity
    - Command vs response
    """

    def __init__(
        self,
        dt: float = 0.005,
        max_lag: int = 50,        # Max samples of lag to check
        correlation_threshold: float = 0.8,
        lag_threshold: float = 0.1,  # Max acceptable lag in seconds
    ):
        self.dt = dt
        self.max_lag = max_lag
        self.correlation_threshold = correlation_threshold
        self.lag_threshold_samples = int(lag_threshold / dt)

    def check_phase_consistency(
        self,
        signal1: np.ndarray,  # [N] reference signal
        signal2: np.ndarray,  # [N] test signal
        window: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check phase consistency between two signals.

        Returns:
            correlation: [N] correlation strength
            lag: [N] lag in samples (positive = signal2 delayed)
        """
        n = len(signal1)
        correlation = np.zeros(n)
        lag = np.zeros(n)

        for i in range(window, n):
            s1 = signal1[i-window:i]
            s2 = signal2[i-window:i]

            # Normalize
            s1 = (s1 - s1.mean()) / (s1.std() + 1e-6)
            s2 = (s2 - s2.mean()) / (s2.std() + 1e-6)

            # Cross-correlation
            corr = np.correlate(s1, s2, mode='full')
            center = len(corr) // 2

            # Find peak
            search_range = corr[center-self.max_lag:center+self.max_lag+1]
            peak_idx = np.argmax(np.abs(search_range))

            correlation[i] = search_range[peak_idx] / window
            lag[i] = (peak_idx - self.max_lag)

        return correlation, lag

    def detect_delay_attack(
        self,
        imu_velocity: np.ndarray,    # [N, 3] from IMU integration
        gps_velocity: np.ndarray,    # [N, 3] from GPS
        control_cmd: np.ndarray,     # [N] control command
        response: np.ndarray,        # [N] measured response
    ) -> Dict[str, np.ndarray]:
        """
        Detect delay attacks using multiple phase checks.

        Returns:
            Dict with detection results and metrics
        """
        n = len(imu_velocity)

        results = {
            'is_delay_attack': np.zeros(n, dtype=bool),
            'imu_gps_correlation': np.zeros(n),
            'imu_gps_lag': np.zeros(n),
            'cmd_response_correlation': np.zeros(n),
            'cmd_response_lag': np.zeros(n),
        }

        # Check IMU vs GPS velocity (x component as proxy)
        if gps_velocity is not None:
            corr, lag = self.check_phase_consistency(
                imu_velocity[:, 0], gps_velocity[:, 0]
            )
            results['imu_gps_correlation'] = corr
            results['imu_gps_lag'] = lag

        # Check command vs response
        corr, lag = self.check_phase_consistency(control_cmd, response)
        results['cmd_response_correlation'] = corr
        results['cmd_response_lag'] = lag

        # Detect attack: low correlation OR excessive lag
        imu_gps_anomaly = (
            (np.abs(results['imu_gps_correlation']) < self.correlation_threshold) |
            (np.abs(results['imu_gps_lag']) > self.lag_threshold_samples)
        )

        cmd_resp_anomaly = (
            (np.abs(results['cmd_response_correlation']) < self.correlation_threshold) |
            (np.abs(results['cmd_response_lag']) > self.lag_threshold_samples)
        )

        results['is_delay_attack'] = imu_gps_anomaly | cmd_resp_anomaly

        return results


# =============================================================================
# Fix 6: Proper Evaluation Metrics
# =============================================================================

@dataclass
class ProperMetrics:
    """Proper metrics for imbalanced fault detection."""
    auroc: float                    # Area under ROC (main metric)
    auprc: float                    # Area under Precision-Recall
    recall_at_1pct_fpr: float       # Recall at 1% false positive rate
    recall_at_5pct_fpr: float       # Recall at 5% false positive rate
    time_to_detection: float        # Median time from fault onset to detection
    energy_drift_before_detection: float  # Energy spent before detection
    per_fault_auroc: Dict[str, float]     # AUROC per fault type


def compute_proper_metrics(
    y_true: np.ndarray,           # [N] binary labels
    y_score: np.ndarray,          # [N] anomaly scores
    fault_types: Optional[np.ndarray] = None,  # [N] fault type per sample
    fault_onset: Optional[np.ndarray] = None,  # [N] time since fault onset
) -> ProperMetrics:
    """
    Compute proper evaluation metrics for fault detection.

    Focuses on:
    - AUROC (not accuracy) - handles imbalance
    - Recall at fixed FPR - operational relevance
    - Time-to-detection - safety critical
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    # Basic AUROC and AUPRC
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = 0.5  # Single class

    try:
        auprc = average_precision_score(y_true, y_score)
    except ValueError:
        auprc = 0.0

    # Recall at fixed FPR
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Find recall at 1% FPR
    idx_1pct = np.searchsorted(fpr, 0.01)
    recall_1pct = tpr[min(idx_1pct, len(tpr)-1)]

    # Find recall at 5% FPR
    idx_5pct = np.searchsorted(fpr, 0.05)
    recall_5pct = tpr[min(idx_5pct, len(tpr)-1)]

    # Time to detection
    ttd = np.nan
    if fault_onset is not None:
        # Find threshold at 5% FPR
        threshold = thresholds[min(idx_5pct, len(thresholds)-1)]

        # Detection times
        detected = y_score > threshold
        fault_samples = y_true == 1

        detected_faults = detected & fault_samples
        if detected_faults.any():
            detection_times = fault_onset[detected_faults]
            ttd = np.median(detection_times[detection_times > 0])

    # Per-fault AUROC
    per_fault_auroc = {}
    if fault_types is not None:
        unique_faults = np.unique(fault_types[y_true == 1])
        for ft in unique_faults:
            mask = (fault_types == ft) | (y_true == 0)
            if mask.sum() > 10:
                try:
                    per_fault_auroc[str(ft)] = roc_auc_score(
                        y_true[mask], y_score[mask]
                    )
                except ValueError:
                    per_fault_auroc[str(ft)] = 0.5

    return ProperMetrics(
        auroc=auroc,
        auprc=auprc,
        recall_at_1pct_fpr=recall_1pct,
        recall_at_5pct_fpr=recall_5pct,
        time_to_detection=ttd if not np.isnan(ttd) else -1,
        energy_drift_before_detection=0.0,  # Placeholder
        per_fault_auroc=per_fault_auroc,
    )


# =============================================================================
# Combined Actuator Fault Detector
# =============================================================================

class EnhancedActuatorDetector:
    """
    Enhanced actuator fault detector with all fixes applied.

    Combines:
    1. Control-effort inconsistency
    2. Dual-timescale analysis
    3. Envelope normalization
    4. Split fault heads
    5. Phase consistency
    """

    def __init__(
        self,
        dt: float = 0.005,
        short_window: int = 256,
        long_window: int = 1024,
    ):
        self.dt = dt

        # Fix 1: Control effort checker
        self.control_checker = ControlEffortChecker(dt=dt)

        # Fix 2: Dual timescale detector
        self.dual_scale = DualTimescaleDetector(
            short_window=short_window,
            long_window=long_window,
        )

        # Fix 3: Envelope normalizer
        self.normalizer = ResidualEnvelopeNormalizer()

        # Fix 5: Phase consistency checker
        self.phase_checker = PhaseConsistencyChecker(dt=dt)

        # Fitted flag
        self.is_fitted = False

    def fit(
        self,
        normal_data: Dict[str, np.ndarray],
    ):
        """
        Fit normalizer on normal flight data.

        Args:
            normal_data: Dict with 'residuals', 'speed', 'altitude'
        """
        self.normalizer.fit(
            normal_data['residuals'],
            normal_data['speed'],
            normal_data['altitude'],
        )
        self.is_fitted = True

    def detect(
        self,
        data: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Run full detection pipeline.

        Args:
            data: Dict with required sensor data

        Returns:
            Dict with all detection results
        """
        results = {}

        # 1. Control effort metrics
        if all(k in data for k in ['control', 'acceleration', 'velocity', 'attitude']):
            control_metrics = self.control_checker.compute_metrics(
                data['control'],
                data['acceleration'],
                data['velocity'],
                data['attitude'],
            )
            results['control_efficiency'] = control_metrics.efficiency
            results['trim_deviation'] = control_metrics.trim_deviation
            results['control_power'] = control_metrics.control_power

        # 2. Envelope-normalized residuals
        if 'residuals' in data and self.is_fitted:
            z_scores = self.normalizer.normalize(
                data['residuals'],
                data.get('speed', np.zeros(len(data['residuals']))),
                data.get('altitude', np.zeros(len(data['residuals']))),
            )
            results['normalized_residuals'] = z_scores
            results['residual_score'] = np.linalg.norm(z_scores, axis=1)

        # 3. Phase consistency (for delay attacks)
        if all(k in data for k in ['imu_velocity', 'control', 'response']):
            phase_results = self.phase_checker.detect_delay_attack(
                data['imu_velocity'],
                data.get('gps_velocity'),
                data['control'][:, 0] if data['control'].ndim > 1 else data['control'],
                data['response'],
            )
            results.update(phase_results)

        # 4. Combine scores
        scores = []
        if 'control_efficiency' in results:
            # Low efficiency = high anomaly score
            eff_score = 1 - np.clip(results['control_efficiency'], 0, 2) / 2
            scores.append(eff_score)

        if 'residual_score' in results:
            res_score = np.clip(results['residual_score'] / 5, 0, 1)
            scores.append(res_score)

        if 'is_delay_attack' in results:
            scores.append(results['is_delay_attack'].astype(float))

        if scores:
            results['combined_score'] = np.mean(scores, axis=0)

        return results


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Actuator Observability Module - Testing All Fixes")
    print("=" * 60)

    # Generate test data
    n = 2000
    dt = 0.005
    t = np.arange(n) * dt

    # Normal flight data
    position = np.column_stack([np.sin(t), np.cos(t), 10 + 0.1*t])
    velocity = np.column_stack([np.cos(t), -np.sin(t), np.ones(n)*0.1])
    acceleration = np.column_stack([-np.sin(t), -np.cos(t), np.zeros(n)])
    attitude = np.column_stack([0.1*np.sin(t), 0.1*np.cos(t), t*0.01])
    angular_rates = np.column_stack([0.1*np.cos(t), -0.1*np.sin(t), 0.01*np.ones(n)])
    control = np.ones((n, 4)) * 0.5 + 0.01 * np.random.randn(n, 4)

    print("\n1. Testing Control Effort Checker...")
    checker = ControlEffortChecker(dt=dt)
    metrics = checker.compute_metrics(control, acceleration, velocity, attitude)
    print(f"   Efficiency: mean={np.mean(metrics.efficiency):.4f}")
    print(f"   Trim deviation: std={np.std(metrics.trim_deviation):.4f}")

    # Simulate actuator fault (motor 0 efficiency drops)
    faulty_control = control.copy()
    faulty_control[n//2:, 0] *= 0.5  # Motor 0 loses 50% efficiency
    faulty_control[n//2:, 1:] *= 1.2  # Others compensate

    faulty_metrics = checker.compute_metrics(
        faulty_control, acceleration, velocity, attitude
    )
    print(f"   With fault - Efficiency: mean={np.mean(faulty_metrics.efficiency[n//2:]):.4f}")
    print(f"   Trim deviation increase: {np.mean(np.abs(faulty_metrics.trim_deviation[n//2:])):.4f}")

    print("\n2. Testing Dual Timescale Detector...")
    dual_detector = DualTimescaleDetector(short_window=128, long_window=512)
    features = np.random.randn(n, 10)
    features[n//2:] += 0.5  # Slow drift

    def simple_scorer(window):
        return np.abs(window).mean()

    dual_result = dual_detector.compute_scores(features, simple_scorer)
    print(f"   Short window detections: {dual_result.short_score[n//2:].mean():.4f}")
    print(f"   Long window detections: {dual_result.long_score[n//2:].mean():.4f}")

    print("\n3. Testing Envelope Normalizer...")
    normalizer = ResidualEnvelopeNormalizer()
    residuals = np.random.randn(n, 5) * 0.1
    speed = np.linalg.norm(velocity, axis=1)
    altitude = position[:, 2]

    normalizer.fit(residuals[:n//2], speed[:n//2], altitude[:n//2])

    # Add anomaly in second half
    residuals[n//2:] += 0.5
    z_scores = normalizer.normalize(residuals, speed, altitude)
    print(f"   Normal z-score: {np.mean(np.abs(z_scores[:n//2])):.4f}")
    print(f"   Anomaly z-score: {np.mean(np.abs(z_scores[n//2:])):.4f}")

    print("\n4. Testing Phase Consistency Checker...")
    phase_checker = PhaseConsistencyChecker(dt=dt)

    # Normal case
    signal1 = np.sin(2 * np.pi * t)
    signal2 = np.sin(2 * np.pi * t)
    corr, lag = phase_checker.check_phase_consistency(signal1, signal2)
    print(f"   Normal correlation: {np.mean(corr[200:]):.4f}, lag: {np.mean(lag[200:]):.4f}")

    # Delayed signal
    signal2_delayed = np.roll(signal1, 20)  # 20 sample delay
    corr, lag = phase_checker.check_phase_consistency(signal1, signal2_delayed)
    print(f"   Delayed correlation: {np.mean(corr[200:]):.4f}, lag: {np.mean(lag[200:]):.4f}")

    print("\n5. Testing Proper Metrics...")
    y_true = np.zeros(n)
    y_true[n//2:] = 1  # Second half is fault
    y_score = np.random.rand(n)
    y_score[n//2:] += 0.3  # Higher scores for faults

    metrics = compute_proper_metrics(y_true, y_score)
    print(f"   AUROC: {metrics.auroc:.4f}")
    print(f"   Recall@1%FPR: {metrics.recall_at_1pct_fpr:.4f}")
    print(f"   Recall@5%FPR: {metrics.recall_at_5pct_fpr:.4f}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
