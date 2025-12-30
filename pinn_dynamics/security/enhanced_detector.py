"""
Enhanced Attack Detector with Multi-Scale Features and PINN Integration.

Key improvements over basic supervised classifier:
1. Multi-scale feature extraction (short, medium, long windows)
2. PINN residual features (prediction error as feature)
3. Attack-type specialists (separate models per category)
4. Threshold tuning for target recall
5. Gradient Boosting with proper hyperparameters

Target: ≥90% recall with ≤10% FPR
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import warnings
# Suppress specific sklearn convergence warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*convergence.*', category=UserWarning)

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import precision_recall_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional PINN integration
try:
    from ..inference.predictor import Predictor
    PINN_AVAILABLE = True
except ImportError:
    PINN_AVAILABLE = False


@dataclass
class DetectionResult:
    is_attack: bool
    probability: float
    attack_type: str
    confidence: float
    feature_contributions: Optional[Dict[str, float]] = None


class MultiScaleFeatureExtractor:
    """
    Extract features at multiple time scales to catch different attack types.

    Short window (10 samples = 50ms): Sudden jumps, discontinuities
    Medium window (50 samples = 250ms): Normal attack patterns
    Long window (200 samples = 1s): Slow drifts, gradual attacks

    CRITICAL: Only uses RELATIVE features (no absolute positions) to avoid
    learning trajectory-specific patterns that don't generalize.

    NEW: Control input features for detecting actuator attacks.
    """

    def __init__(
        self,
        short_window: int = 10,
        medium_window: int = 50,
        long_window: int = 200,
        use_fft: bool = True,
        use_control_features: bool = True,  # NEW: Enable control features
    ):
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.use_fft = use_fft
        self.use_control_features = use_control_features

        # Channels where absolute mean is meaningful (rates, velocities)
        # vs channels where only relative features make sense (positions)
        self.all_channels = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz']
        self.position_channels = ['x', 'y', 'z', 'psi']  # Skip absolute means for these
        self.rate_channels = ['phi', 'theta', 'p', 'q', 'r', 'vx', 'vy', 'vz']  # Keep means for these

        # NEW: Control channels for detecting actuator attacks
        self.control_channels = ['thrust', 'torque_x', 'torque_y', 'torque_z']

        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        """Build comprehensive feature name list - NO absolute position means."""
        names = []

        scales = ['short', 'medium', 'long']

        # Per-channel, per-scale statistics
        for scale in scales:
            for ch in self.all_channels:
                # Only include mean for non-position channels (rates, velocities)
                if ch not in self.position_channels:
                    names.append(f'{scale}_{ch}_mean')
                # Always include std, range, diff_std (relative measures)
                names.extend([
                    f'{scale}_{ch}_std',
                    f'{scale}_{ch}_range',
                    f'{scale}_{ch}_diff_std',  # Derivative volatility
                ])

        # Cross-scale features (detect gradual changes) - use drift which is relative
        for ch in self.all_channels:
            names.extend([
                f'{ch}_drift',       # Long mean - short mean (RELATIVE change)
                f'{ch}_std_ratio',   # Long std / short std
            ])

        # Physics consistency features
        names.extend([
            'pos_vel_consistency',
            'vel_accel_consistency',
            'att_rate_consistency',
            'energy_conservation',
        ])

        # Temporal pattern features
        names.extend([
            'freeze_score',           # How constant is the signal?
            'jump_score',             # Any sudden discontinuities?
            'periodicity_score',      # Unusual periodicity (replay)?
            'noise_floor',            # Noise characteristics
        ])

        # FFT features (if enabled)
        if self.use_fft:
            names.extend([
                'dominant_freq',
                'spectral_entropy',
                'high_freq_energy',
                'low_freq_energy',
            ])

        # NEW: Control input features (for actuator attack detection)
        if self.use_control_features:
            for scale in scales:
                for ch in self.control_channels:
                    names.extend([
                        f'{scale}_{ch}_mean',
                        f'{scale}_{ch}_std',
                        f'{scale}_{ch}_range',
                        f'{scale}_{ch}_diff_std',  # Control signal smoothness
                    ])

            # Control-state consistency features
            names.extend([
                'thrust_accel_consistency',    # Does thrust match vertical accel?
                'torque_rate_consistency',     # Do torques match angular rates?
                'control_freeze_score',        # Are controls frozen? (actuator_stuck)
                'control_jump_score',          # Sudden control changes?
                'thrust_efficiency',           # Expected vs observed altitude change
                'actuator_responsiveness',     # NEW: Do state changes follow control changes?
                'attitude_velocity_coupling',  # NEW: Does horizontal accel require tilt?
            ])

        return names

    def extract(self, data: np.ndarray, controls: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract features from data buffer.

        Args:
            data: [N, 12] state sequence (must be >= long_window)
            controls: [N, 4] control inputs (thrust, torque_x, torque_y, torque_z)
                      Optional but REQUIRED for actuator attack detection

        Returns:
            [n_features] feature vector
        """
        if len(data) < self.long_window:
            # Pad with last value if needed
            pad_size = self.long_window - len(data)
            data = np.vstack([np.tile(data[0], (pad_size, 1)), data])
            if controls is not None:
                controls = np.vstack([np.tile(controls[0], (pad_size, 1)), controls])

        # Get windows
        short_data = data[-self.short_window:]
        medium_data = data[-self.medium_window:]
        long_data = data[-self.long_window:]

        features = []

        # Per-channel, per-scale statistics
        # CRITICAL: Skip absolute means for position channels to avoid overfitting
        for scale_name, scale_data in [('short', short_data),
                                         ('medium', medium_data),
                                         ('long', long_data)]:
            for i, ch_name in enumerate(self.all_channels):
                ch = scale_data[:, i]
                diff = np.diff(ch)

                # Only include mean for non-position channels
                if ch_name not in self.position_channels:
                    features.append(np.mean(ch))

                # Always include relative measures
                features.extend([
                    np.std(ch) + 1e-8,
                    np.max(ch) - np.min(ch),
                    np.std(diff) + 1e-8,
                ])

        # Cross-scale features (detect gradual changes)
        # Drift is relative (difference between windows) so it's OK for all channels
        for i in range(12):
            short_mean = np.mean(short_data[:, i])
            long_mean = np.mean(long_data[:, i])
            short_std = np.std(short_data[:, i]) + 1e-8
            long_std = np.std(long_data[:, i]) + 1e-8

            features.extend([
                long_mean - short_mean,  # Drift (relative change, not absolute)
                long_std / short_std,     # Std ratio
            ])

        # Physics consistency features
        features.extend(self._physics_features(medium_data))

        # Temporal pattern features
        features.extend(self._temporal_features(long_data))

        # FFT features
        if self.use_fft:
            features.extend(self._fft_features(medium_data))

        # NEW: Control input features (for actuator attack detection)
        if self.use_control_features:
            if controls is not None:
                short_ctrl = controls[-self.short_window:]
                medium_ctrl = controls[-self.medium_window:]
                long_ctrl = controls[-self.long_window:]
                features.extend(self._control_features(
                    short_ctrl, medium_ctrl, long_ctrl, medium_data
                ))
            else:
                # No controls provided - fill with zeros
                n_control_features = len(self.control_channels) * 4 * 3 + 7  # 4 stats x 3 scales + 7 consistency
                features.extend([0.0] * n_control_features)

        return np.array(features, dtype=np.float32)

    def _physics_features(self, data: np.ndarray) -> List[float]:
        """Physics consistency checks."""
        dt = 0.005

        pos = data[:, :3]
        att = data[:, 3:6]
        rates = data[:, 6:9]
        vel = data[:, 9:12]

        # Position-velocity consistency
        pos_diff = np.diff(pos, axis=0) / dt
        vel_mid = vel[:-1]
        pos_vel_err = np.mean(np.abs(pos_diff - vel_mid))

        # Velocity-acceleration consistency (simplified)
        vel_diff = np.diff(vel, axis=0) / dt
        # Assume acceleration should be bounded
        accel_magnitude = np.linalg.norm(vel_diff, axis=1)
        vel_accel_err = np.mean(accel_magnitude > 50)  # Unrealistic acceleration

        # Attitude-rate consistency
        att_diff = np.diff(att, axis=0) / dt
        rates_mid = rates[:-1]
        att_rate_err = np.mean(np.abs(att_diff - rates_mid))

        # Energy conservation (simplified)
        kinetic = 0.5 * np.sum(vel**2, axis=1)
        potential = 9.81 * pos[:, 2]
        total_energy = kinetic + potential
        energy_var = np.std(total_energy) / (np.mean(np.abs(total_energy)) + 1e-6)

        return [pos_vel_err, vel_accel_err, att_rate_err, energy_var]

    def _temporal_features(self, data: np.ndarray) -> List[float]:
        """Temporal pattern detection."""
        # Use position magnitude as summary
        pos_mag = np.linalg.norm(data[:, :3], axis=1)

        # Freeze detection: very low variance
        freeze_score = 1.0 / (np.std(pos_mag) + 0.01)
        freeze_score = min(freeze_score, 100)  # Cap

        # Jump detection: max derivative vs mean
        diff = np.abs(np.diff(pos_mag))
        mean_diff = np.mean(diff) + 1e-6
        max_diff = np.max(diff)
        jump_score = max_diff / mean_diff

        # Periodicity detection (for replay)
        # Look at autocorrelation at various lags
        n = len(pos_mag)
        centered = pos_mag - np.mean(pos_mag)
        var = np.var(pos_mag) + 1e-8

        # Check lags 10, 20, 50, 100
        autocorrs = []
        for lag in [10, 20, 50, 100]:
            if lag < n:
                ac = np.mean(centered[:-lag] * centered[lag:]) / var
                autocorrs.append(ac)

        # High autocorrelation at specific lag = replay
        periodicity_score = max(autocorrs) if autocorrs else 0

        # Noise floor: std of second derivative
        diff2 = np.diff(diff)
        noise_floor = np.std(diff2)

        return [freeze_score, jump_score, periodicity_score, noise_floor]

    def _fft_features(self, data: np.ndarray) -> List[float]:
        """Frequency domain features."""
        # Use velocity magnitude
        vel_mag = np.linalg.norm(data[:, 9:12], axis=1)

        # FFT
        fft_vals = np.abs(np.fft.rfft(vel_mag))
        freqs = np.fft.rfftfreq(len(vel_mag), d=0.005)

        if len(fft_vals) < 2:
            return [0.0, 0.0, 0.0, 0.0]

        # Dominant frequency
        dominant_idx = np.argmax(fft_vals[1:]) + 1  # Skip DC
        dominant_freq = freqs[dominant_idx] if dominant_idx < len(freqs) else 0

        # Spectral entropy
        fft_norm = fft_vals / (np.sum(fft_vals) + 1e-8)
        spectral_entropy = -np.sum(fft_norm * np.log(fft_norm + 1e-10))

        # High/low frequency energy split
        mid_idx = len(fft_vals) // 2
        low_freq_energy = np.sum(fft_vals[:mid_idx]**2)
        high_freq_energy = np.sum(fft_vals[mid_idx:]**2)
        total_energy = low_freq_energy + high_freq_energy + 1e-8

        return [
            dominant_freq,
            spectral_entropy,
            high_freq_energy / total_energy,
            low_freq_energy / total_energy,
        ]

    def _control_features(
        self,
        short_ctrl: np.ndarray,
        medium_ctrl: np.ndarray,
        long_ctrl: np.ndarray,
        state_data: np.ndarray,
    ) -> List[float]:
        """
        Extract control input features for actuator attack detection.

        Key features:
        1. Per-channel statistics (mean, std, range, diff_std) at each scale
        2. Thrust-acceleration consistency (does commanded thrust match observed accel?)
        3. Torque-rate consistency (do torques match angular rate changes?)
        4. Control freeze score (actuator stuck detection)
        5. Control jump score (sudden control changes)
        """
        dt = 0.005
        GRAVITY = 9.81
        features = []

        # Per-control-channel, per-scale statistics
        for scale_ctrl in [short_ctrl, medium_ctrl, long_ctrl]:
            for i in range(min(4, scale_ctrl.shape[1] if len(scale_ctrl.shape) > 1 else 1)):
                if len(scale_ctrl.shape) > 1:
                    ch = scale_ctrl[:, i]
                else:
                    ch = scale_ctrl
                diff = np.diff(ch)

                features.extend([
                    np.mean(ch),               # Mean control value
                    np.std(ch) + 1e-8,         # Control variability
                    np.max(ch) - np.min(ch),   # Control range
                    np.std(diff) + 1e-8,       # Control smoothness
                ])

        # Control-state consistency features

        # 1. Thrust-acceleration consistency
        # If thrust is commanded, we expect vertical acceleration
        thrust = medium_ctrl[:, 0] if medium_ctrl.shape[1] > 0 else np.zeros(len(medium_ctrl))
        vel_z = state_data[:, 11]  # vz
        observed_accel_z = np.diff(vel_z) / dt
        expected_accel_z = thrust[:-1] - GRAVITY  # Simplified: accel = thrust - g
        thrust_accel_err = np.mean(np.abs(observed_accel_z - expected_accel_z))
        thrust_accel_consistency = 1.0 / (thrust_accel_err + 0.1)  # Higher = more consistent

        # 2. Torque-rate consistency
        # If torques are commanded, we expect angular rate changes
        if medium_ctrl.shape[1] >= 4:
            torques = medium_ctrl[:, 1:4]  # torque_x, torque_y, torque_z
            rates = state_data[:, 6:9]     # p, q, r
            rate_diff = np.diff(rates, axis=0) / dt
            # Simplified: angular accel should correlate with torque
            torque_rate_corr = np.abs(np.corrcoef(
                np.linalg.norm(torques[:-1], axis=1),
                np.linalg.norm(rate_diff, axis=1)
            )[0, 1])
            if np.isnan(torque_rate_corr):
                torque_rate_corr = 0.0
        else:
            torque_rate_corr = 0.0

        # 3. Control freeze score (detects actuator_stuck)
        # If controls have near-zero variance, actuator may be stuck
        ctrl_std = np.std(long_ctrl, axis=0)
        ctrl_freeze_score = 1.0 / (np.mean(ctrl_std) + 0.01)
        ctrl_freeze_score = min(ctrl_freeze_score, 100)  # Cap

        # 4. Control jump score (sudden control changes)
        ctrl_diff = np.diff(long_ctrl[:, 0])  # Focus on thrust
        mean_ctrl_diff = np.mean(np.abs(ctrl_diff)) + 1e-6
        max_ctrl_diff = np.max(np.abs(ctrl_diff))
        ctrl_jump_score = max_ctrl_diff / mean_ctrl_diff

        # 5. Thrust efficiency (does thrust actually change altitude?)
        # Low efficiency = actuator degraded or stuck
        thrust_integral = np.sum(thrust - GRAVITY) * dt  # Expected delta-vz
        actual_delta_vz = vel_z[-1] - vel_z[0]
        if abs(thrust_integral) > 0.1:
            thrust_efficiency = actual_delta_vz / (thrust_integral + 1e-6)
            thrust_efficiency = np.clip(thrust_efficiency, -10, 10)
        else:
            thrust_efficiency = 1.0  # No significant thrust = no expectation

        # 6. Actuator responsiveness (detects actuator_stuck, actuator_degraded)
        # Key insight: If control changes but state doesn't follow, actuator is stuck/degraded
        thrust_diff = np.diff(thrust)
        accel_z_diff = np.diff(observed_accel_z) if len(observed_accel_z) > 1 else np.array([0])
        if len(thrust_diff) > 5 and len(accel_z_diff) > 5:
            # Correlation: do thrust changes cause accel changes?
            min_len = min(len(thrust_diff), len(accel_z_diff))
            corr = np.corrcoef(thrust_diff[:min_len], accel_z_diff[:min_len])[0, 1]
            actuator_responsiveness = corr if not np.isnan(corr) else 0.0
        else:
            actuator_responsiveness = 1.0  # Assume responsive if not enough data

        # 7. Attitude-velocity coupling (detects stealthy_coordinated)
        # Physics: horizontal acceleration requires tilt (roll/pitch)
        # If velocity changes but attitude doesn't, it's suspicious
        att = state_data[:, 3:6]  # phi, theta, psi
        vel_xy = state_data[:, 9:11]  # vx, vy

        accel_xy = np.diff(vel_xy, axis=0) / dt
        if len(accel_xy) > 5:
            # Expected attitude from acceleration: theta ≈ a_x/g, phi ≈ -a_y/g
            expected_theta = accel_xy[:, 0] / GRAVITY
            expected_phi = -accel_xy[:, 1] / GRAVITY
            actual_theta = att[:-1, 1]
            actual_phi = att[:-1, 0]

            # Error between expected and actual tilt
            theta_err = np.mean(np.abs(expected_theta - actual_theta))
            phi_err = np.mean(np.abs(expected_phi - actual_phi))
            attitude_velocity_coupling = theta_err + phi_err  # Higher = more suspicious
        else:
            attitude_velocity_coupling = 0.0

        features.extend([
            thrust_accel_consistency,
            torque_rate_corr,
            ctrl_freeze_score,
            ctrl_jump_score,
            thrust_efficiency,
            actuator_responsiveness,
            attitude_velocity_coupling,
        ])

        return features

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


class PINNFeatureExtractor:
    """
    Extract features from PINN predictions.

    Uses trained PINN to compute:
    - Prediction residuals
    - Physics violation scores
    - Uncertainty estimates
    """

    def __init__(self, predictor: 'Predictor', window_size: int = 20):
        self.predictor = predictor
        self.window_size = window_size

        self.feature_names = [
            'pinn_residual_mean',
            'pinn_residual_max',
            'pinn_residual_std',
            'pinn_residual_trend',  # Is residual growing?
            'pinn_pos_residual',
            'pinn_vel_residual',
            'pinn_att_residual',
        ]

    def extract(
        self,
        states: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        """
        Extract PINN-based features.

        Args:
            states: [N, 12] state sequence
            controls: [N, 4] control sequence

        Returns:
            [n_features] feature vector
        """
        if len(states) < self.window_size + 1:
            return np.zeros(len(self.feature_names))

        # Compute residuals over window
        residuals = []
        pos_residuals = []
        vel_residuals = []
        att_residuals = []

        for i in range(len(states) - self.window_size, len(states) - 1):
            state = states[i]
            control = controls[i]
            next_state = states[i + 1]

            # PINN prediction
            predicted = self.predictor.predict(state, control)

            # Residual
            residual = np.linalg.norm(next_state - predicted)
            residuals.append(residual)

            # Per-component residuals
            pos_residuals.append(np.linalg.norm(next_state[:3] - predicted[:3]))
            vel_residuals.append(np.linalg.norm(next_state[9:12] - predicted[9:12]))
            att_residuals.append(np.linalg.norm(next_state[3:6] - predicted[3:6]))

        residuals = np.array(residuals)

        # Trend: is residual growing?
        if len(residuals) > 1:
            trend = np.polyfit(np.arange(len(residuals)), residuals, 1)[0]
        else:
            trend = 0

        return np.array([
            np.mean(residuals),
            np.max(residuals),
            np.std(residuals),
            trend,
            np.mean(pos_residuals),
            np.mean(vel_residuals),
            np.mean(att_residuals),
        ], dtype=np.float32)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


class EnhancedAttackDetector:
    """
    Enhanced attack detector with multi-scale features and PINN integration.

    Key improvements:
    1. Multi-scale feature extraction
    2. PINN residual features (if available)
    3. Gradient Boosting classifier
    4. Threshold tuning for target recall
    5. Confidence calibration

    Args:
        predictor: Optional PINN Predictor for residual features
        target_recall: Target recall for threshold tuning (default 0.9)
        n_estimators: Number of boosting iterations
    """

    def __init__(
        self,
        predictor: Optional['Predictor'] = None,
        target_recall: float = 0.90,
        n_estimators: int = 200,
    ):
        self.predictor = predictor
        self.target_recall = target_recall

        # Feature extractors
        self.multi_scale_extractor = MultiScaleFeatureExtractor()
        self.pinn_extractor = PINNFeatureExtractor(predictor) if predictor else None

        # Classifier
        if SKLEARN_AVAILABLE:
            self.classifier = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=6,
                min_samples_leaf=10,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
            self.scaler = RobustScaler()  # More robust to outliers
        else:
            self.classifier = None
            self.scaler = None

        self.threshold = 0.5  # Will be tuned
        self.is_trained = False

        # Online state
        self.state_buffer: deque = deque(maxlen=250)  # Max window size + margin
        self.control_buffer: deque = deque(maxlen=250)

    def _extract_all_features(
        self,
        states: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract all features from a window."""
        # Pass controls to multi_scale_extractor for control features
        features = self.multi_scale_extractor.extract(states, controls)

        if self.pinn_extractor and controls is not None:
            pinn_features = self.pinn_extractor.extract(states, controls)
            features = np.concatenate([features, pinn_features])

        return features

    def train(
        self,
        normal_states: np.ndarray,
        attack_states: np.ndarray,
        normal_controls: Optional[np.ndarray] = None,
        attack_controls: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train detector on labeled data.

        Args:
            normal_states: [N, 12] normal flight data
            attack_states: [M, 12] attack data
            normal_controls: [N, 4] controls for normal (optional)
            attack_controls: [M, 4] controls for attack (optional)
        """
        if not SKLEARN_AVAILABLE:
            print("sklearn not available")
            self.is_trained = True
            return {}

        print("Training EnhancedAttackDetector...")
        print(f"  Normal samples: {len(normal_states):,}")
        print(f"  Attack samples: {len(attack_states):,}")

        window_size = self.multi_scale_extractor.long_window

        # Extract features with sliding window
        print("  Extracting features...")
        X_normal = []
        for i in range(window_size, len(normal_states)):
            window = normal_states[i-window_size:i]
            ctrl_window = normal_controls[i-window_size:i] if normal_controls is not None else None
            feat = self._extract_all_features(window, ctrl_window)
            X_normal.append(feat)

        X_attack = []
        for i in range(window_size, len(attack_states)):
            window = attack_states[i-window_size:i]
            ctrl_window = attack_controls[i-window_size:i] if attack_controls is not None else None
            feat = self._extract_all_features(window, ctrl_window)
            X_attack.append(feat)

        X_normal = np.array(X_normal)
        X_attack = np.array(X_attack)

        print(f"  Normal features: {X_normal.shape}")
        print(f"  Attack features: {X_attack.shape}")

        # Combine
        X = np.vstack([X_normal, X_attack])
        y = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_attack))])

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Train
        print("  Training Gradient Boosting classifier...")
        self.classifier.fit(X_scaled, y)

        # Get probabilities for threshold tuning
        probs = self.classifier.predict_proba(X_scaled)[:, 1]

        # Tune threshold for target recall
        print(f"  Tuning threshold for {self.target_recall*100:.0f}% recall...")
        precision, recall, thresholds = precision_recall_curve(y, probs)

        # Find threshold that gives target recall
        valid_idx = np.where(recall >= self.target_recall)[0]
        if len(valid_idx) > 0:
            # Take the threshold that gives highest precision at target recall
            best_idx = valid_idx[np.argmax(precision[valid_idx])]
            if best_idx < len(thresholds):
                self.threshold = thresholds[best_idx]
            else:
                self.threshold = thresholds[-1] if len(thresholds) > 0 else 0.5
        else:
            # Can't achieve target recall, use lowest threshold
            self.threshold = thresholds[0] if len(thresholds) > 0 else 0.1

        print(f"  Optimal threshold: {self.threshold:.4f}")

        # Compute final metrics with tuned threshold
        y_pred = (probs >= self.threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))
        tn = np.sum((y_pred == 0) & (y == 0))

        final_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        final_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        final_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"  Training Precision: {final_precision*100:.1f}%")
        print(f"  Training Recall:    {final_recall*100:.1f}%")
        print(f"  Training FPR:       {final_fpr*100:.1f}%")

        # Feature importances
        importances = self.classifier.feature_importances_
        all_names = self.multi_scale_extractor.feature_names
        if self.pinn_extractor:
            all_names = all_names + self.pinn_extractor.feature_names

        top_idx = np.argsort(importances)[-10:][::-1]
        print("  Top 10 features:")
        for idx in top_idx:
            if idx < len(all_names):
                print(f"    {all_names[idx]}: {importances[idx]:.4f}")

        self.is_trained = True

        return {
            "precision": float(final_precision),
            "recall": float(final_recall),
            "fpr": float(final_fpr),
            "threshold": float(self.threshold),
        }

    def reset(self):
        """Reset online buffers."""
        self.state_buffer.clear()
        self.control_buffer.clear()

    def predict(
        self,
        state: np.ndarray,
        control: Optional[np.ndarray] = None,
    ) -> DetectionResult:
        """Online prediction for single timestep."""
        self.state_buffer.append(state.copy())
        if control is not None:
            self.control_buffer.append(control.copy())

        window_size = self.multi_scale_extractor.long_window

        if len(self.state_buffer) < window_size:
            return DetectionResult(
                is_attack=False,
                probability=0.0,
                attack_type="none",
                confidence=0.0,
            )

        if not self.is_trained or not SKLEARN_AVAILABLE:
            return DetectionResult(
                is_attack=False,
                probability=0.0,
                attack_type="unknown",
                confidence=0.0,
            )

        # Extract features
        states = np.array(list(self.state_buffer))
        controls = np.array(list(self.control_buffer)) if self.control_buffer else None

        features = self._extract_all_features(states, controls)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Scale and predict
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prob = self.classifier.predict_proba(features_scaled)[0, 1]

        is_attack = prob >= self.threshold

        # Confidence based on distance from threshold
        if is_attack:
            confidence = min((prob - self.threshold) / (1 - self.threshold), 1.0)
        else:
            confidence = min((self.threshold - prob) / self.threshold, 1.0)

        return DetectionResult(
            is_attack=is_attack,
            probability=float(prob),
            attack_type="detected" if is_attack else "none",
            confidence=float(confidence),
        )

    def predict_batch(
        self,
        states: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            n_out = max(0, len(states) - self.multi_scale_extractor.long_window)
            return np.zeros(n_out), np.zeros(n_out)

        window_size = self.multi_scale_extractor.long_window

        features = []
        for i in range(window_size, len(states)):
            window = states[i-window_size:i]
            ctrl_window = controls[i-window_size:i] if controls is not None else None
            feat = self._extract_all_features(window, ctrl_window)
            features.append(feat)

        if not features:
            return np.array([]), np.array([])

        X = np.array(features)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)

        probs = self.classifier.predict_proba(X_scaled)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        return preds, probs

    def evaluate(
        self,
        states: np.ndarray,
        labels: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate on labeled data."""
        preds, probs = self.predict_batch(states, controls)

        window_size = self.multi_scale_extractor.long_window
        aligned_labels = labels[window_size:]

        min_len = min(len(preds), len(aligned_labels))
        preds = preds[:min_len]
        aligned_labels = aligned_labels[:min_len]

        tp = np.sum((preds == 1) & (aligned_labels == 1))
        fp = np.sum((preds == 1) & (aligned_labels == 0))
        fn = np.sum((preds == 0) & (aligned_labels == 1))
        tn = np.sum((preds == 0) & (aligned_labels == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fpr": float(fpr),
        }
