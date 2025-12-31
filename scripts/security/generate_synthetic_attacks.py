"""
Generate realistic synthetic attacks by injecting faults into clean EuRoC data.

Comprehensive attack types based on published literature (30 total):

GPS ATTACKS (7):
1. GPS Gradual Drift - Slow position shift (Tippenhauer CCS'11)
2. GPS Sudden Jump - Instantaneous position change
3. GPS Oscillating - Sinusoidal position manipulation
4. GPS Meaconing - Delayed signal replay
5. GPS Jamming - Complete signal loss
6. GPS Freeze - Position stuck at one location
7. GPS Multipath - Urban canyon signal reflections

IMU ATTACKS (7):
8. IMU Constant Bias - Fixed offset (Shoukry CHES'13)
9. IMU Gradual Drift - Slowly increasing bias
10. IMU Sinusoidal - Oscillating sensor values
11. IMU Noise Injection - Increased sensor noise
12. IMU Scale Factor - Multiplicative attack
13. Gyro Saturation - Angular rate sensor maxed out
14. Accelerometer Saturation - Linear acceleration maxed out

MAGNETOMETER/BAROMETER ATTACKS (2):
15. Magnetometer Spoofing - Compass/heading manipulation
16. Barometer Spoofing - Altitude manipulation

ACTUATOR/CONTROL ATTACKS (4):
17. Actuator Stuck - Motor/servo jammed at fixed value
18. Actuator Degraded - Reduced actuator effectiveness
19. Control Hijack - False control command injection
20. Thrust Manipulation - Engine power attack

COORDINATED ATTACKS (2):
21. GPS+IMU Coordinated - Both sensors attacked
22. Stealthy Coordinated - Physics-consistent attack (FDI-style)

TEMPORAL ATTACKS (3):
23. Replay Attack - Repeat old trajectory
24. Time Delay - Delayed sensor readings
25. Sensor Dropout - Random data loss

STEALTH/ADVANCED ATTACKS (5):
26. Adaptive Attack - Magnitude grows to avoid detection
27. Intermittent Attack - On-off pattern
28. Slow Ramp - Very gradual manipulation
29. Resonance Attack - Excite natural frequencies
30. False Data Injection - Optimized stealthy attack (Liu CCS'09)

Usage:
    python scripts/security/generate_synthetic_attacks.py \\
        --input data/euroc_mav/ \\
        --output data/attack_datasets/synthetic/ \\
        --randomize  # Enable parameter randomization
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Physical constants
GRAVITY = 9.81  # m/s^2

# Attack scaling factors
Z_SCALE = 0.3  # Vertical axis scaling (z-axis is more constrained in flight)


class SyntheticAttackGenerator:
    """Generate comprehensive physics-based attack scenarios from clean data."""

    # Column name mappings for different data formats
    COLUMN_ALIASES = {
        "phi": ["phi", "roll"],
        "theta": ["theta", "pitch"],
        "psi": ["psi", "yaw"],
        "thrust": ["thrust"],  # Will compute from az if not present
        "torque_x": ["torque_x"],
        "torque_y": ["torque_y"],
        "torque_z": ["torque_z"],
    }

    def __init__(self, clean_data: pd.DataFrame, seed: int = 42, randomize: bool = False):
        """
        Args:
            clean_data: Clean flight trajectory with columns:
                Standard: [timestamp, x, y, z, phi, theta, psi, p, q, r, vx, vy, vz,
                          thrust, torque_x, torque_y, torque_z]
                EuRoC:    [timestamp, x, y, z, roll, pitch, yaw, p, q, r, vx, vy, vz,
                          ax, ay, az]
            seed: Random seed for reproducibility
            randomize: If True, randomize attack parameters for robustness
        """
        self.clean_data = self._normalize_columns(clean_data.copy())
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize

        # Calculate dt properly - use median of small diffs to handle sequence gaps
        diffs = self.clean_data["timestamp"].diff().dropna()
        # Filter out large gaps (sequence boundaries) - keep only diffs < 1 second
        small_diffs = diffs[diffs < 1.0]
        if len(small_diffs) > 0:
            self.dt = small_diffs.median()
        else:
            self.dt = 0.005  # Default 200Hz for EuRoC

    def _normalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard format."""
        # Map alternative column names to standard names
        for standard_name, aliases in self.COLUMN_ALIASES.items():
            if standard_name not in data.columns:
                for alias in aliases:
                    if alias in data.columns:
                        data[standard_name] = data[alias]
                        break

        # Generate missing control columns if needed
        if "thrust" not in data.columns:
            if "az" in data.columns:
                # EuRoC az is body-frame, add gravity for total thrust
                # thrust = m * (az + g) ≈ az + GRAVITY for unit mass
                data["thrust"] = data["az"] + GRAVITY
            else:
                data["thrust"] = GRAVITY * np.ones(len(data))

        if "torque_x" not in data.columns:
            data["torque_x"] = np.zeros(len(data))
        if "torque_y" not in data.columns:
            data["torque_y"] = np.zeros(len(data))
        if "torque_z" not in data.columns:
            data["torque_z"] = np.zeros(len(data))

        return data

    def handle_nan_values(self, data: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
        """
        Handle NaN values created by jamming/dropout attacks.

        Args:
            data: DataFrame with potential NaN values
            method: 'interpolate', 'ffill', 'drop', or 'zero'

        Returns:
            New DataFrame with NaN handled (original is not modified)
        """
        # Always work on a copy to ensure consistent behavior
        data = data.copy()

        if method == "interpolate":
            # Linear interpolation (best for continuous signals)
            # Only interpolate numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].interpolate(
                method="linear", limit_direction="both"
            )
        elif method == "ffill":
            # Forward fill (use last known value)
            data = data.ffill().bfill()
        elif method == "drop":
            # Drop rows with NaN
            data = data.dropna()
        elif method == "zero":
            # Replace with zero (numeric only)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(0)
        else:
            raise ValueError(f"Unknown NaN handling method: {method}")

        return data

    def _get_attack_window(
        self, attack_start_ratio: float, attack_duration: float
    ) -> Tuple[int, int, int]:
        """Helper to compute attack window indices."""
        n_total = len(self.clean_data)
        attack_start_idx = int(n_total * attack_start_ratio)
        attack_duration_samples = int(attack_duration / self.dt)
        attack_end_idx = min(attack_start_idx + attack_duration_samples, n_total)
        n_attack = attack_end_idx - attack_start_idx
        return attack_start_idx, attack_end_idx, n_attack

    def _randomize_params(self, base_value: float, variation: float = 0.3) -> float:
        """Randomize parameter if randomization is enabled."""
        if self.randomize:
            return base_value * (1 + self.rng.uniform(-variation, variation))
        return base_value

    def _init_data(self) -> pd.DataFrame:
        """Initialize data copy with labels."""
        data = self.clean_data.copy()
        data["label"] = 0
        data["attack_type"] = "Normal"
        return data

    # =========================================================================
    # GPS ATTACKS
    # =========================================================================

    def gps_gradual_drift(
        self,
        drift_magnitude: float = 10.0,
        drift_duration: float = 30.0,
        attack_start_ratio: float = 0.3,
        drift_direction: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        GPS Spoofing: Gradual position drift.

        Reference: Tippenhauer et al. "On the Requirements for Successful
        GPS Spoofing Attacks" (CCS 2011)
        """
        data = self._init_data()

        drift_magnitude = self._randomize_params(drift_magnitude)
        attack_start_ratio = self._randomize_params(attack_start_ratio, 0.2)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, drift_duration
        )

        # Random drift direction if not specified
        if drift_direction is None:
            drift_direction = self.rng.standard_normal(3)
            drift_direction = drift_direction / np.linalg.norm(drift_direction)

        # Create smooth drift trajectory
        drift_profile = np.linspace(0, drift_magnitude, n_attack)

        data.loc[attack_start_idx : attack_end_idx - 1, "x"] += drift_profile * drift_direction[0]
        data.loc[attack_start_idx : attack_end_idx - 1, "y"] += drift_profile * drift_direction[1]
        data.loc[attack_start_idx : attack_end_idx - 1, "z"] += (
            drift_profile * drift_direction[2] * 0.3
        )

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "GPS_Gradual_Drift"

        return data

    def gps_sudden_jump(
        self,
        jump_magnitude: float = 5.0,
        attack_start_ratio: float = 0.4,
        attack_duration: float = 20.0,
    ) -> pd.DataFrame:
        """
        GPS Spoofing: Sudden position jump.

        Instantaneous position change - easier to detect but sometimes
        used in unsophisticated attacks.
        """
        data = self._init_data()

        jump_magnitude = self._randomize_params(jump_magnitude)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Random jump direction
        jump_direction = self.rng.standard_normal(3)
        jump_direction = jump_direction / np.linalg.norm(jump_direction)

        # Apply sudden constant offset
        data.loc[attack_start_idx : attack_end_idx - 1, "x"] += jump_magnitude * jump_direction[0]
        data.loc[attack_start_idx : attack_end_idx - 1, "y"] += jump_magnitude * jump_direction[1]
        data.loc[attack_start_idx : attack_end_idx - 1, "z"] += (
            jump_magnitude * jump_direction[2] * 0.3
        )

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "GPS_Sudden_Jump"

        return data

    def gps_oscillating(
        self,
        amplitude: float = 3.0,
        frequency: float = 0.5,
        attack_start_ratio: float = 0.3,
        attack_duration: float = 25.0,
    ) -> pd.DataFrame:
        """
        GPS Spoofing: Oscillating position manipulation.

        Sinusoidal position deviation - can be used to confuse
        navigation filters.
        """
        data = self._init_data()

        amplitude = self._randomize_params(amplitude)
        frequency = self._randomize_params(frequency)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Create oscillating pattern
        t = np.arange(n_attack) * self.dt
        oscillation_x = amplitude * np.sin(2 * np.pi * frequency * t)
        oscillation_y = amplitude * np.cos(2 * np.pi * frequency * t)
        oscillation_z = amplitude * 0.3 * np.sin(2 * np.pi * frequency * t + np.pi / 4)

        data.loc[attack_start_idx : attack_end_idx - 1, "x"] += oscillation_x
        data.loc[attack_start_idx : attack_end_idx - 1, "y"] += oscillation_y
        data.loc[attack_start_idx : attack_end_idx - 1, "z"] += oscillation_z

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "GPS_Oscillating"

        return data

    def gps_meaconing(
        self,
        delay_samples: int = 50,
        attack_start_ratio: float = 0.4,
        attack_duration: float = 30.0,
    ) -> pd.DataFrame:
        """
        GPS Meaconing: Delayed signal replay.

        Attacker records and replays GPS signals with a delay,
        causing position to lag behind actual location.
        """
        data = self._init_data()

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Ensure we have enough history
        if attack_start_idx < delay_samples:
            attack_start_idx = delay_samples
            attack_end_idx = min(attack_start_idx + n_attack, len(data))
            n_attack = attack_end_idx - attack_start_idx

        # Replay delayed GPS data
        source_start = attack_start_idx - delay_samples
        source_end = attack_end_idx - delay_samples

        for col in ["x", "y", "z"]:
            data.loc[attack_start_idx : attack_end_idx - 1, col] = data.loc[
                source_start : source_end - 1, col
            ].values

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "GPS_Meaconing"

        return data

    def gps_jamming(
        self,
        attack_start_ratio: float = 0.35,
        attack_duration: float = 15.0,
    ) -> pd.DataFrame:
        """
        GPS Jamming: Complete signal loss.

        All GPS data becomes unavailable (NaN).
        Detector should handle missing data.
        """
        data = self._init_data()

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Complete GPS loss
        data.loc[attack_start_idx : attack_end_idx - 1, "x"] = np.nan
        data.loc[attack_start_idx : attack_end_idx - 1, "y"] = np.nan
        data.loc[attack_start_idx : attack_end_idx - 1, "z"] = np.nan

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "GPS_Jamming"

        return data

    def gps_freeze(
        self,
        attack_start_ratio: float = 0.4,
        attack_duration: float = 20.0,
    ) -> pd.DataFrame:
        """
        GPS Freeze: Position stuck at one location.

        GPS reports the same position continuously while
        the drone is actually moving.
        """
        data = self._init_data()

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Freeze at attack start position
        frozen_x = data.loc[attack_start_idx, "x"]
        frozen_y = data.loc[attack_start_idx, "y"]
        frozen_z = data.loc[attack_start_idx, "z"]

        data.loc[attack_start_idx : attack_end_idx - 1, "x"] = frozen_x
        data.loc[attack_start_idx : attack_end_idx - 1, "y"] = frozen_y
        data.loc[attack_start_idx : attack_end_idx - 1, "z"] = frozen_z

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "GPS_Freeze"

        return data

    def gps_multipath(
        self,
        jitter_std: float = 2.0,
        correlation_time: float = 0.5,
        attack_start_ratio: float = 0.3,
        attack_duration: float = 30.0,
    ) -> pd.DataFrame:
        """
        GPS Multipath: Urban canyon signal reflections.

        Simulates GPS errors caused by signal reflections off buildings,
        causing correlated position jitter (not white noise).

        Reference: Common in urban UAV operations
        """
        data = self._init_data()

        jitter_std = self._randomize_params(jitter_std)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Generate correlated noise (exponential correlation)
        alpha = self.dt / correlation_time
        noise_x = np.zeros(n_attack)
        noise_y = np.zeros(n_attack)
        noise_z = np.zeros(n_attack)

        for i in range(1, n_attack):
            noise_x[i] = (1 - alpha) * noise_x[
                i - 1
            ] + alpha * self.rng.standard_normal() * jitter_std
            noise_y[i] = (1 - alpha) * noise_y[
                i - 1
            ] + alpha * self.rng.standard_normal() * jitter_std
            noise_z[i] = (1 - alpha) * noise_z[
                i - 1
            ] + alpha * self.rng.standard_normal() * jitter_std * 0.5

        data.loc[attack_start_idx : attack_end_idx - 1, "x"] += noise_x
        data.loc[attack_start_idx : attack_end_idx - 1, "y"] += noise_y
        data.loc[attack_start_idx : attack_end_idx - 1, "z"] += noise_z

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "GPS_Multipath"

        return data

    # =========================================================================
    # IMU ATTACKS
    # =========================================================================

    def imu_constant_bias(
        self,
        accel_bias: float = 0.5,
        gyro_bias: float = 0.1,
        attack_start_ratio: float = 0.3,
        attack_duration: float = 20.0,
    ) -> pd.DataFrame:
        """
        IMU Injection: Constant bias added to accelerometer/gyro.

        Reference: Shoukry et al. "Non-invasive Spoofing Attacks for
        Anti-lock Braking Systems" (CHES 2013)
        """
        data = self._init_data()

        accel_bias = self._randomize_params(accel_bias)
        gyro_bias = self._randomize_params(gyro_bias)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Inject constant bias - affects velocity integration
        velocity_change = accel_bias * self.dt * np.arange(n_attack)

        data.loc[attack_start_idx : attack_end_idx - 1, "vx"] += velocity_change
        data.loc[attack_start_idx : attack_end_idx - 1, "p"] += gyro_bias
        data.loc[attack_start_idx : attack_end_idx - 1, "q"] += gyro_bias * 0.7

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "IMU_Constant_Bias"

        return data

    def imu_gradual_drift(
        self,
        max_accel_drift: float = 1.0,
        max_gyro_drift: float = 0.2,
        attack_start_ratio: float = 0.25,
        attack_duration: float = 40.0,
    ) -> pd.DataFrame:
        """
        IMU Gradual Drift: Slowly increasing bias.

        Simulates sensor degradation or stealthy attack where
        bias increases gradually over time.
        """
        data = self._init_data()

        max_accel_drift = self._randomize_params(max_accel_drift)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Linearly increasing bias
        accel_drift_profile = np.linspace(0, max_accel_drift, n_attack)
        gyro_drift_profile = np.linspace(0, max_gyro_drift, n_attack)

        # Integrate acceleration drift to velocity
        velocity_change = np.cumsum(accel_drift_profile * self.dt)

        data.loc[attack_start_idx : attack_end_idx - 1, "vx"] += velocity_change
        data.loc[attack_start_idx : attack_end_idx - 1, "vy"] += velocity_change * 0.5
        data.loc[attack_start_idx : attack_end_idx - 1, "p"] += gyro_drift_profile
        data.loc[attack_start_idx : attack_end_idx - 1, "q"] += gyro_drift_profile * 0.7
        data.loc[attack_start_idx : attack_end_idx - 1, "r"] += gyro_drift_profile * 0.5

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "IMU_Gradual_Drift"

        return data

    def imu_sinusoidal(
        self,
        accel_amplitude: float = 0.8,
        gyro_amplitude: float = 0.15,
        frequency: float = 2.0,
        attack_start_ratio: float = 0.35,
        attack_duration: float = 20.0,
    ) -> pd.DataFrame:
        """
        IMU Sinusoidal: Oscillating sensor injection.

        Periodic false sensor values that may confuse
        state estimation filters.
        """
        data = self._init_data()

        accel_amplitude = self._randomize_params(accel_amplitude)
        frequency = self._randomize_params(frequency)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        t = np.arange(n_attack) * self.dt
        accel_wave = accel_amplitude * np.sin(2 * np.pi * frequency * t)
        gyro_wave = gyro_amplitude * np.sin(2 * np.pi * frequency * t)

        # Integrate to get velocity effect
        velocity_effect = np.cumsum(accel_wave * self.dt)

        data.loc[attack_start_idx : attack_end_idx - 1, "vx"] += velocity_effect
        data.loc[attack_start_idx : attack_end_idx - 1, "p"] += gyro_wave
        data.loc[attack_start_idx : attack_end_idx - 1, "q"] += gyro_wave * np.cos(
            2 * np.pi * frequency * t
        )

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "IMU_Sinusoidal"

        return data

    def imu_noise_injection(
        self,
        accel_noise_std: float = 0.5,
        gyro_noise_std: float = 0.1,
        attack_start_ratio: float = 0.3,
        attack_duration: float = 25.0,
    ) -> pd.DataFrame:
        """
        IMU Noise Injection: Increased sensor noise.

        Adds excessive noise to degrade state estimation
        accuracy and potentially mask other attacks.
        """
        data = self._init_data()

        accel_noise_std = self._randomize_params(accel_noise_std)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Generate random noise
        accel_noise = self.rng.standard_normal(n_attack) * accel_noise_std
        gyro_noise_p = self.rng.standard_normal(n_attack) * gyro_noise_std
        gyro_noise_q = self.rng.standard_normal(n_attack) * gyro_noise_std
        gyro_noise_r = self.rng.standard_normal(n_attack) * gyro_noise_std

        # Integrate acceleration noise to velocity
        velocity_noise = np.cumsum(accel_noise * self.dt)

        data.loc[attack_start_idx : attack_end_idx - 1, "vx"] += velocity_noise
        data.loc[attack_start_idx : attack_end_idx - 1, "vy"] += np.cumsum(
            self.rng.standard_normal(n_attack) * accel_noise_std * self.dt
        )
        data.loc[attack_start_idx : attack_end_idx - 1, "vz"] += np.cumsum(
            self.rng.standard_normal(n_attack) * accel_noise_std * 0.5 * self.dt
        )
        data.loc[attack_start_idx : attack_end_idx - 1, "p"] += gyro_noise_p
        data.loc[attack_start_idx : attack_end_idx - 1, "q"] += gyro_noise_q
        data.loc[attack_start_idx : attack_end_idx - 1, "r"] += gyro_noise_r

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "IMU_Noise_Injection"

        return data

    def imu_scale_factor(
        self,
        accel_scale: float = 1.2,
        gyro_scale: float = 1.15,
        attack_start_ratio: float = 0.3,
        attack_duration: float = 30.0,
    ) -> pd.DataFrame:
        """
        IMU Scale Factor Attack: Multiplicative manipulation.

        Sensor readings are scaled by a factor, simulating
        calibration tampering or sophisticated spoofing.
        """
        data = self._init_data()

        accel_scale = self._randomize_params(accel_scale, 0.1)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Scale velocities and angular rates
        data.loc[attack_start_idx : attack_end_idx - 1, "vx"] *= accel_scale
        data.loc[attack_start_idx : attack_end_idx - 1, "vy"] *= accel_scale
        data.loc[attack_start_idx : attack_end_idx - 1, "vz"] *= accel_scale
        data.loc[attack_start_idx : attack_end_idx - 1, "p"] *= gyro_scale
        data.loc[attack_start_idx : attack_end_idx - 1, "q"] *= gyro_scale
        data.loc[attack_start_idx : attack_end_idx - 1, "r"] *= gyro_scale

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "IMU_Scale_Factor"

        return data

    def gyro_saturation(
        self,
        max_rate: float = 4.0,
        attack_start_ratio: float = 0.35,
        attack_duration: float = 15.0,
    ) -> pd.DataFrame:
        """
        Gyro Saturation: Angular rate sensors maxed out.

        Simulates gyroscope hitting maximum measurable rate,
        causing loss of attitude information.
        """
        data = self._init_data()

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Saturate gyro readings at max value based on original sign
        # (simulates sensor hitting its measurement limits)
        original_p = data.loc[attack_start_idx : attack_end_idx - 1, "p"].values
        data.loc[attack_start_idx : attack_end_idx - 1, "p"] = max_rate * np.sign(original_p + 0.1)
        data.loc[attack_start_idx : attack_end_idx - 1, "q"] = max_rate * 0.8
        data.loc[attack_start_idx : attack_end_idx - 1, "r"] = -max_rate * 0.5

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Gyro_Saturation"

        return data

    def accel_saturation(
        self,
        max_accel: float = 50.0,
        attack_start_ratio: float = 0.4,
        attack_duration: float = 10.0,
    ) -> pd.DataFrame:
        """
        Accelerometer Saturation: Linear acceleration maxed out.

        Simulates accelerometer hitting limits during high-g maneuvers
        or attack, causing velocity estimation errors.
        """
        data = self._init_data()

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Create saturated acceleration effect on velocity
        t = np.arange(n_attack) * self.dt
        saturated_accel = max_accel * np.ones(n_attack)
        velocity_effect = np.cumsum(saturated_accel * self.dt)

        data.loc[attack_start_idx : attack_end_idx - 1, "vx"] += velocity_effect
        data.loc[attack_start_idx : attack_end_idx - 1, "vy"] += velocity_effect * 0.3
        data.loc[attack_start_idx : attack_end_idx - 1, "vz"] += velocity_effect * 0.2

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Accel_Saturation"

        return data

    # =========================================================================
    # MAGNETOMETER/BAROMETER ATTACKS
    # =========================================================================

    def magnetometer_spoofing(
        self,
        heading_offset: float = 0.5,
        attack_start_ratio: float = 0.3,
        attack_duration: float = 25.0,
    ) -> pd.DataFrame:
        """
        Magnetometer Spoofing: Compass/heading manipulation.

        Attacker uses magnetic field to corrupt heading (yaw) estimate.
        Can cause drone to fly in wrong direction.
        """
        data = self._init_data()

        heading_offset = self._randomize_params(heading_offset)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Gradually introduce heading error
        offset_profile = np.linspace(0, heading_offset, n_attack)

        data.loc[attack_start_idx : attack_end_idx - 1, "psi"] += offset_profile

        # Heading error also affects velocity direction perception
        cos_offset = np.cos(offset_profile)
        sin_offset = np.sin(offset_profile)
        vx = data.loc[attack_start_idx : attack_end_idx - 1, "vx"].values
        vy = data.loc[attack_start_idx : attack_end_idx - 1, "vy"].values

        data.loc[attack_start_idx : attack_end_idx - 1, "vx"] = vx * cos_offset - vy * sin_offset
        data.loc[attack_start_idx : attack_end_idx - 1, "vy"] = vx * sin_offset + vy * cos_offset

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Magnetometer_Spoofing"

        return data

    def barometer_spoofing(
        self,
        altitude_offset: float = 5.0,
        attack_start_ratio: float = 0.35,
        attack_duration: float = 30.0,
    ) -> pd.DataFrame:
        """
        Barometer Spoofing: Altitude manipulation.

        Attacker manipulates pressure sensor to report wrong altitude.
        Can cause dangerous altitude deviations.
        """
        data = self._init_data()

        altitude_offset = self._randomize_params(altitude_offset)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Gradual altitude offset
        offset_profile = np.linspace(0, altitude_offset, n_attack)

        data.loc[attack_start_idx : attack_end_idx - 1, "z"] += offset_profile

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Barometer_Spoofing"

        return data

    # =========================================================================
    # ACTUATOR/CONTROL ATTACKS
    # =========================================================================

    def actuator_stuck(
        self,
        stuck_value_ratio: float = 0.7,
        attack_start_ratio: float = 0.4,
        attack_duration: float = 20.0,
    ) -> pd.DataFrame:
        """
        Actuator Stuck: Motor/servo jammed at fixed value.

        Simulates mechanical failure where actuator freezes.
        Similar to ALFA dataset stuck control surface faults.
        """
        data = self._init_data()

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Freeze thrust at a fixed value
        if "thrust" in data.columns:
            stuck_thrust = data.loc[attack_start_idx, "thrust"] * stuck_value_ratio
            data.loc[attack_start_idx : attack_end_idx - 1, "thrust"] = stuck_thrust

        # Also freeze one torque channel
        if "torque_x" in data.columns:
            stuck_torque = data.loc[attack_start_idx, "torque_x"]
            data.loc[attack_start_idx : attack_end_idx - 1, "torque_x"] = stuck_torque

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Actuator_Stuck"

        return data

    def actuator_degraded(
        self,
        efficiency_factor: float = 0.5,
        attack_start_ratio: float = 0.3,
        attack_duration: float = 40.0,
    ) -> pd.DataFrame:
        """
        Actuator Degraded: Reduced actuator effectiveness.

        Simulates motor wear, propeller damage, or partial failure.
        Actuators produce less force/torque than commanded.
        """
        data = self._init_data()

        efficiency_factor = self._randomize_params(efficiency_factor, 0.2)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Reduce control effectiveness
        if "thrust" in data.columns:
            data.loc[attack_start_idx : attack_end_idx - 1, "thrust"] *= efficiency_factor

        for col in ["torque_x", "torque_y", "torque_z"]:
            if col in data.columns:
                data.loc[attack_start_idx : attack_end_idx - 1, col] *= efficiency_factor

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Actuator_Degraded"

        return data

    def control_hijack(
        self,
        hijack_magnitude: float = 5.0,
        attack_start_ratio: float = 0.45,
        attack_duration: float = 15.0,
    ) -> pd.DataFrame:
        """
        Control Hijack: False control command injection.

        Attacker injects malicious control commands to
        deviate drone from intended trajectory.
        """
        data = self._init_data()

        hijack_magnitude = self._randomize_params(hijack_magnitude)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Inject false control commands
        if "thrust" in data.columns:
            data.loc[attack_start_idx : attack_end_idx - 1, "thrust"] += hijack_magnitude

        if "torque_x" in data.columns:
            data.loc[attack_start_idx : attack_end_idx - 1, "torque_x"] += hijack_magnitude * 0.1
        if "torque_y" in data.columns:
            data.loc[attack_start_idx : attack_end_idx - 1, "torque_y"] += hijack_magnitude * 0.1

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Control_Hijack"

        return data

    def thrust_manipulation(
        self,
        thrust_scale: float = 0.3,
        attack_start_ratio: float = 0.4,
        attack_duration: float = 20.0,
    ) -> pd.DataFrame:
        """
        Thrust Manipulation: Engine power attack.

        Attacker reduces or increases engine power causing
        altitude control issues.
        """
        data = self._init_data()

        thrust_scale = self._randomize_params(thrust_scale, 0.2)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        if "thrust" in data.columns:
            # Sudden thrust reduction
            data.loc[attack_start_idx : attack_end_idx - 1, "thrust"] *= thrust_scale

        # This affects vertical velocity
        t = np.arange(n_attack) * self.dt
        vz_effect = -GRAVITY * (1 - thrust_scale) * t  # Gravity takes over
        data.loc[attack_start_idx : attack_end_idx - 1, "vz"] += vz_effect

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Thrust_Manipulation"

        return data

    # =========================================================================
    # COORDINATED ATTACKS
    # =========================================================================

    def coordinated_gps_imu(
        self,
        gps_drift: float = 5.0,
        imu_bias: float = 0.3,
        attack_start_ratio: float = 0.35,
        attack_duration: float = 25.0,
    ) -> pd.DataFrame:
        """
        Coordinated GPS+IMU Attack: Both sensors compromised.

        Sophisticated attack targeting multiple sensors simultaneously.
        More difficult to detect with single-sensor monitoring.
        """
        data = self._init_data()

        gps_drift = self._randomize_params(gps_drift)
        imu_bias = self._randomize_params(imu_bias)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # GPS gradual drift
        drift_direction = self.rng.standard_normal(3)
        drift_direction = drift_direction / np.linalg.norm(drift_direction)
        drift_profile = np.linspace(0, gps_drift, n_attack)

        data.loc[attack_start_idx : attack_end_idx - 1, "x"] += drift_profile * drift_direction[0]
        data.loc[attack_start_idx : attack_end_idx - 1, "y"] += drift_profile * drift_direction[1]
        data.loc[attack_start_idx : attack_end_idx - 1, "z"] += (
            drift_profile * drift_direction[2] * 0.3
        )

        # IMU bias injection
        velocity_change = imu_bias * self.dt * np.arange(n_attack)
        data.loc[attack_start_idx : attack_end_idx - 1, "vx"] += velocity_change
        data.loc[attack_start_idx : attack_end_idx - 1, "p"] += imu_bias * 0.2
        data.loc[attack_start_idx : attack_end_idx - 1, "q"] += imu_bias * 0.15

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Coordinated_GPS_IMU"

        return data

    def stealthy_coordinated(
        self,
        max_drift: float = 3.0,
        attack_start_ratio: float = 0.3,
        attack_duration: float = 40.0,
    ) -> pd.DataFrame:
        """
        Stealthy Coordinated Attack: Physics-consistent manipulation.

        Attack modifies GPS and IMU in a coordinated way that
        maintains physical consistency (position matches integrated velocity).
        This is the hardest attack to detect.

        Reference: Inspired by "Stealthy Attacks on GPS-Based Systems" literature
        """
        data = self._init_data()

        max_drift = self._randomize_params(max_drift)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Need at least 2 samples to compute smooth trajectory
        if n_attack < 2:
            data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
            data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Stealthy_Coordinated"
            return data

        # Create smooth position deviation that's physically consistent
        t = np.arange(n_attack) * self.dt
        t_max = t[-1] if t[-1] > 0 else 1.0  # Avoid division by zero

        # Use smooth polynomial trajectory for stealth
        # Position: starts at 0, smoothly ramps to max_drift
        t_normalized = t / t_max
        position_drift = max_drift * (3 * t_normalized**2 - 2 * t_normalized**3)  # Smooth step

        # Velocity is derivative of position drift
        velocity_drift = np.gradient(position_drift, self.dt)

        # Apply to GPS position
        data.loc[attack_start_idx : attack_end_idx - 1, "x"] += position_drift
        data.loc[attack_start_idx : attack_end_idx - 1, "y"] += position_drift * 0.5

        # Apply matching velocity change (makes attack physics-consistent)
        data.loc[attack_start_idx : attack_end_idx - 1, "vx"] += velocity_drift
        data.loc[attack_start_idx : attack_end_idx - 1, "vy"] += velocity_drift * 0.5

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Stealthy_Coordinated"

        return data

    # =========================================================================
    # TEMPORAL ATTACKS
    # =========================================================================

    def replay_attack(
        self,
        replay_window: int = 100,
        attack_start_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """
        Replay Attack: Repeat old sensor trajectory.

        Attacker replays previously recorded sensor data.
        Creates temporal inconsistency detectable by physics prediction.
        """
        data = self._init_data()

        n_total = len(data)

        # We need enough data for: source window + gap + attack window
        # Minimum requirement: replay_window + 100 (gap) + replay_window
        min_required = replay_window * 2 + 100
        if n_total < min_required:
            # Data too short for replay attack, return with minimal label
            mid_idx = n_total // 2
            data.loc[mid_idx, "label"] = 1
            data.loc[mid_idx, "attack_type"] = "Replay_Attack"
            return data

        attack_start_idx = int(n_total * attack_start_ratio)

        # Ensure we have enough history for replay source
        min_attack_start = replay_window + 100  # Need this much history
        if attack_start_idx < min_attack_start:
            attack_start_idx = min_attack_start

        # Ensure attack doesn't exceed data bounds
        attack_end_idx = min(attack_start_idx + replay_window, n_total)

        # Calculate replay source indices (data from the past to replay)
        replay_source_start = max(0, attack_start_idx - replay_window - 100)
        replay_source_end = replay_source_start + (attack_end_idx - attack_start_idx)

        # Ensure source end doesn't exceed source start + available data
        if replay_source_end > attack_start_idx - 100:
            replay_source_end = attack_start_idx - 100
            replay_source_start = max(0, replay_source_end - (attack_end_idx - attack_start_idx))

        state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
        for col in state_cols:
            if col in data.columns:
                data.loc[attack_start_idx : attack_end_idx - 1, col] = data.loc[
                    replay_source_start : replay_source_end - 1, col
                ].values

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Replay_Attack"

        return data

    def time_delay_attack(
        self,
        delay_samples: int = 20,
        attack_start_ratio: float = 0.4,
        attack_duration: float = 30.0,
    ) -> pd.DataFrame:
        """
        Time Delay Attack: Delayed sensor readings.

        All sensor data is delayed by a fixed amount,
        causing control system to operate on stale information.
        """
        data = self._init_data()

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        if attack_start_idx < delay_samples:
            attack_start_idx = delay_samples
            attack_end_idx = min(attack_start_idx + n_attack, len(data))

        source_start = attack_start_idx - delay_samples
        source_end = attack_end_idx - delay_samples

        # Delay all sensor readings
        all_state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
        for col in all_state_cols:
            if col in data.columns:
                data.loc[attack_start_idx : attack_end_idx - 1, col] = data.loc[
                    source_start : source_end - 1, col
                ].values

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Time_Delay"

        return data

    def sensor_dropout(
        self,
        dropout_probability: float = 0.3,
        attack_start_ratio: float = 0.4,
        attack_duration: float = 15.0,
        affect_gps: bool = True,
        affect_imu: bool = False,
    ) -> pd.DataFrame:
        """
        Sensor Dropout: Random data loss (jamming attack).

        Simulates GPS/IMU jamming where sensor data is intermittently lost.
        """
        data = self._init_data()

        dropout_probability = self._randomize_params(dropout_probability)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        dropout_mask = self.rng.random(n_attack) < dropout_probability

        if affect_gps:
            for col in ["x", "y", "z"]:
                original = data.loc[attack_start_idx : attack_end_idx - 1, col].values
                data.loc[attack_start_idx : attack_end_idx - 1, col] = np.where(
                    dropout_mask, np.nan, original
                )

        if affect_imu:
            for col in ["p", "q", "r", "vx", "vy", "vz"]:
                if col in data.columns:
                    original = data.loc[attack_start_idx : attack_end_idx - 1, col].values
                    data.loc[attack_start_idx : attack_end_idx - 1, col] = np.where(
                        dropout_mask, np.nan, original
                    )

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Sensor_Dropout"

        return data

    # =========================================================================
    # STEALTH ATTACKS
    # =========================================================================

    def adaptive_attack(
        self,
        initial_magnitude: float = 1.0,
        growth_rate: float = 1.5,
        attack_start_ratio: float = 0.25,
        attack_duration: float = 50.0,
    ) -> pd.DataFrame:
        """
        Adaptive Attack: Magnitude grows over time.

        Starts with small perturbations and gradually increases,
        designed to stay below detection thresholds initially.
        """
        data = self._init_data()

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Exponentially growing attack magnitude
        t = np.arange(n_attack) * self.dt
        magnitude_profile = initial_magnitude * np.exp(growth_rate * t / t[-1])

        # Apply to GPS
        direction = self.rng.standard_normal(3)
        direction = direction / np.linalg.norm(direction)

        data.loc[attack_start_idx : attack_end_idx - 1, "x"] += magnitude_profile * direction[0]
        data.loc[attack_start_idx : attack_end_idx - 1, "y"] += magnitude_profile * direction[1]
        data.loc[attack_start_idx : attack_end_idx - 1, "z"] += (
            magnitude_profile * direction[2] * 0.3
        )

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Adaptive_Attack"

        return data

    def intermittent_attack(
        self,
        on_duration: float = 5.0,
        off_duration: float = 10.0,
        attack_magnitude: float = 4.0,
        attack_start_ratio: float = 0.2,
        attack_duration: float = 60.0,
    ) -> pd.DataFrame:
        """
        Intermittent Attack: On-off pattern.

        Attack activates and deactivates periodically,
        making it harder to confirm as a consistent attack.
        """
        data = self._init_data()

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        on_samples = int(on_duration / self.dt)
        off_samples = int(off_duration / self.dt)
        cycle_length = on_samples + off_samples

        # Create on-off pattern
        attack_active = np.zeros(n_attack, dtype=bool)
        for i in range(0, n_attack, cycle_length):
            end_on = min(i + on_samples, n_attack)
            attack_active[i:end_on] = True

        # Apply attack only when active
        direction = self.rng.standard_normal(3)
        direction = direction / np.linalg.norm(direction)

        offset_x = np.where(attack_active, attack_magnitude * direction[0], 0)
        offset_y = np.where(attack_active, attack_magnitude * direction[1], 0)
        offset_z = np.where(attack_active, attack_magnitude * direction[2] * 0.3, 0)

        data.loc[attack_start_idx : attack_end_idx - 1, "x"] += offset_x
        data.loc[attack_start_idx : attack_end_idx - 1, "y"] += offset_y
        data.loc[attack_start_idx : attack_end_idx - 1, "z"] += offset_z

        # Only mark active periods as attack
        attack_indices = data.index[attack_start_idx:attack_end_idx]
        data.loc[attack_indices[attack_active], "label"] = 1
        data.loc[attack_indices[attack_active], "attack_type"] = "Intermittent_Attack"

        return data

    def slow_ramp_attack(
        self,
        final_offset: float = 8.0,
        attack_start_ratio: float = 0.1,
        attack_duration: float = 80.0,
    ) -> pd.DataFrame:
        """
        Slow Ramp Attack: Very gradual manipulation.

        Extremely slow drift designed to stay below
        any reasonable detection threshold.
        """
        data = self._init_data()

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Very slow linear ramp
        ramp_profile = np.linspace(0, final_offset, n_attack)

        # Random direction
        direction = self.rng.standard_normal(3)
        direction = direction / np.linalg.norm(direction)

        data.loc[attack_start_idx : attack_end_idx - 1, "x"] += ramp_profile * direction[0]
        data.loc[attack_start_idx : attack_end_idx - 1, "y"] += ramp_profile * direction[1]
        data.loc[attack_start_idx : attack_end_idx - 1, "z"] += ramp_profile * direction[2] * 0.3

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Slow_Ramp"

        return data

    def resonance_attack(
        self,
        frequency: float = 5.0,
        amplitude: float = 0.5,
        attack_start_ratio: float = 0.3,
        attack_duration: float = 30.0,
    ) -> pd.DataFrame:
        """
        Resonance Attack: Excite natural frequencies.

        Attacker injects oscillations at drone's natural frequency
        to destabilize control system and amplify vibrations.

        Reference: Control system destabilization attacks
        """
        data = self._init_data()

        frequency = self._randomize_params(frequency)
        amplitude = self._randomize_params(amplitude)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        t = np.arange(n_attack) * self.dt

        # Growing oscillation (resonance builds up)
        envelope = np.linspace(0.1, 1.0, n_attack)
        resonance_p = amplitude * envelope * np.sin(2 * np.pi * frequency * t)
        resonance_q = amplitude * envelope * np.sin(2 * np.pi * frequency * t + np.pi / 3)
        resonance_r = amplitude * envelope * np.sin(2 * np.pi * frequency * t + 2 * np.pi / 3)

        data.loc[attack_start_idx : attack_end_idx - 1, "p"] += resonance_p
        data.loc[attack_start_idx : attack_end_idx - 1, "q"] += resonance_q
        data.loc[attack_start_idx : attack_end_idx - 1, "r"] += resonance_r

        # Resonance also affects attitude
        data.loc[attack_start_idx : attack_end_idx - 1, "phi"] += np.cumsum(resonance_p * self.dt)
        data.loc[attack_start_idx : attack_end_idx - 1, "theta"] += np.cumsum(resonance_q * self.dt)

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "Resonance_Attack"

        return data

    def false_data_injection(
        self,
        attack_vector: Optional[np.ndarray] = None,
        magnitude: float = 2.0,
        attack_start_ratio: float = 0.35,
        attack_duration: float = 25.0,
    ) -> pd.DataFrame:
        """
        False Data Injection (FDI): Optimized stealthy attack.

        Sophisticated attack designed to maximize impact while
        minimizing detectability. Attack vector can be optimized
        to lie in system's null space.

        Reference: Liu et al. "False Data Injection Attacks" (CCS 2009)
        """
        data = self._init_data()

        magnitude = self._randomize_params(magnitude)

        attack_start_idx, attack_end_idx, n_attack = self._get_attack_window(
            attack_start_ratio, attack_duration
        )

        # Default attack vector (can be optimized externally)
        if attack_vector is None:
            # Attack multiple correlated states
            attack_vector = self.rng.standard_normal(6)
            attack_vector = attack_vector / np.linalg.norm(attack_vector)

        # Smooth attack profile
        t = np.arange(n_attack) / n_attack
        profile = magnitude * (3 * t**2 - 2 * t**3)  # Smooth step

        # Apply to position and velocity consistently
        data.loc[attack_start_idx : attack_end_idx - 1, "x"] += profile * attack_vector[0]
        data.loc[attack_start_idx : attack_end_idx - 1, "y"] += profile * attack_vector[1]
        data.loc[attack_start_idx : attack_end_idx - 1, "z"] += profile * attack_vector[2] * 0.3

        # Matching velocity changes (FDI is physics-aware)
        velocity_profile = np.gradient(profile, self.dt)
        data.loc[attack_start_idx : attack_end_idx - 1, "vx"] += velocity_profile * attack_vector[0]
        data.loc[attack_start_idx : attack_end_idx - 1, "vy"] += velocity_profile * attack_vector[1]
        data.loc[attack_start_idx : attack_end_idx - 1, "vz"] += (
            velocity_profile * attack_vector[2] * 0.3
        )

        data.loc[attack_start_idx : attack_end_idx - 1, "label"] = 1
        data.loc[attack_start_idx : attack_end_idx - 1, "attack_type"] = "False_Data_Injection"

        return data

    # =========================================================================
    # GENERATION METHODS
    # =========================================================================

    def generate_all_attacks(
        self, handle_nan: bool = False, nan_method: str = "interpolate"
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all attack types (30 total + clean baseline).

        Args:
            handle_nan: If True, handle NaN values for PINN compatibility
            nan_method: Method for NaN handling ('interpolate', 'ffill', 'drop', 'zero')
        """
        attacks = {
            # GPS Attacks (7)
            "gps_gradual_drift": self.gps_gradual_drift(),
            "gps_sudden_jump": self.gps_sudden_jump(),
            "gps_oscillating": self.gps_oscillating(),
            "gps_meaconing": self.gps_meaconing(),
            "gps_jamming": self.gps_jamming(),
            "gps_freeze": self.gps_freeze(),
            "gps_multipath": self.gps_multipath(),
            # IMU Attacks (7)
            "imu_constant_bias": self.imu_constant_bias(),
            "imu_gradual_drift": self.imu_gradual_drift(),
            "imu_sinusoidal": self.imu_sinusoidal(),
            "imu_noise_injection": self.imu_noise_injection(),
            "imu_scale_factor": self.imu_scale_factor(),
            "gyro_saturation": self.gyro_saturation(),
            "accel_saturation": self.accel_saturation(),
            # Magnetometer/Barometer Attacks (2)
            "magnetometer_spoofing": self.magnetometer_spoofing(),
            "barometer_spoofing": self.barometer_spoofing(),
            # Actuator/Control Attacks (4)
            "actuator_stuck": self.actuator_stuck(),
            "actuator_degraded": self.actuator_degraded(),
            "control_hijack": self.control_hijack(),
            "thrust_manipulation": self.thrust_manipulation(),
            # Coordinated Attacks (2)
            "coordinated_gps_imu": self.coordinated_gps_imu(),
            "stealthy_coordinated": self.stealthy_coordinated(),
            # Temporal Attacks (3)
            "replay_attack": self.replay_attack(),
            "time_delay": self.time_delay_attack(),
            "sensor_dropout": self.sensor_dropout(),
            # Stealth/Advanced Attacks (5)
            "adaptive_attack": self.adaptive_attack(),
            "intermittent_attack": self.intermittent_attack(),
            "slow_ramp": self.slow_ramp_attack(),
            "resonance_attack": self.resonance_attack(),
            "false_data_injection": self.false_data_injection(),
            # Clean baseline
            "clean": self._init_data(),
        }

        # Handle NaN values if requested (for PINN compatibility)
        if handle_nan:
            for name, data in attacks.items():
                attacks[name] = self.handle_nan_values(data, method=nan_method)

        return attacks

    def generate_pinn_ready_dataset(
        self,
        attacks_to_include: Optional[List[str]] = None,
        nan_method: str = "interpolate",
    ) -> pd.DataFrame:
        """
        Generate a combined dataset ready for PINN training/evaluation.

        Automatically handles NaN values and ensures all required columns exist.

        Args:
            attacks_to_include: List of attack names to include (None = all)
            nan_method: Method for NaN handling

        Returns:
            Combined DataFrame with columns:
            [timestamp, x, y, z, phi, theta, psi, p, q, r, vx, vy, vz,
             thrust, torque_x, torque_y, torque_z, label, attack_type]
        """
        attacks = self.generate_all_attacks(handle_nan=True, nan_method=nan_method)

        if attacks_to_include is not None:
            attacks = {k: v for k, v in attacks.items() if k in attacks_to_include}

        # Combine all attacks
        all_data = []
        for name, data in attacks.items():
            all_data.append(data)

        combined = pd.concat(all_data, ignore_index=True)

        # Ensure standard column order for PINN
        pinn_cols = [
            "timestamp",
            "x",
            "y",
            "z",
            "phi",
            "theta",
            "psi",
            "p",
            "q",
            "r",
            "vx",
            "vy",
            "vz",
            "thrust",
            "torque_x",
            "torque_y",
            "torque_z",
            "label",
            "attack_type",
        ]

        # Keep only columns that exist
        available_cols = [c for c in pinn_cols if c in combined.columns]
        combined = combined[available_cols]

        return combined

    def generate_attack_variants(self, attack_name: str, n_variants: int = 5) -> List[pd.DataFrame]:
        """Generate multiple variants of a specific attack with randomized parameters."""
        original_randomize = self.randomize
        self.randomize = True

        variants = []
        attack_method = getattr(self, attack_name, None)

        if attack_method is None:
            raise ValueError(f"Unknown attack type: {attack_name}")

        for i in range(n_variants):
            self.rng = np.random.RandomState(42 + i)
            variants.append(attack_method())

        self.randomize = original_randomize
        return variants

    def generate_combined_dataset(self, attacks_per_type: int = 3) -> pd.DataFrame:
        """Generate a combined dataset with multiple attack instances."""
        all_data = []

        attack_methods = [
            # GPS Attacks (7)
            "gps_gradual_drift",
            "gps_sudden_jump",
            "gps_oscillating",
            "gps_meaconing",
            "gps_freeze",
            "gps_multipath",
            # IMU Attacks (7)
            "imu_constant_bias",
            "imu_gradual_drift",
            "imu_sinusoidal",
            "imu_noise_injection",
            "imu_scale_factor",
            "gyro_saturation",
            "accel_saturation",
            # Magnetometer/Barometer (2)
            "magnetometer_spoofing",
            "barometer_spoofing",
            # Actuator/Control (4)
            "actuator_stuck",
            "actuator_degraded",
            "control_hijack",
            "thrust_manipulation",
            # Coordinated (2)
            "coordinated_gps_imu",
            "stealthy_coordinated",
            # Temporal (3)
            "replay_attack",
            "time_delay_attack",
            "sensor_dropout",
            # Stealth/Advanced (4)
            "adaptive_attack",
            "intermittent_attack",
            "slow_ramp_attack",
            "resonance_attack",
            "false_data_injection",
        ]

        # Add clean samples
        clean_data = self.clean_data.copy()
        clean_data["label"] = 0
        clean_data["attack_type"] = "Normal"
        all_data.append(clean_data)

        # Add attack variants
        for attack_name in attack_methods:
            variants = self.generate_attack_variants(attack_name, attacks_per_type)
            all_data.extend(variants)

        combined = pd.concat(all_data, ignore_index=True)
        return combined


def load_euroc_data(euroc_path: Path) -> pd.DataFrame:
    """Load preprocessed EuRoC data."""
    csv_files = list(euroc_path.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {euroc_path}")

    df = pd.read_csv(csv_files[0])
    print(f"Loaded {len(df)} samples from {csv_files[0].name}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive synthetic attacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Attack Types:
  GPS: gradual_drift, sudden_jump, oscillating, meaconing, jamming, freeze
  IMU: constant_bias, gradual_drift, sinusoidal, noise_injection, scale_factor
  Coordinated: gps_imu, stealthy_coordinated
  Temporal: replay, time_delay, sensor_dropout
  Stealth: adaptive, intermittent, slow_ramp
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to clean EuRoC data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/attack_datasets/synthetic",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Enable parameter randomization for robustness",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Number of variants per attack type (for robustness)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Generate a single combined dataset with all attacks",
    )
    parser.add_argument(
        "--pinn-ready",
        action="store_true",
        help="Generate PINN-compatible dataset (handles NaN, standard columns)",
    )
    parser.add_argument(
        "--handle-nan",
        action="store_true",
        help="Handle NaN values created by jamming/dropout attacks",
    )
    parser.add_argument(
        "--nan-method",
        type=str,
        default="interpolate",
        choices=["interpolate", "ffill", "drop", "zero"],
        help="Method for handling NaN values (default: interpolate)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Comprehensive Synthetic Attack Generator")
    print("=" * 70)
    print(f"Randomization: {'ENABLED' if args.randomize else 'DISABLED'}")
    print(f"Variants per attack: {args.variants}")

    # Load clean data
    print("\nLoading clean EuRoC data...")
    clean_data = load_euroc_data(input_path)

    if "label" not in clean_data.columns:
        clean_data["label"] = 0
    if "attack_type" not in clean_data.columns:
        clean_data["attack_type"] = "Normal"

    # Generate attacks
    print("\nGenerating attacks...")
    generator = SyntheticAttackGenerator(clean_data, seed=args.seed, randomize=args.randomize)

    if args.pinn_ready:
        # Generate PINN-compatible dataset
        print("\nGenerating PINN-ready dataset...")
        combined = generator.generate_pinn_ready_dataset(nan_method=args.nan_method)
        output_file = output_path / "pinn_ready_attacks.csv"
        combined.to_csv(output_file, index=False)

        attack_counts = combined["attack_type"].value_counts()
        print(f"\nPINN-ready dataset: {len(combined)} samples")
        print(f"NaN handling: {args.nan_method}")
        print(f"Columns: {list(combined.columns)}")
        print("\nAttack distribution:")
        for attack_type, count in attack_counts.items():
            print(f"  {attack_type}: {count}")

    elif args.combined:
        # Generate combined dataset
        combined = generator.generate_combined_dataset(attacks_per_type=args.variants)

        # Handle NaN if requested
        if args.handle_nan:
            combined = generator.handle_nan_values(combined, method=args.nan_method)

        output_file = output_path / "combined_attacks.csv"
        combined.to_csv(output_file, index=False)

        # Summary statistics
        attack_counts = combined["attack_type"].value_counts()
        print(f"\nCombined dataset: {len(combined)} samples")
        print("\nAttack distribution:")
        for attack_type, count in attack_counts.items():
            print(f"  {attack_type}: {count}")
    else:
        # Generate individual attack files
        attacks = generator.generate_all_attacks(
            handle_nan=args.handle_nan, nan_method=args.nan_method
        )

        for attack_name, attack_data in attacks.items():
            output_file = output_path / f"{attack_name}.csv"
            attack_data.to_csv(output_file, index=False)

            meta = {
                "attack_type": attack_name,
                "n_samples": len(attack_data),
                "n_attack_samples": int(attack_data["label"].sum()),
                "attack_ratio": float(attack_data["label"].mean()),
                "randomized": args.randomize,
            }

            meta_file = output_path / f"{attack_name}_meta.json"
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=2)

            print(
                f"  {attack_name}: {len(attack_data)} samples "
                f"({meta['attack_ratio']*100:.1f}% attack)"
            )

    print("\n" + "=" * 70)
    print("Attack generation complete!")
    print(f"Output: {output_path.absolute()}")
    print("=" * 70)
    print("\nAttack categories generated:")
    print("  - GPS Attacks: 7 types")
    print("  - IMU Attacks: 7 types")
    print("  - Magnetometer/Barometer: 2 types")
    print("  - Actuator/Control: 4 types")
    print("  - Coordinated Attacks: 2 types")
    print("  - Temporal Attacks: 3 types")
    print("  - Stealth/Advanced: 5 types")
    print("  - Total: 30 attack types + clean baseline")
    print("\nNext steps:")
    print("1. Train PINN: python scripts/security/train_detector.py")
    print("2. Evaluate: python scripts/security/evaluate_detector.py")


if __name__ == "__main__":
    main()
