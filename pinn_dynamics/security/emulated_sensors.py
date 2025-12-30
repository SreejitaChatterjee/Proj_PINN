"""
Emulated Sensors for CPU-Friendly Spoofing Detection.

Since EuRoC lacks barometer and magnetometer, we emulate them from available
state data with realistic noise models. This enables L4 fusion testing without
requiring new datasets.

CPU Impact: O(N) single pass, no heavy models.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class BarometerConfig:
    """Barometer emulation parameters."""
    alpha: float = 0.98          # IIR smoothing factor
    noise_std: float = 0.05      # Measurement noise (m)
    drift_rate: float = 0.001    # Drift per second (m/s)
    quantization: float = 0.01   # ADC quantization (m)
    delay_samples: int = 2       # Sensor latency (samples)
    temp_coefficient: float = 0.0065  # ISA lapse rate (K/m)


@dataclass
class MagnetometerConfig:
    """Magnetometer emulation parameters."""
    hard_iron: np.ndarray = None      # Hard iron offset (3,)
    soft_iron: np.ndarray = None      # Soft iron matrix (3,3)
    noise_std: float = 0.02           # Measurement noise (normalized)
    bias_drift_rate: float = 0.0001   # Bias drift per second
    declination: float = 0.0          # Magnetic declination (rad)
    inclination: float = 1.05         # Magnetic inclination (rad, ~60 deg)

    def __post_init__(self):
        if self.hard_iron is None:
            self.hard_iron = np.array([0.02, -0.01, 0.015])
        if self.soft_iron is None:
            # Slight axis scaling asymmetry
            self.soft_iron = np.array([
                [1.02, 0.01, 0.00],
                [0.01, 0.98, 0.01],
                [0.00, 0.01, 1.00]
            ])


class BarometerEmulator:
    """
    Emulate barometric altitude from ground truth z.

    Model:
    1. IIR low-pass filter (sensor bandwidth)
    2. Slow drift (temperature/pressure changes)
    3. Gaussian noise
    4. Quantization
    5. Delay
    """

    def __init__(self, config: Optional[BarometerConfig] = None, dt: float = 0.005):
        self.config = config or BarometerConfig()
        self.dt = dt
        self.state = None
        self.drift = 0.0
        self.buffer = []

    def reset(self):
        """Reset emulator state."""
        self.state = None
        self.drift = 0.0
        self.buffer = []

    def emulate(self, z: np.ndarray) -> np.ndarray:
        """
        Emulate barometric altitude from true z.

        Args:
            z: True altitude array (N,)

        Returns:
            baro_z: Emulated barometric altitude (N,)
        """
        N = len(z)
        baro = np.zeros(N)
        cfg = self.config

        # Initialize
        if self.state is None:
            self.state = z[0]

        for t in range(N):
            # IIR filter (sensor bandwidth limitation)
            self.state = cfg.alpha * self.state + (1 - cfg.alpha) * z[t]

            # Slow drift (temperature-induced)
            self.drift += cfg.drift_rate * self.dt * np.random.randn()

            # Measurement noise (altitude-dependent - higher altitude = more noise)
            altitude_factor = 1.0 + 0.1 * abs(z[t]) / 10.0  # Scale with altitude
            noise = cfg.noise_std * altitude_factor * np.random.randn()

            # Combine
            raw = self.state + self.drift + noise

            # Quantization
            baro[t] = np.round(raw / cfg.quantization) * cfg.quantization

        # Apply delay
        if cfg.delay_samples > 0:
            baro = np.concatenate([
                np.full(cfg.delay_samples, baro[0]),
                baro[:-cfg.delay_samples]
            ])

        return baro

    @staticmethod
    def altitude_to_pressure(z: np.ndarray, P0: float = 101325.0) -> np.ndarray:
        """Convert altitude to pressure using ISA model."""
        # ISA: P = P0 * (1 - L*h/T0)^(g*M/(R*L))
        # Simplified: P = P0 * (1 - 2.2558e-5 * h)^5.2559
        return P0 * (1 - 2.2558e-5 * z) ** 5.2559

    @staticmethod
    def pressure_to_altitude(P: np.ndarray, P0: float = 101325.0) -> np.ndarray:
        """Convert pressure to altitude using ISA model."""
        return 44330.0 * (1 - (P / P0) ** 0.19029)


class MagnetometerEmulator:
    """
    Emulate magnetometer readings from attitude.

    Model:
    1. Earth's magnetic field in NED frame
    2. Rotate to body frame using attitude
    3. Apply hard/soft iron distortions
    4. Add noise and bias drift
    """

    def __init__(self, config: Optional[MagnetometerConfig] = None, dt: float = 0.005):
        self.config = config or MagnetometerConfig()
        self.dt = dt
        self.bias = np.zeros(3)

    def reset(self):
        """Reset emulator state."""
        self.bias = np.zeros(3)

    def _rotation_matrix(self, phi: float, theta: float, psi: float) -> np.ndarray:
        """Compute rotation matrix from Euler angles (ZYX convention)."""
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        R = np.array([
            [cth*cpsi, cth*spsi, -sth],
            [sphi*sth*cpsi - cphi*spsi, sphi*sth*spsi + cphi*cpsi, sphi*cth],
            [cphi*sth*cpsi + sphi*spsi, cphi*sth*spsi - sphi*cpsi, cphi*cth]
        ])
        return R

    def emulate(self, phi: np.ndarray, theta: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """
        Emulate magnetometer readings from attitude.

        Args:
            phi: Roll angle array (N,)
            theta: Pitch angle array (N,)
            psi: Yaw angle array (N,)

        Returns:
            mag: Magnetometer readings (N, 3) - normalized
        """
        N = len(phi)
        mag = np.zeros((N, 3))
        cfg = self.config

        # Earth's magnetic field in NED (normalized, with inclination)
        # Horizontal component points to magnetic north
        m_ned = np.array([
            np.cos(cfg.inclination) * np.cos(cfg.declination),
            np.cos(cfg.inclination) * np.sin(cfg.declination),
            np.sin(cfg.inclination)
        ])

        for t in range(N):
            # Rotate magnetic field to body frame
            R = self._rotation_matrix(phi[t], theta[t], psi[t])
            m_body = R @ m_ned

            # Apply soft iron distortion
            m_distorted = cfg.soft_iron @ m_body

            # Apply hard iron offset
            m_distorted += cfg.hard_iron

            # Bias drift
            self.bias += cfg.bias_drift_rate * self.dt * np.random.randn(3)
            m_distorted += self.bias

            # Measurement noise
            m_distorted += cfg.noise_std * np.random.randn(3)

            # Normalize (magnetometer typically outputs normalized)
            norm = np.linalg.norm(m_distorted)
            if norm > 1e-6:
                mag[t] = m_distorted / norm
            else:
                mag[t] = m_distorted

        return mag

    def compute_heading(self, mag: np.ndarray, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute tilt-compensated heading from magnetometer.

        Args:
            mag: Magnetometer readings (N, 3)
            phi: Roll angles (N,)
            theta: Pitch angles (N,)

        Returns:
            heading: Magnetic heading (N,) in radians
        """
        N = len(mag)
        heading = np.zeros(N)

        for t in range(N):
            # Tilt compensation
            cphi, sphi = np.cos(phi[t]), np.sin(phi[t])
            cth, sth = np.cos(theta[t]), np.sin(theta[t])

            mx, my, mz = mag[t]

            # Compensate for tilt
            mx_h = mx * cth + my * sphi * sth + mz * cphi * sth
            my_h = my * cphi - mz * sphi

            # Compute heading
            heading[t] = np.arctan2(-my_h, mx_h)

        return heading


class SensorEmulationPipeline:
    """
    Complete sensor emulation pipeline for EuRoC data.

    Adds emulated barometer and magnetometer to existing IMU + position data.
    """

    def __init__(
        self,
        baro_config: Optional[BarometerConfig] = None,
        mag_config: Optional[MagnetometerConfig] = None,
        dt: float = 0.005
    ):
        self.baro = BarometerEmulator(baro_config, dt)
        self.mag = MagnetometerEmulator(mag_config, dt)
        self.dt = dt

    def emulate(self, df) -> dict:
        """
        Add emulated sensors to dataframe.

        Args:
            df: DataFrame with columns [x,y,z,phi,theta,psi,...]

        Returns:
            dict with emulated sensor arrays
        """
        self.baro.reset()
        self.mag.reset()

        # Emulate barometer
        baro_z = self.baro.emulate(df['z'].values)

        # Emulate magnetometer
        mag = self.mag.emulate(
            df['phi'].values,
            df['theta'].values,
            df['psi'].values
        )

        # Compute derived quantities
        heading = self.mag.compute_heading(
            mag,
            df['phi'].values,
            df['theta'].values
        )

        return {
            'baro_z': baro_z,
            'mag_x': mag[:, 0],
            'mag_y': mag[:, 1],
            'mag_z': mag[:, 2],
            'mag_heading': heading,
            # Integrity metrics
            'baro_pos_diff': np.abs(baro_z - df['z'].values),
            'heading_yaw_diff': np.abs(np.unwrap(heading - df['psi'].values))
        }


def inject_sensor_attack(
    emulated: dict,
    attack_type: str,
    start_idx: int,
    duration: int,
    magnitude: float = 1.0
) -> Tuple[dict, np.ndarray]:
    """
    Inject attack into emulated sensors.

    Args:
        emulated: Dict from SensorEmulationPipeline.emulate()
        attack_type: 'baro_drift', 'baro_jump', 'mag_bias', 'mag_rotation'
        start_idx: Attack start index
        duration: Attack duration in samples
        magnitude: Attack magnitude

    Returns:
        attacked: Modified sensor dict
        labels: Binary attack labels (N,)
    """
    attacked = {k: v.copy() for k, v in emulated.items()}
    N = len(attacked['baro_z'])
    labels = np.zeros(N)

    end_idx = min(start_idx + duration, N)
    labels[start_idx:end_idx] = 1

    if attack_type == 'baro_drift':
        # Gradual barometric altitude drift
        ramp = np.linspace(0, magnitude, end_idx - start_idx)
        attacked['baro_z'][start_idx:end_idx] += ramp

    elif attack_type == 'baro_jump':
        # Sudden barometric altitude offset
        attacked['baro_z'][start_idx:end_idx] += magnitude

    elif attack_type == 'mag_bias':
        # Magnetometer hard iron injection
        attacked['mag_x'][start_idx:end_idx] += magnitude * 0.1
        attacked['mag_y'][start_idx:end_idx] += magnitude * 0.05

    elif attack_type == 'mag_rotation':
        # Heading spoofing via magnetic field rotation
        rotation = magnitude * np.pi / 180  # degrees to radians
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        mx = attacked['mag_x'][start_idx:end_idx].copy()
        my = attacked['mag_y'][start_idx:end_idx].copy()
        attacked['mag_x'][start_idx:end_idx] = cos_r * mx - sin_r * my
        attacked['mag_y'][start_idx:end_idx] = sin_r * mx + cos_r * my

    # Recompute integrity metrics
    attacked['baro_pos_diff'] = np.abs(attacked['baro_z'] - emulated['baro_z'] + emulated['baro_pos_diff'])

    return attacked, labels
