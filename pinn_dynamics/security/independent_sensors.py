"""
Independent Sensor Generation (Non-Circular).

Generates baro/mag proxies WITHOUT using ground truth pose/attitude.
This breaks the circular dependency that plagued emulated_sensors.py.

Key changes:
- Baro: Integrate vertical velocity + slow drift anchors
- Mag: Geomagnetic model + environmental noise (not derived from attitude)

CPU Impact: O(N) single pass.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class IndependentBaroConfig:
    """Independent barometer configuration."""
    # Integration from IMU
    accel_noise_std: float = 0.1       # m/s^2 noise on vertical accel
    velocity_noise_std: float = 0.02    # m/s noise on integrated velocity

    # Anchor correction (prevents unbounded drift)
    anchor_tau: float = 30.0           # Time constant for drift correction (s)
    anchor_noise: float = 0.5          # Noise on anchor measurements (m)

    # Sensor characteristics
    quantization: float = 0.01         # ADC quantization (m)
    bias: float = 0.0                  # Initial bias (m)
    drift_rate: float = 0.001          # Drift per second (m/s)


@dataclass
class IndependentMagConfig:
    """Independent magnetometer configuration."""
    # Geomagnetic model (WMM approximation for mid-latitudes)
    field_strength: float = 50000.0    # nT (typical Earth field)
    inclination: float = 60.0          # degrees (typical mid-latitude)
    declination: float = 0.0           # degrees (local variation)

    # Noise model
    hard_iron: np.ndarray = None       # Hard iron offset (nT)
    noise_std: float = 100.0           # Measurement noise (nT)

    # Environmental disturbances (not from attitude!)
    disturbance_freq: float = 0.1      # Hz - slow environmental variation
    disturbance_amp: float = 500.0     # nT - amplitude

    def __post_init__(self):
        if self.hard_iron is None:
            self.hard_iron = np.array([200.0, -150.0, 100.0])  # nT


class IndependentBarometer:
    """
    Generate barometric altitude WITHOUT using ground truth z.

    Method:
    1. Integrate IMU vertical acceleration to get velocity
    2. Integrate velocity to get position
    3. Apply slow anchor corrections (simulates pressure-based reference)
    4. Add realistic sensor noise and drift

    This is INDEPENDENT because:
    - Uses IMU acceleration (real sensor)
    - Does NOT use ground truth position
    - Anchor is noisy and slow (like real pressure reference)
    """

    def __init__(self, config: Optional[IndependentBaroConfig] = None, dt: float = 0.005):
        self.config = config or IndependentBaroConfig()
        self.dt = dt
        self.reset()

    def reset(self):
        """Reset integrator state."""
        self.velocity_z = 0.0
        self.position_z = 0.0
        self.bias = self.config.bias
        self.anchor_z = 0.0

    def generate(
        self,
        accel_z: np.ndarray,
        initial_z: float = 0.0,
        anchor_updates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate barometric altitude from IMU acceleration.

        Args:
            accel_z: Vertical acceleration from IMU (N,) - in body frame, corrected for gravity
            initial_z: Initial altitude estimate
            anchor_updates: Optional sparse anchor measurements (like GPS altitude)

        Returns:
            baro_z: Estimated barometric altitude (N,)
        """
        N = len(accel_z)
        baro_z = np.zeros(N)
        cfg = self.config

        # Initialize
        self.velocity_z = 0.0
        self.position_z = initial_z
        self.anchor_z = initial_z

        for t in range(N):
            # Add noise to acceleration
            noisy_accel = accel_z[t] + cfg.accel_noise_std * np.random.randn()

            # Integrate acceleration to velocity
            self.velocity_z += noisy_accel * self.dt
            self.velocity_z += cfg.velocity_noise_std * np.random.randn() * np.sqrt(self.dt)

            # Integrate velocity to position
            self.position_z += self.velocity_z * self.dt

            # Slow anchor correction (prevents unbounded drift)
            # This simulates slow pressure reference without using ground truth
            if anchor_updates is not None and not np.isnan(anchor_updates[t]):
                self.anchor_z = anchor_updates[t] + cfg.anchor_noise * np.random.randn()

            # Apply anchor as slow correction
            alpha = self.dt / cfg.anchor_tau
            self.position_z = (1 - alpha) * self.position_z + alpha * self.anchor_z

            # Add sensor drift
            self.bias += cfg.drift_rate * self.dt * np.random.randn()

            # Quantize output
            raw = self.position_z + self.bias
            baro_z[t] = np.round(raw / cfg.quantization) * cfg.quantization

        return baro_z


class IndependentMagnetometer:
    """
    Generate magnetometer readings WITHOUT using ground truth attitude.

    Method:
    1. Use geomagnetic field model (constant + slow variation)
    2. Add environmental disturbances (time-varying, NOT attitude-derived)
    3. Add hard/soft iron distortions (constant offsets)
    4. Add sensor noise

    This is INDEPENDENT because:
    - Geomagnetic field is constant/slowly varying (not from attitude)
    - Environmental disturbances are random (not from attitude)
    - Only real dependency would be rotation (which we DON'T use)
    """

    def __init__(self, config: Optional[IndependentMagConfig] = None, dt: float = 0.005):
        self.config = config or IndependentMagConfig()
        self.dt = dt

    def generate(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate magnetometer readings independent of attitude.

        Args:
            N: Number of samples

        Returns:
            mag: Magnetometer readings (N, 3) in nT
            heading: Derived heading estimate (N,) - noisy
        """
        cfg = self.config
        mag = np.zeros((N, 3))
        t = np.arange(N) * self.dt

        # Base geomagnetic field in NED frame
        inc_rad = np.radians(cfg.inclination)
        dec_rad = np.radians(cfg.declination)

        # Horizontal component
        H = cfg.field_strength * np.cos(inc_rad)
        # Vertical component
        Z = cfg.field_strength * np.sin(inc_rad)

        # Base field
        base_field = np.array([
            H * np.cos(dec_rad),  # North
            H * np.sin(dec_rad),  # East
            Z                      # Down
        ])

        for i in range(N):
            # Start with base field
            field = base_field.copy()

            # Add slow environmental disturbance (simulates moving near metal, etc.)
            # This is NOT derived from attitude - it's environmental noise
            disturbance = cfg.disturbance_amp * np.array([
                np.sin(2 * np.pi * cfg.disturbance_freq * t[i]),
                np.cos(2 * np.pi * cfg.disturbance_freq * t[i] + 0.5),
                np.sin(2 * np.pi * cfg.disturbance_freq * t[i] * 0.7)
            ])
            field += disturbance

            # Add hard iron offset (constant)
            field += cfg.hard_iron

            # Add measurement noise
            field += cfg.noise_std * np.random.randn(3)

            mag[i] = field

        # Compute noisy heading (from horizontal components only)
        heading = np.arctan2(mag[:, 1], mag[:, 0])

        return mag, heading


class IndependentSensorFusion:
    """
    Generate independent sensor suite for testing.

    Key principle: None of these sensors use ground truth pose/attitude.
    They are noisy, drifty, but INDEPENDENT.
    """

    def __init__(self, dt: float = 0.005):
        self.baro = IndependentBarometer(dt=dt)
        self.mag = IndependentMagnetometer(dt=dt)
        self.dt = dt

    def generate(
        self,
        imu_accel_z: np.ndarray,
        initial_z: float = 0.0
    ) -> dict:
        """
        Generate independent sensor suite.

        Args:
            imu_accel_z: Vertical acceleration from IMU (real sensor)
            initial_z: Initial altitude

        Returns:
            Dict with independent sensor data
        """
        N = len(imu_accel_z)

        # Generate baro from IMU integration (not ground truth)
        baro_z = self.baro.generate(imu_accel_z, initial_z)

        # Generate mag independent of attitude
        mag, heading = self.mag.generate(N)

        return {
            'baro_z': baro_z,
            'mag': mag,
            'mag_heading': heading,
            'source': 'independent_generation',
            'warning': 'These are noisy estimates, not ground truth'
        }


def validate_independence(df, independent_sensors: dict) -> dict:
    """
    Validate that generated sensors are truly independent of ground truth.

    If correlation with ground truth is low, sensors are independent.
    """
    results = {}

    # Check baro vs ground truth z
    if 'z' in df.columns:
        gt_z = df['z'].values
        baro_z = independent_sensors['baro_z']

        # Compute correlation
        corr = np.corrcoef(gt_z[:len(baro_z)], baro_z)[0, 1]
        results['baro_gt_correlation'] = corr
        results['baro_independent'] = abs(corr) < 0.9  # Should be somewhat correlated but not perfectly

    # Check mag heading vs ground truth yaw
    if 'psi' in df.columns:
        gt_psi = df['psi'].values
        mag_heading = independent_sensors['mag_heading']

        # Normalize angles
        gt_psi_norm = np.mod(gt_psi + np.pi, 2*np.pi) - np.pi
        mag_heading_norm = np.mod(mag_heading + np.pi, 2*np.pi) - np.pi

        corr = np.corrcoef(gt_psi_norm[:len(mag_heading)], mag_heading_norm)[0, 1]
        results['mag_gt_correlation'] = corr
        results['mag_independent'] = abs(corr) < 0.5  # Should have low correlation

    return results


if __name__ == "__main__":
    # Test independence
    N = 10000

    # Simulate IMU vertical acceleration (real sensor data would go here)
    # This includes gravity-corrected acceleration
    imu_accel_z = 0.1 * np.sin(2 * np.pi * 0.1 * np.arange(N) * 0.005)  # Slight vertical motion

    fusion = IndependentSensorFusion(dt=0.005)
    sensors = fusion.generate(imu_accel_z, initial_z=0.0)

    print("Independent Sensor Generation Test")
    print("="*50)
    print(f"Baro altitude range: [{sensors['baro_z'].min():.2f}, {sensors['baro_z'].max():.2f}] m")
    print(f"Mag field range: [{sensors['mag'].min():.0f}, {sensors['mag'].max():.0f}] nT")
    print(f"Heading range: [{np.degrees(sensors['mag_heading'].min()):.1f}, "
          f"{np.degrees(sensors['mag_heading'].max()):.1f}] deg")
    print()
    print("Key: These sensors are NOT derived from ground truth pose/attitude.")
    print("They use IMU integration + noise models, breaking circular dependency.")
