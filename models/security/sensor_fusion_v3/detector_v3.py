"""
Sensor Fusion Detector V3 - Comprehensive Attack Detection

Based on deep analysis of attack signatures:

DETECTABLE (have clear signals):
1. gyro_saturation    - Direct p,q,r threshold (|rate| > 4 rad/s)
2. control_hijack     - Thrust anomaly (|thrust - expected| > 2)
3. time_delay         - Attitude integration mismatch
4. intermittent       - pos_vel consistency (168x ratio!)
5. resonance          - Angular rate periodicity

TRULY HARD (maintain consistency):
6. stealthy_coordinated - Coordinates position AND velocity drift
7. false_data_injection - Small coordinated drift
8. sensor_dropout       - Interpolation hides it
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class DetectorV3Config:
    dt: float = 0.005

    # Physics thresholds (from training data)
    pos_vel_threshold: float = 0.02
    kinematic_threshold: float = 0.004

    # Direct value thresholds
    gyro_rate_max: float = 3.0  # rad/s - rates above this are suspicious
    thrust_nominal: float = 10.0  # N - expected hover thrust
    thrust_deviation_max: float = 3.0  # N - max deviation from nominal


class ComprehensiveDetector:
    """
    Multi-signal attack detector.

    Combines:
    1. Physics consistency (handles 67% of attacks)
    2. Direct value anomalies (handles gyro saturation, control hijack)
    3. Integration consistency (handles time delay)
    4. Statistical drift detection (helps with stealthy attacks)
    """

    def __init__(self, config: DetectorV3Config = None):
        self.config = config or DetectorV3Config()

    def detect(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run all detection algorithms.

        Args:
            data: [N, 16] array - state(12) + control(4)

        Returns:
            Dict with predictions and individual detector outputs
        """
        N = len(data)
        dt = self.config.dt

        # Extract components
        pos = data[:, 0:3]      # x, y, z
        att = data[:, 3:6]      # phi, theta, psi
        rate = data[:, 6:9]     # p, q, r
        vel = data[:, 9:12]     # vx, vy, vz
        thrust = data[:, 12]    # thrust

        results = {}

        # =====================================================================
        # 1. PHYSICS CONSISTENCY (from v2 - handles 67% of attacks)
        # =====================================================================

        # Position-velocity consistency
        pos_deriv = (pos[1:] - pos[:-1]) / dt
        pos_vel_diff = np.linalg.norm(pos_deriv - vel[1:], axis=1)
        physics_pred = (pos_vel_diff > self.config.pos_vel_threshold).astype(int)

        # Kinematic consistency
        window = 20
        kinematic_diff = np.zeros(N - 1)
        for i in range(window, N - 1):
            vel_integral = vel[i-window+1:i+1].sum(axis=0) * dt
            pos_change = pos[i+1] - pos[i-window+1]
            kinematic_diff[i] = np.linalg.norm(vel_integral - pos_change)
        kinematic_pred = (kinematic_diff > self.config.kinematic_threshold).astype(int)

        results['physics'] = np.maximum(physics_pred, kinematic_pred)

        # =====================================================================
        # 2. GYRO SATURATION DETECTOR
        # =====================================================================
        # Direct check: are angular rates above physical limits?

        p, q, r = rate[:, 0], rate[:, 1], rate[:, 2]

        # Rate magnitude check
        rate_magnitude = np.sqrt(p**2 + q**2 + r**2)
        gyro_saturated = (rate_magnitude > self.config.gyro_rate_max).astype(int)

        # Rate jump check (sudden changes)
        rate_jump = np.zeros(N)
        rate_jump[1:] = np.abs(np.diff(rate_magnitude)) > 1.0  # 1 rad/s jump

        results['gyro_saturation'] = np.maximum(gyro_saturated, rate_jump)[:-1]

        # =====================================================================
        # 3. CONTROL HIJACK DETECTOR
        # =====================================================================
        # Check if thrust deviates significantly from expected

        thrust_deviation = np.abs(thrust - self.config.thrust_nominal)
        thrust_anomaly = (thrust_deviation > self.config.thrust_deviation_max).astype(int)

        results['control_hijack'] = thrust_anomaly[:-1]

        # =====================================================================
        # 4. TIME DELAY DETECTOR
        # =====================================================================
        # Attitude integration check: integral(rate) should match attitude change

        phi, theta, psi = att[:, 0], att[:, 1], att[:, 2]

        # Simplified attitude integration (valid for small angles)
        att_deriv = np.zeros((N-1, 3))
        att_deriv[:, 0] = (phi[1:] - phi[:-1]) / dt
        att_deriv[:, 1] = (theta[1:] - theta[:-1]) / dt
        att_deriv[:, 2] = (psi[1:] - psi[:-1]) / dt

        att_rate_diff = np.linalg.norm(att_deriv - rate[1:], axis=1)

        # Use adaptive threshold based on local statistics
        window = 100
        att_rate_threshold = np.zeros(N - 1)
        for i in range(window, N - 1):
            local_std = np.std(att_rate_diff[max(0, i-window):i])
            att_rate_threshold[i] = 3 * local_std if local_std > 0 else 1.0

        time_delay_pred = (att_rate_diff > np.maximum(att_rate_threshold, 0.5)).astype(int)
        results['time_delay'] = time_delay_pred

        # =====================================================================
        # 5. RESONANCE DETECTOR (Frequency analysis)
        # =====================================================================
        # Look for unusual periodicity in angular rates

        resonance_pred = np.zeros(N - 1)
        fft_window = 128

        for i in range(fft_window, N - 1, fft_window // 2):
            for rate_signal in [p, q, r]:
                segment = rate_signal[i-fft_window:i]

                # Detrend
                segment = segment - np.mean(segment)

                # FFT
                fft_mag = np.abs(np.fft.rfft(segment))
                fft_mag[0] = 0  # Remove DC

                if fft_mag.sum() > 0:
                    # Check if single frequency dominates
                    peak_ratio = fft_mag.max() / (fft_mag.sum() + 1e-10)
                    if peak_ratio > 0.4:  # Single frequency > 40% of power
                        resonance_pred[i-fft_window:i] = 1

        results['resonance'] = resonance_pred

        # =====================================================================
        # 6. STATISTICAL DRIFT DETECTOR (for stealthy attacks)
        # =====================================================================
        # CUSUM on position error

        cusum_threshold = 0.5
        cusum_x = np.zeros(N - 1)
        cusum_y = np.zeros(N - 1)

        # Position should follow velocity integration
        x_residual = pos_deriv[:, 0] - vel[1:, 0]
        y_residual = pos_deriv[:, 1] - vel[1:, 1]

        for i in range(1, N - 1):
            cusum_x[i] = max(0, cusum_x[i-1] + abs(x_residual[i]) - 0.01)
            cusum_y[i] = max(0, cusum_y[i-1] + abs(y_residual[i]) - 0.01)

        drift_pred = ((cusum_x > cusum_threshold) | (cusum_y > cusum_threshold)).astype(int)
        results['drift'] = drift_pred

        # =====================================================================
        # 7. INTERMITTENT DETECTOR (transition detection)
        # =====================================================================
        # Detect sudden jumps in any state variable

        jump_threshold = 0.1
        state_jump = np.zeros(N - 1)

        for i in range(12):  # All state variables
            col_diff = np.abs(np.diff(data[:, i]))
            col_std = np.std(col_diff) + 1e-10

            # Jumps > 5 sigma
            col_jump = (col_diff / col_std) > 5
            state_jump = np.maximum(state_jump, col_jump.astype(float))

        results['intermittent'] = state_jump.astype(int)

        # =====================================================================
        # COMBINE ALL DETECTORS
        # =====================================================================

        combined = np.zeros(N - 1)

        # Weight detectors by reliability
        weights = {
            'physics': 1.0,
            'gyro_saturation': 1.0,
            'control_hijack': 0.8,
            'time_delay': 0.7,
            'resonance': 0.5,
            'drift': 0.3,
            'intermittent': 0.6,
        }

        for name, pred in results.items():
            if name in weights:
                combined = np.maximum(combined, pred * weights[name])

        # Final prediction: any detector fires
        final_pred = (combined > 0).astype(int)

        results['combined'] = combined
        results['predictions'] = final_pred

        return results

    def calibrate(self, clean_data: np.ndarray):
        """Calibrate thresholds on clean data."""
        N = len(clean_data)
        dt = self.config.dt

        pos = clean_data[:, 0:3]
        vel = clean_data[:, 9:12]
        rate = clean_data[:, 6:9]
        thrust = clean_data[:, 12]

        # Position-velocity threshold
        pos_deriv = (pos[1:] - pos[:-1]) / dt
        pos_vel_diff = np.linalg.norm(pos_deriv - vel[1:], axis=1)
        self.config.pos_vel_threshold = np.percentile(pos_vel_diff, 99)

        # Kinematic threshold
        window = 20
        kinematic_diff = []
        for i in range(window, N - 1):
            vel_integral = vel[i-window+1:i+1].sum(axis=0) * dt
            pos_change = pos[i+1] - pos[i-window+1]
            kinematic_diff.append(np.linalg.norm(vel_integral - pos_change))
        self.config.kinematic_threshold = np.percentile(kinematic_diff, 99)

        # Gyro rate max
        rate_magnitude = np.linalg.norm(rate, axis=1)
        self.config.gyro_rate_max = np.percentile(rate_magnitude, 99.9)

        # Thrust nominal and deviation
        self.config.thrust_nominal = np.median(thrust)
        thrust_deviation = np.abs(thrust - self.config.thrust_nominal)
        self.config.thrust_deviation_max = np.percentile(thrust_deviation, 99)

        print(f"Calibrated thresholds:")
        print(f"  pos_vel_threshold:     {self.config.pos_vel_threshold:.4f}")
        print(f"  kinematic_threshold:   {self.config.kinematic_threshold:.6f}")
        print(f"  gyro_rate_max:         {self.config.gyro_rate_max:.2f} rad/s")
        print(f"  thrust_nominal:        {self.config.thrust_nominal:.2f} N")
        print(f"  thrust_deviation_max:  {self.config.thrust_deviation_max:.2f} N")


def evaluate_detector(detector: ComprehensiveDetector, attacks: dict) -> dict:
    """Evaluate on all attacks."""
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    results = {}

    print(f"\n{'Attack':<25} {'Recall':>8} {'Precision':>10} {'F1':>8}")
    print("-" * 55)

    for attack_name, attack_data in attacks.items():
        data = attack_data[state_cols + control_cols].values
        labels = attack_data["label"].values[1:]  # Align with predictions

        out = detector.detect(data)
        preds = out['predictions']

        min_len = min(len(preds), len(labels))
        preds = preds[:min_len]
        labels = labels[:min_len]

        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[attack_name] = {'recall': recall, 'precision': precision, 'f1': f1}

        if attack_name != "clean":
            print(f"{attack_name:<25} {recall*100:>7.1f}% {precision*100:>9.1f}% {f1*100:>7.1f}%")

    # Overall
    attack_results = [v for k, v in results.items() if k != "clean"]
    avg_recall = np.mean([r['recall'] for r in attack_results])
    avg_precision = np.mean([r['precision'] for r in attack_results])
    avg_f1 = np.mean([r['f1'] for r in attack_results])

    print("-" * 55)
    print(f"{'AVERAGE':<25} {avg_recall*100:>7.1f}% {avg_precision*100:>9.1f}% {avg_f1*100:>7.1f}%")

    return results
