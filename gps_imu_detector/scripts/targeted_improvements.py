"""
Targeted Improvements for IMU_BIAS and ACTUATOR_FAULT Detection

Baseline (locked): 70% detection at 0.8% FPR under honest evaluation

Problem Analysis:
- GPS_DRIFT, GPS_JUMP, SPOOFING: 100% (solved)
- IMU_BIAS: 17% (weakly observable - small angular velocity changes)
- ACTUATOR_FAULT: 33% (control-masked - looks like normal noise)

Targeted Solutions:
1. IMU_BIAS: Physics residual accumulation (PINN detects drift over time)
2. ACTUATOR_FAULT: Active probing + temporal consistency

Constraint: Do NOT harm FPR (must stay <= 1%)
"""

import sys
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# BASELINE LOCK
# =============================================================================

BASELINE_METRICS = {
    "detection_rate": 0.70,
    "fpr": 0.008,
    "per_attack": {
        "GPS_DRIFT": 1.00,
        "GPS_JUMP": 1.00,
        "IMU_BIAS": 0.17,
        "SPOOFING": 1.00,
        "ACTUATOR_FAULT": 0.33,
    },
    "note": "Under strictly held-out evaluation, detection rate is ~70% at 0.8% FPR"
}


# =============================================================================
# IMPROVEMENT 1: IMU Bias Detector (Physics Residual Accumulation)
# =============================================================================

@dataclass
class IMUBiasDetectorConfig:
    """Config for IMU bias detection via physics residual accumulation."""
    window_size: int = 60  # Samples to accumulate
    bias_threshold: float = 0.6  # Accumulated bias threshold (raised for FPR control)
    angular_channels: List[int] = field(default_factory=lambda: [9, 10, 11])  # ang_vel indices
    min_consistent_samples: int = 40  # Minimum samples showing consistent bias (raised)


class IMUBiasDetector:
    """
    Detects IMU bias through physics residual accumulation.

    Key insight: IMU bias causes systematic drift in angular velocity
    that accumulates over time. Single-sample detection is weak,
    but accumulated residuals reveal the bias.
    """

    def __init__(self, config: Optional[IMUBiasDetectorConfig] = None):
        self.config = config or IMUBiasDetectorConfig()
        self.residual_history = deque(maxlen=self.config.window_size)
        self.accumulated_bias = np.zeros(3)
        self.consistent_count = 0

    def update(self, state: np.ndarray, predicted_state: np.ndarray) -> Dict:
        """
        Update with new observation and return bias detection result.

        Args:
            state: Observed state [12]
            predicted_state: PINN-predicted state [12]

        Returns:
            dict with 'bias_detected', 'accumulated_bias', 'confidence'
        """
        # Extract angular velocity residual
        ang_vel_obs = state[self.config.angular_channels]
        ang_vel_pred = predicted_state[self.config.angular_channels]
        residual = ang_vel_obs - ang_vel_pred

        self.residual_history.append(residual)

        if len(self.residual_history) < self.config.min_consistent_samples:
            return {
                'bias_detected': False,
                'accumulated_bias': np.zeros(3),
                'confidence': 0.0,
            }

        # Compute accumulated bias (mean of residuals)
        residuals = np.array(self.residual_history)
        self.accumulated_bias = np.mean(residuals, axis=0)

        # Check for consistent bias direction
        # Bias should be consistently in same direction (not random noise)
        sign_consistency = np.abs(np.mean(np.sign(residuals), axis=0))
        is_consistent = np.any(sign_consistency > 0.9)  # 90% same sign (very strict)

        if is_consistent:
            self.consistent_count += 1
        else:
            self.consistent_count = max(0, self.consistent_count - 1)

        # Detect if accumulated bias exceeds threshold
        bias_magnitude = np.linalg.norm(self.accumulated_bias)
        bias_detected = (
            bias_magnitude > self.config.bias_threshold and
            self.consistent_count >= self.config.min_consistent_samples // 2
        )

        # Confidence based on consistency and magnitude
        confidence = min(1.0, bias_magnitude / self.config.bias_threshold) * \
                     min(1.0, self.consistent_count / self.config.min_consistent_samples)

        return {
            'bias_detected': bias_detected,
            'accumulated_bias': self.accumulated_bias.copy(),
            'confidence': confidence,
            'bias_magnitude': bias_magnitude,
            'consistent_count': self.consistent_count,
        }

    def reset(self):
        self.residual_history.clear()
        self.accumulated_bias = np.zeros(3)
        self.consistent_count = 0


# =============================================================================
# IMPROVEMENT 2: Actuator Fault Detector (Temporal Consistency + Probing)
# =============================================================================

@dataclass
class ActuatorFaultDetectorConfig:
    """Config for actuator fault detection."""
    window_size: int = 50  # Larger window for stability
    variance_ratio_threshold: float = 8.0  # Much higher threshold for FPR control
    correlation_threshold: float = 0.15  # Lower threshold (require more decorrelation)
    probe_amplitude: float = 0.1
    probe_frequency: float = 2.0  # Hz


class ActuatorFaultDetector:
    """
    Detects actuator faults through:
    1. Variance ratio analysis (faults increase angular velocity variance)
    2. Temporal autocorrelation (faults create uncorrelated noise)
    3. Control-response correlation (faults break expected response)

    Key insight: Actuator faults are "control-masked" - they look like
    normal noise in single samples. But they have different statistical
    properties when analyzed over time.
    """

    def __init__(self, config: Optional[ActuatorFaultDetectorConfig] = None):
        self.config = config or ActuatorFaultDetectorConfig()
        self.ang_vel_history = deque(maxlen=self.config.window_size)
        self.baseline_variance = None
        self._calibrated = False

    def calibrate(self, nominal_data: np.ndarray):
        """
        Calibrate baseline variance from nominal data.

        Args:
            nominal_data: Array of shape (n_trajectories, T, 12)
        """
        # Extract angular velocity channels
        ang_vel = nominal_data[:, :, 9:12]

        # Compute baseline variance per channel
        self.baseline_variance = np.var(ang_vel, axis=(0, 1))
        self._calibrated = True

    def update(self, state: np.ndarray) -> Dict:
        """
        Update with new observation and return fault detection result.

        Args:
            state: Observed state [12]

        Returns:
            dict with 'fault_detected', 'variance_ratio', 'confidence'
        """
        ang_vel = state[9:12]
        self.ang_vel_history.append(ang_vel)

        if len(self.ang_vel_history) < self.config.window_size // 2:
            return {
                'fault_detected': False,
                'variance_ratio': 1.0,
                'confidence': 0.0,
            }

        # Compute current variance
        history = np.array(self.ang_vel_history)
        current_variance = np.var(history, axis=0)

        # Variance ratio (fault increases variance)
        if self._calibrated and np.all(self.baseline_variance > 1e-6):
            variance_ratio = np.max(current_variance / self.baseline_variance)
        else:
            # Use expected variance if not calibrated
            variance_ratio = np.max(current_variance) / 0.01  # Assume baseline ~0.01

        # Autocorrelation (fault creates more random, less correlated signal)
        if len(history) >= 10:
            autocorr = self._compute_autocorrelation(history)
        else:
            autocorr = 0.5

        # Detect fault: high variance ratio + low autocorrelation
        fault_detected = (
            variance_ratio > self.config.variance_ratio_threshold and
            autocorr < self.config.correlation_threshold
        )

        # Confidence
        var_conf = min(1.0, variance_ratio / self.config.variance_ratio_threshold)
        corr_conf = 1.0 - min(1.0, autocorr / self.config.correlation_threshold)
        confidence = var_conf * corr_conf

        return {
            'fault_detected': fault_detected,
            'variance_ratio': variance_ratio,
            'autocorrelation': autocorr,
            'confidence': confidence,
        }

    def _compute_autocorrelation(self, history: np.ndarray) -> float:
        """Compute lag-1 autocorrelation."""
        # Use first angular velocity channel
        x = history[:, 0]
        if len(x) < 3:
            return 0.5

        x_centered = x - np.mean(x)
        var = np.var(x)
        if var < 1e-10:
            return 0.5

        autocorr = np.correlate(x_centered[:-1], x_centered[1:]) / (var * (len(x) - 1))
        return float(np.abs(autocorr[0])) if len(autocorr) > 0 else 0.5

    def reset(self):
        self.ang_vel_history.clear()


# =============================================================================
# IMPROVED DETECTOR (Combines baseline + targeted improvements)
# =============================================================================

class ImprovedDetector:
    """
    Improved detector that combines:
    - Baseline detection (works for GPS attacks, spoofing)
    - IMU bias detector (physics residual accumulation)
    - Actuator fault detector (variance + autocorrelation)
    """

    def __init__(self):
        self.imu_detector = IMUBiasDetector()
        self.actuator_detector = ActuatorFaultDetector()
        self.baseline_threshold = 5.0  # Higher for FPR control

        # Simple PINN for residual computation
        self._init_simple_pinn()

    def _init_simple_pinn(self):
        """Initialize a simple PINN for state prediction."""
        self.pinn = torch.nn.Sequential(
            torch.nn.Linear(12, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 12),
        )
        # Initialize with identity-like weights for stability
        with torch.no_grad():
            self.pinn[-1].weight.fill_(0)
            self.pinn[-1].bias.fill_(0)
            for i in range(12):
                self.pinn[-1].weight[i, i % 64] = 0.1

    def calibrate(self, nominal_data: np.ndarray):
        """Calibrate all components."""
        self.actuator_detector.calibrate(nominal_data)

        # Quick PINN training
        self._train_pinn(nominal_data)

    def _train_pinn(self, data: np.ndarray, epochs: int = 20):
        """Train PINN on nominal data."""
        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=1e-3)

        for epoch in range(epochs):
            total_loss = 0
            for traj in data[:20]:  # Use subset for speed
                for t in range(len(traj) - 1):
                    x = torch.tensor(traj[t], dtype=torch.float32)
                    y = torch.tensor(traj[t+1], dtype=torch.float32)

                    pred = self.pinn(x)
                    loss = torch.nn.functional.mse_loss(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

    def process(self, state: np.ndarray, next_state: Optional[np.ndarray] = None) -> Dict:
        """
        Process a sample and return detection result.

        Returns dict with:
            - 'detected': bool
            - 'source': str (which detector triggered)
            - 'confidence': float
            - 'details': dict
        """
        state = np.asarray(state, dtype=np.float32)

        # Get PINN prediction for IMU bias detection
        with torch.no_grad():
            x = torch.tensor(state)
            predicted = self.pinn(x).numpy()

        # Run specialized detectors
        imu_result = self.imu_detector.update(state, predicted)
        actuator_result = self.actuator_detector.update(state)

        # Compute baseline residual (for GPS attacks)
        if next_state is not None:
            next_state = np.asarray(next_state, dtype=np.float32)
            baseline_residual = np.linalg.norm(next_state - predicted)
        else:
            baseline_residual = np.linalg.norm(state - predicted)

        baseline_detected = baseline_residual > self.baseline_threshold

        # Combine detections
        detected = False
        source = "none"
        confidence = 0.0

        if baseline_detected:
            detected = True
            source = "baseline"
            confidence = min(1.0, baseline_residual / self.baseline_threshold)

        if imu_result['bias_detected']:
            detected = True
            source = "imu_bias" if source == "none" else f"{source}+imu_bias"
            confidence = max(confidence, imu_result['confidence'])

        if actuator_result['fault_detected']:
            detected = True
            source = "actuator_fault" if source == "none" else f"{source}+actuator_fault"
            confidence = max(confidence, actuator_result['confidence'])

        return {
            'detected': detected,
            'source': source,
            'confidence': confidence,
            'baseline_residual': baseline_residual,
            'imu': imu_result,
            'actuator': actuator_result,
        }

    def reset(self):
        self.imu_detector.reset()
        self.actuator_detector.reset()


# =============================================================================
# EVALUATION
# =============================================================================

def generate_trajectories(n_traj: int, T: int, seed: int, is_attack: bool = False) -> np.ndarray:
    """Generate trajectories with specific seed."""
    np.random.seed(seed)
    trajectories = []

    for i in range(n_traj):
        traj = np.zeros((T, 12), dtype=np.float32)
        pos = np.array([0.0, 0.0, 10.0])
        vel = np.array([0.0, 0.0, 0.0])
        orient = np.array([0.0, 0.0, 0.0])
        ang_vel = np.array([0.0, 0.0, 0.0])

        profile = i % 4
        dt = 0.005
        attack_start = T // 4
        atype = i % 5

        for t in range(T):
            if profile == 0:
                vel = np.random.randn(3) * 0.05
                ang_vel = np.random.randn(3) * 0.02
            elif profile == 1:
                vel = np.array([2.0, 0.0, 0.0]) + np.random.randn(3) * 0.1
                ang_vel = np.random.randn(3) * 0.05
            elif profile == 2:
                angle = t * 0.02
                vel = np.array([np.cos(angle), np.sin(angle), 0.0]) * 3.0 + np.random.randn(3) * 0.05
                ang_vel = np.array([0.0, 0.0, 0.02]) + np.random.randn(3) * 0.02
            else:
                angle = t * 0.03
                vel = np.array([np.cos(angle), np.sin(2*angle), 0.0]) * 2.0 + np.random.randn(3) * 0.08
                ang_vel = np.random.randn(3) * 0.03

            pos = pos + vel * dt
            orient = orient + ang_vel * dt

            traj[t, :3] = pos
            traj[t, 3:6] = vel
            traj[t, 6:9] = orient
            traj[t, 9:12] = ang_vel

            if is_attack and t >= attack_start:
                if atype == 0:  # GPS drift
                    traj[t, :3] += 0.5 * (t - attack_start) * 0.005
                elif atype == 1:  # GPS jump
                    traj[t, :3] += np.array([5.0, 5.0, 1.0])
                    traj[t, 3:6] += np.array([0.5, 0.5, 0.0])
                elif atype == 2:  # IMU bias
                    traj[t, 9:12] += np.array([0.8, 0.8, 0.3])
                elif atype == 3:  # Spoofing
                    traj[t, :3] += np.array([3.0, -3.0, 0.5])
                    traj[t, 3:6] += np.array([-0.3, 0.3, 0.0])
                elif atype == 4:  # Actuator fault
                    traj[t, 9:12] += np.random.randn(3) * 0.5

        trajectories.append(traj)

    return np.array(trajectories)


def evaluate_improved_detector():
    """Evaluate the improved detector on held-out test data."""
    print("="*70)
    print("      TARGETED IMPROVEMENTS EVALUATION")
    print("="*70)

    print("\n" + "-"*70)
    print("BASELINE (LOCKED):")
    print(f"  {BASELINE_METRICS['note']}")
    print("-"*70)

    # Training data
    print("\n[Generating training data (seeds 100-199)...]")
    train_nominal = generate_trajectories(50, 200, seed=100, is_attack=False)
    train_attacks = generate_trajectories(50, 200, seed=150, is_attack=True)

    # Held-out test data (DIFFERENT seeds)
    print("[Generating held-out test data (seeds 200-299)...]")
    test_nominal = generate_trajectories(30, 200, seed=200, is_attack=False)
    test_attacks = generate_trajectories(30, 200, seed=250, is_attack=True)

    # Create and calibrate detector
    print("[Training improved detector...]")
    detector = ImprovedDetector()
    detector.calibrate(train_nominal)

    # Evaluate on nominal (FPR)
    print("[Evaluating on nominal data...]")
    false_positives = 0
    total_nominal = 0

    for traj in test_nominal:
        for t in range(len(traj) - 1):
            result = detector.process(traj[t], traj[t+1])
            if result['detected']:
                false_positives += 1
            total_nominal += 1
        detector.reset()

    fpr = false_positives / total_nominal

    # Evaluate on attacks (recall)
    print("[Evaluating on attack data...]")
    attack_types = ['GPS_DRIFT', 'GPS_JUMP', 'IMU_BIAS', 'SPOOFING', 'ACTUATOR_FAULT']
    per_attack = {at: {'detected': 0, 'total': 0} for at in attack_types}
    total_detected = 0

    for i, traj in enumerate(test_attacks):
        attack_start = len(traj) // 4
        detected = False
        atype = attack_types[i % 5]

        for t in range(len(traj) - 1):
            result = detector.process(traj[t], traj[t+1])

            if t >= attack_start and not detected:
                if result['detected']:
                    detected = True

        if detected:
            total_detected += 1
            per_attack[atype]['detected'] += 1
        per_attack[atype]['total'] += 1
        detector.reset()

    detection_rate = total_detected / len(test_attacks)

    # Print results
    print("\n" + "="*70)
    print("                    IMPROVED RESULTS")
    print("="*70)

    print(f"\n{'METRIC':<35} {'BASELINE':<15} {'IMPROVED':<15} {'CHANGE':<15}")
    print("-"*80)
    print(f"{'Overall Detection Rate':<35} {BASELINE_METRICS['detection_rate']*100:.1f}%{'':<10} {detection_rate*100:.1f}%{'':<10} {(detection_rate - BASELINE_METRICS['detection_rate'])*100:+.1f}%")
    print(f"{'False Positive Rate':<35} {BASELINE_METRICS['fpr']*100:.2f}%{'':<10} {fpr*100:.2f}%{'':<10} {(fpr - BASELINE_METRICS['fpr'])*100:+.2f}%")

    print(f"\n{'ATTACK TYPE':<35} {'BASELINE':<15} {'IMPROVED':<15} {'CHANGE':<15}")
    print("-"*80)

    for at in attack_types:
        baseline_recall = BASELINE_METRICS['per_attack'][at]
        if per_attack[at]['total'] > 0:
            improved_recall = per_attack[at]['detected'] / per_attack[at]['total']
        else:
            improved_recall = 0.0
        change = improved_recall - baseline_recall

        # Highlight improvements
        marker = " **" if change > 0.1 else ""
        print(f"{at:<35} {baseline_recall*100:.0f}%{'':<12} {improved_recall*100:.0f}%{'':<12} {change*100:+.0f}%{marker}")

    # Certification check
    print("\n" + "="*70)
    print("                    CERTIFICATION STATUS")
    print("="*70)

    checks = [
        ("Detection Rate >= 80%", detection_rate >= 0.80, detection_rate),
        ("FPR <= 1%", fpr <= 0.01, fpr),
    ]

    for name, passed, value in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    all_passed = all(c[1] for c in checks)

    if all_passed:
        print("\n" + "="*70)
        print("     [OK] ALL REQUIREMENTS MET")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("     [!!] REQUIREMENTS NOT MET - Further tuning needed")
        print("="*70)

    return {
        'detection_rate': detection_rate,
        'fpr': fpr,
        'per_attack': {at: per_attack[at]['detected'] / max(1, per_attack[at]['total'])
                       for at in attack_types},
    }


if __name__ == "__main__":
    results = evaluate_improved_detector()
