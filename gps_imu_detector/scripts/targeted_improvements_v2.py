"""
Targeted Improvements v2 - Statistical Approach

Key insight: Don't rely on poorly-trained PINN residuals.
Instead, use direct statistical tests that characterize each attack type.

IMU_BIAS: Consistent offset from calibrated mean
ACTUATOR_FAULT: Elevated variance from calibrated baseline

This is simpler, more robust, and easier to tune.
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List
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
# Statistical Detector for IMU Bias
# =============================================================================

class IMUBiasStatDetector:
    """
    Detects IMU bias using cumulative sum (CUSUM) of angular velocity.

    Key insight: IMU bias creates a consistent offset that accumulates.
    Normal noise averages to zero; bias accumulates in one direction.
    """

    def __init__(self, threshold: float = 15.0, window: int = 50):
        self.threshold = threshold  # CUSUM threshold
        self.window = window
        self.cusum_pos = np.zeros(3)  # Positive CUSUM
        self.cusum_neg = np.zeros(3)  # Negative CUSUM
        self.calibrated_mean = np.zeros(3)
        self.calibrated_std = np.ones(3)
        self._count = 0

    def calibrate(self, nominal_data: np.ndarray):
        """Calibrate from nominal trajectories."""
        ang_vel = nominal_data[:, :, 9:12].reshape(-1, 3)
        self.calibrated_mean = np.mean(ang_vel, axis=0)
        self.calibrated_std = np.std(ang_vel, axis=0) + 1e-6

    def update(self, state: np.ndarray) -> Dict:
        """Update with new observation."""
        ang_vel = state[9:12]

        # Normalize by calibrated statistics
        z = (ang_vel - self.calibrated_mean) / self.calibrated_std

        # CUSUM update
        self.cusum_pos = np.maximum(0, self.cusum_pos + z - 0.5)
        self.cusum_neg = np.maximum(0, self.cusum_neg - z - 0.5)

        self._count += 1

        # Detect if CUSUM exceeds threshold
        max_cusum = max(np.max(self.cusum_pos), np.max(self.cusum_neg))
        detected = max_cusum > self.threshold and self._count > self.window

        return {
            'detected': detected,
            'cusum': max_cusum,
            'confidence': min(1.0, max_cusum / self.threshold) if self._count > self.window else 0.0,
        }

    def reset(self):
        self.cusum_pos = np.zeros(3)
        self.cusum_neg = np.zeros(3)
        self._count = 0


# =============================================================================
# Statistical Detector for Actuator Fault
# =============================================================================

class ActuatorFaultStatDetector:
    """
    Detects actuator faults via variance ratio test.

    Key insight: Actuator faults inject additional noise that increases
    variance beyond the calibrated baseline.
    """

    def __init__(self, variance_threshold: float = 3.0, window: int = 40):
        self.variance_threshold = variance_threshold
        self.window = window
        self.history = deque(maxlen=window)
        self.baseline_variance = np.ones(3)

    def calibrate(self, nominal_data: np.ndarray):
        """Calibrate baseline variance from nominal data."""
        ang_vel = nominal_data[:, :, 9:12].reshape(-1, 3)
        self.baseline_variance = np.var(ang_vel, axis=0) + 1e-6

    def update(self, state: np.ndarray) -> Dict:
        """Update with new observation."""
        ang_vel = state[9:12]
        self.history.append(ang_vel)

        if len(self.history) < self.window:
            return {'detected': False, 'variance_ratio': 1.0, 'confidence': 0.0}

        # Compute current variance
        history = np.array(self.history)
        current_var = np.var(history, axis=0)

        # Variance ratio
        variance_ratio = np.max(current_var / self.baseline_variance)
        detected = variance_ratio > self.variance_threshold

        return {
            'detected': detected,
            'variance_ratio': variance_ratio,
            'confidence': min(1.0, variance_ratio / self.variance_threshold),
        }

    def reset(self):
        self.history.clear()


# =============================================================================
# Combined Improved Detector
# =============================================================================

class ImprovedStatDetector:
    """
    Improved detector combining:
    - GPS drift: CUSUM on position deviation from expected trajectory
    - GPS jump: Sudden position discontinuity
    - Spoofing: Velocity anomaly
    - IMU bias: CUSUM on angular velocity
    - Actuator fault: Variance ratio on angular velocity
    """

    def __init__(self,
                 gps_threshold: float = 2.0,
                 gps_drift_threshold: float = 10.0,
                 imu_cusum_threshold: float = 12.0,
                 actuator_var_threshold: float = 4.0):
        self.gps_threshold = gps_threshold
        self.gps_drift_threshold = gps_drift_threshold
        self.imu_detector = IMUBiasStatDetector(threshold=imu_cusum_threshold)
        self.actuator_detector = ActuatorFaultStatDetector(variance_threshold=actuator_var_threshold)

        # Baseline calibration
        self.baseline_pos_mean = np.zeros(3)
        self.baseline_vel_mean = np.zeros(3)
        self.baseline_pos_std = np.ones(3)
        self.baseline_vel_std = np.ones(3)

        # GPS drift detection: track cumulative position error
        self.integrated_pos = None  # Position integrated from velocity
        self.cumulative_error = np.zeros(3)

        # State tracking for GPS attacks
        self.prev_state = None
        self.velocity_residuals = deque(maxlen=20)
        self._count = 0

    def calibrate(self, nominal_data: np.ndarray):
        """Calibrate all components."""
        # GPS baseline (position changes and velocity)
        pos = nominal_data[:, :, 0:3].reshape(-1, 3)
        vel = nominal_data[:, :, 3:6].reshape(-1, 3)

        self.baseline_pos_mean = np.mean(pos, axis=0)
        self.baseline_vel_mean = np.mean(vel, axis=0)
        self.baseline_pos_std = np.std(pos, axis=0) + 1e-6
        self.baseline_vel_std = np.std(vel, axis=0) + 1e-6

        # Specialized detectors
        self.imu_detector.calibrate(nominal_data)
        self.actuator_detector.calibrate(nominal_data)

    def process(self, state: np.ndarray, next_state: Optional[np.ndarray] = None) -> Dict:
        """Process a sample."""
        state = np.asarray(state, dtype=np.float32)
        self._count += 1

        gps_detected = False
        gps_drift_detected = False

        # GPS/Spoofing detection: velocity anomaly
        vel = state[3:6]
        vel_z = np.abs((vel - self.baseline_vel_mean) / self.baseline_vel_std)
        gps_anomaly = np.max(vel_z)
        if gps_anomaly > self.gps_threshold:
            gps_detected = True

        # GPS drift detection: track cumulative position vs velocity integration
        if self.prev_state is not None:
            # Update integrated position using velocity
            if self.integrated_pos is None:
                self.integrated_pos = self.prev_state[0:3].copy()
            self.integrated_pos = self.integrated_pos + self.prev_state[3:6] * 0.005

            # Cumulative error = reported position - velocity-integrated position
            reported_pos = state[0:3]
            self.cumulative_error = reported_pos - self.integrated_pos

            # GPS drift creates growing cumulative error
            drift_magnitude = np.linalg.norm(self.cumulative_error)
            if drift_magnitude > self.gps_drift_threshold and self._count > 30:
                gps_drift_detected = True

            # Also check for sudden jumps
            pos_change = np.linalg.norm(state[0:3] - self.prev_state[0:3])
            expected_change = np.linalg.norm(self.prev_state[3:6]) * 0.005 + 0.02
            if pos_change > expected_change * 10:  # Sudden large jump
                gps_detected = True

        self.prev_state = state.copy()

        # IMU bias detection
        imu_result = self.imu_detector.update(state)

        # Actuator fault detection
        actuator_result = self.actuator_detector.update(state)

        # Combine detections
        detected = gps_detected or gps_drift_detected or imu_result['detected'] or actuator_result['detected']

        source = []
        if gps_detected:
            source.append('gps_jump')
        if gps_drift_detected:
            source.append('gps_drift')
        if imu_result['detected']:
            source.append('imu_bias')
        if actuator_result['detected']:
            source.append('actuator_fault')

        return {
            'detected': detected,
            'source': '+'.join(source) if source else 'none',
            'gps_anomaly': gps_anomaly,
            'gps_drift_error': np.linalg.norm(self.cumulative_error),
            'imu': imu_result,
            'actuator': actuator_result,
        }

    def reset(self):
        self.imu_detector.reset()
        self.actuator_detector.reset()
        self.prev_state = None
        self.velocity_residuals.clear()
        self.integrated_pos = None
        self.cumulative_error = np.zeros(3)
        self._count = 0


# =============================================================================
# Data Generation (same as before)
# =============================================================================

def generate_trajectories(n_traj: int, T: int, seed: int, is_attack: bool = False) -> np.ndarray:
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


# =============================================================================
# Threshold Search
# =============================================================================

def search_thresholds():
    """Search for optimal thresholds using validation set."""
    print("\n" + "="*60)
    print("THRESHOLD SEARCH (Grid Search on Validation Set)")
    print("="*60)

    # Validation data (different from test)
    val_nominal = generate_trajectories(30, 200, seed=300, is_attack=False)
    val_attacks = generate_trajectories(30, 200, seed=350, is_attack=True)

    # Training data
    train_nominal = generate_trajectories(50, 200, seed=100, is_attack=False)

    best_config = None
    best_score = -float('inf')

    # Grid search (including GPS drift threshold)
    for gps_th in [2.5, 3.0, 3.5]:
        for gps_drift_th in [0.5, 1.0, 2.0, 3.0]:  # Lower thresholds for cumulative error
            for imu_th in [12.0, 15.0, 18.0]:
                for act_th in [5.0, 6.0, 7.0]:
                    detector = ImprovedStatDetector(
                        gps_threshold=gps_th,
                        gps_drift_threshold=gps_drift_th,
                        imu_cusum_threshold=imu_th,
                        actuator_var_threshold=act_th,
                    )
                    detector.calibrate(train_nominal)

                    # Evaluate FPR
                    fp = 0
                    total = 0
                    for traj in val_nominal:
                        for t in range(len(traj) - 1):
                            result = detector.process(traj[t], traj[t+1])
                            if result['detected']:
                                fp += 1
                            total += 1
                        detector.reset()
                    fpr = fp / total

                    # Evaluate detection rate
                    detected_count = 0
                    for traj in val_attacks:
                        attack_start = len(traj) // 4
                        detected = False
                        for t in range(len(traj) - 1):
                            result = detector.process(traj[t], traj[t+1])
                            if t >= attack_start and result['detected']:
                                detected = True
                                break
                        if detected:
                            detected_count += 1
                        detector.reset()
                    detection_rate = detected_count / len(val_attacks)

                    # Score: maximize detection while FPR <= 1%
                    if fpr <= 0.01:
                        score = detection_rate
                    else:
                        score = detection_rate - 10 * (fpr - 0.01)

                    if score > best_score:
                        best_score = score
                        best_config = {
                            'gps_threshold': gps_th,
                            'gps_drift_threshold': gps_drift_th,
                            'imu_cusum_threshold': imu_th,
                            'actuator_var_threshold': act_th,
                            'fpr': fpr,
                            'detection_rate': detection_rate,
                        }

    print(f"\nBest configuration:")
    print(f"  GPS velocity threshold: {best_config['gps_threshold']}")
    print(f"  GPS drift CUSUM threshold: {best_config['gps_drift_threshold']}")
    print(f"  IMU CUSUM threshold: {best_config['imu_cusum_threshold']}")
    print(f"  Actuator variance threshold: {best_config['actuator_var_threshold']}")
    print(f"  Validation FPR: {best_config['fpr']*100:.2f}%")
    print(f"  Validation Detection: {best_config['detection_rate']*100:.1f}%")

    return best_config


# =============================================================================
# Final Evaluation
# =============================================================================

def evaluate_improved():
    """Evaluate with tuned thresholds on held-out test set."""
    print("="*70)
    print("      TARGETED IMPROVEMENTS v2 - STATISTICAL APPROACH")
    print("="*70)

    print("\n" + "-"*70)
    print("BASELINE (LOCKED):")
    print(f"  {BASELINE_METRICS['note']}")
    print("-"*70)

    # Search for best thresholds
    config = search_thresholds()

    print("\n" + "="*60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*60)

    # Training data
    train_nominal = generate_trajectories(50, 200, seed=100, is_attack=False)

    # HELD-OUT test data (seeds 200-299, never seen during threshold search)
    test_nominal = generate_trajectories(30, 200, seed=200, is_attack=False)
    test_attacks = generate_trajectories(30, 200, seed=250, is_attack=True)

    # Create detector with tuned thresholds
    detector = ImprovedStatDetector(
        gps_threshold=config['gps_threshold'],
        gps_drift_threshold=config['gps_drift_threshold'],
        imu_cusum_threshold=config['imu_cusum_threshold'],
        actuator_var_threshold=config['actuator_var_threshold'],
    )
    detector.calibrate(train_nominal)

    # Evaluate FPR
    print("\n[Evaluating on held-out nominal data...]")
    fp = 0
    total = 0
    for traj in test_nominal:
        for t in range(len(traj) - 1):
            result = detector.process(traj[t], traj[t+1])
            if result['detected']:
                fp += 1
            total += 1
        detector.reset()
    fpr = fp / total

    # Evaluate detection per attack type
    print("[Evaluating on held-out attack data...]")
    attack_types = ['GPS_DRIFT', 'GPS_JUMP', 'IMU_BIAS', 'SPOOFING', 'ACTUATOR_FAULT']
    per_attack = {at: {'detected': 0, 'total': 0} for at in attack_types}
    total_detected = 0

    for i, traj in enumerate(test_attacks):
        attack_start = len(traj) // 4
        detected = False
        atype = attack_types[i % 5]

        for t in range(len(traj) - 1):
            result = detector.process(traj[t], traj[t+1])
            if t >= attack_start and result['detected']:
                detected = True
                break

        if detected:
            total_detected += 1
            per_attack[atype]['detected'] += 1
        per_attack[atype]['total'] += 1
        detector.reset()

    detection_rate = total_detected / len(test_attacks)

    # Print results
    print("\n" + "="*70)
    print("                    IMPROVED RESULTS (v2)")
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
        marker = " **" if change > 0.1 else ""
        print(f"{at:<35} {baseline_recall*100:.0f}%{'':<12} {improved_recall*100:.0f}%{'':<12} {change*100:+.0f}%{marker}")

    # Certification
    print("\n" + "="*70)
    print("                    CERTIFICATION STATUS")
    print("="*70)

    checks = [
        ("Detection Rate >= 80%", detection_rate >= 0.80),
        ("FPR <= 1%", fpr <= 0.01),
    ]

    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    all_passed = all(c[1] for c in checks)

    if all_passed:
        print("\n" + "="*70)
        print("     [OK] ALL REQUIREMENTS MET")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("     [!!] REQUIREMENTS NOT MET")
        print("="*70)

    return {
        'detection_rate': detection_rate,
        'fpr': fpr,
        'per_attack': {at: per_attack[at]['detected'] / max(1, per_attack[at]['total'])
                       for at in attack_types},
        'config': config,
    }


if __name__ == "__main__":
    results = evaluate_improved()
