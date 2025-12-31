"""
Robustness Stress Coverage Module (Phase 4)

Implements:
4.1 Physics-Consistent Counterfactual Generator
    - Generates attack samples that respect physics constraints
    - PINN-guided perturbations
    - Covers diverse attack types

4.2 Stress Test Suite
    - Edge case coverage
    - Attack intensity sweep
    - Regime-specific testing
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Phase 4.1: Physics-Consistent Counterfactual Generator
# =============================================================================

class AttackType(Enum):
    """Types of attacks to generate."""
    GPS_DRIFT = auto()        # Slow GPS position drift
    GPS_JUMP = auto()         # Sudden GPS position jump
    GPS_FREEZE = auto()       # GPS stuck at old value
    IMU_BIAS = auto()         # Constant IMU bias injection
    IMU_SCALE = auto()        # IMU scale factor attack
    IMU_NOISE = auto()        # Increased IMU noise
    SPOOFING = auto()         # Coordinated GPS spoofing
    JAMMING = auto()          # GPS jamming (loss of signal)
    ACTUATOR_FAULT = auto()   # Actuator degradation


@dataclass
class AttackConfig:
    """Configuration for attack generation."""
    attack_type: AttackType
    intensity: float = 1.0        # Attack strength multiplier
    onset_time: int = 50          # When attack starts (timesteps)
    duration: int = 100           # Attack duration (timesteps)
    ramp_time: int = 10           # Ramp-up time for gradual attacks
    physics_compliance: float = 0.8  # How much to respect physics (0-1)


@dataclass
class CounterfactualSample:
    """A generated counterfactual attack sample."""
    original: np.ndarray         # [T, state_dim] original trajectory
    attacked: np.ndarray         # [T, state_dim] attacked trajectory
    attack_type: AttackType
    attack_mask: np.ndarray      # [T] binary mask of attack periods
    intensity: float
    physics_violation: float     # How much physics was violated (0 = consistent)


class PhysicsConsistentGenerator:
    """
    Generates attack samples that respect physics constraints.

    Uses PINN predictions to guide perturbations and ensure
    generated attacks are physically plausible.
    """

    def __init__(
        self,
        state_dim: int = 12,
        dt: float = 0.005,
        device: str = 'cpu',
    ):
        self.state_dim = state_dim
        self.dt = dt
        self.device = device

        # Simple dynamics model for physics consistency
        self.dynamics_model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
        ).to(device)

    def generate(
        self,
        trajectory: np.ndarray,
        config: AttackConfig,
    ) -> CounterfactualSample:
        """
        Generate a physics-consistent attack sample.

        Args:
            trajectory: [T, state_dim] nominal trajectory
            config: Attack configuration

        Returns:
            CounterfactualSample with attacked trajectory
        """
        T = trajectory.shape[0]
        attacked = trajectory.copy()
        attack_mask = np.zeros(T, dtype=np.float32)

        # Determine attack window
        start = config.onset_time
        end = min(start + config.duration, T)

        if start >= T:
            return CounterfactualSample(
                original=trajectory,
                attacked=trajectory,
                attack_type=config.attack_type,
                attack_mask=attack_mask,
                intensity=0.0,
                physics_violation=0.0,
            )

        # Generate attack perturbation
        perturbation = self._generate_perturbation(
            trajectory[start:end],
            config,
        )

        # Apply with physics compliance
        attacked[start:end] = self._apply_with_physics(
            trajectory[start:end],
            perturbation,
            config.physics_compliance,
        )

        attack_mask[start:end] = 1.0

        # Compute physics violation
        physics_violation = self._compute_physics_violation(
            trajectory[start:end],
            attacked[start:end],
        )

        return CounterfactualSample(
            original=trajectory,
            attacked=attacked,
            attack_type=config.attack_type,
            attack_mask=attack_mask,
            intensity=config.intensity,
            physics_violation=physics_violation,
        )

    def _generate_perturbation(
        self,
        segment: np.ndarray,
        config: AttackConfig,
    ) -> np.ndarray:
        """Generate attack perturbation based on type."""
        T = segment.shape[0]
        perturbation = np.zeros_like(segment)

        if config.attack_type == AttackType.GPS_DRIFT:
            # Slow drift in position (first 3 states)
            drift_rate = 0.1 * config.intensity
            for t in range(T):
                perturbation[t, :3] = drift_rate * t * self.dt

        elif config.attack_type == AttackType.GPS_JUMP:
            # Sudden position jump
            jump = np.array([2.0, 2.0, 0.5]) * config.intensity
            ramp = min(config.ramp_time, T)
            for t in range(ramp):
                perturbation[t, :3] = jump * (t / ramp)
            perturbation[ramp:, :3] = jump

        elif config.attack_type == AttackType.GPS_FREEZE:
            # Freeze at initial position
            initial_pos = segment[0, :3]
            for t in range(T):
                perturbation[t, :3] = initial_pos - segment[t, :3]

        elif config.attack_type == AttackType.IMU_BIAS:
            # Constant bias on accelerometer (affects velocity)
            bias = np.array([0.0, 0.0, 0.5]) * config.intensity
            perturbation[:, 3:6] = bias

        elif config.attack_type == AttackType.IMU_SCALE:
            # Scale factor error
            scale = 0.1 * config.intensity
            perturbation[:, 3:6] = segment[:, 3:6] * scale
            perturbation[:, 9:12] = segment[:, 9:12] * scale

        elif config.attack_type == AttackType.IMU_NOISE:
            # Increased noise
            noise_std = 0.5 * config.intensity
            perturbation[:, 3:6] = np.random.randn(T, 3) * noise_std
            perturbation[:, 9:12] = np.random.randn(T, 3) * noise_std * 0.1

        elif config.attack_type == AttackType.SPOOFING:
            # Coordinated spoofing (position and velocity)
            spoof_pos = np.array([5.0, 5.0, 0.0]) * config.intensity
            spoof_vel = np.array([1.0, 1.0, 0.0]) * config.intensity
            perturbation[:, :3] = spoof_pos
            perturbation[:, 3:6] = spoof_vel

        elif config.attack_type == AttackType.JAMMING:
            # Jamming (add large noise to GPS)
            noise_std = 10.0 * config.intensity
            perturbation[:, :3] = np.random.randn(T, 3) * noise_std

        elif config.attack_type == AttackType.ACTUATOR_FAULT:
            # Actuator degradation (affects dynamics response)
            degradation = 0.3 * config.intensity
            perturbation[:, 3:6] = -segment[:, 3:6] * degradation

        return perturbation

    def _apply_with_physics(
        self,
        original: np.ndarray,
        perturbation: np.ndarray,
        physics_compliance: float,
    ) -> np.ndarray:
        """Apply perturbation while respecting physics."""
        T = original.shape[0]
        result = original + perturbation

        if physics_compliance < 0.5:
            # Low compliance - just apply perturbation
            return result

        # High compliance - enforce physics consistency
        for t in range(1, T):
            # Position should be consistent with velocity
            expected_pos = result[t-1, :3] + result[t-1, 3:6] * self.dt
            pos_error = result[t, :3] - expected_pos

            # Blend towards physics-consistent
            blend = physics_compliance
            result[t, :3] = (1 - blend) * result[t, :3] + blend * expected_pos

            # Velocity should be somewhat consistent with acceleration
            if t > 1:
                expected_vel = result[t-1, 3:6] + (result[t-1, 3:6] - result[t-2, 3:6])
                result[t, 3:6] = (1 - blend) * result[t, 3:6] + blend * expected_vel

        return result

    def _compute_physics_violation(
        self,
        original: np.ndarray,
        attacked: np.ndarray,
    ) -> float:
        """Compute how much physics constraints are violated."""
        T = original.shape[0]

        violations = []
        for t in range(1, T):
            # Check position-velocity consistency
            expected_pos = attacked[t-1, :3] + attacked[t-1, 3:6] * self.dt
            pos_error = np.linalg.norm(attacked[t, :3] - expected_pos)
            violations.append(pos_error)

        return float(np.mean(violations)) if violations else 0.0


# =============================================================================
# Phase 4.2: Stress Test Suite
# =============================================================================

@dataclass
class StressTestResult:
    """Result of a single stress test."""
    attack_type: AttackType
    intensity: float
    detection_rate: float
    false_positive_rate: float
    mean_detection_delay: float
    physics_violation: float


@dataclass
class StressTestSuite:
    """Complete stress test suite results."""
    results: Dict[str, List[StressTestResult]]
    coverage: Dict[str, float]
    weaknesses: List[str]


class RobustnessStressTester:
    """
    Comprehensive robustness stress testing.

    Tests detector against diverse attack types, intensities,
    and edge cases.
    """

    def __init__(
        self,
        detector_fn: Optional[callable] = None,
        generator: Optional[PhysicsConsistentGenerator] = None,
    ):
        self.generator = generator or PhysicsConsistentGenerator()
        self.detector_fn = detector_fn or self._default_detector

        # Attack types to test
        self.attack_types = list(AttackType)

        # Intensity levels
        self.intensity_levels = [0.5, 1.0, 2.0, 5.0]

    def _default_detector(self, trajectory: np.ndarray) -> np.ndarray:
        """Default detector for testing (simple residual-based)."""
        T = trajectory.shape[0]
        scores = np.zeros(T)

        for t in range(1, T):
            # Simple velocity-based residual
            expected_pos = trajectory[t-1, :3] + trajectory[t-1, 3:6] * 0.005
            residual = np.linalg.norm(trajectory[t, :3] - expected_pos)
            scores[t] = residual

        return scores

    def run_stress_tests(
        self,
        nominal_trajectories: np.ndarray,
        n_samples_per_attack: int = 10,
    ) -> StressTestSuite:
        """
        Run comprehensive stress tests.

        Args:
            nominal_trajectories: [N, T, state_dim] nominal data
            n_samples_per_attack: Samples per attack type/intensity

        Returns:
            StressTestSuite with all results
        """
        results = {}

        for attack_type in self.attack_types:
            results[attack_type.name] = []

            for intensity in self.intensity_levels:
                # Generate attack samples
                config = AttackConfig(
                    attack_type=attack_type,
                    intensity=intensity,
                )

                detections = []
                false_positives = []
                delays = []
                physics_violations = []

                for i in range(min(n_samples_per_attack, len(nominal_trajectories))):
                    traj = nominal_trajectories[i]

                    # Generate counterfactual
                    sample = self.generator.generate(traj, config)

                    # Run detector
                    scores = self.detector_fn(sample.attacked)

                    # Compute metrics
                    threshold = 0.5
                    detected = np.any(scores[sample.attack_mask > 0] > threshold)
                    detections.append(float(detected))

                    # False positives (detections before attack)
                    pre_attack = sample.attack_mask == 0
                    fp = np.sum(scores[pre_attack] > threshold) / max(1, np.sum(pre_attack))
                    false_positives.append(fp)

                    # Detection delay
                    if detected:
                        attack_start = np.argmax(sample.attack_mask > 0)
                        detect_idx = np.argmax(scores > threshold)
                        delay = max(0, detect_idx - attack_start) * 0.005  # to seconds
                        delays.append(delay)

                    physics_violations.append(sample.physics_violation)

                result = StressTestResult(
                    attack_type=attack_type,
                    intensity=intensity,
                    detection_rate=np.mean(detections),
                    false_positive_rate=np.mean(false_positives),
                    mean_detection_delay=np.mean(delays) if delays else float('inf'),
                    physics_violation=np.mean(physics_violations),
                )
                results[attack_type.name].append(result)

        # Compute coverage metrics
        coverage = self._compute_coverage(results)

        # Identify weaknesses
        weaknesses = self._identify_weaknesses(results)

        return StressTestSuite(
            results=results,
            coverage=coverage,
            weaknesses=weaknesses,
        )

    def _compute_coverage(
        self,
        results: Dict[str, List[StressTestResult]],
    ) -> Dict[str, float]:
        """Compute coverage metrics."""
        coverage = {}

        for attack_name, attack_results in results.items():
            # Coverage = fraction of intensities with >50% detection
            detected = sum(1 for r in attack_results if r.detection_rate > 0.5)
            coverage[attack_name] = detected / len(attack_results)

        coverage['overall'] = np.mean(list(coverage.values()))

        return coverage

    def _identify_weaknesses(
        self,
        results: Dict[str, List[StressTestResult]],
    ) -> List[str]:
        """Identify detector weaknesses."""
        weaknesses = []

        for attack_name, attack_results in results.items():
            # Check for low detection at high intensity
            high_intensity_results = [r for r in attack_results if r.intensity >= 2.0]
            if high_intensity_results:
                mean_detection = np.mean([r.detection_rate for r in high_intensity_results])
                if mean_detection < 0.5:
                    weaknesses.append(f"Low detection for {attack_name} at high intensity")

            # Check for high false positive rate
            mean_fpr = np.mean([r.false_positive_rate for r in attack_results])
            if mean_fpr > 0.1:
                weaknesses.append(f"High FPR ({mean_fpr:.2%}) for {attack_name}")

            # Check for long detection delay
            mean_delay = np.mean([
                r.mean_detection_delay for r in attack_results
                if r.mean_detection_delay < float('inf')
            ] or [float('inf')])
            if mean_delay > 0.5:  # 500ms
                weaknesses.append(f"Long detection delay ({mean_delay:.2f}s) for {attack_name}")

        return weaknesses

    def generate_edge_cases(
        self,
        trajectory: np.ndarray,
    ) -> List[CounterfactualSample]:
        """Generate edge case samples for comprehensive testing."""
        edge_cases = []

        # Edge case 1: Very short attack
        edge_cases.append(self.generator.generate(
            trajectory,
            AttackConfig(
                attack_type=AttackType.GPS_JUMP,
                duration=5,  # Very short
                intensity=2.0,
            ),
        ))

        # Edge case 2: Very subtle attack
        edge_cases.append(self.generator.generate(
            trajectory,
            AttackConfig(
                attack_type=AttackType.GPS_DRIFT,
                intensity=0.1,  # Very subtle
                duration=200,
            ),
        ))

        # Edge case 3: Attack at trajectory start
        edge_cases.append(self.generator.generate(
            trajectory,
            AttackConfig(
                attack_type=AttackType.IMU_BIAS,
                onset_time=0,
                duration=50,
            ),
        ))

        # Edge case 4: Rapid attack sequence
        for i in range(3):
            edge_cases.append(self.generator.generate(
                trajectory,
                AttackConfig(
                    attack_type=AttackType.GPS_JUMP,
                    onset_time=i * 30,
                    duration=10,
                    intensity=1.5,
                ),
            ))

        return edge_cases


def evaluate_robustness(
    nominal_trajectories: np.ndarray,
    detector_fn: callable,
) -> Dict:
    """
    Evaluate detector robustness.

    Args:
        nominal_trajectories: [N, T, state_dim] nominal data
        detector_fn: Function mapping trajectory to detection scores

    Returns:
        Robustness evaluation metrics
    """
    tester = RobustnessStressTester(detector_fn=detector_fn)

    # Run full stress test suite
    suite = tester.run_stress_tests(nominal_trajectories, n_samples_per_attack=5)

    # Aggregate results
    all_detection_rates = []
    all_fprs = []

    for attack_results in suite.results.values():
        for result in attack_results:
            all_detection_rates.append(result.detection_rate)
            all_fprs.append(result.false_positive_rate)

    return {
        'coverage': suite.coverage,
        'mean_detection_rate': np.mean(all_detection_rates),
        'mean_fpr': np.mean(all_fprs),
        'weaknesses': suite.weaknesses,
        'n_weaknesses': len(suite.weaknesses),
        'attack_types_tested': len(suite.results),
        'total_test_cases': sum(len(r) for r in suite.results.values()),
    }
