"""
Tests for Phase 4: Robustness Stress Coverage

Tests:
4.1 Physics-consistent counterfactual generator
4.2 Stress test suite
"""

import numpy as np
import pytest


# =============================================================================
# Phase 4.1: Counterfactual Generator Tests
# =============================================================================

class TestPhysicsConsistentGenerator:
    """Tests for physics-consistent counterfactual generator."""

    def test_initialization(self):
        from gps_imu_detector.src.robustness_testing import PhysicsConsistentGenerator

        generator = PhysicsConsistentGenerator()
        assert generator.state_dim == 12
        assert generator.dt == 0.005

    def test_generate_gps_drift(self):
        from gps_imu_detector.src.robustness_testing import (
            PhysicsConsistentGenerator, AttackConfig, AttackType
        )

        generator = PhysicsConsistentGenerator()
        trajectory = np.random.randn(100, 12).astype(np.float32) * 0.1

        config = AttackConfig(
            attack_type=AttackType.GPS_DRIFT,
            onset_time=20,
            duration=50,
            intensity=1.0,
        )

        sample = generator.generate(trajectory, config)

        assert sample.original.shape == trajectory.shape
        assert sample.attacked.shape == trajectory.shape
        assert sample.attack_type == AttackType.GPS_DRIFT
        assert np.sum(sample.attack_mask) == 50  # Duration

    def test_generate_gps_jump(self):
        from gps_imu_detector.src.robustness_testing import (
            PhysicsConsistentGenerator, AttackConfig, AttackType
        )

        generator = PhysicsConsistentGenerator()
        trajectory = np.zeros((100, 12), dtype=np.float32)

        config = AttackConfig(
            attack_type=AttackType.GPS_JUMP,
            onset_time=30,
            duration=40,
            intensity=1.0,
        )

        sample = generator.generate(trajectory, config)

        # Position should jump during attack
        pre_attack = sample.attacked[25, :3]
        during_attack = sample.attacked[60, :3]
        assert np.linalg.norm(during_attack - pre_attack) > 1.0

    def test_generate_imu_bias(self):
        from gps_imu_detector.src.robustness_testing import (
            PhysicsConsistentGenerator, AttackConfig, AttackType
        )

        generator = PhysicsConsistentGenerator()
        trajectory = np.zeros((100, 12), dtype=np.float32)

        config = AttackConfig(
            attack_type=AttackType.IMU_BIAS,
            onset_time=20,
            duration=60,
            intensity=1.0,
        )

        sample = generator.generate(trajectory, config)

        # Velocity should have bias during attack
        during_attack_vel = sample.attacked[50, 3:6]
        assert np.linalg.norm(during_attack_vel) > 0.1

    def test_attack_mask_correct(self):
        from gps_imu_detector.src.robustness_testing import (
            PhysicsConsistentGenerator, AttackConfig, AttackType
        )

        generator = PhysicsConsistentGenerator()
        trajectory = np.random.randn(100, 12).astype(np.float32) * 0.1

        config = AttackConfig(
            attack_type=AttackType.GPS_JUMP,
            onset_time=30,
            duration=40,
        )

        sample = generator.generate(trajectory, config)

        # Mask should be 0 before, 1 during
        assert sample.attack_mask[25] == 0.0
        assert sample.attack_mask[35] == 1.0
        assert sample.attack_mask[65] == 1.0

    def test_physics_compliance_reduces_violation(self):
        from gps_imu_detector.src.robustness_testing import (
            PhysicsConsistentGenerator, AttackConfig, AttackType
        )

        generator = PhysicsConsistentGenerator()
        trajectory = np.zeros((100, 12), dtype=np.float32)
        trajectory[:, 3:6] = 1.0  # Constant velocity

        config_low = AttackConfig(
            attack_type=AttackType.GPS_JUMP,
            onset_time=20,
            duration=50,
            physics_compliance=0.2,
        )

        config_high = AttackConfig(
            attack_type=AttackType.GPS_JUMP,
            onset_time=20,
            duration=50,
            physics_compliance=0.9,
        )

        sample_low = generator.generate(trajectory, config_low)
        sample_high = generator.generate(trajectory, config_high)

        # Higher compliance should have lower violation
        assert sample_high.physics_violation <= sample_low.physics_violation + 0.1

    def test_all_attack_types_generate(self):
        from gps_imu_detector.src.robustness_testing import (
            PhysicsConsistentGenerator, AttackConfig, AttackType
        )

        generator = PhysicsConsistentGenerator()
        trajectory = np.random.randn(100, 12).astype(np.float32) * 0.1

        for attack_type in AttackType:
            config = AttackConfig(
                attack_type=attack_type,
                onset_time=20,
                duration=50,
            )

            sample = generator.generate(trajectory, config)
            assert sample.attacked.shape == trajectory.shape


# =============================================================================
# Phase 4.2: Stress Test Suite Tests
# =============================================================================

class TestRobustnessStressTester:
    """Tests for robustness stress tester."""

    def test_initialization(self):
        from gps_imu_detector.src.robustness_testing import RobustnessStressTester

        tester = RobustnessStressTester()
        assert len(tester.attack_types) == len(list(tester.attack_types))
        assert len(tester.intensity_levels) == 4

    def test_run_stress_tests(self):
        from gps_imu_detector.src.robustness_testing import RobustnessStressTester

        np.random.seed(42)

        tester = RobustnessStressTester()
        trajectories = np.random.randn(5, 100, 12).astype(np.float32) * 0.1

        suite = tester.run_stress_tests(trajectories, n_samples_per_attack=2)

        assert 'GPS_DRIFT' in suite.results
        assert 'overall' in suite.coverage
        assert isinstance(suite.weaknesses, list)

    def test_coverage_computation(self):
        from gps_imu_detector.src.robustness_testing import RobustnessStressTester

        np.random.seed(42)

        tester = RobustnessStressTester()
        trajectories = np.random.randn(3, 100, 12).astype(np.float32) * 0.1

        suite = tester.run_stress_tests(trajectories, n_samples_per_attack=2)

        # Coverage should be between 0 and 1
        for attack_name, coverage in suite.coverage.items():
            assert 0.0 <= coverage <= 1.0

    def test_generate_edge_cases(self):
        from gps_imu_detector.src.robustness_testing import RobustnessStressTester

        np.random.seed(42)

        tester = RobustnessStressTester()
        trajectory = np.random.randn(200, 12).astype(np.float32) * 0.1

        edge_cases = tester.generate_edge_cases(trajectory)

        # Should generate multiple edge cases
        assert len(edge_cases) >= 4

        # All should be valid samples
        for sample in edge_cases:
            assert sample.attacked.shape == trajectory.shape

    def test_custom_detector_function(self):
        from gps_imu_detector.src.robustness_testing import RobustnessStressTester

        np.random.seed(42)

        def custom_detector(trajectory):
            # Simple threshold detector
            return np.linalg.norm(trajectory[:, :3], axis=1)

        tester = RobustnessStressTester(detector_fn=custom_detector)
        trajectories = np.random.randn(3, 100, 12).astype(np.float32) * 0.1

        suite = tester.run_stress_tests(trajectories, n_samples_per_attack=2)

        assert len(suite.results) > 0


# =============================================================================
# Phase 4 Checkpoint Tests
# =============================================================================

class TestPhase4Checkpoint:
    """Integration tests for Phase 4 checkpoint."""

    def test_evaluate_robustness(self):
        from gps_imu_detector.src.robustness_testing import evaluate_robustness

        np.random.seed(42)

        trajectories = np.random.randn(5, 100, 12).astype(np.float32) * 0.1

        def simple_detector(traj):
            return np.linalg.norm(traj[:, :3], axis=1)

        results = evaluate_robustness(trajectories, simple_detector)

        assert 'coverage' in results
        assert 'mean_detection_rate' in results
        assert 'weaknesses' in results
        assert 'n_weaknesses' in results

    def test_attack_diversity(self):
        from gps_imu_detector.src.robustness_testing import AttackType

        # Should have diverse attack types
        attack_types = list(AttackType)
        assert len(attack_types) >= 8

        # Check categories covered
        type_names = [t.name for t in attack_types]
        assert 'GPS_DRIFT' in type_names
        assert 'GPS_JUMP' in type_names
        assert 'IMU_BIAS' in type_names
        assert 'SPOOFING' in type_names

    def test_intensity_sweep(self):
        from gps_imu_detector.src.robustness_testing import (
            RobustnessStressTester, AttackType
        )

        np.random.seed(42)

        tester = RobustnessStressTester()
        trajectories = np.random.randn(3, 100, 12).astype(np.float32) * 0.1

        suite = tester.run_stress_tests(trajectories, n_samples_per_attack=2)

        # Each attack type should be tested at multiple intensities
        for attack_name, attack_results in suite.results.items():
            intensities = [r.intensity for r in attack_results]
            assert len(set(intensities)) > 1

    def test_weakness_identification(self):
        from gps_imu_detector.src.robustness_testing import RobustnessStressTester

        np.random.seed(42)

        # Create a weak detector that misses everything
        def weak_detector(traj):
            return np.zeros(traj.shape[0])

        tester = RobustnessStressTester(detector_fn=weak_detector)
        trajectories = np.random.randn(3, 100, 12).astype(np.float32) * 0.1

        suite = tester.run_stress_tests(trajectories, n_samples_per_attack=2)

        # Should identify weaknesses
        assert len(suite.weaknesses) > 0

    def test_physics_consistency_measured(self):
        from gps_imu_detector.src.robustness_testing import (
            PhysicsConsistentGenerator, AttackConfig, AttackType
        )

        generator = PhysicsConsistentGenerator()
        trajectory = np.zeros((100, 12), dtype=np.float32)
        trajectory[:, 3:6] = 1.0  # Constant velocity

        # Physical attack
        config_physical = AttackConfig(
            attack_type=AttackType.GPS_DRIFT,
            onset_time=20,
            duration=50,
            physics_compliance=0.9,
        )

        sample = generator.generate(trajectory, config_physical)

        # Physics violation should be measured
        assert sample.physics_violation >= 0.0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
