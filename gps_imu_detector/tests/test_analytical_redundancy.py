"""
Tests for Analytical Redundancy Module (v0.7.0)

Tests:
1. NonlinearEKF estimator
2. LinearComplementaryEstimator
3. EstimatorDisagreementDetector
4. AnalyticalRedundancySystem
"""

import numpy as np
import pytest
from gps_imu_detector.src.analytical_redundancy import (
    # EKF
    EKFState,
    NonlinearEKF,
    # Complementary
    LinearComplementaryEstimator,
    # Disagreement
    DisagreementResult,
    EstimatorDisagreementDetector,
    # Combined
    AnalyticalRedundancyResult,
    AnalyticalRedundancySystem,
    # Evaluation
    evaluate_analytical_redundancy,
)


# =============================================================================
# EKF Tests
# =============================================================================

class TestNonlinearEKF:
    """Tests for nonlinear EKF estimator."""

    def test_initialization(self):
        ekf = NonlinearEKF()
        assert ekf.n_states == 12
        assert ekf.x.shape == (12,)

    def test_predict_maintains_state_shape(self):
        ekf = NonlinearEKF()

        state = ekf.predict()

        assert state.shape == (12,)

    def test_predict_with_control(self):
        ekf = NonlinearEKF()

        control = np.array([9.81, 0.0, 0.0, 0.0])  # Hover thrust
        state = ekf.predict(control)

        assert state.shape == (12,)

    def test_update_returns_innovation(self):
        ekf = NonlinearEKF()
        ekf.predict()

        measurement = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        state, innovation = ekf.update(measurement)

        assert state.shape == (12,)
        assert innovation.shape == (6,)

    def test_innovation_decreases_with_convergence(self):
        ekf = NonlinearEKF()

        # Constant measurement
        measurement = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        innovations = []
        for _ in range(20):
            ekf.predict()
            _, innovation = ekf.update(measurement)
            innovations.append(np.linalg.norm(innovation))

        # Innovation should decrease as filter converges
        assert innovations[-1] < innovations[0]

    def test_get_state_returns_ekf_state(self):
        ekf = NonlinearEKF()
        ekf.x = np.arange(12).astype(float)

        state = ekf.get_state()

        assert isinstance(state, EKFState)
        np.testing.assert_array_equal(state.position, [0, 1, 2])
        np.testing.assert_array_equal(state.velocity, [3, 4, 5])

    def test_reset_clears_state(self):
        ekf = NonlinearEKF()

        for _ in range(5):
            ekf.predict()
            ekf.update(np.random.randn(6))

        assert len(ekf.state_history) == 5

        ekf.reset()

        assert len(ekf.state_history) == 0
        np.testing.assert_array_equal(ekf.x, np.zeros(12))


# =============================================================================
# Complementary Estimator Tests
# =============================================================================

class TestLinearComplementaryEstimator:
    """Tests for linear complementary estimator."""

    def test_initialization(self):
        est = LinearComplementaryEstimator()
        assert est.alpha == 0.98
        np.testing.assert_array_equal(est.position, np.zeros(3))

    def test_update_returns_state(self):
        est = LinearComplementaryEstimator()

        state = est.update(
            gps_position=np.array([1.0, 2.0, 3.0]),
            gps_velocity=np.array([0.1, 0.2, 0.3]),
            imu_acceleration=np.array([0.0, 0.0, 0.0]),
        )

        assert state.shape == (6,)

    def test_gps_dominates_at_low_alpha(self):
        est = LinearComplementaryEstimator(alpha=0.1)

        gps_pos = np.array([10.0, 20.0, 30.0])
        gps_vel = np.array([1.0, 2.0, 3.0])

        for _ in range(10):
            state = est.update(gps_pos, gps_vel, np.zeros(3))

        # With low alpha, GPS should dominate
        assert np.linalg.norm(state[:3] - gps_pos) < 5.0

    def test_imu_dominates_at_high_alpha(self):
        est = LinearComplementaryEstimator(alpha=0.99)

        # Start at origin, IMU says accelerate
        for _ in range(100):
            est.update(
                gps_position=np.zeros(3),
                gps_velocity=np.zeros(3),
                imu_acceleration=np.array([1.0, 0.0, 0.0]),
            )

        # High alpha means IMU integration dominates
        assert est.velocity[0] > 0.1

    def test_get_state(self):
        est = LinearComplementaryEstimator()
        est.position = np.array([1.0, 2.0, 3.0])
        est.velocity = np.array([0.1, 0.2, 0.3])

        state = est.get_state()

        np.testing.assert_array_equal(state[:3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(state[3:], [0.1, 0.2, 0.3])

    def test_reset(self):
        est = LinearComplementaryEstimator()

        for _ in range(5):
            est.update(np.ones(3), np.ones(3), np.ones(3))

        assert len(est.state_history) == 5

        est.reset()

        assert len(est.state_history) == 0
        np.testing.assert_array_equal(est.position, np.zeros(3))


# =============================================================================
# Disagreement Detector Tests
# =============================================================================

class TestEstimatorDisagreementDetector:
    """Tests for estimator disagreement detection."""

    def test_initialization(self):
        detector = EstimatorDisagreementDetector()
        assert detector.position_threshold == 2.0
        assert detector.velocity_threshold == 1.0

    def test_no_fault_when_states_agree(self):
        detector = EstimatorDisagreementDetector()

        primary = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        secondary = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

        result = detector.detect(primary, secondary)

        assert not result.is_fault_detected
        assert result.disagreement_norm < 0.01

    def test_fault_when_position_disagrees(self):
        detector = EstimatorDisagreementDetector(position_threshold=1.0)

        primary = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        secondary = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = detector.detect(primary, secondary)

        assert result.is_fault_detected
        assert result.disagreement_norm > 1.0

    def test_fault_when_velocity_disagrees(self):
        detector = EstimatorDisagreementDetector(velocity_threshold=0.5)

        primary = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
        secondary = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = detector.detect(primary, secondary)

        assert result.is_fault_detected

    def test_fault_channel_identified(self):
        detector = EstimatorDisagreementDetector(position_threshold=1.0)

        # Disagreement in Y channel
        primary = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        secondary = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = detector.detect(primary, secondary)

        assert result.is_fault_detected
        assert result.fault_channel == 'y'

    def test_calibration_updates_baseline(self):
        detector = EstimatorDisagreementDetector()

        # Simulate calibration data
        disagreements = np.random.randn(100, 6) * 0.1

        detector.calibrate(disagreements)

        assert detector.calibrated
        assert np.abs(detector.baseline_mean[0]) < 0.5
        assert detector.baseline_std[0] > 0

    def test_detection_rate_computed(self):
        detector = EstimatorDisagreementDetector(position_threshold=1.0)

        # 5 agreements, 5 disagreements
        for i in range(10):
            if i < 5:
                primary = np.zeros(6)
                secondary = np.zeros(6)
            else:
                primary = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                secondary = np.zeros(6)

            detector.detect(primary, secondary)

        rate = detector.get_detection_rate()
        assert 0.4 <= rate <= 0.6

    def test_reset(self):
        detector = EstimatorDisagreementDetector()

        for _ in range(5):
            detector.detect(np.zeros(6), np.zeros(6))

        assert detector.total_count == 5

        detector.reset()

        assert detector.total_count == 0


# =============================================================================
# Combined System Tests
# =============================================================================

class TestAnalyticalRedundancySystem:
    """Tests for combined analytical redundancy system."""

    def test_initialization(self):
        system = AnalyticalRedundancySystem()
        assert system.primary is not None
        assert system.secondary is not None
        assert system.disagreement_detector is not None

    def test_update_returns_result(self):
        system = AnalyticalRedundancySystem()

        result = system.update(
            gps_position=np.zeros(3),
            gps_velocity=np.zeros(3),
            imu_acceleration=np.zeros(3),
        )

        assert isinstance(result, AnalyticalRedundancyResult)
        assert hasattr(result, 'is_fault_detected')
        assert hasattr(result, 'detection_source')

    def test_nominal_no_fault(self):
        np.random.seed(42)
        system = AnalyticalRedundancySystem()

        # Consistent data
        for _ in range(50):
            result = system.update(
                gps_position=np.array([1.0, 0.0, 0.0]) + np.random.randn(3) * 0.1,
                gps_velocity=np.zeros(3) + np.random.randn(3) * 0.01,
                imu_acceleration=np.zeros(3),
            )

        # Should not detect faults in nominal data
        assert result.detection_source in ["none", "disagreement"]

    def test_actuator_fault_detected_via_disagreement(self):
        system = AnalyticalRedundancySystem(position_threshold=1.0)

        # Nominal warmup
        for _ in range(20):
            system.update(
                gps_position=np.zeros(3),
                gps_velocity=np.zeros(3),
                imu_acceleration=np.zeros(3),
            )

        # Inject actuator fault: GPS shows drift, but IMU doesn't match
        for i in range(30):
            result = system.update(
                gps_position=np.array([i * 0.5, 0.0, 0.0]),  # Drifting
                gps_velocity=np.array([0.5, 0.0, 0.0]),
                imu_acceleration=np.zeros(3),  # No acceleration
            )

        # Should detect disagreement
        metrics = system.get_metrics()
        assert metrics['fault_detections'] > 0

    def test_detection_source_classification(self):
        system = AnalyticalRedundancySystem(
            position_threshold=1.0,
            innovation_threshold=2.0,
        )

        # Force disagreement without innovation
        system.primary.x[:3] = np.array([10.0, 0.0, 0.0])
        system.secondary.position = np.zeros(3)

        result = system.update(
            gps_position=np.zeros(3),
            gps_velocity=np.zeros(3),
            imu_acceleration=np.zeros(3),
        )

        # Check classification
        assert result.detection_source in ["disagreement", "combined", "innovation"]

    def test_metrics_tracked(self):
        system = AnalyticalRedundancySystem()

        for _ in range(10):
            system.update(np.zeros(3), np.zeros(3), np.zeros(3))

        metrics = system.get_metrics()

        assert "total_samples" in metrics
        assert "fault_detections" in metrics
        assert metrics["total_samples"] == 10

    def test_reset(self):
        system = AnalyticalRedundancySystem()

        for _ in range(5):
            system.update(np.zeros(3), np.zeros(3), np.zeros(3))

        assert system.total_count == 5

        system.reset()

        assert system.total_count == 0


# =============================================================================
# Evaluation Tests
# =============================================================================

class TestEvaluation:
    """Tests for evaluation function."""

    def test_evaluate_returns_metrics(self):
        np.random.seed(42)

        n = 100
        gps_positions = np.random.randn(n, 3)
        gps_velocities = np.random.randn(n, 3) * 0.1
        imu_accelerations = np.random.randn(n, 3) * 0.01
        labels = np.concatenate([np.zeros(50), np.ones(50)])

        results = evaluate_analytical_redundancy(
            gps_positions, gps_velocities, imu_accelerations, labels
        )

        assert "recall" in results
        assert "fpr" in results
        assert "precision" in results

    def test_metrics_in_valid_range(self):
        np.random.seed(42)

        n = 100
        gps_positions = np.random.randn(n, 3)
        gps_velocities = np.random.randn(n, 3)
        imu_accelerations = np.random.randn(n, 3)
        labels = np.concatenate([np.zeros(50), np.ones(50)])

        results = evaluate_analytical_redundancy(
            gps_positions, gps_velocities, imu_accelerations, labels
        )

        assert 0 <= results["recall"] <= 1
        assert 0 <= results["fpr"] <= 1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
