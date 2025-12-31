"""
Tests for Final Improvements Module (v0.5.1)

Tests:
1. Fault Persistence Scoring
2. Cost-Aware Asymmetric Thresholds
3. TTD Metrics
4. Controller-in-Loop Prediction
5. Cross-Axis Coupling
"""

import numpy as np
import pytest
from gps_imu_detector.src.final_improvements import (
    # Persistence
    FaultPersistenceScorer,
    PersistenceResult,
    # Thresholds
    FaultClass,
    AsymmetricThresholds,
    CostAwareThresholder,
    CostAwareResult,
    # TTD
    TTDMetrics,
    TTDAnalyzer,
    # Combined
    FinalDetector,
    FinalDetectionResult,
    # Controller
    ControllerPredictor,
    ControllerResidualResult,
    # Cross-axis
    CrossAxisCouplingChecker,
    CrossAxisResult,
    # Evaluation
    evaluate_with_final_improvements,
)


# =============================================================================
# Persistence Scoring Tests
# =============================================================================

class TestFaultPersistenceScorer:
    """Tests for fault persistence scoring."""

    def test_initialization(self):
        scorer = FaultPersistenceScorer(k=3, n=10)
        assert scorer.k == 3
        assert scorer.n == 10

    def test_single_alarm_not_persistent(self):
        scorer = FaultPersistenceScorer(k=3, n=10, base_threshold=0.5)

        # Single high score
        result = scorer.update(0.8)
        assert result.raw_score == 0.8
        assert result.alarm_count == 1
        assert not result.is_persistent  # Need k=3

    def test_persistent_after_k_alarms(self):
        scorer = FaultPersistenceScorer(k=3, n=10, base_threshold=0.5)

        # Send k high scores
        for i in range(3):
            result = scorer.update(0.8)

        assert result.alarm_count >= 3
        assert result.is_persistent

    def test_transient_filtered(self):
        scorer = FaultPersistenceScorer(k=3, n=10, base_threshold=0.5)

        # High-low-high pattern (transient)
        scorer.update(0.8)  # High
        scorer.update(0.2)  # Low
        result = scorer.update(0.8)  # High

        assert result.alarm_count == 2
        assert not result.is_persistent

    def test_reset_clears_history(self):
        scorer = FaultPersistenceScorer(k=3, n=10)

        for _ in range(5):
            scorer.update(0.8)

        assert len(scorer.history) == 5

        scorer.reset()
        assert len(scorer.history) == 0
        assert scorer.smoothed_score == 0.0

    def test_persistence_ratio(self):
        scorer = FaultPersistenceScorer(k=3, n=10, base_threshold=0.5)

        # 5 high, 5 low
        for _ in range(5):
            scorer.update(0.8)
        for _ in range(5):
            scorer.update(0.2)

        ratio = scorer.get_persistence_ratio()
        assert 0.4 <= ratio <= 0.6  # ~50%


# =============================================================================
# Cost-Aware Thresholds Tests
# =============================================================================

class TestCostAwareThresholder:
    """Tests for cost-aware asymmetric thresholds."""

    def test_initialization(self):
        thresholder = CostAwareThresholder()
        assert FaultClass.ACTUATOR in thresholder.cost_weights

    def test_actuator_has_lower_threshold(self):
        thresholds = AsymmetricThresholds()
        assert thresholds.actuator < thresholds.sensor
        assert thresholds.actuator < thresholds.default

    def test_actuator_detected_at_lower_score(self):
        thresholder = CostAwareThresholder()

        # Same score, different classes
        actuator_result = thresholder.detect(0.35, FaultClass.ACTUATOR)
        sensor_result = thresholder.detect(0.35, FaultClass.SENSOR)

        assert actuator_result.is_detected  # Threshold 0.3
        assert not sensor_result.is_detected  # Threshold 0.45

    def test_cost_weights_adjust_score(self):
        thresholder = CostAwareThresholder()

        result = thresholder.detect(0.5, FaultClass.ACTUATOR)

        # Actuator has cost weight 3.0
        assert result.adjusted_score == 0.5 * 3.0

    def test_get_threshold_returns_correct_values(self):
        thresholder = CostAwareThresholder()

        assert thresholder.get_threshold(FaultClass.ACTUATOR) == 0.3
        assert thresholder.get_threshold(FaultClass.GPS) == 0.4
        assert thresholder.get_threshold(FaultClass.UNKNOWN) == 0.5


# =============================================================================
# TTD Metrics Tests
# =============================================================================

class TestTTDAnalyzer:
    """Tests for time-to-detection analysis."""

    def test_initialization(self):
        analyzer = TTDAnalyzer(dt=0.005)
        assert analyzer.dt == 0.005

    def test_compute_ttd_simple(self):
        analyzer = TTDAnalyzer(dt=0.005)

        # Scores ramp up after onset
        scores = np.concatenate([
            np.zeros(50),  # Clean
            np.linspace(0, 1, 50),  # Ramp after fault
        ])
        labels = np.concatenate([np.zeros(50), np.ones(50)])
        fault_onsets = np.array([50])

        metrics = analyzer.compute_ttd(scores, labels, fault_onsets, threshold=0.5)

        assert metrics.n_detected == 1
        assert metrics.n_total == 1
        assert metrics.median_ttd < 50  # Should detect before end

    def test_ttd_to_seconds(self):
        analyzer = TTDAnalyzer(dt=0.005)

        # 100 samples at 200 Hz = 0.5 seconds
        assert analyzer.ttd_to_seconds(100) == 0.5

    def test_no_detections_returns_inf(self):
        analyzer = TTDAnalyzer()

        scores = np.zeros(100)  # Never triggers
        labels = np.concatenate([np.zeros(50), np.ones(50)])
        fault_onsets = np.array([50])

        metrics = analyzer.compute_ttd(scores, labels, fault_onsets, threshold=0.5)

        assert metrics.n_detected == 0
        assert metrics.median_ttd == float('inf')

    def test_format_report(self):
        analyzer = TTDAnalyzer()

        metrics = TTDMetrics(
            median_ttd=20,
            ttd_95=40,
            ttd_99=60,
            mean_ttd=25,
            std_ttd=10,
            energy_at_detection=0.5,
            n_detected=8,
            n_total=10,
            detection_rate=0.8,
        )

        report = analyzer.format_ttd_report(metrics)
        assert "Median TTD" in report
        assert "80.0%" in report


# =============================================================================
# Controller Predictor Tests
# =============================================================================

class TestControllerPredictor:
    """Tests for controller-in-loop prediction."""

    def test_initialization(self):
        predictor = ControllerPredictor(state_dim=6, control_dim=4)
        assert predictor.state_dim == 6
        assert predictor.control_dim == 4

    def test_predict_without_fit(self):
        predictor = ControllerPredictor()

        state = np.zeros(6)
        state_dot = np.zeros(6)
        control = np.ones(4)

        result = predictor.predict(state, state_dot, control)

        assert not result.is_anomalous
        assert result.control_residual == 0.0

    def test_fit_and_predict(self):
        predictor = ControllerPredictor(state_dim=3, control_dim=2)

        # Generate simple training data
        np.random.seed(42)
        n = 100
        states = np.random.randn(n, 3)
        state_dots = np.random.randn(n, 3)
        controls = states[:, :2] * 0.5 + np.random.randn(n, 2) * 0.1

        predictor.fit(states, state_dots, controls)

        # Predict on similar data
        result = predictor.predict(
            np.zeros(3),
            np.zeros(3),
            np.zeros(2),
        )

        assert predictor.K is not None
        assert isinstance(result, ControllerResidualResult)

    def test_anomalous_control_detected(self):
        predictor = ControllerPredictor(state_dim=3, control_dim=2, threshold=0.3)

        # Fit on nominal data
        np.random.seed(42)
        n = 100
        states = np.random.randn(n, 3) * 0.1
        state_dots = np.random.randn(n, 3) * 0.1
        controls = np.ones((n, 2)) * 5  # Constant control

        predictor.fit(states, state_dots, controls)

        # Test with very different control
        result = predictor.predict(
            np.zeros(3),
            np.zeros(3),
            np.ones(2) * 100,  # Way off
        )

        assert result.control_residual > 1.0  # High residual


# =============================================================================
# Cross-Axis Coupling Tests
# =============================================================================

class TestCrossAxisCouplingChecker:
    """Tests for cross-axis coupling consistency."""

    def test_initialization(self):
        checker = CrossAxisCouplingChecker(window_size=100)
        assert checker.window_size == 100

    def test_compute_correlation(self):
        checker = CrossAxisCouplingChecker()

        # Perfect correlation
        x = np.arange(100).astype(float)
        y = x * 2

        corr = checker.compute_correlation(x, y)
        assert abs(corr - 1.0) < 0.01

    def test_compute_correlation_negative(self):
        checker = CrossAxisCouplingChecker()

        x = np.arange(100).astype(float)
        y = -x

        corr = checker.compute_correlation(x, y)
        assert abs(corr - (-1.0)) < 0.01

    def test_check_coupling_nominal(self):
        checker = CrossAxisCouplingChecker()

        # Generate coupled data
        np.random.seed(42)
        n = 200
        t = np.linspace(0, 10, n)

        # omega_x correlates with v_y (roll causes lateral motion)
        omega = np.column_stack([
            np.sin(t),
            np.cos(t),
            np.zeros(n),
        ])
        velocity = np.column_stack([
            np.zeros(n),
            np.sin(t) * 0.8 + np.random.randn(n) * 0.1,  # Correlated with omega_x
            np.zeros(n),
        ])
        acceleration = np.column_stack([
            np.zeros(n),
            np.zeros(n),
            np.cos(t) * 0.8 + np.random.randn(n) * 0.1,  # Correlated with omega_y
        ])

        result = checker.check_coupling(omega, velocity, acceleration)

        assert isinstance(result, CrossAxisResult)
        assert abs(result.roll_lateral_corr) > 0.5  # Should be correlated

    def test_calibrate(self):
        checker = CrossAxisCouplingChecker()

        np.random.seed(42)
        n = 500
        omega = np.random.randn(n, 3)
        velocity = np.random.randn(n, 3)
        acceleration = np.random.randn(n, 3)

        checker.calibrate(omega, velocity, acceleration)

        # Baseline should be updated
        assert checker.baseline_correlations["roll_lateral"] != 0.5


# =============================================================================
# Final Detector Tests
# =============================================================================

class TestFinalDetector:
    """Tests for combined final detector."""

    def test_initialization(self):
        detector = FinalDetector()
        assert detector.persistence is not None
        assert detector.thresholder is not None

    def test_detect_returns_result(self):
        detector = FinalDetector()

        result = detector.detect(0.5)

        assert isinstance(result, FinalDetectionResult)
        assert hasattr(result, 'is_final_detection')
        assert hasattr(result, 'detection_confidence')

    def test_persistent_detection(self):
        detector = FinalDetector(
            persistence_k=3,
            persistence_n=10,
            base_threshold=0.4,
        )

        # Send multiple high scores for actuator
        for _ in range(5):
            result = detector.detect(0.6, FaultClass.ACTUATOR)

        assert result.is_final_detection
        assert result.detection_confidence > 0.5

    def test_reset(self):
        detector = FinalDetector()

        for _ in range(5):
            detector.detect(0.7)

        assert len(detector.score_history) == 5

        detector.reset()

        assert len(detector.score_history) == 0


# =============================================================================
# Evaluation Function Tests
# =============================================================================

class TestEvaluation:
    """Tests for evaluation function."""

    def test_evaluate_returns_metrics(self):
        np.random.seed(42)

        scores = np.concatenate([
            np.random.rand(50) * 0.3,  # Clean
            np.random.rand(50) * 0.7 + 0.3,  # Attack
        ])
        labels = np.concatenate([np.zeros(50), np.ones(50)])
        fault_onsets = np.array([50])

        results = evaluate_with_final_improvements(scores, labels, fault_onsets)

        assert "recall" in results
        assert "fpr" in results
        assert "ttd" in results
        assert "persistence" in results

    def test_recall_in_valid_range(self):
        np.random.seed(42)

        scores = np.concatenate([
            np.random.rand(100) * 0.3,
            np.random.rand(100) * 0.8,
        ])
        labels = np.concatenate([np.zeros(100), np.ones(100)])
        fault_onsets = np.array([100])

        results = evaluate_with_final_improvements(scores, labels, fault_onsets)

        assert 0 <= results["recall"] <= 1
        assert 0 <= results["fpr"] <= 1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
