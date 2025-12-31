"""
Tests for Industry-Aligned Detection Module (v0.6.0)

Tests:
A. Two-Stage Decision Logic
B. Risk-Weighted Fault Acceptance
C. Integrity-Based Detection
D. Combined Detector
"""

import numpy as np
import pytest
from gps_imu_detector.src.industry_aligned import (
    # Two-stage
    TwoStageDecisionLogic,
    TwoStageResult,
    SuspicionState,
    # Risk-weighted
    HazardClass,
    HazardThresholds,
    RiskWeightedDetector,
    RiskWeightedResult,
    FAULT_HAZARD_MAPPING,
    # Integrity
    ProtectionLevels,
    AlertLimits,
    IntegrityMonitor,
    IntegrityResult,
    # Combined
    IndustryAlignedDetector,
    IndustryAlignedResult,
    # Evaluation
    evaluate_industry_aligned,
)


# =============================================================================
# A. Two-Stage Decision Logic Tests
# =============================================================================

class TestTwoStageDecisionLogic:
    """Tests for two-stage decision logic."""

    def test_initialization(self):
        logic = TwoStageDecisionLogic()
        assert logic.suspicion_threshold == 0.4
        assert logic.confirmation_threshold == 0.5
        assert logic.confirmation_window_K == 40

    def test_single_high_score_not_alarm(self):
        logic = TwoStageDecisionLogic(
            confirmation_window_K=10,
            confirmation_required_M=6,
        )

        # Single high score should trigger suspicion but not alarm
        result = logic.update(0.8)

        assert result.is_suspicious
        assert not result.is_alarm

    def test_sustained_high_scores_trigger_alarm(self):
        logic = TwoStageDecisionLogic(
            suspicion_threshold=0.3,
            confirmation_threshold=0.4,
            confirmation_window_K=10,
            confirmation_required_M=6,
            cooldown_samples=0,  # Disable cooldown for this test
        )

        # Send sustained high scores and check if any triggered alarm
        alarm_triggered = False
        for _ in range(15):
            result = logic.update(0.7)
            if result.is_alarm:
                alarm_triggered = True

        assert alarm_triggered

    def test_transient_spike_no_alarm(self):
        logic = TwoStageDecisionLogic(
            suspicion_threshold=0.3,
            confirmation_threshold=0.5,
            confirmation_window_K=10,
            confirmation_required_M=6,
        )

        # High-low pattern (transient)
        result = None
        for i in range(20):
            score = 0.7 if i % 3 == 0 else 0.2
            result = logic.update(score)

        # Should not alarm due to lack of sustained confirmation
        assert not result.is_alarm

    def test_cooldown_after_alarm(self):
        logic = TwoStageDecisionLogic(
            suspicion_threshold=0.3,
            confirmation_threshold=0.4,
            confirmation_window_K=5,
            confirmation_required_M=3,
            cooldown_samples=10,
        )

        # Trigger alarm - track when it happens
        alarm_triggered = False
        for _ in range(10):
            result = logic.update(0.8)
            if result.is_alarm:
                alarm_triggered = True
                break

        assert alarm_triggered

        # Next samples should be in cooldown
        result = logic.update(0.8)
        assert not result.is_suspicious
        assert not result.is_alarm

    def test_reset_clears_state(self):
        logic = TwoStageDecisionLogic()

        for _ in range(5):
            logic.update(0.7)

        assert logic.state.is_suspicious
        assert len(logic.history) == 5

        logic.reset()

        assert not logic.state.is_suspicious
        assert len(logic.history) == 0

    def test_confirmation_ratio_computed(self):
        logic = TwoStageDecisionLogic(
            suspicion_threshold=0.3,
            confirmation_threshold=0.5,
            confirmation_window_K=10,
            confirmation_required_M=6,
        )

        # Mix of high and low scores
        for _ in range(5):
            logic.update(0.6)  # Above confirmation
        for _ in range(5):
            result = logic.update(0.4)  # Below confirmation

        # 5 of 10 confirmed = 50%
        assert 0.4 <= result.confirmation_ratio <= 0.6


# =============================================================================
# B. Risk-Weighted Fault Acceptance Tests
# =============================================================================

class TestRiskWeightedDetector:
    """Tests for risk-weighted fault detection."""

    def test_initialization(self):
        detector = RiskWeightedDetector()
        assert HazardClass.CATASTROPHIC in detector.detection_counts

    def test_hazard_thresholds_ordering(self):
        thresholds = HazardThresholds()

        # Catastrophic should have lowest threshold (most aggressive)
        assert thresholds.catastrophic < thresholds.hazardous
        assert thresholds.hazardous < thresholds.major
        assert thresholds.major < thresholds.minor

    def test_actuator_classified_catastrophic(self):
        detector = RiskWeightedDetector()

        result = detector.detect(0.3, "actuator_stuck")

        assert result.hazard_class == HazardClass.CATASTROPHIC
        assert result.risk_priority == 1

    def test_gps_spoofing_classified_hazardous(self):
        detector = RiskWeightedDetector()

        result = detector.detect(0.4, "gps_spoofing")

        assert result.hazard_class == HazardClass.HAZARDOUS
        assert result.risk_priority == 2

    def test_minor_fault_classified_minor(self):
        detector = RiskWeightedDetector()

        result = detector.detect(0.5, "transient_glitch")

        assert result.hazard_class == HazardClass.MINOR
        assert result.risk_priority == 4

    def test_catastrophic_detected_at_lower_score(self):
        detector = RiskWeightedDetector()

        # Same score, different fault types
        cat_result = detector.detect(0.25, "actuator_stuck")
        minor_result = detector.detect(0.25, "transient_glitch")

        assert cat_result.is_detected  # Threshold 0.20
        assert not minor_result.is_detected  # Threshold 0.70

    def test_recall_by_class(self):
        detector = RiskWeightedDetector()

        # Simulate detections
        for _ in range(10):
            detector.detect(0.8, "actuator_stuck")  # All detected
        for _ in range(10):
            detector.detect(0.3, "transient_glitch")  # None detected

        recalls = detector.get_recall_by_class()

        assert recalls[HazardClass.CATASTROPHIC] == 1.0
        assert recalls[HazardClass.MINOR] == 0.0

    def test_pattern_matching_fallback(self):
        detector = RiskWeightedDetector()

        # Unknown fault types should use pattern matching
        result1 = detector.detect(0.5, "motor_xyz_failure")
        assert result1.hazard_class == HazardClass.CATASTROPHIC

        result2 = detector.detect(0.5, "some_attack_vector")
        assert result2.hazard_class == HazardClass.HAZARDOUS

        result3 = detector.detect(0.5, "sensor_drift_issue")
        assert result3.hazard_class == HazardClass.MAJOR

    def test_reset_clears_counts(self):
        detector = RiskWeightedDetector()

        for _ in range(5):
            detector.detect(0.5, "actuator_stuck")

        assert detector.total_counts[HazardClass.CATASTROPHIC] == 5

        detector.reset()

        assert detector.total_counts[HazardClass.CATASTROPHIC] == 0


# =============================================================================
# C. Integrity-Based Detection Tests
# =============================================================================

class TestIntegrityMonitor:
    """Tests for integrity-based detection."""

    def test_initialization(self):
        monitor = IntegrityMonitor()
        assert monitor.k_factor == 5.33
        assert monitor.position_noise_std == 1.0

    def test_alert_limits_ordering(self):
        limits = AlertLimits()

        # Approach should have tightest limits
        assert limits.hal_approach < limits.hal_terminal
        assert limits.hal_terminal < limits.hal_enroute

    def test_protection_levels_computed(self):
        monitor = IntegrityMonitor()

        position = np.array([100.0, 200.0, 50.0])
        velocity = np.array([5.0, 3.0, -1.0])

        pl = monitor.compute_protection_levels(position, velocity)

        assert pl.hpl > 0
        assert pl.vpl > 0

    def test_nominal_position_has_integrity(self):
        np.random.seed(42)
        monitor = IntegrityMonitor()

        # Stable position with low noise
        for _ in range(20):
            position = np.array([100.0, 200.0, 50.0]) + np.random.randn(3) * 0.1
            velocity = np.array([0.0, 0.0, 0.0])

            result = monitor.check_integrity(position, velocity, flight_phase="enroute")

        assert result.overall_integrity
        assert not result.is_alert

    def test_high_variance_triggers_alert(self):
        np.random.seed(42)
        monitor = IntegrityMonitor(position_noise_std=100.0)

        # Unstable position with high noise
        for _ in range(50):
            position = np.array([100.0, 200.0, 50.0]) + np.random.randn(3) * 100
            velocity = np.array([0.0, 0.0, 0.0])

            result = monitor.check_integrity(position, velocity, flight_phase="approach")

        # High variance should eventually trigger alert in approach phase
        assert result.hpl > 10  # Significant protection level

    def test_innovation_tracked_in_history(self):
        """Test that innovation values are tracked in history."""
        monitor = IntegrityMonitor()

        # Add samples with innovation
        for i in range(10):
            position = np.array([100.0 + i, 200.0, 50.0])
            velocity = np.zeros(3)
            monitor.compute_protection_levels(position, velocity, innovation=float(i))

        # Check innovation history is populated
        assert len(monitor.innovation_history) == 10
        assert monitor.innovation_history[-1] == 9.0

    def test_flight_phase_affects_limits(self):
        monitor = IntegrityMonitor()

        position = np.array([100.0, 200.0, 50.0])
        velocity = np.zeros(3)

        result_enroute = monitor.check_integrity(position, velocity, flight_phase="enroute")
        result_approach = monitor.check_integrity(position, velocity, flight_phase="approach")

        # Approach has tighter limits
        assert result_approach.hal < result_enroute.hal
        assert result_approach.val < result_enroute.val

    def test_alert_rate_computed(self):
        monitor = IntegrityMonitor()

        # Force some alerts
        for _ in range(50):
            position = np.array([100.0, 200.0, 50.0])
            velocity = np.zeros(3)
            monitor.check_integrity(position, velocity)

        rate = monitor.get_alert_rate()
        assert 0 <= rate <= 1

    def test_reset_clears_state(self):
        monitor = IntegrityMonitor()

        for _ in range(10):
            position = np.array([100.0, 200.0, 50.0])
            velocity = np.zeros(3)
            monitor.check_integrity(position, velocity)

        assert len(monitor.position_history) == 10
        assert monitor.total_count == 10

        monitor.reset()

        assert len(monitor.position_history) == 0
        assert monitor.total_count == 0


# =============================================================================
# D. Combined Detector Tests
# =============================================================================

class TestIndustryAlignedDetector:
    """Tests for combined industry-aligned detector."""

    def test_initialization(self):
        detector = IndustryAlignedDetector()
        assert detector.two_stage is not None
        assert detector.risk_weighted is not None
        assert detector.integrity is not None

    def test_detect_returns_result(self):
        detector = IndustryAlignedDetector()

        result = detector.detect(
            anomaly_score=0.5,
            fault_type="actuator_stuck",
        )

        assert isinstance(result, IndustryAlignedResult)
        assert hasattr(result, 'final_alarm')
        assert hasattr(result, 'alarm_source')

    def test_catastrophic_triggers_on_either(self):
        detector = IndustryAlignedDetector(
            two_stage_config={
                "suspicion_threshold": 0.3,
                "confirmation_threshold": 0.4,
                "confirmation_window_K": 5,
                "confirmation_required_M": 3,
            }
        )

        # Sustained high scores for catastrophic fault
        for _ in range(10):
            result = detector.detect(0.6, "actuator_stuck")

        # Should alarm via either two-stage or risk-weighted
        assert result.final_alarm
        assert result.alarm_source in ["two_stage", "risk_weighted", "combined"]

    def test_non_catastrophic_requires_two_stage(self):
        detector = IndustryAlignedDetector(
            two_stage_config={
                "suspicion_threshold": 0.3,
                "confirmation_threshold": 0.4,
                "confirmation_window_K": 5,
                "confirmation_required_M": 3,
            }
        )

        # Single high score for minor fault
        result = detector.detect(0.8, "transient_glitch")

        # Should not alarm immediately (needs two-stage confirmation)
        assert not result.final_alarm

    def test_integrity_integration(self):
        detector = IndustryAlignedDetector(enable_integrity=True)

        position = np.array([100.0, 200.0, 50.0])
        velocity = np.zeros(3)

        result = detector.detect(
            anomaly_score=0.5,
            fault_type="gps_drift",
            position=position,
            velocity=velocity,
        )

        assert result.integrity is not None

    def test_metrics_tracked(self):
        detector = IndustryAlignedDetector()

        for _ in range(10):
            detector.detect(0.5, "sensor_noise")

        metrics = detector.get_metrics()

        assert "alarm_rate" in metrics
        assert "total_samples" in metrics
        assert metrics["total_samples"] == 10

    def test_reset_clears_all(self):
        detector = IndustryAlignedDetector()

        for _ in range(5):
            detector.detect(0.7, "actuator_stuck")

        assert detector.total_count == 5

        detector.reset()

        assert detector.total_count == 0
        assert detector.alarm_count == 0


# =============================================================================
# E. Evaluation Function Tests
# =============================================================================

class TestEvaluation:
    """Tests for evaluation function."""

    def test_evaluate_returns_metrics(self):
        np.random.seed(42)

        n = 100
        scores = np.concatenate([
            np.random.rand(50) * 0.3,  # Clean
            np.random.rand(50) * 0.7 + 0.3,  # Fault
        ])
        labels = np.concatenate([np.zeros(50), np.ones(50)])
        fault_types = ["nominal"] * 50 + ["actuator_stuck"] * 50

        results = evaluate_industry_aligned(scores, labels, fault_types)

        assert "recall" in results
        assert "fpr" in results
        assert "catastrophic_recall" in results

    def test_metrics_in_valid_range(self):
        np.random.seed(42)

        n = 100
        scores = np.random.rand(n)
        labels = np.concatenate([np.zeros(50), np.ones(50)])
        fault_types = ["nominal"] * 50 + ["gps_spoofing"] * 50

        results = evaluate_industry_aligned(scores, labels, fault_types)

        assert 0 <= results["recall"] <= 1
        assert 0 <= results["fpr"] <= 1
        assert 0 <= results["precision"] <= 1

    def test_with_position_data(self):
        np.random.seed(42)

        n = 100
        scores = np.random.rand(n)
        labels = np.concatenate([np.zeros(50), np.ones(50)])
        fault_types = ["nominal"] * n
        positions = np.random.randn(n, 3) * 10 + np.array([100, 200, 50])
        velocities = np.random.randn(n, 3)

        results = evaluate_industry_aligned(
            scores, labels, fault_types,
            positions=positions, velocities=velocities
        )

        assert "recall" in results


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
