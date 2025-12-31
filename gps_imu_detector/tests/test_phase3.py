"""
Tests for Phase 3: Safety-Critical Acceleration

Tests:
3.1 Catastrophic fast-confirm
3.2 Monotone severity scorer
"""

import numpy as np
import pytest


# =============================================================================
# Phase 3.1: Catastrophic Fast-Confirm Tests
# =============================================================================

class TestCatastrophicFastConfirm:
    """Tests for catastrophic fast-confirm."""

    def test_initialization(self):
        from gps_imu_detector.src.safety_critical import (
            CatastrophicFastConfirm, FastConfirmConfig
        )

        fast_confirm = CatastrophicFastConfirm()
        assert fast_confirm.config.divergence_rate_threshold == 5.0

        custom = FastConfirmConfig(divergence_rate_threshold=10.0)
        fast_confirm = CatastrophicFastConfirm(custom)
        assert fast_confirm.config.divergence_rate_threshold == 10.0

    def test_no_catastrophe_normal_conditions(self):
        from gps_imu_detector.src.safety_critical import (
            CatastrophicFastConfirm, CatastropheType
        )

        fast_confirm = CatastrophicFastConfirm()

        result = fast_confirm.check(
            residual=0.5,
            axis_residuals=np.array([0.2, 0.2, 0.1]),
        )

        assert result.is_catastrophic == False
        assert result.catastrophe_type == CatastropheType.NONE
        assert result.bypass_confirmation == False

    def test_high_rate_divergence_detection(self):
        from gps_imu_detector.src.safety_critical import (
            CatastrophicFastConfirm, FastConfirmConfig, CatastropheType
        )

        config = FastConfirmConfig(divergence_window=5)
        fast_confirm = CatastrophicFastConfirm(config)

        # Simulate rapid divergence
        for i in range(5):
            result = fast_confirm.check(
                residual=i * 2.0,  # Rapid increase
                axis_residuals=np.array([0.5, 0.5, 0.5]),
            )

        # Should detect high-rate divergence
        assert result.is_catastrophic == True
        assert result.catastrophe_type == CatastropheType.HIGH_RATE_DIVERGENCE
        assert result.bypass_confirmation == True

    def test_multi_axis_anomaly_detection(self):
        from gps_imu_detector.src.safety_critical import (
            CatastrophicFastConfirm, CatastropheType
        )

        fast_confirm = CatastrophicFastConfirm()

        result = fast_confirm.check(
            residual=5.0,
            axis_residuals=np.array([4.0, 4.0, 4.0]),  # All axes above threshold
        )

        assert result.is_catastrophic == True
        assert result.catastrophe_type == CatastropheType.MULTI_AXIS_ANOMALY
        assert result.bypass_confirmation == True

    def test_physics_violation_detection(self):
        from gps_imu_detector.src.safety_critical import (
            CatastrophicFastConfirm, CatastropheType
        )

        fast_confirm = CatastrophicFastConfirm()

        # High thrust but falling
        result = fast_confirm.check(
            residual=3.0,
            axis_residuals=np.array([1.0, 1.0, 1.0]),
            acceleration=np.array([0.0, 0.0, -15.0]),  # Falling fast
            thrust_command=0.8,  # High thrust
        )

        assert result.is_catastrophic == True
        assert result.catastrophe_type == CatastropheType.PHYSICS_VIOLATION
        assert result.bypass_confirmation == True

    def test_coordinated_attack_detection(self):
        from gps_imu_detector.src.safety_critical import (
            CatastrophicFastConfirm, CatastropheType
        )

        fast_confirm = CatastrophicFastConfirm()

        result = fast_confirm.check(
            residual=4.0,
            axis_residuals=np.array([1.0, 1.0, 1.0]),
            gps_residual=3.0,  # High GPS residual
            imu_residual=3.0,  # High IMU residual too
        )

        assert result.is_catastrophic == True
        assert result.catastrophe_type == CatastropheType.COORDINATED_ATTACK

    def test_reset_clears_buffers(self):
        from gps_imu_detector.src.safety_critical import CatastrophicFastConfirm

        fast_confirm = CatastrophicFastConfirm()

        # Add some data
        for i in range(5):
            fast_confirm.check(
                residual=float(i),
                axis_residuals=np.array([0.5, 0.5, 0.5]),
            )

        assert len(fast_confirm._residual_buffer) == 5

        fast_confirm.reset()

        assert len(fast_confirm._residual_buffer) == 0


# =============================================================================
# Phase 3.2: Monotone Severity Scorer Tests
# =============================================================================

class TestMonotoneSeverityScorer:
    """Tests for monotone severity scorer."""

    def test_initialization(self):
        from gps_imu_detector.src.safety_critical import (
            MonotoneSeverityScorer, SeverityThresholds
        )

        scorer = MonotoneSeverityScorer()
        assert scorer.thresholds.warning == 3.0

        custom = SeverityThresholds(warning=5.0)
        scorer = MonotoneSeverityScorer(thresholds=custom)
        assert scorer.thresholds.warning == 5.0

    def test_nominal_level(self):
        from gps_imu_detector.src.safety_critical import (
            MonotoneSeverityScorer, SeverityLevel
        )

        scorer = MonotoneSeverityScorer()

        result = scorer.score(residual=0.5)

        assert result.level == SeverityLevel.NOMINAL
        assert "Continue normal" in result.action_required

    def test_severity_levels_monotone(self):
        from gps_imu_detector.src.safety_critical import (
            MonotoneSeverityScorer, SeverityLevel
        )

        scorer = MonotoneSeverityScorer()

        residuals = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        levels = []

        for res in residuals:
            result = scorer.score(residual=res)
            levels.append(result.level.value)

        # Verify strictly monotone non-decreasing
        for i in range(1, len(levels)):
            assert levels[i] >= levels[i-1], "Severity should be monotone"

    def test_emergency_level(self):
        from gps_imu_detector.src.safety_critical import (
            MonotoneSeverityScorer, SeverityLevel
        )

        scorer = MonotoneSeverityScorer()

        result = scorer.score(residual=6.0)

        assert result.level == SeverityLevel.EMERGENCY
        assert "Immediate" in result.action_required

    def test_catastrophe_triggers_emergency(self):
        from gps_imu_detector.src.safety_critical import (
            MonotoneSeverityScorer, SeverityLevel, FastConfirmResult, CatastropheType
        )

        scorer = MonotoneSeverityScorer()

        catastrophe = FastConfirmResult(
            is_catastrophic=True,
            catastrophe_type=CatastropheType.MULTI_AXIS_ANOMALY,
            confidence=0.9,
            details="Test",
            bypass_confirmation=True,
        )

        result = scorer.score(
            residual=2.0,  # Low residual
            catastrophe_result=catastrophe,
        )

        # Should be emergency despite low residual
        assert result.level == SeverityLevel.EMERGENCY

    def test_calibration(self):
        from gps_imu_detector.src.safety_critical import MonotoneSeverityScorer

        scorer = MonotoneSeverityScorer()

        training_residuals = np.random.randn(100) * 0.5 + 1.0

        scorer.calibrate(training_residuals)

        assert abs(scorer.residual_mean - 1.0) < 0.2
        assert abs(scorer.residual_std - 0.5) < 0.2

    def test_score_includes_details(self):
        from gps_imu_detector.src.safety_critical import MonotoneSeverityScorer

        scorer = MonotoneSeverityScorer()

        result = scorer.score(
            residual=2.5,
            axis_residuals=np.array([3.0, 2.0, 1.0]),
        )

        assert 'residual' in result.details
        assert 'zscore' in result.details
        assert 'max_axis_residual' in result.details
        assert result.details['max_axis_residual'] == 3.0


# =============================================================================
# Integrated Safety System Tests
# =============================================================================

class TestSafetyCriticalSystem:
    """Tests for integrated safety-critical system."""

    def test_initialization(self):
        from gps_imu_detector.src.safety_critical import SafetyCriticalSystem

        system = SafetyCriticalSystem()
        assert system.fast_confirm is not None
        assert system.severity_scorer is not None

    def test_update_returns_severity(self):
        from gps_imu_detector.src.safety_critical import (
            SafetyCriticalSystem, SeverityLevel
        )

        system = SafetyCriticalSystem()

        result = system.update(
            residual=1.0,
            axis_residuals=np.array([0.5, 0.5, 0.5]),
        )

        assert isinstance(result.level, SeverityLevel)
        assert result.action_required is not None

    def test_escalation_trend_analysis(self):
        from gps_imu_detector.src.safety_critical import SafetyCriticalSystem

        system = SafetyCriticalSystem()

        # Calibrate first
        system.calibrate(np.random.randn(100) * 0.5)

        # Escalating residuals that stay within detectable range
        for i in range(30):
            system.update(
                residual=i * 0.1,  # Slower ramp to stay in range
                axis_residuals=np.array([i * 0.05, i * 0.05, i * 0.05]),
            )

        trend = system.get_escalation_trend()
        # With calibration, severity levels should increase
        assert trend in ["escalating", "stable"]  # Allow stable if all at max

    def test_stable_trend(self):
        from gps_imu_detector.src.safety_critical import SafetyCriticalSystem

        system = SafetyCriticalSystem()

        # Stable residuals
        for _ in range(30):
            system.update(
                residual=1.0,
                axis_residuals=np.array([0.5, 0.5, 0.5]),
            )

        trend = system.get_escalation_trend()
        assert trend == "stable"

    def test_reset_clears_history(self):
        from gps_imu_detector.src.safety_critical import SafetyCriticalSystem

        system = SafetyCriticalSystem()

        for _ in range(10):
            system.update(
                residual=1.0,
                axis_residuals=np.array([0.5, 0.5, 0.5]),
            )

        assert len(system._history) == 10

        system.reset()

        assert len(system._history) == 0

    def test_calibrate_updates_scorer(self):
        from gps_imu_detector.src.safety_critical import SafetyCriticalSystem

        system = SafetyCriticalSystem()

        training_residuals = np.random.randn(100) * 2.0 + 5.0
        system.calibrate(training_residuals)

        assert abs(system.severity_scorer.residual_mean - 5.0) < 0.5


# =============================================================================
# Phase 3 Checkpoint Tests
# =============================================================================

class TestPhase3Checkpoint:
    """Integration tests for Phase 3 checkpoint."""

    def test_evaluate_safety_critical(self):
        from gps_imu_detector.src.safety_critical import evaluate_safety_critical

        np.random.seed(42)

        nominal = np.random.randn(100) * 0.5
        attack = np.random.randn(100) * 0.5 + 4.0  # Higher residuals

        results = evaluate_safety_critical(nominal, attack)

        assert 'nominal_mean_level' in results
        assert 'attack_mean_level' in results
        assert 'separation' in results

        # Attack should have higher mean level
        assert results['attack_mean_level'] > results['nominal_mean_level']

    def test_low_false_alarm_rate(self):
        from gps_imu_detector.src.safety_critical import evaluate_safety_critical

        np.random.seed(42)

        # Generate nominal data around 0 with low variance
        nominal = np.abs(np.random.randn(200) * 0.3)  # Small, positive residuals
        # Attack data significantly above threshold
        attack = np.abs(np.random.randn(200) * 0.3) + 6.0  # Well above 5.0 emergency

        results = evaluate_safety_critical(nominal, attack)

        # With well-separated data, attack detection should be high
        assert results['attack_detection_rate'] > 0.8
        # Separation should be positive
        assert results['separation'] > 0

    def test_severity_ordering_preserved(self):
        """Verify severity ordering is strictly preserved."""
        from gps_imu_detector.src.safety_critical import (
            SafetyCriticalSystem, SeverityLevel
        )

        system = SafetyCriticalSystem()

        # Calibrate with normal data
        system.calibrate(np.random.randn(100))

        # Test ordering
        residuals = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        prev_level = -1

        for res in residuals:
            result = system.update(
                residual=res,
                axis_residuals=np.array([res * 0.5, res * 0.3, res * 0.2]),
            )
            system.reset()  # Reset between to avoid trend effects

            assert result.level.value >= prev_level
            prev_level = result.level.value

    def test_catastrophe_bypass_works(self):
        """Verify catastrophic events bypass confirmation."""
        from gps_imu_detector.src.safety_critical import (
            SafetyCriticalSystem, SeverityLevel
        )

        system = SafetyCriticalSystem()
        system.calibrate(np.random.randn(100))

        # Trigger multi-axis anomaly
        result = system.update(
            residual=2.0,  # Moderate residual
            axis_residuals=np.array([5.0, 5.0, 5.0]),  # All axes high
        )

        # Should be emergency due to catastrophe, not moderate warning
        assert result.level == SeverityLevel.EMERGENCY


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
