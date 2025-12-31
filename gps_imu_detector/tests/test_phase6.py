"""
Tests for Phase 6: Integration and Certification-Aligned Validation

Tests:
6.1 Unified detection pipeline
6.2 Certification-aligned validation
"""

import numpy as np
import pytest


# =============================================================================
# Phase 6.1: Unified Pipeline Tests
# =============================================================================

class TestUnifiedDetectionPipeline:
    """Tests for unified detection pipeline."""

    def test_initialization(self):
        from gps_imu_detector.src.integration import UnifiedDetectionPipeline

        pipeline = UnifiedDetectionPipeline()
        assert pipeline.config.enable_regime_detection == True
        assert pipeline.regime_classifier is not None
        assert pipeline.safety_system is not None

    def test_process_returns_result(self):
        from gps_imu_detector.src.integration import (
            UnifiedDetectionPipeline, DetectionDecision
        )

        pipeline = UnifiedDetectionPipeline()
        state = np.random.randn(12).astype(np.float32) * 0.1

        result = pipeline.process(state)

        assert result.decision in list(DetectionDecision)
        assert result.trace_id.startswith("DET-")
        assert result.latency_ms > 0
        assert 0 <= result.confidence <= 1

    def test_process_with_next_state(self):
        from gps_imu_detector.src.integration import UnifiedDetectionPipeline

        pipeline = UnifiedDetectionPipeline()
        state = np.random.randn(12).astype(np.float32) * 0.1
        next_state = np.random.randn(12).astype(np.float32) * 0.1

        result = pipeline.process(state, next_state)

        assert 'residual' in result.components
        assert result.components['residual'] >= 0

    def test_process_with_sensor_data(self):
        from gps_imu_detector.src.integration import UnifiedDetectionPipeline

        pipeline = UnifiedDetectionPipeline()
        state = np.random.randn(12).astype(np.float32) * 0.1
        gps_data = np.random.randn(3).astype(np.float32) * 0.1
        imu_data = np.random.randn(6).astype(np.float32) * 0.1

        result = pipeline.process(state, gps_data=gps_data, imu_data=imu_data)

        assert result is not None

    def test_regime_detection(self):
        from gps_imu_detector.src.integration import UnifiedDetectionPipeline

        pipeline = UnifiedDetectionPipeline()

        # Hover state
        hover_state = np.zeros(12, dtype=np.float32)
        hover_state[3:6] = 0.1  # Low velocity

        result = pipeline.process(hover_state)
        assert result.regime.name == 'HOVER'

        # Aggressive state
        aggressive_state = np.zeros(12, dtype=np.float32)
        aggressive_state[3:6] = 10.0  # High velocity

        result = pipeline.process(aggressive_state)
        assert result.regime.name == 'AGGRESSIVE'

    def test_calibration(self):
        from gps_imu_detector.src.integration import UnifiedDetectionPipeline

        np.random.seed(42)

        pipeline = UnifiedDetectionPipeline()
        nominal = np.random.randn(3, 50, 12).astype(np.float32) * 0.1

        pipeline.calibrate(nominal)

        # Should have calibrated safety system
        assert pipeline.safety_system.severity_scorer.residual_std > 0

    def test_statistics(self):
        from gps_imu_detector.src.integration import UnifiedDetectionPipeline

        pipeline = UnifiedDetectionPipeline()

        for _ in range(10):
            state = np.random.randn(12).astype(np.float32) * 0.1
            pipeline.process(state)

        stats = pipeline.get_statistics()

        assert 'total_processed' in stats
        assert stats['total_processed'] == 10
        assert 'mean_latency_ms' in stats

    def test_reset(self):
        from gps_imu_detector.src.integration import UnifiedDetectionPipeline

        pipeline = UnifiedDetectionPipeline()

        for _ in range(5):
            state = np.random.randn(12).astype(np.float32) * 0.1
            pipeline.process(state)

        pipeline.reset()

        stats = pipeline.get_statistics()
        assert stats['total_processed'] == 0


# =============================================================================
# Phase 6.2: Certification Validation Tests
# =============================================================================

class TestCertificationValidator:
    """Tests for certification-aligned validation."""

    def test_initialization(self):
        from gps_imu_detector.src.integration import (
            CertificationValidator, ValidationConfig
        )

        validator = CertificationValidator()
        assert validator.config.min_detection_rate == 0.95

        config = ValidationConfig(min_detection_rate=0.90)
        validator = CertificationValidator(config)
        assert validator.config.min_detection_rate == 0.90

    def test_validate_runs(self):
        from gps_imu_detector.src.integration import (
            CertificationValidator, UnifiedDetectionPipeline
        )

        np.random.seed(42)

        pipeline = UnifiedDetectionPipeline()
        validator = CertificationValidator()

        nominal = np.random.randn(3, 30, 12).astype(np.float32) * 0.1
        attack = np.random.randn(3, 30, 12).astype(np.float32) * 0.1 + 2.0

        passed, metrics = validator.validate(pipeline, nominal, attack)

        assert isinstance(passed, bool)
        assert metrics.detection_rate_overall >= 0
        assert metrics.false_positive_rate >= 0

    def test_metrics_complete(self):
        from gps_imu_detector.src.integration import (
            CertificationValidator, UnifiedDetectionPipeline
        )

        np.random.seed(42)

        pipeline = UnifiedDetectionPipeline()
        validator = CertificationValidator()

        nominal = np.random.randn(2, 20, 12).astype(np.float32) * 0.1
        attack = np.random.randn(2, 20, 12).astype(np.float32) * 0.1 + 3.0

        _, metrics = validator.validate(pipeline, nominal, attack)

        # Check all metrics present
        assert hasattr(metrics, 'detection_rate_overall')
        assert hasattr(metrics, 'false_positive_rate')
        assert hasattr(metrics, 'mean_detection_latency_ms')
        assert hasattr(metrics, 'coverage_by_attack_type')
        assert hasattr(metrics, 'availability')

    def test_generate_report(self):
        from gps_imu_detector.src.integration import (
            CertificationValidator, UnifiedDetectionPipeline
        )

        np.random.seed(42)

        pipeline = UnifiedDetectionPipeline()
        validator = CertificationValidator()

        nominal = np.random.randn(2, 20, 12).astype(np.float32) * 0.1
        attack = np.random.randn(2, 20, 12).astype(np.float32) * 0.1 + 3.0

        validator.validate(pipeline, nominal, attack)
        report = validator.generate_report()

        assert "CERTIFICATION VALIDATION REPORT" in report
        assert "Detection Rate:" in report
        assert "False Positive Rate:" in report


# =============================================================================
# Phase 6 Checkpoint Tests
# =============================================================================

class TestPhase6Checkpoint:
    """Integration tests for Phase 6 checkpoint."""

    def test_run_certification_validation(self):
        from gps_imu_detector.src.integration import run_certification_validation

        np.random.seed(42)

        nominal = np.random.randn(3, 30, 12).astype(np.float32) * 0.1
        attack = np.random.randn(3, 30, 12).astype(np.float32) * 0.1 + 2.0

        passed, metrics, report = run_certification_validation(nominal, attack)

        assert isinstance(passed, bool)
        assert metrics is not None
        assert len(report) > 0

    def test_pipeline_latency_acceptable(self):
        from gps_imu_detector.src.integration import UnifiedDetectionPipeline

        pipeline = UnifiedDetectionPipeline()

        latencies = []
        for _ in range(20):
            state = np.random.randn(12).astype(np.float32) * 0.1
            result = pipeline.process(state)
            latencies.append(result.latency_ms)

        # Mean latency should be reasonable
        mean_latency = np.mean(latencies)
        assert mean_latency < 10.0  # <10ms mean

    def test_all_components_integrated(self):
        from gps_imu_detector.src.integration import UnifiedDetectionPipeline

        pipeline = UnifiedDetectionPipeline()
        state = np.random.randn(12).astype(np.float32) * 0.1
        next_state = np.random.randn(12).astype(np.float32) * 0.1

        result = pipeline.process(state, next_state)

        # Check all components ran
        assert 'regime' in result.components
        assert 'residual' in result.components
        assert 'safety' in result.components
        assert 'shadow' in result.components

    def test_decision_escalation(self):
        from gps_imu_detector.src.integration import (
            UnifiedDetectionPipeline, DetectionDecision
        )

        pipeline = UnifiedDetectionPipeline()

        # Calibrate with normal data
        nominal = np.random.randn(3, 30, 12).astype(np.float32) * 0.1
        pipeline.calibrate(nominal)

        # Process increasingly anomalous states
        decisions = []
        for multiplier in [0.1, 1.0, 3.0, 5.0, 10.0]:
            state = np.ones(12, dtype=np.float32) * multiplier
            result = pipeline.process(state)
            decisions.append(result.decision.value)
            pipeline.reset()

        # Later decisions should be higher (more severe)
        assert decisions[-1] >= decisions[0]

    def test_certification_metrics_structure(self):
        from gps_imu_detector.src.integration import CertificationMetrics

        # Check all required fields exist
        metrics = CertificationMetrics(
            detection_rate_overall=0.95,
            detection_rate_per_attack={'GPS_DRIFT': 0.9},
            false_positive_rate=0.01,
            false_negative_rate=0.05,
            mean_detection_latency_ms=100.0,
            p95_detection_latency_ms=200.0,
            worst_case_latency_ms=500.0,
            coverage_by_attack_type={'overall': 0.9},
            weakness_count=2,
            availability=0.999,
            mtbf_samples=10000,
            test_coverage=0.95,
            requirements_traced=12,
        )

        assert metrics.detection_rate_overall == 0.95
        assert metrics.availability == 0.999


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
