"""
Tests for Phase 7: Governance and Rollout

Tests:
7.1 Drift monitoring
7.2 Recalibration and version management
"""

import numpy as np
import pytest


# =============================================================================
# Phase 7.1: Drift Monitoring Tests
# =============================================================================

class TestDriftMonitor:
    """Tests for drift monitoring."""

    def test_initialization(self):
        from gps_imu_detector.src.governance import DriftMonitor, DriftConfig

        monitor = DriftMonitor()
        assert monitor.config.reference_window == 1000

        config = DriftConfig(reference_window=500)
        monitor = DriftMonitor(config)
        assert monitor.config.reference_window == 500

    def test_set_reference(self):
        from gps_imu_detector.src.governance import DriftMonitor

        monitor = DriftMonitor()

        residuals = np.random.randn(1000)
        regimes = np.random.randint(1, 5, 1000)

        monitor.set_reference(residuals, regimes)

        assert len(monitor._ref_residuals) == 1000
        assert len(monitor._ref_regimes) == 1000

    def test_update_no_drift(self):
        from gps_imu_detector.src.governance import DriftMonitor, DriftConfig, DriftType

        config = DriftConfig(test_window=50, alert_cooldown=10)
        monitor = DriftMonitor(config)

        # Set reference
        ref_residuals = np.random.randn(1000)
        ref_regimes = np.random.randint(1, 5, 1000)
        monitor.set_reference(ref_residuals, ref_regimes)

        # Feed similar data
        for _ in range(60):
            result = monitor.update(
                residual=np.random.randn(),
                regime=np.random.randint(1, 5),
            )

        # Should not detect drift on similar data
        assert result.drift_type == DriftType.NONE

    def test_detect_mean_shift(self):
        from gps_imu_detector.src.governance import (
            DriftMonitor, DriftConfig, DriftType
        )

        config = DriftConfig(test_window=50, alert_cooldown=10)
        monitor = DriftMonitor(config)

        # Set reference with mean 0
        monitor.set_reference(
            np.random.randn(1000),
            np.random.randint(1, 5, 1000),
        )

        # Feed data with shifted mean
        for _ in range(60):
            result = monitor.update(
                residual=np.random.randn() + 5.0,  # Shifted mean
                regime=np.random.randint(1, 5),
            )

        assert result.drift_detected == True
        assert result.drift_type == DriftType.RESIDUAL_MEAN_SHIFT

    def test_detect_variance_shift(self):
        from gps_imu_detector.src.governance import (
            DriftMonitor, DriftConfig, DriftType
        )

        config = DriftConfig(
            test_window=50,
            alert_cooldown=10,
            mean_shift_threshold=10.0,  # High to not trigger mean shift
        )
        monitor = DriftMonitor(config)

        # Set reference with std 1.0
        monitor.set_reference(
            np.random.randn(1000),
            np.random.randint(1, 5, 1000),
        )

        # Feed data with higher variance
        for _ in range(60):
            result = monitor.update(
                residual=np.random.randn() * 5.0,  # Higher variance
                regime=np.random.randint(1, 5),
            )

        assert result.drift_detected == True
        assert result.drift_type == DriftType.RESIDUAL_VARIANCE_SHIFT

    def test_cooldown_respected(self):
        from gps_imu_detector.src.governance import DriftMonitor, DriftConfig

        config = DriftConfig(test_window=20, alert_cooldown=50)
        monitor = DriftMonitor(config)

        monitor.set_reference(np.random.randn(1000), np.random.randint(1, 5, 1000))

        # Fill window and trigger alert
        for _ in range(30):
            monitor.update(np.random.randn() + 10.0, 1)

        first_alert = monitor._last_alert_sample

        # Immediately after, should be in cooldown
        result = monitor.update(np.random.randn() + 10.0, 1)
        assert 'in_cooldown' in result.details or not result.drift_detected

    def test_reset(self):
        from gps_imu_detector.src.governance import DriftMonitor

        monitor = DriftMonitor()
        monitor.set_reference(np.random.randn(100), np.random.randint(1, 5, 100))

        for _ in range(50):
            monitor.update(np.random.randn(), 1)

        monitor.reset()

        assert len(monitor._cur_residuals) == 0
        assert monitor._sample_count == 0


# =============================================================================
# Phase 7.2: Version Management Tests
# =============================================================================

class TestVersionManager:
    """Tests for version management."""

    def test_initialization(self):
        from gps_imu_detector.src.governance import VersionManager

        manager = VersionManager()
        assert manager._active_version is None
        assert len(manager._versions) == 0

    def test_create_version(self):
        from gps_imu_detector.src.governance import VersionManager

        manager = VersionManager()

        version = manager.create_version(
            calibration_samples=1000,
            metrics={'fpr': 0.01, 'detection_rate': 0.95},
            config_hash="abc123",
        )

        assert version.version_id.startswith("v")
        assert version.calibration_samples == 1000
        assert version.metrics['fpr'] == 0.01

    def test_activate_version(self):
        from gps_imu_detector.src.governance import VersionManager

        manager = VersionManager()

        version = manager.create_version(1000, {'fpr': 0.01}, "abc123")
        success = manager.activate_version(version.version_id)

        assert success == True
        assert manager._active_version == version.version_id
        assert manager._versions[version.version_id].status == "active"

    def test_rollback(self):
        from gps_imu_detector.src.governance import VersionManager

        manager = VersionManager()

        v1 = manager.create_version(1000, {'fpr': 0.01}, "abc123")
        manager.activate_version(v1.version_id)

        v2 = manager.create_version(1000, {'fpr': 0.02}, "def456")
        manager.activate_version(v2.version_id)

        rolled_back = manager.rollback()

        assert rolled_back == v1.version_id
        assert manager._active_version == v1.version_id

    def test_list_versions(self):
        from gps_imu_detector.src.governance import VersionManager

        manager = VersionManager()

        for i in range(3):
            manager.create_version(1000 + i, {'fpr': 0.01 * i}, f"hash{i}")

        versions = manager.list_versions()

        assert len(versions) == 3


class TestRecalibrationManager:
    """Tests for recalibration management."""

    def test_initialization(self):
        from gps_imu_detector.src.governance import RecalibrationManager

        manager = RecalibrationManager()
        assert manager.version_manager is not None

    def test_recalibrate_insufficient_data(self):
        from gps_imu_detector.src.governance import (
            RecalibrationManager, RecalibrationConfig
        )
        from gps_imu_detector.src.integration import UnifiedDetectionPipeline

        config = RecalibrationConfig(min_samples_for_recalibration=100)
        manager = RecalibrationManager(config=config)

        pipeline = UnifiedDetectionPipeline()
        small_data = np.random.randn(5, 20, 12).astype(np.float32)

        success, details = manager.recalibrate(pipeline, small_data)

        assert success == False
        assert 'insufficient_data' in details.get('error', '')


# =============================================================================
# Phase 7 Checkpoint Tests
# =============================================================================

class TestPhase7Checkpoint:
    """Integration tests for Phase 7 checkpoint."""

    def test_governance_system_initialization(self):
        from gps_imu_detector.src.governance import GovernanceSystem
        from gps_imu_detector.src.integration import UnifiedDetectionPipeline

        np.random.seed(42)

        system = GovernanceSystem()
        pipeline = UnifiedDetectionPipeline()
        data = np.random.randn(5, 30, 12).astype(np.float32) * 0.1

        system.initialize(pipeline, data)

        status = system.get_status()
        assert status['active_version'] is not None
        assert status['version_count'] == 1

    def test_governance_monitor_flow(self):
        from gps_imu_detector.src.governance import GovernanceSystem
        from gps_imu_detector.src.integration import UnifiedDetectionPipeline

        np.random.seed(42)

        system = GovernanceSystem()
        pipeline = UnifiedDetectionPipeline()
        data = np.random.randn(5, 30, 12).astype(np.float32) * 0.1

        system.initialize(pipeline, data)

        # Monitor some samples
        for _ in range(50):
            result = system.monitor(
                residual=np.random.randn() * 0.5,
                regime=1,
                detection=False,
            )

        assert result is not None

    def test_drift_types_comprehensive(self):
        from gps_imu_detector.src.governance import DriftType

        types = list(DriftType)
        assert len(types) >= 4
        assert DriftType.RESIDUAL_MEAN_SHIFT in types
        assert DriftType.RESIDUAL_VARIANCE_SHIFT in types

    def test_drift_severity_levels(self):
        from gps_imu_detector.src.governance import DriftSeverity

        assert DriftSeverity.NONE.value < DriftSeverity.MINOR.value
        assert DriftSeverity.MINOR.value < DriftSeverity.MODERATE.value
        assert DriftSeverity.MODERATE.value < DriftSeverity.SEVERE.value

    def test_version_lineage(self):
        from gps_imu_detector.src.governance import VersionManager

        manager = VersionManager()

        v1 = manager.create_version(1000, {}, "hash1")
        manager.activate_version(v1.version_id)

        v2 = manager.create_version(2000, {}, "hash2")

        # v2 should have v1 as parent
        assert v2.parent_version == v1.version_id


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
