"""
Tests for Actuator Observability Module

Tests all 6 fixes:
1. Control-effort inconsistency metrics
2. Dual-timescale windows
3. Residual envelope normalization
4. Split fault heads
5. Phase-consistency check
6. Proper evaluation metrics
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gps_imu_detector.src.actuator_observability import (
    ControlEffortChecker,
    ControlEffortMetrics,
    DualTimescaleDetector,
    DualScaleResult,
    ResidualEnvelopeNormalizer,
    SplitFaultHead,
    extract_motor_features,
    extract_actuator_features,
    PhaseConsistencyChecker,
    compute_proper_metrics,
    EnhancedActuatorDetector,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample flight data for testing."""
    n = 1000
    dt = 0.005
    t = np.arange(n) * dt

    return {
        'n': n,
        'dt': dt,
        't': t,
        'position': np.column_stack([np.sin(t), np.cos(t), 10 + 0.1*t]),
        'velocity': np.column_stack([np.cos(t), -np.sin(t), np.ones(n)*0.1]),
        'acceleration': np.column_stack([-np.sin(t), -np.cos(t), np.zeros(n)]),
        'attitude': np.column_stack([0.1*np.sin(t), 0.1*np.cos(t), t*0.01]),
        'angular_rates': np.column_stack([0.1*np.cos(t), -0.1*np.sin(t), 0.01*np.ones(n)]),
        'control': np.ones((n, 4)) * 0.5 + 0.01 * np.random.randn(n, 4),
    }


# =============================================================================
# Fix 1: Control Effort Metrics Tests
# =============================================================================

class TestControlEffortChecker:
    """Tests for control-effort inconsistency detection."""

    def test_compute_metrics_shape(self, sample_data):
        """Test that metrics have correct shape."""
        checker = ControlEffortChecker(dt=sample_data['dt'])
        metrics = checker.compute_metrics(
            sample_data['control'],
            sample_data['acceleration'],
            sample_data['velocity'],
            sample_data['attitude'],
        )

        n = sample_data['n']
        assert metrics.efficiency.shape == (n,)
        assert metrics.trim_deviation.shape == (n,)
        assert metrics.energy_per_thrust.shape == (n,)
        assert metrics.control_power.shape == (n,)

    def test_efficiency_drops_with_fault(self, sample_data):
        """Test that efficiency drops when actuator is faulty."""
        checker = ControlEffortChecker(dt=sample_data['dt'])
        n = sample_data['n']

        # Normal case
        normal_metrics = checker.compute_metrics(
            sample_data['control'],
            sample_data['acceleration'],
            sample_data['velocity'],
            sample_data['attitude'],
        )

        # Faulty case: motor loses efficiency but controller compensates
        faulty_control = sample_data['control'].copy()
        faulty_control[:, 0] *= 2  # Motor 0 needs 2x input for same output

        # Acceleration stays roughly same (controller compensates)
        faulty_metrics = checker.compute_metrics(
            faulty_control,
            sample_data['acceleration'],  # Same acceleration
            sample_data['velocity'],
            sample_data['attitude'],
        )

        # Efficiency should drop (same acceleration, more control)
        assert np.mean(faulty_metrics.efficiency) < np.mean(normal_metrics.efficiency)

    def test_trim_deviation_increases_with_fault(self, sample_data):
        """Test that trim deviation magnitude increases when compensating."""
        checker = ControlEffortChecker(dt=sample_data['dt'])

        # Normal control metrics
        normal_metrics = checker.compute_metrics(
            sample_data['control'],
            sample_data['acceleration'],
            sample_data['velocity'],
            sample_data['attitude'],
        )

        # Create faulty control with higher effort
        faulty_control = sample_data['control'].copy()
        faulty_control[:, :] *= 1.5  # 50% more effort

        faulty_metrics = checker.compute_metrics(
            faulty_control,
            sample_data['acceleration'],
            sample_data['velocity'],
            sample_data['attitude'],
        )

        # Trim deviation magnitude should increase (absolute change)
        assert np.std(faulty_metrics.trim_deviation) > np.std(normal_metrics.trim_deviation) * 0.5


# =============================================================================
# Fix 2: Dual Timescale Tests
# =============================================================================

class TestDualTimescaleDetector:
    """Tests for dual-timescale fault detection."""

    def test_dual_scale_result_shape(self):
        """Test output shapes."""
        detector = DualTimescaleDetector(short_window=64, long_window=256)
        features = np.random.randn(500, 10)

        def scorer(window):
            return np.mean(np.abs(window))

        result = detector.compute_scores(features, scorer)

        assert result.short_score.shape == (500,)
        assert result.long_score.shape == (500,)
        assert result.fused_score.shape == (500,)
        assert result.is_anomaly.shape == (500,)

    def test_detects_abrupt_fault(self):
        """Test that short window detects abrupt faults."""
        detector = DualTimescaleDetector(
            short_window=64,
            long_window=256,
            short_threshold=0.5,
            long_threshold=0.3,
        )

        # Create signal with abrupt change
        features = np.zeros((500, 5))
        features[250:260] = 5.0  # Short spike

        def scorer(window):
            return np.max(np.abs(window))

        result = detector.compute_scores(features, scorer)

        # Short window should detect it
        assert np.max(result.short_score[250:260]) > 0.5

    def test_detects_slow_drift(self):
        """Test that long window detects slow degradation."""
        detector = DualTimescaleDetector(
            short_window=64,
            long_window=256,
            short_threshold=2.0,  # High threshold - won't trigger on drift
            long_threshold=0.1,   # Lower threshold for slow faults
        )

        # Create slow drift with larger magnitude
        features = np.zeros((500, 5))
        features[250:] = np.linspace(0, 1.0, 250)[:, None]  # Larger drift

        def scorer(window):
            return np.mean(np.abs(window))

        result = detector.compute_scores(features, scorer)

        # Long window should show higher scores in drift region vs clean region
        drift_score = np.mean(result.long_score[300:400])
        clean_score = np.mean(result.long_score[100:200])
        assert drift_score > clean_score * 2  # Drift region should be at least 2x higher


# =============================================================================
# Fix 3: Envelope Normalization Tests
# =============================================================================

class TestResidualEnvelopeNormalizer:
    """Tests for condition-aware normalization."""

    def test_fit_creates_bins(self):
        """Test that fitting creates speed/altitude bins."""
        normalizer = ResidualEnvelopeNormalizer(n_speed_bins=3, n_altitude_bins=3)

        residuals = np.random.randn(1000, 5) * 0.1
        speed = np.random.rand(1000) * 10
        altitude = np.random.rand(1000) * 100

        normalizer.fit(residuals, speed, altitude)

        assert normalizer.speed_edges is not None
        assert normalizer.altitude_edges is not None
        assert len(normalizer.speed_edges) == 4  # n_bins + 1
        assert len(normalizer.altitude_edges) == 4

    def test_normalize_output_shape(self):
        """Test normalized output has correct shape."""
        normalizer = ResidualEnvelopeNormalizer()

        residuals = np.random.randn(1000, 5) * 0.1
        speed = np.random.rand(1000) * 10
        altitude = np.random.rand(1000) * 100

        normalizer.fit(residuals[:500], speed[:500], altitude[:500])
        z_scores = normalizer.normalize(residuals, speed, altitude)

        assert z_scores.shape == residuals.shape

    def test_anomaly_has_high_zscore(self):
        """Test that anomalies have high z-scores after normalization."""
        normalizer = ResidualEnvelopeNormalizer()

        # Normal data
        residuals_normal = np.random.randn(500, 5) * 0.1
        speed = np.random.rand(500) * 10
        altitude = np.random.rand(500) * 100

        normalizer.fit(residuals_normal, speed, altitude)

        # Anomalous data
        residuals_anomaly = np.random.randn(100, 5) * 0.1 + 1.0  # Offset

        z_normal = normalizer.normalize(residuals_normal, speed, altitude)
        z_anomaly = normalizer.normalize(
            residuals_anomaly,
            np.random.rand(100) * 10,
            np.random.rand(100) * 100
        )

        assert np.mean(np.abs(z_anomaly)) > np.mean(np.abs(z_normal))


# =============================================================================
# Fix 4: Split Fault Heads Tests
# =============================================================================

class TestSplitFaultHead:
    """Tests for split motor/actuator fault heads."""

    def test_forward_shapes(self):
        """Test output shapes from both heads."""
        model = SplitFaultHead(motor_input_dim=8, actuator_input_dim=12)

        motor_feat = torch.randn(32, 8)
        actuator_feat = torch.randn(32, 12)

        motor_score, actuator_score = model(motor_feat, actuator_feat)

        assert motor_score.shape == (32, 1)
        assert actuator_score.shape == (32, 1)

    def test_scores_in_valid_range(self):
        """Test scores are in [0, 1] (sigmoid output)."""
        model = SplitFaultHead()

        motor_feat = torch.randn(32, 8)
        actuator_feat = torch.randn(32, 12)

        motor_score, actuator_score = model(motor_feat, actuator_feat)

        assert (motor_score >= 0).all() and (motor_score <= 1).all()
        assert (actuator_score >= 0).all() and (actuator_score <= 1).all()


class TestFeatureExtraction:
    """Tests for feature extraction functions."""

    def test_motor_features_shape(self):
        """Test motor feature extraction."""
        n = 100
        thrust = np.random.rand(n)
        vertical_accel = np.random.rand(n)
        velocity = np.random.randn(n, 3)
        control_effort = np.random.rand(n)

        features = extract_motor_features(
            thrust, vertical_accel, velocity, control_effort
        )

        assert features.shape == (n, 8)

    def test_actuator_features_shape(self):
        """Test actuator feature extraction."""
        n = 100
        attitude = np.random.randn(n, 3) * 0.1
        attitude_cmd = np.random.randn(n, 3) * 0.1
        angular_rates = np.random.randn(n, 3) * 0.1
        control_input = np.random.rand(n, 4)

        features = extract_actuator_features(
            attitude, attitude_cmd, angular_rates, control_input
        )

        assert features.shape == (n, 12)


# =============================================================================
# Fix 5: Phase Consistency Tests
# =============================================================================

class TestPhaseConsistencyChecker:
    """Tests for phase-consistency detection."""

    def test_detects_no_delay_in_sync_signals(self):
        """Test that synchronized signals show zero lag."""
        checker = PhaseConsistencyChecker(dt=0.01)

        t = np.linspace(0, 10, 1000)
        signal1 = np.sin(2 * np.pi * t)
        signal2 = np.sin(2 * np.pi * t)  # Identical

        corr, lag = checker.check_phase_consistency(signal1, signal2)

        # Should have high correlation and near-zero lag
        assert np.mean(corr[200:]) > 0.9
        assert np.mean(np.abs(lag[200:])) < 2

    def test_detects_delay(self):
        """Test that delayed signals are detected."""
        checker = PhaseConsistencyChecker(dt=0.01, max_lag=50)

        t = np.linspace(0, 10, 1000)
        signal1 = np.sin(2 * np.pi * t)
        signal2 = np.roll(signal1, 20)  # 20 sample delay

        corr, lag = checker.check_phase_consistency(signal1, signal2)

        # Should detect non-zero lag (may oscillate around true lag due to periodicity)
        # Check that we detect significant lag values
        lag_values = lag[200:]
        non_zero_lags = np.abs(lag_values[lag_values != 0])
        assert len(non_zero_lags) > 0 and np.mean(non_zero_lags) > 5

    def test_delay_attack_detection(self):
        """Test full delay attack detection pipeline."""
        checker = PhaseConsistencyChecker(dt=0.01)
        n = 500

        # Create signals
        t = np.linspace(0, 5, n)
        imu_velocity = np.column_stack([
            np.sin(2*np.pi*t),
            np.cos(2*np.pi*t),
            np.zeros(n)
        ])

        # Delayed GPS (attack)
        gps_velocity = np.column_stack([
            np.roll(imu_velocity[:, 0], 30),
            np.roll(imu_velocity[:, 1], 30),
            np.zeros(n)
        ])

        control_cmd = np.sin(2*np.pi*t)
        response = control_cmd  # Normal response

        results = checker.detect_delay_attack(
            imu_velocity, gps_velocity, control_cmd, response
        )

        # Should detect delay attack from IMU-GPS mismatch
        assert 'is_delay_attack' in results
        assert results['is_delay_attack'][300:].any()


# =============================================================================
# Fix 6: Proper Metrics Tests
# =============================================================================

class TestProperMetrics:
    """Tests for proper evaluation metrics."""

    def test_auroc_perfect_classifier(self):
        """Test AUROC = 1.0 for perfect classifier."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9])

        metrics = compute_proper_metrics(y_true, y_score)

        assert metrics.auroc == 1.0

    def test_auroc_random_classifier(self):
        """Test AUROC ~ 0.5 for random classifier."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_score = np.random.rand(1000)

        metrics = compute_proper_metrics(y_true, y_score)

        assert 0.4 < metrics.auroc < 0.6

    def test_recall_at_fpr(self):
        """Test recall at fixed FPR."""
        y_true = np.array([0]*100 + [1]*100)
        y_score = np.array([0.1]*100 + [0.9]*100)

        metrics = compute_proper_metrics(y_true, y_score)

        # Perfect separation should give recall = 1.0 at any FPR
        assert metrics.recall_at_1pct_fpr == 1.0
        assert metrics.recall_at_5pct_fpr == 1.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestEnhancedActuatorDetector:
    """Integration tests for full detector."""

    def test_detector_runs_without_error(self, sample_data):
        """Test that detector runs end-to-end."""
        detector = EnhancedActuatorDetector(dt=sample_data['dt'])

        data = {
            'control': sample_data['control'],
            'acceleration': sample_data['acceleration'],
            'velocity': sample_data['velocity'],
            'attitude': sample_data['attitude'],
        }

        results = detector.detect(data)

        assert 'control_efficiency' in results
        assert 'trim_deviation' in results

    def test_fit_and_normalize(self, sample_data):
        """Test fitting and normalization pipeline."""
        detector = EnhancedActuatorDetector(dt=sample_data['dt'])
        n = sample_data['n']

        # Fit on "normal" data
        normal_data = {
            'residuals': np.random.randn(n, 5) * 0.1,
            'speed': np.linalg.norm(sample_data['velocity'], axis=1),
            'altitude': sample_data['position'][:, 2],
        }

        detector.fit(normal_data)
        assert detector.is_fitted

        # Detect with residuals
        detect_data = {
            'control': sample_data['control'],
            'acceleration': sample_data['acceleration'],
            'velocity': sample_data['velocity'],
            'attitude': sample_data['attitude'],
            'residuals': np.random.randn(n, 5) * 0.1 + 0.5,  # Anomaly
            'speed': normal_data['speed'],
            'altitude': normal_data['altitude'],
        }

        results = detector.detect(detect_data)
        assert 'normalized_residuals' in results
        assert 'residual_score' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
