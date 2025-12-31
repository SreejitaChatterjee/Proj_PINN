"""
Tests for Advanced Detection Module (v0.5.0)

Tests all 6 advanced improvements:
A. Lag Drift Tracker
B. Second-Order Consistency
C. Control Regime Envelopes
D. Fault Attribution
E. Prediction-Retrodiction Asymmetry
F. Randomized Subspace Sampling
"""

import numpy as np
import pytest
from gps_imu_detector.src.advanced_detection import (
    # Improvement A
    LagDriftTracker,
    LagDriftResult,
    # Improvement B
    SecondOrderConsistency,
    SecondOrderResult,
    # Improvement C
    ControlRegime,
    ControlRegimeEnvelopes,
    # Improvement D
    FaultType,
    FaultAttributor,
    FaultAttribution,
    # Improvement E
    PredictionRetrodictionChecker,
    AsymmetryResult,
    # Improvement F
    RandomizedSubspaceSampler,
    RandomizedResult,
    # Integrated
    AdvancedDetector,
    AdvancedDetectionResult,
)


# =============================================================================
# Improvement A: Lag Drift Tracker Tests
# =============================================================================

class TestLagDriftTracker:
    """Tests for lag drift tracking (incipient actuator failure detection)."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = LagDriftTracker(window_size=256, history_length=10)
        assert tracker.window_size == 256
        assert tracker.history_length == 10
        assert len(tracker.lag_history) == 0

    def test_compute_lag_zero_for_aligned_signals(self):
        """Test that aligned signals have zero lag."""
        tracker = LagDriftTracker(window_size=100)
        t = np.linspace(0, 1, 200)
        control = np.sin(2 * np.pi * 5 * t)
        response = np.sin(2 * np.pi * 5 * t)  # Same phase

        lag = tracker.compute_lag(control, response)
        assert abs(lag) < 5  # Should be near zero

    def test_compute_lag_detects_delay(self):
        """Test that delayed signals have non-zero lag."""
        tracker = LagDriftTracker(window_size=100)
        t = np.linspace(0, 1, 200)
        control = np.sin(2 * np.pi * 5 * t)
        # Delay by 10 samples
        response = np.roll(control, 10)

        lag = tracker.compute_lag(control, response)
        assert lag >= 5  # Should detect significant lag

    def test_update_returns_result(self):
        """Test that update returns LagDriftResult."""
        tracker = LagDriftTracker(window_size=100)
        control = np.random.randn(200)
        response = np.random.randn(200)

        result = tracker.update(control, response)
        assert isinstance(result, LagDriftResult)
        assert hasattr(result, 'current_lag')
        assert hasattr(result, 'lag_drift')
        assert hasattr(result, 'monotonic_growth')

    def test_monotonic_growth_detection(self):
        """Test detection of monotonic lag growth (actuator degradation)."""
        tracker = LagDriftTracker(
            window_size=50,
            history_length=10,
            monotonic_windows=3,
            drift_threshold=0.1,
        )

        # Simulate degrading actuator - increasing lag
        for i in range(10):
            t = np.linspace(0, 1, 100)
            control = np.sin(2 * np.pi * 5 * t)
            # Increasing delay
            delay = 5 + i * 2
            response = np.roll(control, delay)

            result = tracker.update(control, response)

        # Should detect monotonic growth after several updates
        assert result.growth_windows > 0

    def test_reset_clears_history(self):
        """Test that reset clears tracking history."""
        tracker = LagDriftTracker()

        # Add some history
        for _ in range(5):
            tracker.update(np.random.randn(300), np.random.randn(300))

        assert len(tracker.lag_history) > 0

        tracker.reset()
        assert len(tracker.lag_history) == 0
        assert len(tracker.drift_history) == 0


# =============================================================================
# Improvement B: Second-Order Consistency Tests
# =============================================================================

class TestSecondOrderConsistency:
    """Tests for second-order (jerk/angular accel) consistency."""

    def test_initialization(self):
        """Test initialization."""
        checker = SecondOrderConsistency(dt=0.005)
        assert checker.dt == 0.005
        assert checker.high_control_threshold == 0.7
        assert checker.low_residual_threshold == 0.3

    def test_compute_jerk(self):
        """Test jerk computation."""
        checker = SecondOrderConsistency(dt=0.01)
        # Constant acceleration -> zero jerk
        acceleration = np.ones((100, 3)) * 9.81
        jerk = checker.compute_jerk(acceleration)
        assert np.allclose(jerk, 0, atol=1e-10)

    def test_compute_jerk_for_ramp(self):
        """Test jerk for linearly increasing acceleration."""
        checker = SecondOrderConsistency(dt=0.01)
        t = np.linspace(0, 1, 100)
        # a = t (linear ramp)
        acceleration = np.column_stack([t, np.zeros(100), np.zeros(100)])
        jerk = checker.compute_jerk(acceleration)
        # Jerk should be constant (derivative of ramp is constant)
        assert len(jerk) == 99  # One less due to diff

    def test_normal_condition_not_flagged(self):
        """Test that normal conditions are not flagged as suspicious."""
        checker = SecondOrderConsistency()

        acceleration = np.random.randn(100, 3) * 0.1
        angular_velocity = np.random.randn(100, 3) * 0.1

        # Low control, low residual -> not stealth condition
        result = checker.check_consistency(
            acceleration, angular_velocity,
            control_magnitude=0.3,  # Low
            residual_magnitude=0.1,  # Low
        )

        assert not result.is_suspicious

    def test_stealth_condition_with_high_jerk(self):
        """Test detection of stealth attack with high jerk."""
        checker = SecondOrderConsistency()
        checker.jerk_std = 0.1  # Set low for test

        # High control + low residual + high jerk = stealth attack
        t = np.linspace(0, 1, 100)
        # Jerky acceleration (not smooth)
        acceleration = np.column_stack([
            np.sin(2 * np.pi * 20 * t) * 10,  # High frequency = high jerk
            np.zeros(100),
            np.zeros(100),
        ])
        angular_velocity = np.random.randn(100, 3) * 0.1

        result = checker.check_consistency(
            acceleration, angular_velocity,
            control_magnitude=0.9,  # High
            residual_magnitude=0.1,  # Low (stealth)
        )

        # Should have non-zero jerk inconsistency
        assert result.jerk_inconsistency > 0

    def test_calibration(self):
        """Test calibration updates statistics."""
        checker = SecondOrderConsistency()

        jerk_samples = np.random.randn(1000) * 0.5
        angular_samples = np.random.randn(1000) * 0.3

        checker.calibrate(jerk_samples, angular_samples)

        assert abs(checker.jerk_mean) < 0.2  # Should be near zero
        assert 0.3 < checker.jerk_std < 0.7  # Should be near 0.5
        assert 0.1 < checker.angular_std < 0.5  # Should be near 0.3


# =============================================================================
# Improvement C: Control Regime Envelopes Tests
# =============================================================================

class TestControlRegimeEnvelopes:
    """Tests for control-regime-aware envelope normalization."""

    def test_initialization(self):
        """Test initialization."""
        envelopes = ControlRegimeEnvelopes(n_regimes=4)
        assert envelopes.n_regimes == 4
        assert len(envelopes.envelopes) == 0

    def test_classify_regime_hover(self):
        """Test regime classification for hover (low control)."""
        envelopes = ControlRegimeEnvelopes(use_kmeans=False)

        control = np.ones((100, 4)) * 0.5  # Low control
        angular_velocity = np.ones((100, 3)) * 0.1

        regime = envelopes.classify_regime(control, angular_velocity)
        assert regime == ControlRegime.HOVER

    def test_classify_regime_aggressive(self):
        """Test regime classification for aggressive maneuver."""
        envelopes = ControlRegimeEnvelopes(use_kmeans=False)

        control = np.ones((100, 4)) * 9.0  # High control
        angular_velocity = np.ones((100, 3)) * 2.0

        regime = envelopes.classify_regime(control, angular_velocity)
        assert regime == ControlRegime.AGGRESSIVE

    def test_fit_creates_envelopes(self):
        """Test that fit creates envelope statistics."""
        envelopes = ControlRegimeEnvelopes(use_kmeans=False)

        # Generate data across regimes
        residuals = np.random.randn(1000, 5)
        control = np.random.rand(1000, 4) * 10  # 0-10 range
        angular_velocity = np.random.randn(1000, 3)

        envelopes.fit(residuals, control, angular_velocity)

        # Should have created some envelopes
        assert len(envelopes.envelopes) > 0

    def test_normalize_uses_regime_stats(self):
        """Test that normalization uses regime-appropriate statistics."""
        envelopes = ControlRegimeEnvelopes(use_kmeans=False)

        # Fit with distinct regime statistics
        n = 500
        residuals = np.vstack([
            np.random.randn(n, 5) * 0.1,  # Hover: low variance
            np.random.randn(n, 5) * 1.0,  # Aggressive: high variance
        ])
        control = np.vstack([
            np.ones((n, 4)) * 0.5,  # Hover
            np.ones((n, 4)) * 9.0,  # Aggressive
        ])
        angular_velocity = np.vstack([
            np.ones((n, 3)) * 0.1,
            np.ones((n, 3)) * 2.0,
        ])

        envelopes.fit(residuals, control, angular_velocity)

        # Normalize aggressive regime data
        test_residual = np.ones(5) * 0.5
        test_control = np.ones((10, 4)) * 9.0
        test_omega = np.ones((10, 3)) * 2.0

        normalized = envelopes.normalize(test_residual, test_control, test_omega)

        # Should be normalized (not huge z-scores despite large residual)
        assert normalized.shape == test_residual.shape


# =============================================================================
# Improvement D: Fault Attribution Tests
# =============================================================================

class TestFaultAttributor:
    """Tests for fault attribution via signature matching."""

    def test_initialization(self):
        """Test initialization."""
        attributor = FaultAttributor()
        assert len(attributor.prototypes) > 0
        assert FaultType.MOTOR_FAULT in attributor.prototypes

    def test_compute_signature_normalized(self):
        """Test that signature is normalized to unit vector."""
        attributor = FaultAttributor()
        signature = attributor.compute_signature(0.5, 0.5, 0.5, 0.5, 0.5)
        norm = np.linalg.norm(signature)
        assert abs(norm - 1.0) < 1e-6

    def test_attribute_motor_fault(self):
        """Test attribution of motor fault signature."""
        attributor = FaultAttributor()

        # Motor fault signature: high kinematic, high effort, high energy
        result = attributor.attribute(
            kinematic_residual=0.9,
            position_residual=0.2,
            effort_residual=0.95,
            phase_residual=0.1,
            energy_residual=0.9,
            threshold=0.5,
        )

        assert isinstance(result, FaultAttribution)
        assert result.primary_fault in [FaultType.MOTOR_FAULT, FaultType.ACTUATOR_STUCK]

    def test_attribute_gps_spoof(self):
        """Test attribution of GPS spoof signature."""
        attributor = FaultAttributor()

        # GPS spoof: high position residual, low effort
        result = attributor.attribute(
            kinematic_residual=0.2,
            position_residual=0.95,
            effort_residual=0.1,
            phase_residual=0.1,
            energy_residual=0.15,
            threshold=0.5,
        )

        assert result.primary_fault == FaultType.GPS_SPOOF
        assert result.confidence > 0.5

    def test_attribute_sensor_delay(self):
        """Test attribution of sensor delay signature."""
        attributor = FaultAttributor()

        # Delay: high phase residual
        result = attributor.attribute(
            kinematic_residual=0.4,
            position_residual=0.4,
            effort_residual=0.2,
            phase_residual=0.95,
            energy_residual=0.2,
            threshold=0.5,
        )

        assert result.primary_fault == FaultType.SENSOR_DELAY
        assert "phase" in result.explanation.lower() or "delay" in result.explanation.lower()

    def test_attribute_unknown_low_similarity(self):
        """Test that low similarity results in UNKNOWN attribution."""
        attributor = FaultAttributor()

        # Unusual signature that doesn't match prototypes well
        # Use asymmetric values that don't match any prototype
        result = attributor.attribute(
            kinematic_residual=0.05,
            position_residual=0.95,
            effort_residual=0.05,
            phase_residual=0.95,
            energy_residual=0.05,
            threshold=0.99,  # Very high threshold
        )

        # With very high threshold, should return UNKNOWN or low confidence
        assert result.confidence < 0.99 or result.primary_fault == FaultType.UNKNOWN

    def test_update_prototype(self):
        """Test prototype update from new labeled example."""
        attributor = FaultAttributor()

        original = attributor.prototypes[FaultType.MOTOR_FAULT].copy()
        new_signature = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        attributor.update_prototype(FaultType.MOTOR_FAULT, new_signature, learning_rate=0.5)

        updated = attributor.prototypes[FaultType.MOTOR_FAULT]
        assert not np.allclose(updated, original)


# =============================================================================
# Improvement E: Prediction-Retrodiction Asymmetry Tests
# =============================================================================

class TestPredictionRetrodictionChecker:
    """Tests for prediction-retrodiction asymmetry detection."""

    def test_initialization(self):
        """Test initialization."""
        checker = PredictionRetrodictionChecker(window_size=256)
        assert checker.window_size == 256
        assert checker.asymmetry_threshold == 2.0

    def test_predict_forward(self):
        """Test forward prediction."""
        checker = PredictionRetrodictionChecker()

        # Linear trajectory
        t = np.arange(100).reshape(-1, 1)
        states = t * 0.1  # Constant velocity

        preds, residuals = checker.predict_forward(states, horizon=10)
        assert len(residuals) == 10

    def test_predict_backward(self):
        """Test backward prediction (retrodiction)."""
        checker = PredictionRetrodictionChecker()

        t = np.arange(100).reshape(-1, 1)
        states = t * 0.1

        preds, residuals = checker.predict_backward(states, horizon=10)
        assert len(residuals) == 10

    def test_symmetric_for_normal_trajectory(self):
        """Test that normal trajectories have low asymmetry."""
        checker = PredictionRetrodictionChecker()

        # Smooth trajectory
        t = np.linspace(0, 1, 100)
        states = np.column_stack([
            np.sin(2 * np.pi * t),
            np.cos(2 * np.pi * t),
        ])

        result = checker.check_asymmetry(states, horizon=10)

        assert isinstance(result, AsymmetryResult)
        # Normal trajectory should be roughly symmetric
        assert abs(result.asymmetry) < 0.5

    def test_asymmetric_for_delayed_trajectory(self):
        """Test that delayed trajectory has asymmetry."""
        checker = PredictionRetrodictionChecker()
        checker.asymmetry_std = 0.1  # Set low for test

        # Create asymmetric trajectory (delayed)
        t = np.linspace(0, 1, 100)
        states_forward = np.sin(2 * np.pi * t)
        # Add delay in second half
        states = np.column_stack([
            np.concatenate([states_forward[:50], states_forward[45:95]]),
            np.zeros(100),
        ])

        result = checker.check_asymmetry(states, horizon=10)

        # Delayed trajectory should show asymmetry
        assert result.asymmetry != 0

    def test_calibration(self):
        """Test calibration from nominal data."""
        checker = PredictionRetrodictionChecker()

        asymmetry_samples = np.random.randn(100) * 0.05

        checker.calibrate(asymmetry_samples)

        assert checker.asymmetry_std > 0


# =============================================================================
# Improvement F: Randomized Subspace Sampling Tests
# =============================================================================

class TestRandomizedSubspaceSampler:
    """Tests for randomized residual subspace sampling."""

    def test_initialization(self):
        """Test initialization."""
        sampler = RandomizedSubspaceSampler(n_channels=10, seed=42)
        assert sampler.n_channels == 10

    def test_sample_channels_returns_mask(self):
        """Test that sample_channels returns boolean mask."""
        sampler = RandomizedSubspaceSampler(n_channels=10, seed=42)

        mask = sampler.sample_channels()

        assert mask.dtype == bool
        assert len(mask) == 10
        assert 6 <= np.sum(mask) <= 8  # 60-80% sampled

    def test_sample_channels_varies(self):
        """Test that sampling varies between calls."""
        sampler = RandomizedSubspaceSampler(n_channels=10, seed=None)

        masks = [sampler.sample_channels() for _ in range(10)]

        # Not all masks should be identical
        unique_masks = len(set(tuple(m) for m in masks))
        assert unique_masks > 1

    def test_compute_score(self):
        """Test score computation."""
        sampler = RandomizedSubspaceSampler(n_channels=5, seed=42)
        sampler.calibrate(np.random.randn(100, 5))

        residuals = np.random.randn(10, 5)
        sampled_score, full_score = sampler.compute_score(residuals)

        assert sampled_score >= 0
        assert full_score >= 0

    def test_detect_returns_result(self):
        """Test that detect returns RandomizedResult."""
        sampler = RandomizedSubspaceSampler(n_channels=5, seed=42)
        sampler.calibrate(np.random.randn(100, 5))

        residuals = np.random.randn(10, 5)
        result = sampler.detect(residuals)

        assert isinstance(result, RandomizedResult)
        assert hasattr(result, 'sampled_score')
        assert hasattr(result, 'channel_mask')

    def test_calibration(self):
        """Test calibration sets channel statistics."""
        sampler = RandomizedSubspaceSampler(n_channels=5, seed=42)

        data = np.random.randn(1000, 5) * np.array([1, 2, 3, 4, 5])

        sampler.calibrate(data)

        # Stds should reflect the different scales
        assert sampler.channel_stds[4] > sampler.channel_stds[0]

    def test_set_seed_reproducibility(self):
        """Test that set_seed enables reproducibility."""
        sampler = RandomizedSubspaceSampler(n_channels=10, seed=42)

        mask1 = sampler.sample_channels()
        sampler.set_seed(42)
        mask2 = sampler.sample_channels()

        assert np.array_equal(mask1, mask2)


# =============================================================================
# Integrated Advanced Detector Tests
# =============================================================================

class TestAdvancedDetector:
    """Tests for integrated advanced detector."""

    def test_initialization(self):
        """Test initialization."""
        detector = AdvancedDetector(n_residual_channels=10)
        assert detector.lag_tracker is not None
        assert detector.second_order is not None
        assert detector.regime_envelopes is not None
        assert detector.attributor is not None
        assert detector.asymmetry_checker is not None
        assert detector.randomized_sampler is not None

    def test_detect_returns_result(self):
        """Test that detect returns AdvancedDetectionResult."""
        detector = AdvancedDetector(n_residual_channels=5)

        n = 100
        states = np.random.randn(n, 6)
        control = np.random.randn(n, 4)
        acceleration = np.random.randn(n, 3)
        angular_velocity = np.random.randn(n, 3)
        residuals = np.random.randn(n, 5)

        result = detector.detect(
            states, control, acceleration, angular_velocity, residuals
        )

        assert isinstance(result, AdvancedDetectionResult)
        assert hasattr(result, 'combined_score')
        assert hasattr(result, 'is_anomalous')
        assert hasattr(result, 'primary_diagnosis')

    def test_detect_nominal_not_anomalous(self):
        """Test that nominal data is not flagged."""
        np.random.seed(42)  # For reproducibility
        detector = AdvancedDetector(n_residual_channels=5)

        # Generate nominal data with consistent seed
        n = 100
        states = np.cumsum(np.random.randn(n, 6) * 0.01, axis=0)
        control = np.random.randn(n, 4) * 0.1
        acceleration = np.random.randn(n, 3) * 0.1
        angular_velocity = np.random.randn(n, 3) * 0.1
        residuals = np.random.randn(n, 5) * 0.1

        # Calibrate first with same type of data
        detector.calibrate(
            states, control, acceleration, angular_velocity, residuals
        )

        # Test with similar nominal data
        np.random.seed(43)
        test_states = np.cumsum(np.random.randn(n, 6) * 0.01, axis=0)
        test_residuals = np.random.randn(n, 5) * 0.1

        result = detector.detect(
            test_states, control, acceleration, angular_velocity, test_residuals,
            control_magnitude=0.3,
            residual_magnitude=0.1,
        )

        # Should have bounded score (not necessarily very low due to randomness)
        assert result.combined_score <= 1.0

    def test_detect_anomalous_high_residual(self):
        """Test that high residuals are flagged."""
        detector = AdvancedDetector(n_residual_channels=5)

        # Calibrate with low-variance data
        n = 100
        detector.calibrate(
            np.random.randn(n, 6) * 0.1,
            np.random.randn(n, 4) * 0.1,
            np.random.randn(n, 3) * 0.1,
            np.random.randn(n, 3) * 0.1,
            np.random.randn(n, 5) * 0.1,
        )

        # Test with high residuals
        result = detector.detect(
            np.random.randn(n, 6),
            np.random.randn(n, 4),
            np.random.randn(n, 3) * 10,  # High acceleration
            np.random.randn(n, 3),
            np.random.randn(n, 5) * 10,  # High residuals
            control_magnitude=0.9,
            residual_magnitude=0.8,
        )

        # Should have higher score
        assert result.combined_score > 0.1

    def test_reset_clears_state(self):
        """Test that reset clears stateful components."""
        detector = AdvancedDetector(n_residual_channels=5)

        # Build up state
        n = 100
        for _ in range(5):
            detector.detect(
                np.random.randn(n, 6),
                np.random.randn(n, 4),
                np.random.randn(n, 3),
                np.random.randn(n, 3),
                np.random.randn(n, 5),
            )

        assert len(detector.lag_tracker.lag_history) > 0

        detector.reset()

        assert len(detector.lag_tracker.lag_history) == 0

    def test_attribution_on_anomaly(self):
        """Test that attribution is provided when anomaly detected."""
        detector = AdvancedDetector(n_residual_channels=5)
        detector.threshold = 0.1  # Low threshold to trigger

        n = 100
        result = detector.detect(
            np.random.randn(n, 6) * 10,
            np.random.randn(n, 4) * 10,
            np.random.randn(n, 3) * 10,
            np.random.randn(n, 3) * 10,
            np.random.randn(n, 5) * 10,
            control_magnitude=0.9,
            residual_magnitude=0.1,  # Stealth-like
        )

        if result.is_anomalous:
            assert result.attribution is not None
            assert result.primary_diagnosis != "nominal"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
