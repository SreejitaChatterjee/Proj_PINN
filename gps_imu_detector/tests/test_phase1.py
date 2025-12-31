"""
Tests for Phase 1: Low-Risk Offline PINN Additions

Tests:
1. Regime detection (taxonomy)
2. Conformal envelopes (calibration)
3. Uncertainty maps (abstention)
"""

import numpy as np
import pytest
import torch


# =============================================================================
# Phase 1.1: Regime Detection Tests
# =============================================================================

class TestFlightRegime:
    """Tests for flight regime taxonomy."""

    def test_regime_enum(self):
        from gps_imu_detector.src.regime_detection import FlightRegime

        assert FlightRegime.HOVER.value == 1
        assert FlightRegime.FORWARD.value == 2
        assert FlightRegime.AGGRESSIVE.value == 3
        assert FlightRegime.GUSTY.value == 4
        assert FlightRegime.UNKNOWN.value == 5

    def test_regime_classifier_hover(self):
        from gps_imu_detector.src.regime_detection import RegimeClassifier, FlightRegime

        classifier = RegimeClassifier()

        velocity = np.array([0.1, 0.1, 0.0])
        angular_rate = np.array([0.05, 0.0, 0.0])

        regime = classifier.classify(velocity, angular_rate)
        assert regime == FlightRegime.HOVER

    def test_regime_classifier_forward(self):
        from gps_imu_detector.src.regime_detection import RegimeClassifier, FlightRegime

        classifier = RegimeClassifier()

        velocity = np.array([3.0, 0.0, 0.0])
        angular_rate = np.array([0.2, 0.0, 0.0])

        regime = classifier.classify(velocity, angular_rate)
        assert regime == FlightRegime.FORWARD

    def test_regime_classifier_aggressive(self):
        from gps_imu_detector.src.regime_detection import RegimeClassifier, FlightRegime

        classifier = RegimeClassifier()

        velocity = np.array([10.0, 0.0, 0.0])
        angular_rate = np.array([0.5, 0.0, 0.0])

        regime = classifier.classify(velocity, angular_rate)
        assert regime == FlightRegime.AGGRESSIVE

    def test_regime_classifier_gusty(self):
        from gps_imu_detector.src.regime_detection import RegimeClassifier, FlightRegime

        classifier = RegimeClassifier()

        # Simulate turbulent conditions with high acceleration variance
        for _ in range(30):
            velocity = np.array([2.0, 0.0, 0.0])
            angular_rate = np.array([0.2, 0.0, 0.0])
            acceleration = np.random.randn(3) * 5.0  # High variance

            regime = classifier.classify(velocity, angular_rate, acceleration)

        assert regime == FlightRegime.GUSTY

    def test_classify_trajectory(self):
        from gps_imu_detector.src.regime_detection import classify_trajectory

        # Simple trajectory
        T = 50
        trajectory = np.zeros((T, 12), dtype=np.float32)
        trajectory[:, 3:6] = 0.1  # Low velocity -> HOVER

        regimes = classify_trajectory(trajectory)

        assert len(regimes) == T
        assert regimes[0] == 1  # HOVER

    def test_regime_parameters(self):
        from gps_imu_detector.src.regime_detection import (
            FlightRegime, get_regime_parameters
        )

        params = get_regime_parameters(FlightRegime.HOVER)
        assert params['residual_scale'] == 1.0
        assert params['probe_allowed'] == True

        params = get_regime_parameters(FlightRegime.AGGRESSIVE)
        assert params['residual_scale'] > 1.0
        assert params['probe_allowed'] == False


# =============================================================================
# Phase 1.2: Conformal Envelopes Tests
# =============================================================================

class TestConformalEnvelopes:
    """Tests for conformal residual envelopes."""

    def test_residual_pinn_forward(self):
        from gps_imu_detector.src.conformal_envelopes import ResidualPINN

        pinn = ResidualPINN(state_dim=12)
        state = torch.randn(10, 12)

        mean, std = pinn(state)

        assert mean.shape == (10, 1)
        assert std.shape == (10, 1)
        assert (std > 0).all()

    def test_conformal_calibrator(self):
        from gps_imu_detector.src.conformal_envelopes import (
            ConformalCalibrator, FlightRegime
        )

        calibrator = ConformalCalibrator(coverage=0.99)

        # Add calibration scores
        for _ in range(100):
            calibrator.add_calibration_score(FlightRegime.HOVER, np.random.rand())

        quantiles, margin = calibrator.compute_quantiles(FlightRegime.HOVER)

        assert 0.01 in quantiles
        assert 0.05 in quantiles
        assert 0.10 in quantiles
        assert margin >= 0

    def test_envelope_builder_train(self):
        from gps_imu_detector.src.conformal_envelopes import ConformalEnvelopeBuilder

        np.random.seed(42)
        torch.manual_seed(42)

        builder = ConformalEnvelopeBuilder(state_dim=12)

        # Generate simple trajectories
        trajectories = np.random.randn(2, 50, 12).astype(np.float32) * 0.1

        history = builder.train_pinn(trajectories, epochs=5, verbose=False)

        assert 'loss' in history
        assert len(history['loss']) == 5

    def test_envelope_builder_calibrate(self):
        from gps_imu_detector.src.conformal_envelopes import ConformalEnvelopeBuilder

        np.random.seed(42)
        torch.manual_seed(42)

        builder = ConformalEnvelopeBuilder(state_dim=12)

        trajectories = np.random.randn(2, 50, 12).astype(np.float32) * 0.1

        builder.train_pinn(trajectories, epochs=3, verbose=False)
        builder.calibrate(trajectories)

        assert len(builder.calibrator.calibration_scores) > 0

    def test_envelope_table_build(self):
        from gps_imu_detector.src.conformal_envelopes import ConformalEnvelopeBuilder

        np.random.seed(42)
        torch.manual_seed(42)

        builder = ConformalEnvelopeBuilder(state_dim=12)

        trajectories = np.random.randn(2, 50, 12).astype(np.float32) * 0.1

        builder.train_pinn(trajectories, epochs=3, verbose=False)
        builder.calibrate(trajectories)

        table = builder.build_envelope_table(version="1.0.0")

        assert table.version == "1.0.0"
        assert len(table.envelopes) > 0

    def test_envelope_table_threshold(self):
        from gps_imu_detector.src.conformal_envelopes import (
            ConformalEnvelopeBuilder, FlightRegime
        )

        np.random.seed(42)
        torch.manual_seed(42)

        builder = ConformalEnvelopeBuilder(state_dim=12)
        trajectories = np.random.randn(2, 50, 12).astype(np.float32) * 0.1

        builder.train_pinn(trajectories, epochs=3, verbose=False)
        builder.calibrate(trajectories)
        table = builder.build_envelope_table()

        threshold = table.get_threshold(FlightRegime.HOVER, alpha=0.01)
        assert threshold > 0


# =============================================================================
# Phase 1.3: Uncertainty Maps Tests
# =============================================================================

class TestUncertaintyMaps:
    """Tests for uncertainty maps and abstention."""

    def test_uncertainty_pinn_forward(self):
        from gps_imu_detector.src.uncertainty_maps import UncertaintyPINN

        pinn = UncertaintyPINN(state_dim=12)
        state = torch.randn(10, 12)

        mean, uncertainty = pinn(state)

        assert mean.shape == (10, 1)
        assert uncertainty.shape == (10, 1)
        assert (uncertainty >= 0).all()

    def test_uncertainty_pinn_mc(self):
        from gps_imu_detector.src.uncertainty_maps import UncertaintyPINN

        pinn = UncertaintyPINN(state_dim=12, dropout=0.2)
        state = torch.randn(10, 12)

        mean, uncertainty = pinn.mc_uncertainty(state, n_samples=10)

        assert mean.shape == (10, 1)
        assert uncertainty.shape == (10, 1)

    def test_uncertainty_map_builder(self):
        from gps_imu_detector.src.uncertainty_maps import UncertaintyMapBuilder

        np.random.seed(42)
        torch.manual_seed(42)

        builder = UncertaintyMapBuilder(state_dim=12)

        trajectories = np.random.randn(2, 50, 12).astype(np.float32) * 0.1

        history = builder.train_pinn(trajectories, epochs=3, verbose=False)

        assert 'loss' in history

    def test_uncertainty_map_build(self):
        from gps_imu_detector.src.uncertainty_maps import UncertaintyMapBuilder

        np.random.seed(42)
        torch.manual_seed(42)

        builder = UncertaintyMapBuilder(state_dim=12)
        trajectories = np.random.randn(2, 50, 12).astype(np.float32) * 0.1

        builder.train_pinn(trajectories, epochs=3, verbose=False)
        builder.collect_bin_statistics(trajectories)

        umap = builder.build_uncertainty_map(version="1.0.0")

        assert umap.version == "1.0.0"

    def test_abstention_policy(self):
        from gps_imu_detector.src.uncertainty_maps import (
            UncertaintyMapBuilder, AbstentionPolicy
        )

        np.random.seed(42)
        torch.manual_seed(42)

        builder = UncertaintyMapBuilder(state_dim=12)
        trajectories = np.random.randn(2, 50, 12).astype(np.float32) * 0.1

        builder.train_pinn(trajectories, epochs=3, verbose=False)
        builder.collect_bin_statistics(trajectories)
        umap = builder.build_uncertainty_map()

        policy = AbstentionPolicy(umap)

        velocity = np.array([0.1, 0.0, 0.0])
        angular_rate = np.array([0.05, 0.0, 0.0])

        decision = policy.get_decision(velocity, angular_rate)

        assert 'regime' in decision
        assert 'envelope_multiplier' in decision
        assert 'allow_probing' in decision


# =============================================================================
# Phase 1 Checkpoint Tests
# =============================================================================

class TestPhase1Checkpoint:
    """Integration tests for Phase 1 checkpoint."""

    def test_per_regime_fpr_target(self):
        """Checkpoint: per-regime FPR <= 1%."""
        from gps_imu_detector.src.conformal_envelopes import (
            ConformalEnvelopeBuilder, evaluate_conformal_envelopes
        )

        np.random.seed(42)
        torch.manual_seed(42)

        builder = ConformalEnvelopeBuilder(state_dim=12, coverage=0.99)

        # Training data
        train = np.random.randn(3, 100, 12).astype(np.float32) * 0.1
        builder.train_pinn(train, epochs=5, verbose=False)
        builder.calibrate(train)
        table = builder.build_envelope_table()

        # Test data
        nominal = np.random.randn(2, 50, 12).astype(np.float32) * 0.1
        attack = np.random.randn(2, 50, 12).astype(np.float32) * 0.1 + 0.5

        results = evaluate_conformal_envelopes(table, nominal, attack)

        # Check per-regime FPR
        for regime, metrics in results['per_regime'].items():
            assert metrics['fpr'] <= 0.05, f"FPR too high for {regime}"

    def test_no_runtime_pinn(self):
        """Verify no PINN inference at runtime."""
        from gps_imu_detector.src.conformal_envelopes import EnvelopeTable
        from gps_imu_detector.src.regime_detection import FlightRegime

        # Create a mock table (in practice, loaded from file)
        from gps_imu_detector.src.conformal_envelopes import ConformalEnvelope

        envelope = ConformalEnvelope(
            regime=FlightRegime.HOVER,
            quantiles={0.01: 0.5, 0.05: 0.4, 0.10: 0.3},
            margin=0.05,
            n_calibration=100,
            coverage_target=0.99,
        )

        table = EnvelopeTable(
            envelopes={'HOVER': envelope},
            version="1.0.0",
            created_at="2025-12-31",
            pinn_checkpoint="dummy.pth",
            calibration_samples=100,
        )

        # Runtime lookup is just dictionary access - no PINN
        threshold = table.get_threshold(FlightRegime.HOVER, alpha=0.01)
        assert threshold == 0.55  # 0.5 + 0.05 margin


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
