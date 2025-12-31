"""
Tests for PINN Integration Module (v0.9.0)

Tests:
1. Option 1: PINN as shadow residual
2. Option 2: PINN for offline envelope (TODO)
3. Option 3: PINN for probing response (TODO)
"""

import numpy as np
import pytest
import torch


# =============================================================================
# Option 1: Shadow Residual Tests
# =============================================================================

class TestQuadrotorPINNResidual:
    """Tests for PINN residual network."""

    def test_initialization(self):
        from gps_imu_detector.src.pinn_integration import QuadrotorPINNResidual

        pinn = QuadrotorPINNResidual(state_dim=12)
        assert pinn.state_dim == 12
        assert pinn.physics_weight == 0.1

    def test_forward_returns_residuals(self):
        from gps_imu_detector.src.pinn_integration import QuadrotorPINNResidual

        pinn = QuadrotorPINNResidual(state_dim=12)

        state = torch.randn(10, 12)
        next_state = torch.randn(10, 12)

        phys_res, corrected_res = pinn(state, next_state)

        assert phys_res.shape == (10, 6)
        assert corrected_res.shape == (10, 6)

    def test_physics_residual_computation(self):
        from gps_imu_detector.src.pinn_integration import QuadrotorPINNResidual

        pinn = QuadrotorPINNResidual(state_dim=12)

        # Create consistent kinematics: next_pos = pos + vel * dt
        dt = 0.005
        state = torch.zeros(1, 12)
        state[0, :3] = torch.tensor([0., 0., 0.])  # position
        state[0, 3:6] = torch.tensor([1., 0., 0.])  # velocity

        # Perfect kinematics
        next_state = torch.zeros(1, 12)
        next_state[0, :3] = state[0, :3] + state[0, 3:6] * dt
        next_state[0, 3:6] = state[0, 3:6] + torch.tensor([0., 0., -9.81]) * dt

        phys_res = pinn.physics_residual(state, next_state, dt)

        # Position residual should be near zero
        assert phys_res[0, :3].abs().max() < 1e-5

    def test_small_corrections(self):
        """Corrections should be small due to initialization."""
        from gps_imu_detector.src.pinn_integration import QuadrotorPINNResidual

        pinn = QuadrotorPINNResidual(state_dim=12)

        state = torch.randn(100, 12)
        corrections = pinn.correction_net(state)[:, :6]

        # Should be small (initialized with gain=0.1)
        assert corrections.abs().mean() < 1.0


class TestPINNShadowResidual:
    """Tests for PINN shadow residual system."""

    def test_initialization(self):
        from gps_imu_detector.src.pinn_integration import PINNShadowResidual

        shadow = PINNShadowResidual(state_dim=12, alpha=0.15)

        assert shadow.alpha == 0.15
        assert shadow.state_dim == 12

    def test_fit_pinn(self):
        from gps_imu_detector.src.pinn_integration import PINNShadowResidual

        np.random.seed(42)
        torch.manual_seed(42)

        # Generate simple trajectory
        T = 200
        state_dim = 12
        traj = np.random.randn(1, T, state_dim).astype(np.float32) * 0.1

        shadow = PINNShadowResidual(state_dim=state_dim, alpha=0.15)
        history = shadow.fit_pinn(traj, epochs=5, verbose=False)

        assert 'loss' in history
        assert len(history['loss']) == 5
        assert shadow.pinn_std > 0

    def test_compute_residual(self):
        from gps_imu_detector.src.pinn_integration import PINNShadowResidual

        np.random.seed(42)
        torch.manual_seed(42)

        shadow = PINNShadowResidual(state_dim=12, alpha=0.15)

        # Train briefly
        traj = np.random.randn(1, 100, 12).astype(np.float32) * 0.1
        shadow.fit_pinn(traj, epochs=3, verbose=False)

        # Compute residual
        state = np.random.randn(12).astype(np.float32)
        next_state = np.random.randn(12).astype(np.float32)

        residual = shadow.compute_residual(state, next_state)

        assert isinstance(residual, float)

    def test_calibrate_combined(self):
        from gps_imu_detector.src.pinn_integration import PINNShadowResidual

        shadow = PINNShadowResidual(state_dim=12, alpha=0.15)

        # Simulate ICI scores
        ici_scores = np.random.randn(1000) * 0.5 + 2.0

        shadow.calibrate_combined(ici_scores)

        assert shadow.ici_mean > 0
        assert shadow.ici_std > 0
        assert shadow.ici_threshold is not None

    def test_detect_returns_result(self):
        from gps_imu_detector.src.pinn_integration import (
            PINNShadowResidual,
            ShadowResidualResult,
        )

        np.random.seed(42)
        torch.manual_seed(42)

        shadow = PINNShadowResidual(state_dim=12, alpha=0.15)

        # Train briefly
        traj = np.random.randn(1, 100, 12).astype(np.float32) * 0.1
        shadow.fit_pinn(traj, epochs=3, verbose=False)
        shadow.calibrate_combined(np.random.randn(100) * 0.5)

        # Detect
        result = shadow.detect(
            ici_score=0.5,
            state=np.random.randn(12).astype(np.float32),
            next_state=np.random.randn(12).astype(np.float32),
        )

        assert isinstance(result, ShadowResidualResult)
        assert hasattr(result, 'combined_score')
        assert hasattr(result, 'pinn_residual')

    def test_combined_score_formula(self):
        """Test that combined = ici + alpha * pinn."""
        from gps_imu_detector.src.pinn_integration import PINNShadowResidual

        np.random.seed(42)
        torch.manual_seed(42)

        alpha = 0.2
        shadow = PINNShadowResidual(state_dim=12, alpha=alpha)

        # Train briefly
        traj = np.random.randn(1, 100, 12).astype(np.float32) * 0.1
        shadow.fit_pinn(traj, epochs=3, verbose=False)

        # Set known calibration values
        shadow.ici_mean = 0.0
        shadow.ici_std = 1.0
        shadow.combined_threshold = 2.0

        # Detect
        state = np.random.randn(12).astype(np.float32)
        next_state = np.random.randn(12).astype(np.float32)

        result = shadow.detect(
            ici_score=1.0,  # z_ici = 1.0
            state=state,
            next_state=next_state,
        )

        # combined = z_ici + alpha * z_pinn
        expected = 1.0 + alpha * result.pinn_residual
        assert abs(result.combined_score - expected) < 0.01

    def test_score_trajectory(self):
        from gps_imu_detector.src.pinn_integration import PINNShadowResidual

        np.random.seed(42)
        torch.manual_seed(42)

        shadow = PINNShadowResidual(state_dim=12, alpha=0.15)

        # Train
        traj = np.random.randn(1, 200, 12).astype(np.float32) * 0.1
        shadow.fit_pinn(traj, epochs=3, verbose=False)
        shadow.calibrate_combined(np.random.randn(100) * 0.5)

        # Score trajectory
        test_traj = np.random.randn(100, 12).astype(np.float32) * 0.1
        ici_scores = np.random.randn(99) * 0.5

        combined = shadow.score_trajectory(test_traj, ici_scores)

        assert len(combined) == 99

    def test_alpha_controls_pinn_weight(self):
        """Higher alpha should give PINN more weight."""
        from gps_imu_detector.src.pinn_integration import PINNShadowResidual

        np.random.seed(42)
        torch.manual_seed(42)

        # Train two systems with different alpha
        traj = np.random.randn(1, 100, 12).astype(np.float32) * 0.1

        shadow_low = PINNShadowResidual(state_dim=12, alpha=0.1)
        shadow_high = PINNShadowResidual(state_dim=12, alpha=0.3)

        shadow_low.fit_pinn(traj, epochs=3, verbose=False)
        shadow_high.fit_pinn(traj, epochs=3, verbose=False)

        # Same calibration
        ici_scores = np.random.randn(100) * 0.5
        shadow_low.calibrate_combined(ici_scores)
        shadow_high.calibrate_combined(ici_scores)

        # Set same stats for fair comparison
        shadow_low.ici_mean = 0.0
        shadow_low.ici_std = 1.0
        shadow_high.ici_mean = 0.0
        shadow_high.ici_std = 1.0

        state = np.random.randn(12).astype(np.float32)
        next_state = np.random.randn(12).astype(np.float32)

        result_low = shadow_low.detect(1.0, state, next_state)
        result_high = shadow_high.detect(1.0, state, next_state)

        # Both start with z_ici = 1.0, but high alpha adds more PINN weight
        # If pinn_residual is same, high alpha combined should differ more from ICI
        diff_low = abs(result_low.combined_score - 1.0)
        diff_high = abs(result_high.combined_score - 1.0)

        # High alpha should have bigger deviation from ICI alone
        assert diff_high > diff_low * 0.9  # Allow small tolerance

    def test_metrics_tracking(self):
        from gps_imu_detector.src.pinn_integration import PINNShadowResidual

        np.random.seed(42)
        torch.manual_seed(42)

        shadow = PINNShadowResidual(state_dim=12, alpha=0.15)

        traj = np.random.randn(1, 100, 12).astype(np.float32) * 0.1
        shadow.fit_pinn(traj, epochs=3, verbose=False)
        shadow.calibrate_combined(np.random.randn(100) * 0.5)

        # Run some detections
        for _ in range(10):
            shadow.detect(
                np.random.randn(),
                np.random.randn(12).astype(np.float32),
                np.random.randn(12).astype(np.float32),
            )

        metrics = shadow.get_metrics()

        assert metrics['total_samples'] == 10
        assert 'alpha' in metrics

    def test_reset(self):
        from gps_imu_detector.src.pinn_integration import PINNShadowResidual

        shadow = PINNShadowResidual(state_dim=12)

        shadow.total_count = 100
        shadow.combined_detections = 50

        shadow.reset()

        assert shadow.total_count == 0
        assert shadow.combined_detections == 0


class TestEvaluation:
    """Tests for evaluation function."""

    def test_evaluate_pinn_shadow(self):
        from gps_imu_detector.src.pinn_integration import evaluate_pinn_shadow

        np.random.seed(42)
        torch.manual_seed(42)

        # Generate synthetic data
        state_dim = 12
        T = 100

        nominal = np.random.randn(2, T, state_dim).astype(np.float32) * 0.1
        attack = np.random.randn(2, T, state_dim).astype(np.float32) * 0.1 + 0.5

        # Generate ICI scores (attack has higher)
        ici_nominal = np.random.randn(2 * (T-1)).astype(np.float32) * 0.1
        ici_attack = np.random.randn(2 * (T-1)).astype(np.float32) * 0.1 + 1.0

        results = evaluate_pinn_shadow(
            nominal, attack, ici_nominal, ici_attack,
            alpha=0.15,
        )

        assert 'ici_auroc' in results
        assert 'combined_auroc' in results
        assert 'improvement_auroc' in results
        assert 0 <= results['ici_auroc'] <= 1
        assert 0 <= results['combined_auroc'] <= 1


# =============================================================================
# Option 2: Envelope Learning Tests
# =============================================================================

class TestPINNEnvelopeLearner:
    """Tests for PINN envelope learner."""

    def test_initialization(self):
        from gps_imu_detector.src.pinn_integration import PINNEnvelopeLearner

        learner = PINNEnvelopeLearner(state_dim=12)
        assert learner.state_dim == 12
        assert len(learner.envelopes) == 0

    def test_classify_regime_hover(self):
        from gps_imu_detector.src.pinn_integration import PINNEnvelopeLearner, ControlRegime

        learner = PINNEnvelopeLearner()

        velocity = np.array([0.1, 0.1, 0.0])  # Low velocity
        acceleration = np.array([0.0, 0.0, 0.0])

        regime = learner.classify_regime(velocity, acceleration)
        assert regime == ControlRegime.HOVER

    def test_classify_regime_cruise(self):
        from gps_imu_detector.src.pinn_integration import PINNEnvelopeLearner, ControlRegime

        learner = PINNEnvelopeLearner()

        velocity = np.array([2.0, 1.0, 0.0])  # Moderate velocity
        acceleration = np.array([0.5, 0.0, 0.0])  # Low acceleration

        regime = learner.classify_regime(velocity, acceleration)
        assert regime == ControlRegime.CRUISE

    def test_classify_regime_aggressive(self):
        from gps_imu_detector.src.pinn_integration import PINNEnvelopeLearner, ControlRegime

        learner = PINNEnvelopeLearner()

        velocity = np.array([3.0, 2.0, 1.0])
        acceleration = np.array([10.0, 5.0, 3.0])  # High acceleration

        regime = learner.classify_regime(velocity, acceleration)
        assert regime == ControlRegime.AGGRESSIVE

    def test_fit_creates_envelopes(self):
        from gps_imu_detector.src.pinn_integration import PINNEnvelopeLearner

        np.random.seed(42)
        torch.manual_seed(42)

        learner = PINNEnvelopeLearner(state_dim=12)

        # Generate trajectory with varying regimes
        traj = np.random.randn(1, 200, 12).astype(np.float32) * 0.1
        # Increase velocity for some samples
        traj[0, 50:100, 3:6] = np.random.randn(50, 3) * 2.0

        learner.fit(traj, epochs=3, verbose=False)

        assert len(learner.envelopes) > 0

    def test_detect_returns_result(self):
        from gps_imu_detector.src.pinn_integration import (
            PINNEnvelopeLearner,
            EnvelopeResult,
        )

        np.random.seed(42)
        torch.manual_seed(42)

        learner = PINNEnvelopeLearner(state_dim=12)
        traj = np.random.randn(1, 100, 12).astype(np.float32) * 0.1
        learner.fit(traj, epochs=3, verbose=False)

        result = learner.detect(
            np.random.randn(12).astype(np.float32),
            np.random.randn(12).astype(np.float32),
        )

        assert isinstance(result, EnvelopeResult)
        assert hasattr(result, 'regime')
        assert hasattr(result, 'envelope_violation')

    def test_score_trajectory(self):
        from gps_imu_detector.src.pinn_integration import PINNEnvelopeLearner

        np.random.seed(42)
        torch.manual_seed(42)

        learner = PINNEnvelopeLearner(state_dim=12)
        traj = np.random.randn(1, 100, 12).astype(np.float32) * 0.1
        learner.fit(traj, epochs=3, verbose=False)

        test_traj = np.random.randn(50, 12).astype(np.float32) * 0.1
        residuals, violations, regimes = learner.score_trajectory(test_traj)

        assert len(residuals) == 49
        assert len(violations) == 49
        assert len(regimes) == 49

    def test_envelope_threshold_reasonable(self):
        """Envelope p99 threshold should be reasonable."""
        from gps_imu_detector.src.pinn_integration import PINNEnvelopeLearner

        np.random.seed(42)
        torch.manual_seed(42)

        learner = PINNEnvelopeLearner(state_dim=12)
        traj = np.random.randn(1, 500, 12).astype(np.float32) * 0.1
        learner.fit(traj, epochs=5, verbose=False)

        # Check that at least one envelope has reasonable threshold
        for regime, env in learner.envelopes.items():
            if env.residual_p99 > 0:
                assert env.residual_p99 > env.residual_mean


class TestEvaluateEnvelope:
    """Tests for envelope evaluation."""

    def test_evaluate_pinn_envelope(self):
        from gps_imu_detector.src.pinn_integration import evaluate_pinn_envelope

        np.random.seed(42)
        torch.manual_seed(42)

        state_dim = 12
        T = 100

        nominal = np.random.randn(2, T, state_dim).astype(np.float32) * 0.1
        attack = np.random.randn(2, T, state_dim).astype(np.float32) * 0.1 + 0.5

        results = evaluate_pinn_envelope(nominal, attack)

        assert 'auroc' in results
        assert 'recall_1pct' in results
        assert 'violation_separation' in results


# =============================================================================
# Option 3: Probing Response Tests
# =============================================================================

class TestPINNProbingPredictor:
    """Tests for PINN probing response predictor."""

    def test_initialization(self):
        from gps_imu_detector.src.pinn_integration import PINNProbingPredictor

        predictor = PINNProbingPredictor(state_dim=12, control_dim=4)
        assert predictor.state_dim == 12
        assert predictor.control_dim == 4

    def test_fit(self):
        from gps_imu_detector.src.pinn_integration import PINNProbingPredictor

        np.random.seed(42)
        torch.manual_seed(42)

        predictor = PINNProbingPredictor(state_dim=12, control_dim=4)
        traj = np.random.randn(1, 100, 12).astype(np.float32) * 0.1

        history = predictor.fit(traj, epochs=3, verbose=False)

        assert 'loss' in history
        assert len(history['loss']) == 3

    def test_predict_response(self):
        from gps_imu_detector.src.pinn_integration import PINNProbingPredictor

        np.random.seed(42)
        torch.manual_seed(42)

        predictor = PINNProbingPredictor(state_dim=12, control_dim=4)
        traj = np.random.randn(1, 100, 12).astype(np.float32) * 0.1
        predictor.fit(traj, epochs=3, verbose=False)

        state = np.random.randn(12).astype(np.float32)
        excitation = np.random.randn(4).astype(np.float32) * 0.01

        response = predictor.predict_response(state, excitation)

        assert response.shape == (6,)

    def test_detect_returns_result(self):
        from gps_imu_detector.src.pinn_integration import (
            PINNProbingPredictor,
            ProbingPredictionResult,
        )

        np.random.seed(42)
        torch.manual_seed(42)

        predictor = PINNProbingPredictor(state_dim=12, control_dim=4)
        traj = np.random.randn(1, 100, 12).astype(np.float32) * 0.1
        predictor.fit(traj, epochs=3, verbose=False)

        result = predictor.detect(
            np.random.randn(12).astype(np.float32),
            np.random.randn(12).astype(np.float32),
            np.random.randn(4).astype(np.float32) * 0.01,
        )

        assert isinstance(result, ProbingPredictionResult)
        assert hasattr(result, 'predicted_response')
        assert hasattr(result, 'actual_response')

    def test_metrics_tracking(self):
        from gps_imu_detector.src.pinn_integration import PINNProbingPredictor

        np.random.seed(42)
        torch.manual_seed(42)

        predictor = PINNProbingPredictor(state_dim=12, control_dim=4)
        traj = np.random.randn(1, 100, 12).astype(np.float32) * 0.1
        predictor.fit(traj, epochs=3, verbose=False)

        for _ in range(5):
            predictor.detect(
                np.random.randn(12).astype(np.float32),
                np.random.randn(12).astype(np.float32),
                np.random.randn(4).astype(np.float32) * 0.01,
            )

        metrics = predictor.get_metrics()
        assert metrics['total_samples'] == 5


class TestPINNResponsePredictor:
    """Tests for PINN response predictor network."""

    def test_forward(self):
        from gps_imu_detector.src.pinn_integration import PINNResponsePredictor

        net = PINNResponsePredictor(state_dim=12, control_dim=4)

        state = torch.randn(10, 12)
        control = torch.randn(10, 4)

        output = net(state, control)

        assert output.shape == (10, 6)


class TestEvaluateProbing:
    """Tests for probing evaluation."""

    def test_evaluate_pinn_probing(self):
        from gps_imu_detector.src.pinn_integration import evaluate_pinn_probing

        np.random.seed(42)
        torch.manual_seed(42)

        state_dim = 12
        T = 50

        nominal = np.random.randn(2, T, state_dim).astype(np.float32) * 0.1
        attack = np.random.randn(2, T, state_dim).astype(np.float32) * 0.1 + 0.5

        results = evaluate_pinn_probing(nominal, attack)

        assert 'auroc' in results
        assert 'recall_1pct' in results


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
