"""
Smoke Tests for GPS-IMU Anomaly Detector Pipeline

Basic tests to verify core functionality.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from feature_extractor import StreamingFeatureExtractor, BatchFeatureExtractor
from physics_residuals import AnalyticPhysicsChecker, HybridPhysicsChecker
from ekf import SimpleEKF, EKFAnomalyDetector
from model import CNNGRUDetector, StreamingDetector
from hybrid_scorer import HybridScorer, TemporalConsistencyScorer
from data_loader import AttackCatalog


class TestFeatureExtractor:
    """Tests for feature extraction."""

    def test_streaming_extractor_output_shape(self):
        """Test streaming extractor produces correct output shape."""
        n_features = 15
        windows = [5, 10, 25]
        extractor = StreamingFeatureExtractor(n_features, windows)

        # Process 100 samples
        for i in range(100):
            x = np.random.randn(n_features)
            feat = extractor.update(x)

        # After warmup, should get features
        assert feat is not None
        assert len(feat) == extractor.n_output_features

    def test_batch_extractor_consistency(self):
        """Test batch extractor produces valid features."""
        n_samples = 200
        n_features = 15
        windows = [5, 10, 25]

        data = np.random.randn(n_samples, n_features)
        extractor = BatchFeatureExtractor(windows)
        features = extractor.extract(data)

        assert features.shape[0] == n_samples - max(windows) + 1
        assert not np.any(np.isnan(features))

    def test_streaming_vs_batch_approximate(self):
        """Test streaming and batch produce similar results."""
        n_samples = 100
        n_features = 15
        windows = [5, 10]

        data = np.random.randn(n_samples, n_features)

        # Streaming
        streaming_ext = StreamingFeatureExtractor(n_features, windows, include_raw=False)
        streaming_feats = streaming_ext.process_batch(data)

        # Batch
        batch_ext = BatchFeatureExtractor(windows, include_cumsum=False)
        batch_feats = batch_ext.extract(data)

        # Should have same number of outputs
        assert streaming_feats.shape[0] == batch_feats.shape[0]


class TestPhysicsResiduals:
    """Tests for physics residual computation."""

    def test_clean_data_low_residuals(self):
        """Clean data should have low physics residuals."""
        n = 100
        dt = 0.005

        # Generate consistent data
        t = np.arange(n) * dt
        position = np.column_stack([np.sin(t), np.cos(t), np.zeros(n)])
        velocity = np.column_stack([np.cos(t), -np.sin(t), np.zeros(n)])
        acceleration = np.column_stack([-np.sin(t), -np.cos(t), np.zeros(n)])
        attitude = np.zeros((n, 3))
        angular_rates = np.zeros((n, 3))

        checker = AnalyticPhysicsChecker(dt=dt)
        residuals = checker.compute_residuals(
            position, velocity, acceleration, attitude, angular_rates
        )

        # Residuals should be small for consistent data
        assert np.mean(np.abs(residuals.pva_residual)) < 1.0

    def test_attacked_data_higher_residuals(self):
        """Attacked data should have higher physics residuals."""
        n = 100
        dt = 0.005

        t = np.arange(n) * dt
        position = np.column_stack([np.sin(t), np.cos(t), np.zeros(n)])
        velocity = np.column_stack([np.cos(t), -np.sin(t), np.zeros(n)])
        acceleration = np.column_stack([-np.sin(t), -np.cos(t), np.zeros(n)])
        attitude = np.zeros((n, 3))
        angular_rates = np.zeros((n, 3))

        # Add bias to position
        attacked_position = position.copy()
        attacked_position[50:, 0] += 0.5

        checker = AnalyticPhysicsChecker(dt=dt)

        clean_residuals = checker.compute_residuals(
            position, velocity, acceleration, attitude, angular_rates
        )
        attack_residuals = checker.compute_residuals(
            attacked_position, velocity, acceleration, attitude, angular_rates
        )

        # Attack should increase residuals
        assert np.mean(np.abs(attack_residuals.pva_residual[50:])) > \
               np.mean(np.abs(clean_residuals.pva_residual[50:]))


class TestEKF:
    """Tests for EKF and NIS computation."""

    def test_ekf_predict_update(self):
        """Test EKF predict and update cycle."""
        ekf = SimpleEKF(dt=0.005)

        # Single step
        imu_gyro = np.array([0.1, 0.0, 0.0])
        imu_accel = np.array([0.0, 0.0, 9.81])
        gps_pos = np.array([0.0, 0.0, 0.0])
        gps_vel = np.array([0.0, 0.0, 0.0])

        ekf.predict(imu_gyro, imu_accel)
        nis, is_consistent = ekf.update_gps(gps_pos, gps_vel)

        assert isinstance(nis, float)
        assert nis >= 0

    def test_ekf_anomaly_detector(self):
        """Test EKF anomaly detector on sequence."""
        n = 100
        dt = 0.005

        # Generate data
        t = np.arange(n) * dt
        data = np.zeros((n, 15))
        data[:, 14] = 9.81  # Gravity in z acceleration

        detector = EKFAnomalyDetector(dt=dt)
        nis_values, nis_avg, anomaly_flags = detector.process_sequence(data)

        assert len(nis_values) == n
        assert len(anomaly_flags) == n


class TestModel:
    """Tests for ML detector model."""

    def test_model_forward_pass(self):
        """Test model forward pass."""
        import torch

        input_dim = 100
        batch_size = 16
        seq_len = 50

        model = CNNGRUDetector(input_dim=input_dim)
        x = torch.randn(batch_size, seq_len, input_dim)

        output, hidden = model(x)

        assert output.shape == (batch_size, seq_len, 1)
        assert hidden.shape[0] == model.gru.num_layers

    def test_streaming_detector(self):
        """Test streaming detector."""
        input_dim = 100
        model = CNNGRUDetector(input_dim=input_dim)
        detector = StreamingDetector(model)

        # Process samples
        scores = []
        for _ in range(50):
            x = np.random.randn(input_dim).astype(np.float32)
            score = detector.predict_step(x)
            scores.append(score)

        assert len(scores) == 50
        assert all(0 <= s <= 1 for s in scores)


class TestHybridScorer:
    """Tests for hybrid scoring."""

    def test_scorer_fit_score(self):
        """Test scorer fitting and scoring."""
        n = 100

        # Generate scores
        physics = np.abs(np.random.randn(n))
        ekf = np.abs(np.random.randn(n))
        ml = np.random.rand(n)
        temporal = np.abs(np.random.randn(n))

        scorer = HybridScorer()
        scorer.fit(physics, ekf, ml, temporal)

        # Score single sample
        result = scorer.score(0.5, 1.0, 0.3, 0.2)

        assert 0 <= result.total_score <= 1
        assert isinstance(result.is_anomaly, (bool, np.bool_))

    def test_temporal_consistency_scorer(self):
        """Test temporal consistency scoring."""
        n = 100
        d = 15

        features = np.random.randn(n, d)
        scorer = TemporalConsistencyScorer(window_size=10)
        scores = scorer.score_sequence(features)

        assert len(scores) == n
        assert scores[0] == 0.0  # First sample


class TestAttackGeneration:
    """Tests for attack generation."""

    def test_attack_catalog(self):
        """Test attack catalog generation."""
        catalog = AttackCatalog(seed=42)
        attacks = catalog.get_attack_catalog()

        assert len(attacks) > 0
        assert all('attack_type' in a for a in attacks)
        assert all('magnitude' in a for a in attacks)

    def test_bias_attack(self):
        """Test bias attack generation."""
        n = 100
        d = 15

        data = np.random.randn(n, d)
        catalog = AttackCatalog(seed=42)

        attacked, labels = catalog.generate_attack(
            data, 'bias', magnitude=1.0, sensor_group='position'
        )

        assert attacked.shape == data.shape
        assert np.all(labels == 1)  # All attacked
        assert not np.allclose(attacked[:, 0:3], data[:, 0:3])  # Position changed

    def test_drift_attack(self):
        """Test drift attack generation."""
        n = 100
        d = 15

        # Use non-zero data so std is non-zero
        data = np.random.randn(n, d)
        catalog = AttackCatalog(seed=42)

        attacked, labels = catalog.generate_attack(
            data, 'drift', magnitude=1.0, sensor_group='position'
        )

        # Drift should be non-zero
        diff = attacked[:, 0:3] - data[:, 0:3]
        assert np.abs(diff).sum() > 0  # Some drift applied


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_synthetic(self):
        """Test full pipeline on synthetic data."""
        # Generate synthetic sequence
        n = 500
        dt = 0.005
        n_features = 15

        data = np.random.randn(n, n_features) * 0.1
        data[:, 12:15] += np.array([0, 0, 9.81])  # Gravity

        # Extract features
        extractor = BatchFeatureExtractor(windows=[5, 10, 25])
        features = extractor.extract(data)

        # Create simple model
        import torch
        model = CNNGRUDetector(input_dim=features.shape[1])

        # Forward pass
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output, _ = model(x)
            scores = torch.sigmoid(output).squeeze().numpy()

        assert len(scores) == len(features)
        assert all(0 <= s <= 1 for s in scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
