"""
Tests for Phase 3 Hardening Components

Tests:
1. Hard negative generation
2. Domain randomization
3. Attribution module
4. Transfer evaluation
5. Hardened training
"""

import numpy as np
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hard_negatives import HardNegativeGenerator, DomainRandomizer, AdversarialAttackGenerator
from attribution import AttackTypeHead, SensorAttributionHead, MultiTaskDetector, MultiTaskLoss
from transfer import TransferEvaluator, DomainAdaptation, generate_flight_regime_split


class TestHardNegativeGeneration:
    """Tests for hard negative mining."""

    def test_ar1_drift_generation(self):
        """Test AR(1) drift attack generation."""
        n = 500
        d = 15
        data = np.random.randn(n, d)

        generator = HardNegativeGenerator(seed=42)
        attacked, labels = generator.generate_ar1_drift(
            data, ar_coef=0.995, magnitude=0.5
        )

        assert attacked.shape == data.shape
        assert len(labels) == n
        assert np.all(labels == 1)  # All samples attacked
        assert not np.allclose(attacked, data)  # Data changed

    def test_coordinated_attack_generation(self):
        """Test coordinated multi-sensor attack."""
        n = 500
        d = 15
        data = np.random.randn(n, d)

        generator = HardNegativeGenerator(seed=42)
        attacked, labels = generator.generate_coordinated_attack(
            data, magnitude=0.5, consistency_factor=0.9
        )

        assert attacked.shape == data.shape
        assert np.all(labels == 1)
        # Position (0:3) should be attacked
        assert not np.allclose(attacked[:, 0:3], data[:, 0:3])

    def test_intermittent_attack_generation(self):
        """Test intermittent on/off attack."""
        n = 1000
        d = 15
        data = np.random.randn(n, d)

        generator = HardNegativeGenerator(seed=42)
        attacked, labels = generator.generate_intermittent_attack(
            data, on_probability=0.1, magnitude=1.0
        )

        # Should have mix of attacked and clean
        assert 0 < np.sum(labels) < n
        # Attacked portions should differ
        attack_mask = labels == 1
        assert not np.allclose(attacked[attack_mask], data[attack_mask])

    def test_below_threshold_ramp(self):
        """Test below-threshold ramp attack."""
        n = 500
        d = 15
        data = np.random.randn(n, d)

        generator = HardNegativeGenerator(seed=42)
        attacked, labels = generator.generate_below_threshold_ramp(
            data, threshold_estimate=0.5, safety_margin=0.8
        )

        assert attacked.shape == data.shape
        # Ramp should increase over time
        diff = attacked[:, 0:3] - data[:, 0:3]
        assert np.abs(diff[-1]).mean() > np.abs(diff[0]).mean()


class TestDomainRandomization:
    """Tests for domain randomization."""

    def test_randomize_noise(self):
        """Test noise randomization."""
        n = 100
        d = 15
        data = np.random.randn(n, d)

        randomizer = DomainRandomizer(seed=42)
        noisy = randomizer.randomize_noise(data)

        assert noisy.shape == data.shape
        assert not np.allclose(noisy, data)

    def test_randomize_sampling_jitter(self):
        """Test sampling jitter simulation."""
        n = 100
        d = 15
        # Smooth signal so interpolation makes sense
        t = np.linspace(0, 4*np.pi, n)
        data = np.column_stack([np.sin(t + i*0.1) for i in range(d)])

        randomizer = DomainRandomizer(seed=42)
        jittered = randomizer.randomize_sampling_jitter(data, jitter_std=0.001)

        assert jittered.shape == data.shape
        # Should be similar but not identical
        assert np.corrcoef(data.flatten(), jittered.flatten())[0, 1] > 0.99

    def test_augment_batch(self):
        """Test full augmentation pipeline."""
        n = 100
        d = 15
        data = np.random.randn(n, d)

        randomizer = DomainRandomizer(seed=42)
        augmented = randomizer.augment_batch(data, augment_prob=1.0)

        assert augmented.shape == data.shape


class TestAttribution:
    """Tests for attribution module."""

    def test_attack_type_head_forward(self):
        """Test attack type classification head."""
        batch_size = 8
        input_dim = 64

        head = AttackTypeHead(input_dim)
        x = torch.randn(batch_size, input_dim)
        logits = head(x)

        assert logits.shape == (batch_size, len(AttackTypeHead.ATTACK_TYPES))

    def test_attack_type_head_predict(self):
        """Test attack type prediction."""
        batch_size = 8
        input_dim = 64

        head = AttackTypeHead(input_dim)
        x = torch.randn(batch_size, input_dim)
        predicted, probs = head.predict(x)

        assert predicted.shape == (batch_size,)
        assert probs.shape == (batch_size, len(AttackTypeHead.ATTACK_TYPES))
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    def test_sensor_attribution_head(self):
        """Test sensor attribution head."""
        batch_size = 8
        input_dim = 64

        head = SensorAttributionHead(input_dim)
        x = torch.randn(batch_size, input_dim)
        attn = head(x)

        assert attn.shape == (batch_size, len(SensorAttributionHead.SENSOR_GROUPS))
        # Attention should sum to 1
        assert torch.allclose(attn.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    def test_multitask_detector_forward(self):
        """Test multi-task detector forward pass."""
        batch_size = 4
        seq_len = 20
        input_dim = 50

        model = MultiTaskDetector(input_dim=input_dim)
        x = torch.randn(batch_size, seq_len, input_dim)

        anomaly_logits, attack_logits, sensor_attn, hidden = model(x)

        assert anomaly_logits.shape == (batch_size, seq_len, 1)
        assert attack_logits.shape == (batch_size, seq_len, len(AttackTypeHead.ATTACK_TYPES))
        assert sensor_attn.shape == (batch_size, seq_len, len(SensorAttributionHead.SENSOR_GROUPS))

    def test_multitask_detector_predict(self):
        """Test multi-task detector prediction."""
        input_dim = 50
        model = MultiTaskDetector(input_dim=input_dim)
        x = torch.randn(1, 5, input_dim)

        results = model.predict(x)

        assert len(results) == 5
        for r in results:
            assert isinstance(r.is_anomaly, bool)
            assert 0 <= r.anomaly_score <= 1
            assert r.attack_type in AttackTypeHead.ATTACK_TYPES

    def test_multitask_loss(self):
        """Test multi-task loss computation."""
        batch_size = 4
        seq_len = 10
        n_classes = len(AttackTypeHead.ATTACK_TYPES)
        n_sensors = len(SensorAttributionHead.SENSOR_GROUPS)

        loss_fn = MultiTaskLoss()

        anomaly_logits = torch.randn(batch_size, seq_len, 1)
        attack_logits = torch.randn(batch_size, seq_len, n_classes)
        sensor_attn = torch.softmax(torch.randn(batch_size, seq_len, n_sensors), dim=-1)
        anomaly_labels = torch.randint(0, 2, (batch_size, seq_len))
        attack_labels = torch.randint(0, n_classes, (batch_size, seq_len))

        loss, loss_dict = loss_fn(
            anomaly_logits, attack_logits, sensor_attn,
            anomaly_labels, attack_labels
        )

        assert loss.item() >= 0
        assert 'anomaly' in loss_dict
        assert 'attack_type' in loss_dict


class TestTransfer:
    """Tests for transfer evaluation."""

    def test_compute_domain_shift(self):
        """Test MMD computation."""
        n = 200
        d = 20

        evaluator = TransferEvaluator(feature_dim=d)

        # Same distribution
        source = np.random.randn(n, d)
        target_same = np.random.randn(n, d)
        mmd_same = evaluator.compute_domain_shift(source, target_same)

        # Different distribution
        target_diff = np.random.randn(n, d) * 2 + 1
        mmd_diff = evaluator.compute_domain_shift(source, target_diff)

        # Different distribution should have higher MMD
        assert mmd_diff > mmd_same

    def test_evaluate_transfer(self):
        """Test transfer evaluation."""
        n = 500
        d = 20

        evaluator = TransferEvaluator(feature_dim=d)

        source_data = {
            'features': np.random.randn(n, d),
            'labels': np.zeros(n)
        }
        target_data = {
            'features': np.random.randn(n, d),
            'labels': np.concatenate([np.zeros(n//2), np.ones(n//2)])
        }

        def mock_scorer(features):
            return np.random.rand(len(features))

        result = evaluator.evaluate_transfer(
            mock_scorer, source_data, target_data, 'source', 'target'
        )

        assert result.source_domain == 'source'
        assert result.target_domain == 'target'
        assert 0 <= result.auroc <= 1
        assert result.feature_shift >= 0

    def test_domain_adaptation_normalization(self):
        """Test domain adaptation normalization."""
        n = 200
        d = 20

        source = np.random.randn(n, d)
        target = np.random.randn(n, d) * 2 + 1  # Different mean/var

        adapter = DomainAdaptation()
        adapter.fit_normalization(source)
        aligned = adapter.normalize_target(target)

        # Aligned should have similar mean to source
        assert np.abs(np.mean(aligned) - np.mean(source)) < np.abs(np.mean(target) - np.mean(source))

    def test_flight_regime_split(self):
        """Test flight regime splitting."""
        n = 1000
        d = 20

        data = np.random.randn(n, d)
        labels = np.zeros(n)

        regimes = generate_flight_regime_split(data, labels, n_regimes=3)

        assert len(regimes) > 0
        for features, labs in regimes:
            assert len(features) > 0
            assert len(features) == len(labs)


class TestIntegration:
    """Integration tests for hardening components."""

    def test_hard_negative_with_detector(self):
        """Test hard negative mining with real detector."""
        n = 200
        d = 50

        # Create simple detector
        from model import CNNGRUDetector

        model = CNNGRUDetector(input_dim=d)
        model.eval()

        def detector_fn(data):
            with torch.no_grad():
                x = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                output, _ = model(x)
                return torch.sigmoid(output).squeeze().cpu().numpy()

        # Generate data
        data = np.random.randn(n, d).astype(np.float32)

        # Find evasive attacks
        generator = HardNegativeGenerator(seed=42)
        evasive = generator.find_evasive_attacks(data, detector_fn, n_attempts=5)

        # Should find some evasive attacks
        assert isinstance(evasive, list)

    def test_full_augmentation_pipeline(self):
        """Test complete augmentation during training iteration."""
        n = 100
        d = 20

        data = np.random.randn(n, d)
        labels = np.zeros(n)

        # Apply all augmentations
        randomizer = DomainRandomizer(seed=42)

        augmented = randomizer.randomize_noise(data)
        augmented = randomizer.randomize_sampling_jitter(augmented)
        augmented = randomizer.randomize_motion_regime(augmented)

        assert augmented.shape == data.shape
        assert not np.allclose(augmented, data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
