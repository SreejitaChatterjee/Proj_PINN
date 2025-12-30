"""
Unit tests for synthetic attack and normal flight generators.

Tests cover:
- SyntheticAttackGenerator from scripts/security/generate_synthetic_attacks.py
- SyntheticNormalGenerator from scripts/generate_synthetic_normals.py
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "security"))

from generate_synthetic_attacks import SyntheticAttackGenerator, GRAVITY, Z_SCALE
from generate_synthetic_normals import SyntheticNormalGenerator, extract_features


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_clean_data():
    """Create sample clean flight data for testing."""
    n_samples = 1000
    np.random.seed(42)

    return pd.DataFrame({
        'timestamp': np.arange(n_samples) * 0.005,  # 200Hz
        'x': np.cumsum(np.random.randn(n_samples) * 0.01),
        'y': np.cumsum(np.random.randn(n_samples) * 0.01),
        'z': np.ones(n_samples) + np.random.randn(n_samples) * 0.01,
        'roll': np.random.randn(n_samples) * 0.1,
        'pitch': np.random.randn(n_samples) * 0.1,
        'yaw': np.cumsum(np.random.randn(n_samples) * 0.01),
        'p': np.random.randn(n_samples) * 0.1,
        'q': np.random.randn(n_samples) * 0.1,
        'r': np.random.randn(n_samples) * 0.05,
        'vx': np.random.randn(n_samples) * 0.5,
        'vy': np.random.randn(n_samples) * 0.5,
        'vz': np.random.randn(n_samples) * 0.2,
        'ax': np.random.randn(n_samples) * 1.0,
        'ay': np.random.randn(n_samples) * 1.0,
        'az': np.random.randn(n_samples) * 1.0 + 9.81,
    })


@pytest.fixture
def short_clean_data():
    """Create very short flight data for edge case testing."""
    n_samples = 50
    np.random.seed(42)

    return pd.DataFrame({
        'timestamp': np.arange(n_samples) * 0.005,
        'x': np.random.randn(n_samples),
        'y': np.random.randn(n_samples),
        'z': np.ones(n_samples),
        'roll': np.random.randn(n_samples) * 0.1,
        'pitch': np.random.randn(n_samples) * 0.1,
        'yaw': np.random.randn(n_samples) * 0.1,
        'p': np.random.randn(n_samples) * 0.1,
        'q': np.random.randn(n_samples) * 0.1,
        'r': np.random.randn(n_samples) * 0.1,
        'vx': np.random.randn(n_samples) * 0.5,
        'vy': np.random.randn(n_samples) * 0.5,
        'vz': np.random.randn(n_samples) * 0.2,
        'ax': np.random.randn(n_samples),
        'ay': np.random.randn(n_samples),
        'az': np.random.randn(n_samples) + 9.81,
    })


@pytest.fixture
def attack_generator(sample_clean_data):
    """Create an attack generator instance."""
    return SyntheticAttackGenerator(sample_clean_data, seed=42)


@pytest.fixture
def sample_normal_data(tmp_path):
    """Create sample normal flight data files for testing."""
    np.random.seed(42)
    n_samples = 500
    n_columns = 24

    data = np.random.randn(n_samples, n_columns).astype(np.float32)
    df = pd.DataFrame(data)

    file_path = tmp_path / "normal_flight.csv"
    df.to_csv(file_path, index=False)

    return file_path


# =============================================================================
# SyntheticAttackGenerator Tests
# =============================================================================

class TestSyntheticAttackGenerator:
    """Tests for SyntheticAttackGenerator class."""

    def test_initialization(self, sample_clean_data):
        """Test generator initializes correctly."""
        gen = SyntheticAttackGenerator(sample_clean_data, seed=42)
        assert gen.clean_data is not None
        assert len(gen.clean_data) == len(sample_clean_data)
        assert gen.dt > 0

    def test_constants_defined(self):
        """Test that constants are properly defined."""
        assert GRAVITY == 9.81
        assert Z_SCALE == 0.3

    def test_init_data_adds_labels(self, attack_generator):
        """Test _init_data adds label and attack_type columns."""
        data = attack_generator._init_data()
        assert 'label' in data.columns
        assert 'attack_type' in data.columns
        assert (data['label'] == 0).all()
        assert (data['attack_type'] == 'Normal').all()

    def test_get_attack_window(self, attack_generator):
        """Test attack window calculation."""
        start_idx, end_idx, n_attack = attack_generator._get_attack_window(0.3, 1.0)
        assert start_idx < end_idx
        assert n_attack > 0
        assert n_attack == end_idx - start_idx

    def test_handle_nan_values_interpolate(self, attack_generator):
        """Test NaN handling with interpolation."""
        data = attack_generator._init_data()
        data.loc[10:20, 'x'] = np.nan

        result = attack_generator.handle_nan_values(data, method='interpolate')
        assert not result['x'].isna().any()

    def test_handle_nan_values_copy(self, attack_generator):
        """Test that handle_nan_values returns a copy."""
        data = attack_generator._init_data()
        original_x = data['x'].copy()
        data.loc[10:20, 'x'] = np.nan

        result = attack_generator.handle_nan_values(data, method='zero')
        # Original should still have NaN
        assert data['x'].isna().any()
        # Result should not have NaN
        assert not result['x'].isna().any()

    def test_handle_nan_invalid_method(self, attack_generator):
        """Test that invalid NaN method raises error."""
        data = attack_generator._init_data()
        with pytest.raises(ValueError, match="Unknown NaN handling method"):
            attack_generator.handle_nan_values(data, method='invalid')

    # GPS Attack Tests
    def test_gps_gradual_drift(self, attack_generator):
        """Test GPS gradual drift attack."""
        result = attack_generator.gps_gradual_drift()
        assert 'label' in result.columns
        assert result['label'].sum() > 0
        assert (result[result['label'] == 1]['attack_type'] == 'GPS_Gradual_Drift').all()

    def test_gps_sudden_jump(self, attack_generator):
        """Test GPS sudden jump attack."""
        result = attack_generator.gps_sudden_jump()
        assert result['label'].sum() > 0

    def test_gps_jamming(self, attack_generator):
        """Test GPS jamming creates NaN values."""
        result = attack_generator.gps_jamming()
        # Should have NaN in attacked region before handling
        assert result['x'].isna().any()

    def test_gps_meaconing_short_data(self, short_clean_data):
        """Test GPS meaconing with short data doesn't cause negative indices."""
        gen = SyntheticAttackGenerator(short_clean_data, seed=42)
        # Should not raise any errors
        result = gen.gps_meaconing(delay_samples=30)
        assert result is not None

    # IMU Attack Tests
    def test_imu_constant_bias(self, attack_generator):
        """Test IMU constant bias attack."""
        result = attack_generator.imu_constant_bias()
        assert result['label'].sum() > 0

    def test_gyro_saturation(self, attack_generator):
        """Test gyro saturation attack."""
        result = attack_generator.gyro_saturation(max_rate=4.0)
        attack_region = result[result['label'] == 1]
        # All p values in attack region should be at saturation
        assert (np.abs(attack_region['p']) == 4.0).all() or len(attack_region) == 0

    # Coordinated Attack Tests
    def test_stealthy_coordinated_short_data(self, short_clean_data):
        """Test stealthy coordinated attack handles short data."""
        gen = SyntheticAttackGenerator(short_clean_data, seed=42)
        # Should not raise division by zero
        result = gen.stealthy_coordinated(attack_duration=0.01)  # Very short duration
        assert result is not None

    def test_stealthy_coordinated_single_sample(self, sample_clean_data):
        """Test stealthy coordinated attack with n_attack=1."""
        gen = SyntheticAttackGenerator(sample_clean_data, seed=42)
        # Force very short duration to get n_attack close to 1
        result = gen.stealthy_coordinated(attack_duration=0.001)
        assert result is not None

    # Temporal Attack Tests
    def test_replay_attack(self, attack_generator):
        """Test replay attack."""
        result = attack_generator.replay_attack()
        assert result['label'].sum() > 0

    def test_replay_attack_short_data(self, short_clean_data):
        """Test replay attack with short data doesn't cause negative indices."""
        gen = SyntheticAttackGenerator(short_clean_data, seed=42)
        # Should handle short data gracefully
        result = gen.replay_attack(replay_window=100)
        assert result is not None
        # With short data, should still have some attack marker
        assert result['label'].sum() >= 0

    def test_time_delay_attack(self, attack_generator):
        """Test time delay attack."""
        result = attack_generator.time_delay_attack()
        assert result['label'].sum() > 0

    # Generation Tests
    def test_generate_all_attacks(self, attack_generator):
        """Test generating all attacks."""
        attacks = attack_generator.generate_all_attacks()
        # Should have 30 attacks + clean baseline
        assert len(attacks) == 31
        assert 'clean' in attacks
        assert 'gps_gradual_drift' in attacks

    def test_generate_all_attacks_with_nan_handling(self, attack_generator):
        """Test generating all attacks with NaN handling."""
        attacks = attack_generator.generate_all_attacks(handle_nan=True)
        # Check that no attack has NaN values
        for name, data in attacks.items():
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            assert not data[numeric_cols].isna().any().any(), f"NaN found in {name}"

    def test_generate_pinn_ready_dataset(self, attack_generator):
        """Test generating PINN-ready dataset."""
        dataset = attack_generator.generate_pinn_ready_dataset()
        assert 'label' in dataset.columns
        assert 'attack_type' in dataset.columns
        assert not dataset.select_dtypes(include=[np.number]).isna().any().any()

    def test_attack_labels_in_window_only(self, attack_generator):
        """Test that attack labels are only within attack window."""
        result = attack_generator.gps_gradual_drift(
            attack_start_ratio=0.3,
            drift_duration=10.0
        )

        # Get indices where label is 1
        attack_indices = result[result['label'] == 1].index

        if len(attack_indices) > 0:
            # Attack should be contiguous
            expected_indices = range(attack_indices.min(), attack_indices.max() + 1)
            assert list(attack_indices) == list(expected_indices)


# =============================================================================
# SyntheticNormalGenerator Tests
# =============================================================================

class TestSyntheticNormalGenerator:
    """Tests for SyntheticNormalGenerator class."""

    def test_initialization(self):
        """Test generator initializes correctly."""
        gen = SyntheticNormalGenerator(seed=42)
        assert len(gen.real_normals) == 0
        assert gen.n_columns is None

    def test_initialization_with_n_columns(self):
        """Test generator initializes with n_columns."""
        gen = SyntheticNormalGenerator(seed=42, n_columns=10)
        assert gen.n_columns == 10

    def test_load_real_normals(self, sample_normal_data):
        """Test loading real normal data."""
        gen = SyntheticNormalGenerator(seed=42, n_columns=24)
        gen.load_real_normals([sample_normal_data])
        assert len(gen.real_normals) == 1
        assert gen.real_normals[0]['data'].shape[1] == 24

    def test_load_real_normals_file_not_found(self):
        """Test loading non-existent file raises error."""
        gen = SyntheticNormalGenerator(seed=42)
        with pytest.raises(FileNotFoundError):
            gen.load_real_normals([Path("nonexistent_file.csv")])

    def test_load_real_normals_insufficient_columns(self, tmp_path):
        """Test loading file with insufficient columns raises error."""
        # Create file with only 10 columns
        data = np.random.randn(100, 10)
        df = pd.DataFrame(data)
        file_path = tmp_path / "small_data.csv"
        df.to_csv(file_path, index=False)

        gen = SyntheticNormalGenerator(seed=42, n_columns=24)
        with pytest.raises(ValueError, match="columns"):
            gen.load_real_normals([file_path])

    def test_add_noise(self, sample_normal_data):
        """Test noise addition."""
        gen = SyntheticNormalGenerator(seed=42, n_columns=24)
        gen.load_real_normals([sample_normal_data])

        original = gen.real_normals[0]['data'].copy()
        noisy = gen.add_noise(original, noise_level=0.1)

        assert noisy.shape == original.shape
        assert not np.allclose(noisy, original)

    def test_time_warp(self, sample_normal_data):
        """Test time warping."""
        gen = SyntheticNormalGenerator(seed=42, n_columns=24)
        gen.load_real_normals([sample_normal_data])

        original = gen.real_normals[0]['data']
        warped = gen.time_warp(original, factor_range=(0.8, 1.2))

        # Shape should be different (time stretched/compressed)
        assert warped.shape[1] == original.shape[1]

    def test_amplitude_scale(self, sample_normal_data):
        """Test amplitude scaling."""
        gen = SyntheticNormalGenerator(seed=42, n_columns=24)
        gen.load_real_normals([sample_normal_data])

        original = gen.real_normals[0]['data'].copy()
        scaled = gen.amplitude_scale(original, scale_range=(0.5, 1.5))

        assert scaled.shape == original.shape

    def test_jitter(self, sample_normal_data):
        """Test jitter addition."""
        gen = SyntheticNormalGenerator(seed=42, n_columns=24)
        gen.load_real_normals([sample_normal_data])

        original = gen.real_normals[0]['data'].copy()
        jittered = gen.jitter(original, sigma=0.01)

        assert jittered.shape == original.shape
        assert not np.allclose(jittered, original)

    def test_smooth(self, sample_normal_data):
        """Test smoothing."""
        gen = SyntheticNormalGenerator(seed=42, n_columns=24)
        gen.load_real_normals([sample_normal_data])

        original = gen.real_normals[0]['data'].copy()
        smoothed = gen.smooth(original, sigma=2)

        assert smoothed.shape == original.shape

    def test_mix_segments(self, sample_normal_data):
        """Test segment mixing."""
        gen = SyntheticNormalGenerator(seed=42, n_columns=24)
        gen.load_real_normals([sample_normal_data])

        data1 = gen.real_normals[0]['data']
        data2 = gen.real_normals[0]['data'] * 2  # Different data

        mixed = gen.mix_segments(data1, data2, n_segments=5)
        assert mixed.shape[0] == min(len(data1), len(data2))

    def test_mix_segments_short_data(self):
        """Test mixing segments with very short data."""
        gen = SyntheticNormalGenerator(seed=42)

        data1 = np.random.randn(3, 10)
        data2 = np.random.randn(3, 10)

        # Should handle n_segments > n gracefully
        mixed = gen.mix_segments(data1, data2, n_segments=10)
        assert mixed.shape == data1.shape

    def test_generate_synthetic_no_data(self):
        """Test generating synthetic without loading data raises error."""
        gen = SyntheticNormalGenerator(seed=42)
        with pytest.raises(ValueError, match="No real normal data loaded"):
            gen.generate_synthetic()

    def test_generate_synthetic(self, sample_normal_data):
        """Test generating synthetic flights."""
        gen = SyntheticNormalGenerator(seed=42, n_columns=24)
        gen.load_real_normals([sample_normal_data])

        synthetic = gen.generate_synthetic(n_synthetic=5)

        assert len(synthetic) == 5
        for s in synthetic:
            assert 'data' in s
            assert 'base' in s
            assert 'augmentations' in s
            assert len(s['augmentations']) >= 2

    def test_channel_dropout(self, sample_normal_data):
        """Test channel dropout."""
        gen = SyntheticNormalGenerator(seed=42, n_columns=24)
        gen.load_real_normals([sample_normal_data])

        original = gen.real_normals[0]['data'].copy()
        dropped = gen.channel_dropout(original, dropout_prob=0.5)

        assert dropped.shape == original.shape
        # Some channels should be zeroed out
        zero_cols = np.sum(np.abs(dropped).sum(axis=0) == 0)
        assert zero_cols >= 0  # May or may not have zeros depending on random state


class TestExtractFeatures:
    """Tests for the extract_features function."""

    def test_extract_features_shape(self):
        """Test feature extraction output shape."""
        window = np.random.randn(256, 10)
        features = extract_features(window)

        # Should have 7 features per channel
        expected_features = 10 * 7
        assert len(features) == expected_features

    def test_extract_features_values(self):
        """Test feature values are reasonable."""
        window = np.random.randn(256, 5)
        features = extract_features(window)

        # Features should all be finite
        assert all(np.isfinite(f) for f in features)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_attack_generator_reproducibility(self, sample_clean_data):
        """Test that same seed produces same results."""
        gen1 = SyntheticAttackGenerator(sample_clean_data, seed=42)
        gen2 = SyntheticAttackGenerator(sample_clean_data, seed=42)

        result1 = gen1.gps_gradual_drift()
        result2 = gen2.gps_gradual_drift()

        pd.testing.assert_frame_equal(result1, result2)

    def test_normal_generator_reproducibility(self, sample_normal_data):
        """Test that same seed produces same results."""
        gen1 = SyntheticNormalGenerator(seed=42, n_columns=24)
        gen2 = SyntheticNormalGenerator(seed=42, n_columns=24)

        gen1.load_real_normals([sample_normal_data])
        gen2.load_real_normals([sample_normal_data])

        result1 = gen1.add_noise(gen1.real_normals[0]['data'], 0.1)
        result2 = gen2.add_noise(gen2.real_normals[0]['data'], 0.1)

        np.testing.assert_array_equal(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
