"""
Tests for Phase 5 Evaluation Components

Tests:
1. Evaluation configuration
2. Metrics computation
3. Result aggregation
4. Ablation study
"""

import numpy as np
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluate import (
    EvaluationConfig, AttackResults, FoldResults, OverallResults,
    RigorousEvaluator
)
from model import CNNGRUDetector


class TestEvaluationConfig:
    """Tests for evaluation configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluationConfig()

        assert config.dt == 0.005
        assert config.epochs == 100
        assert config.batch_size == 64
        assert len(config.attack_types) > 0
        assert len(config.attack_magnitudes) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = EvaluationConfig(
            epochs=50,
            batch_size=32,
            attack_types=['bias', 'drift']
        )

        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.attack_types == ['bias', 'drift']


class TestAttackResults:
    """Tests for attack results dataclass."""

    def test_attack_results_creation(self):
        """Test creating attack results."""
        result = AttackResults(
            attack_type='bias',
            magnitude=1.0,
            n_samples=100,
            recall_at_1pct_fpr=0.85,
            recall_at_5pct_fpr=0.95,
            recall_at_10pct_fpr=0.98,
            auroc=0.92,
            auprc=0.88,
            detection_delay_mean=5.0,
            detection_delay_std=2.0
        )

        assert result.attack_type == 'bias'
        assert result.magnitude == 1.0
        assert result.recall_at_5pct_fpr == 0.95


class TestFoldResults:
    """Tests for fold results dataclass."""

    def test_fold_results_creation(self):
        """Test creating fold results."""
        attack_results = [
            AttackResults(
                attack_type='bias', magnitude=1.0, n_samples=100,
                recall_at_1pct_fpr=0.8, recall_at_5pct_fpr=0.9,
                recall_at_10pct_fpr=0.95, auroc=0.9, auprc=0.85,
                detection_delay_mean=3.0, detection_delay_std=1.0
            )
        ]

        fold = FoldResults(
            fold_idx=0,
            test_sequence='test_seq_0',
            n_train=1000,
            n_test=200,
            n_test_attacks=100,
            overall_auroc=0.92,
            overall_auprc=0.88,
            attack_results=attack_results,
            training_time_sec=30.0
        )

        assert fold.fold_idx == 0
        assert fold.overall_auroc == 0.92
        assert len(fold.attack_results) == 1


class TestOverallResults:
    """Tests for overall results dataclass."""

    def test_overall_results_creation(self):
        """Test creating overall results."""
        results = OverallResults(
            n_folds=5,
            total_train_samples=5000,
            total_test_samples=1000,
            mean_auroc=0.90,
            std_auroc=0.02,
            mean_auprc=0.85,
            std_auprc=0.03,
            attack_summary={'bias_1.0': {'mean_recall_5pct': 0.9}},
            auroc_ci_lower=0.88,
            auroc_ci_upper=0.92
        )

        assert results.n_folds == 5
        assert results.mean_auroc == 0.90
        assert 'bias_1.0' in results.attack_summary


class TestRigorousEvaluator:
    """Tests for rigorous evaluator."""

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        config = EvaluationConfig(
            output_dir='./test_results',
            epochs=10
        )
        evaluator = RigorousEvaluator(config)

        assert evaluator.device in ['cpu', 'cuda']
        assert evaluator.feature_extractor is not None
        assert evaluator.attack_catalog is not None

    def test_train_model(self):
        """Test model training."""
        config = EvaluationConfig(epochs=5)
        evaluator = RigorousEvaluator(config)

        # Create synthetic data
        n = 500
        d = 50
        features = np.random.randn(n, d).astype(np.float32)
        labels = np.zeros(n)

        model, train_time = evaluator._train_model(features, labels)

        assert isinstance(model, CNNGRUDetector)
        assert train_time > 0

    def test_predict(self):
        """Test model prediction."""
        config = EvaluationConfig(epochs=5)
        evaluator = RigorousEvaluator(config)

        # Create and train model
        n = 500
        d = 50
        features = np.random.randn(n, d).astype(np.float32)
        labels = np.zeros(n)

        model, _ = evaluator._train_model(features, labels)

        # Predict
        test_features = np.random.randn(100, d).astype(np.float32)
        scores = evaluator._predict(model, test_features)

        assert len(scores) == 100
        assert all(0 <= s <= 1 for s in scores)

    def test_compute_attack_metrics(self):
        """Test attack metrics computation."""
        config = EvaluationConfig()
        evaluator = RigorousEvaluator(config)

        # Create mock scores and labels
        n = 200
        scores = np.random.rand(n)
        labels = np.concatenate([np.zeros(100), np.ones(100)])

        # Make attacked scores higher
        scores[100:] += 0.3
        scores = np.clip(scores, 0, 1)

        result = evaluator._compute_attack_metrics(
            scores, labels, 'bias', 1.0
        )

        assert result.attack_type == 'bias'
        assert result.magnitude == 1.0
        assert 0 <= result.auroc <= 1
        assert 0 <= result.recall_at_5pct_fpr <= 1

    def test_aggregate_results(self):
        """Test results aggregation."""
        config = EvaluationConfig()
        evaluator = RigorousEvaluator(config)

        # Create mock fold results
        for i in range(3):
            attack_results = [
                AttackResults(
                    attack_type='bias', magnitude=1.0, n_samples=100,
                    recall_at_1pct_fpr=0.8 + i*0.05,
                    recall_at_5pct_fpr=0.9 + i*0.02,
                    recall_at_10pct_fpr=0.95,
                    auroc=0.85 + i*0.05,
                    auprc=0.80 + i*0.05,
                    detection_delay_mean=3.0, detection_delay_std=1.0
                )
            ]
            fold = FoldResults(
                fold_idx=i,
                test_sequence=f'seq_{i}',
                n_train=1000,
                n_test=200,
                n_test_attacks=100,
                overall_auroc=0.85 + i*0.05,
                overall_auprc=0.80 + i*0.05,
                attack_results=attack_results,
                training_time_sec=10.0
            )
            evaluator.fold_results.append(fold)

        overall = evaluator._aggregate_results()

        assert overall.n_folds == 3
        assert overall.mean_auroc > 0
        assert overall.std_auroc >= 0


class TestMockEvaluation:
    """Integration tests with mock data."""

    def test_mock_data_evaluation(self):
        """Test evaluation with mock data loader."""
        # Create mock data loader
        class MockDataLoader:
            def __init__(self):
                np.random.seed(42)
                self.sequences = {
                    f"seq_{i}": np.random.randn(100, 15).astype(np.float32)
                    for i in range(3)
                }

            def get_loso_splits(self):
                ids = list(self.sequences.keys())
                return [(ids[:i] + ids[i+1:], ids[i]) for i in range(len(ids))]

            def get_train_test_data(self, train_ids, test_id, fit_scaler=True):
                X_train = np.vstack([self.sequences[i] for i in train_ids])
                X_test = self.sequences[test_id]
                return X_train, X_test, None, None

        config = EvaluationConfig(
            epochs=5,
            attack_types=['bias'],
            attack_magnitudes=[1.0],
            n_bootstrap=10,
            output_dir='./test_eval_results'
        )
        evaluator = RigorousEvaluator(config)
        mock_loader = MockDataLoader()

        # Run just one fold
        train_ids, test_id = mock_loader.get_loso_splits()[0]
        fold_result = evaluator._run_fold(mock_loader, train_ids, test_id, 0)

        assert fold_result is not None
        assert fold_result.fold_idx == 0
        assert fold_result.overall_auroc >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
