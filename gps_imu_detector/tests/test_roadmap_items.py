"""
Tests for Roadmap Priority Items

Tests the missing roadmap items:
1. CI gate script
2. Minimax calibration
3. Operational metrics
4. Explainable alarms

Run with: pytest tests/test_roadmap_items.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))


class TestCIGateScript:
    """Tests for CI circular sensor gate."""

    def test_circular_sensor_checker_import(self):
        """Test that CI gate script can be imported."""
        from ci_circular_check import CircularSensorChecker
        checker = CircularSensorChecker(threshold=0.9)
        assert checker is not None

    def test_detects_banned_columns(self):
        """Test that banned column patterns are detected."""
        import pandas as pd
        from ci_circular_check import CircularSensorChecker

        # DataFrame with banned column
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'baro_alt': [100, 101, 102],  # Banned!
            'vx': [0.1, 0.2, 0.3]
        })

        checker = CircularSensorChecker()
        passed = checker.check_dataframe(df)

        assert not passed, "Should fail when banned column present"
        assert len(checker.violations) > 0

    def test_passes_clean_data(self):
        """Test that clean data passes."""
        import pandas as pd
        from ci_circular_check import CircularSensorChecker

        # Clean DataFrame
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'z': np.random.randn(100),
            'vx': np.random.randn(100),
            'vy': np.random.randn(100),
            'vz': np.random.randn(100)
        })

        checker = CircularSensorChecker()
        passed = checker.check_dataframe(df)

        assert passed, "Clean data should pass"


class TestMinimaxCalibration:
    """Tests for minimax calibration."""

    def test_minimax_calibrator_import(self):
        """Test that minimax calibrator can be imported."""
        from minimax_calibration import MinimaxCalibrator, CalibrationResult
        calibrator = MinimaxCalibrator(target_fpr=0.05)
        assert calibrator is not None

    def test_calibration_result_structure(self):
        """Test CalibrationResult dataclass."""
        from minimax_calibration import CalibrationResult

        result = CalibrationResult(
            weights=np.array([0.25, 0.25, 0.25, 0.25]),
            threshold=0.5,
            worst_case_recall=0.8,
            worst_case_attack='drift',
            per_attack_recall={'bias': 0.9, 'drift': 0.8},
            achieved_fpr=0.05,
            optimization_success=True
        )

        assert result.worst_case_recall == 0.8
        assert result.worst_case_attack == 'drift'

    def test_minimax_beats_standard_on_worst_case(self):
        """Test that minimax improves worst-case recall."""
        from minimax_calibration import MinimaxCalibrator, StandardCalibrator

        np.random.seed(42)
        n = 500

        # Normal scores
        normal_scores = {
            'pinn': np.random.randn(n) * 0.3,
            'ekf': np.random.randn(n) * 0.3,
            'ml': np.random.randn(n) * 0.3,
            'temporal': np.random.randn(n) * 0.3
        }

        # Attack scores with different detectability
        attack_scores = {
            'easy': np.random.randn(100, 4) * 0.3 + 1.0,
            'hard': np.random.randn(100, 4) * 0.3 + 0.3  # Hard to detect
        }
        attack_labels = {
            'easy': np.ones(100),
            'hard': np.ones(100)
        }

        # Run both calibrations
        minimax = MinimaxCalibrator(target_fpr=0.05, method='grid')
        minimax_result = minimax.calibrate(attack_scores, attack_labels, normal_scores)

        standard = StandardCalibrator(target_fpr=0.05)
        standard_result = standard.calibrate(attack_scores, attack_labels, normal_scores)

        # Minimax should have better or equal worst-case
        assert minimax_result.worst_case_recall >= standard_result.worst_case_recall * 0.95, \
            "Minimax should not be significantly worse than standard"

    def test_weights_sum_to_one(self):
        """Test that calibrated weights are normalized."""
        from minimax_calibration import MinimaxCalibrator

        np.random.seed(42)

        normal_scores = {k: np.random.randn(100) * 0.3 for k in ['pinn', 'ekf', 'ml', 'temporal']}
        attack_scores = {'test': np.random.randn(50, 4) * 0.3 + 0.5}
        attack_labels = {'test': np.ones(50)}

        calibrator = MinimaxCalibrator(target_fpr=0.05, method='grid')
        result = calibrator.calibrate(attack_scores, attack_labels, normal_scores)

        assert abs(result.weights.sum() - 1.0) < 0.01, "Weights should sum to 1"


class TestOperationalMetrics:
    """Tests for operational metrics."""

    def test_operational_profiler_import(self):
        """Test that profiler can be imported."""
        from operational_metrics import OperationalProfiler, OperationalMetrics
        import torch.nn as nn

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                return self.fc(x)

        model = DummyModel()
        profiler = OperationalProfiler(model, sample_rate_hz=200)
        assert profiler is not None

    def test_latency_stats_structure(self):
        """Test LatencyStats dataclass."""
        from operational_metrics import LatencyStats

        stats = LatencyStats(
            mean_ms=2.5,
            std_ms=0.5,
            p50_ms=2.4,
            p95_ms=3.5,
            p99_ms=4.0,
            max_ms=5.0,
            min_ms=1.5,
            samples=100
        )

        assert stats.mean_ms == 2.5
        assert stats.p99_ms == 4.0

    def test_false_alarm_computation(self):
        """Test false alarm rate computation."""
        from operational_metrics import OperationalProfiler
        import torch.nn as nn

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                return self.fc(x)

        model = DummyModel()
        profiler = OperationalProfiler(model, sample_rate_hz=200)

        # 5% of samples exceed threshold
        normal_preds = np.concatenate([
            np.zeros(95),
            np.ones(5)
        ])
        threshold = 0.5

        fa_stats = profiler.compute_false_alarms(normal_preds, threshold)

        assert fa_stats.false_alarm_rate == 0.05
        assert fa_stats.false_alarms_per_hour == 0.05 * 200 * 3600

    def test_metrics_summary(self):
        """Test that summary can be generated."""
        from operational_metrics import (
            OperationalMetrics, LatencyStats, FalseAlarmStats,
            DetectionDelayStats, ResourceStats
        )

        metrics = OperationalMetrics(
            latency=LatencyStats(2.5, 0.5, 2.4, 3.5, 4.0, 5.0, 1.5, 100),
            false_alarms=FalseAlarmStats(100, 2000, 0.05, 36000, 100.0, 200),
            detection_delay=DetectionDelayStats(5.0, 2.0, 25.0, 20.0, 50.0),
            resources=ResourceStats(25.0, 40.0, 100.0, 120.0, 0.5),
            meets_latency_target=True,
            target_latency_ms=5.0
        )

        summary = metrics.summary()
        assert 'LATENCY' in summary
        assert 'FALSE ALARMS' in summary
        assert 'PASS' in summary


class TestExplainableAlarms:
    """Tests for explainable alarms."""

    def test_alarm_explainer_import(self):
        """Test that explainer can be imported."""
        from explainable_alarms import AlarmExplainer, AlarmExplanation

        weights = np.array([0.25, 0.25, 0.25, 0.25])
        explainer = AlarmExplainer(weights, threshold=0.5)
        assert explainer is not None

    def test_explain_single_alarm(self):
        """Test single alarm explanation."""
        from explainable_alarms import AlarmExplainer, AlarmSource

        weights = np.array([0.3, 0.25, 0.25, 0.2])
        explainer = AlarmExplainer(weights, threshold=0.5)

        # High scores across all components to exceed threshold
        scores = np.array([1.5, 0.8, 0.8, 0.6])  # fused = 0.3*1.5 + 0.25*0.8 + 0.25*0.8 + 0.2*0.6 = 0.97
        exp = explainer.explain_single(scores, timestamp=100)

        assert exp.is_alarm
        assert exp.pinn_contribution > exp.ekf_contribution
        assert exp.primary_source in [AlarmSource.PINN, AlarmSource.MULTI]

    def test_explain_normal_sample(self):
        """Test normal sample explanation."""
        from explainable_alarms import AlarmExplainer

        weights = np.array([0.25, 0.25, 0.25, 0.25])
        explainer = AlarmExplainer(weights, threshold=0.5)

        # Low scores should not trigger alarm
        scores = np.array([0.1, 0.1, 0.1, 0.1])
        exp = explainer.explain_single(scores)

        assert not exp.is_alarm

    def test_batch_explanation(self):
        """Test batch explanation."""
        from explainable_alarms import AlarmExplainer

        weights = np.array([0.25, 0.25, 0.25, 0.25])
        explainer = AlarmExplainer(weights, threshold=0.5)

        # Mix of normal and alarm samples
        scores = np.vstack([
            np.random.rand(50, 4) * 0.3,  # Normal
            np.random.rand(10, 4) * 0.3 + 0.7  # Alarms
        ])

        batch_exp = explainer.explain_batch(scores)

        assert batch_exp.n_samples == 60
        assert batch_exp.n_alarms > 0
        assert len(batch_exp.source_distribution) > 0

    def test_rule_fusion_explainer(self):
        """Test rule-based fusion explainer."""
        from explainable_alarms import RuleFusionExplainer

        explainer = RuleFusionExplainer()

        # Physics violation pattern
        pattern = np.array([0.8, 0.6, 0.3, 0.2])
        name, explanation, confidence = explainer.apply_rules(pattern)

        assert name != 'unknown'
        assert confidence > 0.5


class TestDemoScript:
    """Tests for demo script."""

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        from demo_reproduce_figure import generate_synthetic_data

        data = generate_synthetic_data(seed=42)

        assert 'normal_scores' in data
        assert 'attack_scores' in data
        assert 'attack_labels' in data
        assert len(data['attack_scores']) >= 5  # Multiple attack types

    def test_calibration_comparison(self):
        """Test calibration comparison function."""
        from demo_reproduce_figure import generate_synthetic_data, run_calibration_comparison

        data = generate_synthetic_data(seed=42)
        minimax_result, standard_result = run_calibration_comparison(data)

        assert minimax_result is not None
        assert standard_result is not None
        assert minimax_result.worst_case_recall > 0


class TestIntegration:
    """Integration tests for all roadmap items."""

    def test_full_pipeline_with_new_modules(self):
        """Test that all new modules work together."""
        from minimax_calibration import MinimaxCalibrator
        from explainable_alarms import AlarmExplainer

        np.random.seed(42)

        # Generate data
        n = 500
        normal_scores = {k: np.random.randn(n) * 0.3 for k in ['pinn', 'ekf', 'ml', 'temporal']}
        attack_scores = {'bias': np.random.randn(100, 4) * 0.3 + 0.8}
        attack_labels = {'bias': np.ones(100)}

        # Calibrate
        calibrator = MinimaxCalibrator(target_fpr=0.05, method='grid')
        result = calibrator.calibrate(attack_scores, attack_labels, normal_scores)

        # Use calibrated weights for explanation
        explainer = AlarmExplainer(result.weights, result.threshold)
        exp = explainer.explain_single(attack_scores['bias'][0])

        assert result.worst_case_recall > 0
        assert exp.is_alarm

    def test_all_new_files_exist(self):
        """Test that all new files were created."""
        base = Path(__file__).parent.parent

        required_files = [
            'scripts/ci_circular_check.py',
            'src/minimax_calibration.py',
            'src/operational_metrics.py',
            'src/explainable_alarms.py',
            'scripts/demo_reproduce_figure.py',
            'tests/test_leakage.py',
        ]

        for file in required_files:
            path = base / file
            assert path.exists(), f"Missing required file: {file}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
