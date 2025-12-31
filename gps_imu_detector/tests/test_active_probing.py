"""
Tests for Active Probing Module (v0.8.0)

Tests:
1. Excitation signal generators
2. Probing controller
3. Response analyzer
4. Complete active probing system
"""

import numpy as np
import pytest
from gps_imu_detector.src.active_probing import (
    # Generators
    ExcitationType,
    ExcitationSignal,
    MicroChirpGenerator,
    StepGenerator,
    PRBSGenerator,
    CompositeExcitationGenerator,
    # Controller
    ProbingController,
    ProbingState,
    # Analyzer
    ResponseAnalyzer,
    ResponseAnalysisResult,
    # System
    ActiveProbingSystem,
    ActiveProbingResult,
    # Evaluation
    evaluate_active_probing,
    MAX_EXCITATION,
)


# =============================================================================
# Excitation Generator Tests
# =============================================================================

class TestMicroChirpGenerator:
    """Tests for micro-chirp generator."""

    def test_initialization(self):
        gen = MicroChirpGenerator()
        assert gen.amplitude <= MAX_EXCITATION
        assert gen.f_start < gen.f_end

    def test_generate_returns_signal(self):
        gen = MicroChirpGenerator()
        signal = gen.generate()

        assert isinstance(signal, ExcitationSignal)
        assert signal.excitation_type == ExcitationType.MICRO_CHIRP
        assert len(signal.signal) > 0

    def test_amplitude_limited(self):
        gen = MicroChirpGenerator(amplitude=0.5)  # Way over limit
        signal = gen.generate()

        assert np.max(np.abs(signal.signal)) <= MAX_EXCITATION

    def test_signal_oscillates(self):
        gen = MicroChirpGenerator(duration=0.5)  # Longer duration
        signal = gen.generate()

        # Check for sign changes (oscillation)
        sign_changes = np.sum(np.diff(np.sign(signal.signal)) != 0)
        assert sign_changes > 2  # Should oscillate at least a few times


class TestStepGenerator:
    """Tests for step generator."""

    def test_initialization(self):
        gen = StepGenerator()
        assert gen.amplitude <= MAX_EXCITATION

    def test_generate_positive_step(self):
        gen = StepGenerator(amplitude=0.01)
        signal = gen.generate(positive=True)

        assert np.all(signal.signal > 0)

    def test_generate_negative_step(self):
        gen = StepGenerator(amplitude=0.01)
        signal = gen.generate(positive=False)

        assert np.all(signal.signal < 0)

    def test_step_is_constant(self):
        gen = StepGenerator(amplitude=0.01)
        signal = gen.generate()

        # All values should be equal
        assert np.allclose(signal.signal, signal.signal[0])


class TestPRBSGenerator:
    """Tests for PRBS generator."""

    def test_initialization(self):
        gen = PRBSGenerator()
        assert gen.amplitude <= MAX_EXCITATION

    def test_generate_returns_signal(self):
        gen = PRBSGenerator()
        signal = gen.generate()

        assert isinstance(signal, ExcitationSignal)
        assert signal.excitation_type == ExcitationType.PRBS

    def test_binary_values(self):
        gen = PRBSGenerator(amplitude=0.01)
        signal = gen.generate()

        # Should only have +amplitude or -amplitude
        unique_vals = np.unique(np.abs(signal.signal))
        assert len(unique_vals) == 1
        assert np.isclose(unique_vals[0], 0.01)

    def test_reproducible_with_seed(self):
        gen1 = PRBSGenerator(seed=42)
        gen2 = PRBSGenerator(seed=42)

        signal1 = gen1.generate()
        signal2 = gen2.generate()

        np.testing.assert_array_equal(signal1.signal, signal2.signal)

    def test_switches_occur(self):
        gen = PRBSGenerator(switch_prob=0.2)
        signal = gen.generate()

        # Check for sign changes
        sign_changes = np.sum(np.diff(np.sign(signal.signal)) != 0)
        assert sign_changes > 0


class TestCompositeExcitationGenerator:
    """Tests for composite generator."""

    def test_initialization(self):
        gen = CompositeExcitationGenerator()
        assert gen.total_amplitude <= MAX_EXCITATION

    def test_generate_returns_signal(self):
        gen = CompositeExcitationGenerator()
        signal = gen.generate()

        assert isinstance(signal, ExcitationSignal)
        assert signal.excitation_type == ExcitationType.COMPOSITE

    def test_combines_multiple_types(self):
        gen = CompositeExcitationGenerator()
        signal = gen.generate()

        # Composite should have more variation than single type
        std = np.std(signal.signal)
        assert std > 0


# =============================================================================
# Probing Controller Tests
# =============================================================================

class TestProbingController:
    """Tests for probing controller."""

    def test_initialization(self):
        controller = ProbingController()
        assert controller.probe_interval > 0
        assert not controller.state.is_probing

    def test_should_probe_after_interval(self):
        controller = ProbingController(probe_interval=10)

        # Advance time
        for _ in range(10):
            controller.sample_count += 1

        assert controller.should_probe()

    def test_get_excitation_starts_probe(self):
        controller = ProbingController(probe_interval=10)

        # Advance to trigger
        controller.sample_count = 10

        excitation = controller.get_excitation()

        # Should have started probing
        assert controller.state.is_probing
        assert controller.state.probe_count == 1

    def test_probes_rotate_types(self):
        controller = ProbingController(probe_interval=10)

        probe_types = []
        for _ in range(4):
            controller.sample_count = controller.sample_count + controller.probe_interval + 100
            controller.start_probe()
            probe_types.append(controller.probe_history[-1].excitation_type)
            controller.state.is_probing = False

        # Should have different types
        assert len(set(probe_types)) > 1

    def test_reset_clears_state(self):
        controller = ProbingController()

        for _ in range(20):
            controller.get_excitation()

        assert controller.sample_count > 0

        controller.reset()

        assert controller.sample_count == 0
        assert controller.state.probe_count == 0


# =============================================================================
# Response Analyzer Tests
# =============================================================================

class TestResponseAnalyzer:
    """Tests for response analyzer."""

    def test_initialization(self):
        analyzer = ResponseAnalyzer()
        assert analyzer.gain_estimate == 1.0
        assert analyzer.threshold > 0

    def test_analyze_returns_result(self):
        analyzer = ResponseAnalyzer()

        result = analyzer.analyze(0.01, 0.01)

        assert isinstance(result, ResponseAnalysisResult)

    def test_no_anomaly_when_response_matches(self):
        analyzer = ResponseAnalyzer(response_delay=2)

        # Send excitation
        for i in range(10):
            excitation = 0.01 if i < 5 else 0.0
            # Response matches excitation with delay
            response = 0.01 if 2 <= i < 7 else 0.0
            result = analyzer.analyze(excitation, response)

        assert not result.is_anomalous

    def test_anomaly_when_response_missing(self):
        analyzer = ResponseAnalyzer(response_delay=1, threshold=0.3)

        # Send excitation but no response
        for i in range(10):
            excitation = 0.05
            response = 0.0  # No response to excitation
            result = analyzer.analyze(excitation, response)

        # Should detect anomaly
        assert result.is_anomalous or analyzer.detection_count > 0

    def test_gain_adapts(self):
        analyzer = ResponseAnalyzer(response_delay=1, adaptation_rate=0.1, threshold=10.0)
        initial_gain = analyzer.gain_estimate

        # Send excitation first, then response with delay
        for i in range(100):
            excitation = 0.05  # Larger excitation
            # Response comes with delay and different gain
            response = 0.1 if i > 2 else 0.0  # 2x gain after delay
            analyzer.analyze(excitation, response)

        # Gain should have adapted toward 2.0
        assert abs(analyzer.gain_estimate - initial_gain) > 0.01

    def test_reset(self):
        analyzer = ResponseAnalyzer()

        for _ in range(10):
            analyzer.analyze(0.01, 0.01)

        assert analyzer.total_count == 10

        analyzer.reset()

        assert analyzer.total_count == 0


# =============================================================================
# Active Probing System Tests
# =============================================================================

class TestActiveProbingSystem:
    """Tests for complete active probing system."""

    def test_initialization(self):
        system = ActiveProbingSystem()
        assert system.controller is not None
        assert system.analyzer is not None

    def test_get_excitation_returns_value(self):
        system = ActiveProbingSystem(probe_interval=10)

        excitations = []
        for _ in range(20):
            excitations.append(system.get_excitation())

        # Should have some non-zero excitations
        assert np.sum(np.abs(excitations)) > 0

    def test_analyze_returns_result(self):
        system = ActiveProbingSystem()

        result = system.analyze(0.01, 0.01)

        assert isinstance(result, ActiveProbingResult)
        assert hasattr(result, 'is_stealth_detected')

    def test_nominal_not_detected(self):
        np.random.seed(42)
        system = ActiveProbingSystem(probe_interval=20)

        detections = 0
        for _ in range(100):
            excitation = system.get_excitation()
            # Nominal response: includes excitation effect
            response = excitation * 0.9 + np.random.randn() * 0.001
            result = system.analyze(excitation, response)
            if result.is_stealth_detected:
                detections += 1

        # Should have few/no false detections
        assert detections < 10

    def test_stealth_attack_detected(self):
        np.random.seed(42)
        system = ActiveProbingSystem(
            probe_interval=20,
            response_threshold=0.3,
            consecutive_required=2,
        )

        detections = 0
        for _ in range(200):
            excitation = system.get_excitation()
            # Attack: does NOT include excitation effect
            response = 0.5 + np.random.randn() * 0.01  # Spoofed, no excitation
            result = system.analyze(excitation, response)
            if result.is_stealth_detected:
                detections += 1

        # Should detect stealth attack
        assert detections > 0

    def test_metrics_tracked(self):
        system = ActiveProbingSystem()

        for _ in range(50):
            exc = system.get_excitation()
            system.analyze(exc, 0.0)

        metrics = system.get_metrics()

        assert "total_samples" in metrics
        assert "stealth_detections" in metrics
        assert metrics["total_samples"] == 50

    def test_reset(self):
        system = ActiveProbingSystem()

        for _ in range(10):
            exc = system.get_excitation()
            system.analyze(exc, 0.0)

        assert system.total_count == 10

        system.reset()

        assert system.total_count == 0


# =============================================================================
# Evaluation Tests
# =============================================================================

class TestEvaluation:
    """Tests for evaluation function."""

    def test_evaluate_returns_metrics(self):
        np.random.seed(42)

        nominal = np.random.randn(100) * 0.1
        attack = np.random.randn(100) * 0.1 + 0.5

        results = evaluate_active_probing(nominal, attack)

        assert "recall" in results
        assert "fpr" in results

    def test_metrics_in_valid_range(self):
        np.random.seed(42)

        nominal = np.random.randn(100) * 0.1
        attack = np.random.randn(100) * 0.1

        results = evaluate_active_probing(nominal, attack)

        assert 0 <= results["recall"] <= 1
        assert 0 <= results["fpr"] <= 1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
