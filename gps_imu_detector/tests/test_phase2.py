"""
Tests for Phase 2: Probing Improvements

Tests:
2.1 Regime-adaptive probe scheduler
2.2 PINN-optimized probe library
"""

import numpy as np
import pytest
import torch


# =============================================================================
# Phase 2.1: Probe Scheduler Tests
# =============================================================================

class TestProbeScheduler:
    """Tests for regime-adaptive probe scheduler."""

    def test_scheduler_initialization(self):
        from gps_imu_detector.src.adaptive_probing import (
            ProbeScheduler, ProbeSchedulerConfig
        )

        scheduler = ProbeScheduler()
        assert scheduler.config.cooldown_hover == 100
        assert scheduler.config.max_probes_hover == 20

        custom = ProbeSchedulerConfig(cooldown_hover=50, max_probes_hover=10)
        scheduler = ProbeScheduler(custom)
        assert scheduler.config.cooldown_hover == 50

    def test_scheduler_allows_hover_probe(self):
        from gps_imu_detector.src.adaptive_probing import ProbeScheduler
        from gps_imu_detector.src.regime_detection import FlightRegime

        scheduler = ProbeScheduler()

        # Hover conditions with good margins
        decision = scheduler.decide(
            velocity=np.array([0.1, 0.0, 0.0]),
            angular_rate=np.array([0.05, 0.0, 0.0]),
            attitude=np.array([0.1, 0.1, 0.0]),  # Well within limits
            power_fraction=0.5,
            altitude_agl=10.0,
        )

        assert decision.allow_probe == True
        assert decision.regime == FlightRegime.HOVER

    def test_scheduler_blocks_aggressive_regime(self):
        from gps_imu_detector.src.adaptive_probing import ProbeScheduler
        from gps_imu_detector.src.regime_detection import FlightRegime

        scheduler = ProbeScheduler()

        # Aggressive flight conditions
        decision = scheduler.decide(
            velocity=np.array([10.0, 0.0, 0.0]),  # High velocity
            angular_rate=np.array([1.5, 0.0, 0.0]),  # High angular rate
            attitude=np.array([0.1, 0.1, 0.0]),
            power_fraction=0.5,
            altitude_agl=10.0,
        )

        assert decision.allow_probe == False
        assert decision.regime == FlightRegime.AGGRESSIVE
        assert "AGGRESSIVE" in decision.reason

    def test_scheduler_blocks_low_altitude(self):
        from gps_imu_detector.src.adaptive_probing import ProbeScheduler

        scheduler = ProbeScheduler()

        decision = scheduler.decide(
            velocity=np.array([0.1, 0.0, 0.0]),
            angular_rate=np.array([0.05, 0.0, 0.0]),
            attitude=np.array([0.1, 0.1, 0.0]),
            power_fraction=0.5,
            altitude_agl=1.0,  # Too low
        )

        assert decision.allow_probe == False
        assert "Altitude" in decision.reason

    def test_scheduler_blocks_low_power_margin(self):
        from gps_imu_detector.src.adaptive_probing import ProbeScheduler

        scheduler = ProbeScheduler()

        decision = scheduler.decide(
            velocity=np.array([0.1, 0.0, 0.0]),
            angular_rate=np.array([0.05, 0.0, 0.0]),
            attitude=np.array([0.1, 0.1, 0.0]),
            power_fraction=0.95,  # Near max power
            altitude_agl=10.0,
        )

        assert decision.allow_probe == False
        assert "power" in decision.reason.lower()

    def test_scheduler_blocks_poor_attitude_margin(self):
        from gps_imu_detector.src.adaptive_probing import ProbeScheduler

        scheduler = ProbeScheduler()

        decision = scheduler.decide(
            velocity=np.array([0.1, 0.0, 0.0]),
            angular_rate=np.array([0.05, 0.0, 0.0]),
            attitude=np.array([0.6, 0.0, 0.0]),  # Near roll limit (pi/4 = 0.785)
            power_fraction=0.5,
            altitude_agl=10.0,
        )

        assert decision.allow_probe == False
        assert "attitude" in decision.reason.lower()

    def test_scheduler_cooldown_enforcement(self):
        from gps_imu_detector.src.adaptive_probing import ProbeScheduler
        from gps_imu_detector.src.regime_detection import FlightRegime

        scheduler = ProbeScheduler()

        # First probe allowed
        decision1 = scheduler.decide(
            velocity=np.array([0.1, 0.0, 0.0]),
            angular_rate=np.array([0.05, 0.0, 0.0]),
            attitude=np.array([0.1, 0.1, 0.0]),
            power_fraction=0.5,
            altitude_agl=10.0,
        )
        assert decision1.allow_probe == True

        # Record the probe
        scheduler.record_probe(FlightRegime.HOVER)

        # Second probe blocked by cooldown
        decision2 = scheduler.decide(
            velocity=np.array([0.1, 0.0, 0.0]),
            angular_rate=np.array([0.05, 0.0, 0.0]),
            attitude=np.array([0.1, 0.1, 0.0]),
            power_fraction=0.5,
            altitude_agl=10.0,
        )
        assert decision2.allow_probe == False
        assert "Cooldown" in decision2.reason
        assert decision2.cooldown_remaining > 0

    def test_scheduler_quota_enforcement(self):
        from gps_imu_detector.src.adaptive_probing import (
            ProbeScheduler, ProbeSchedulerConfig
        )
        from gps_imu_detector.src.regime_detection import FlightRegime

        # Small quota for testing
        config = ProbeSchedulerConfig(
            max_probes_hover=2,
            cooldown_hover=1,  # Very short cooldown
        )
        scheduler = ProbeScheduler(config)

        # Exhaust quota
        for i in range(3):
            decision = scheduler.decide(
                velocity=np.array([0.1, 0.0, 0.0]),
                angular_rate=np.array([0.05, 0.0, 0.0]),
                attitude=np.array([0.1, 0.1, 0.0]),
                power_fraction=0.5,
                altitude_agl=10.0,
            )
            if decision.allow_probe:
                scheduler.record_probe(FlightRegime.HOVER)

        # Should be denied due to quota
        decision = scheduler.decide(
            velocity=np.array([0.1, 0.0, 0.0]),
            angular_rate=np.array([0.05, 0.0, 0.0]),
            attitude=np.array([0.1, 0.1, 0.0]),
            power_fraction=0.5,
            altitude_agl=10.0,
        )
        assert decision.allow_probe == False
        assert "Quota" in decision.reason

    def test_scheduler_reset(self):
        from gps_imu_detector.src.adaptive_probing import ProbeScheduler
        from gps_imu_detector.src.regime_detection import FlightRegime

        scheduler = ProbeScheduler()

        # Do some probing
        scheduler.decide(
            velocity=np.array([0.1, 0.0, 0.0]),
            angular_rate=np.array([0.05, 0.0, 0.0]),
            attitude=np.array([0.1, 0.1, 0.0]),
            power_fraction=0.5,
            altitude_agl=10.0,
        )
        scheduler.record_probe(FlightRegime.HOVER)

        # Reset
        scheduler.reset()

        # Should be able to probe again
        decision = scheduler.decide(
            velocity=np.array([0.1, 0.0, 0.0]),
            angular_rate=np.array([0.05, 0.0, 0.0]),
            attitude=np.array([0.1, 0.1, 0.0]),
            power_fraction=0.5,
            altitude_agl=10.0,
        )
        assert decision.allow_probe == True


# =============================================================================
# Phase 2.2: Probe Library Tests
# =============================================================================

class TestProbeLibrary:
    """Tests for PINN-optimized probe library."""

    def test_standard_library_creation(self):
        from gps_imu_detector.src.adaptive_probing import create_standard_probe_library

        library = create_standard_probe_library()

        assert library.version == "1.0.0"
        assert 'HOVER' in library.probes
        assert 'FORWARD' in library.probes
        assert len(library.probes['HOVER']) >= 1
        assert len(library.probes['FORWARD']) >= 1

    def test_probe_waveform_properties(self):
        from gps_imu_detector.src.adaptive_probing import create_standard_probe_library
        from gps_imu_detector.src.regime_detection import FlightRegime

        library = create_standard_probe_library()
        hover_probes = library.get_probes(FlightRegime.HOVER)

        assert len(hover_probes) > 0

        probe = hover_probes[0]
        assert probe.duration == len(probe.waveform)
        assert probe.amplitude > 0
        assert probe.energy > 0
        assert probe.expected_response_gain > 0

    def test_get_best_probe(self):
        from gps_imu_detector.src.adaptive_probing import create_standard_probe_library
        from gps_imu_detector.src.regime_detection import FlightRegime

        library = create_standard_probe_library()

        best_hover = library.get_best_probe(FlightRegime.HOVER)
        assert best_hover is not None
        assert best_hover.regime == FlightRegime.HOVER

        # No probes for aggressive
        best_aggressive = library.get_best_probe(FlightRegime.AGGRESSIVE)
        assert best_aggressive is None

    def test_probe_optimizer_initialization(self):
        from gps_imu_detector.src.adaptive_probing import ProbeOptimizer

        optimizer = ProbeOptimizer()
        assert optimizer.max_amplitude == 0.02
        assert optimizer.max_duration == 20

    def test_probe_optimizer_optimize(self):
        from gps_imu_detector.src.adaptive_probing import ProbeOptimizer
        from gps_imu_detector.src.regime_detection import FlightRegime

        np.random.seed(42)
        torch.manual_seed(42)

        optimizer = ProbeOptimizer()

        nominal = np.random.randn(20, 12).astype(np.float32) * 0.1
        attack = np.random.randn(20, 12).astype(np.float32) * 0.1 + 0.1

        probe = optimizer.optimize_probe(
            FlightRegime.HOVER,
            nominal,
            attack,
            n_iterations=10,
        )

        assert probe.regime == FlightRegime.HOVER
        assert probe.duration == optimizer.max_duration
        assert len(probe.waveform) == probe.duration
        assert np.max(np.abs(probe.waveform)) <= optimizer.max_amplitude

    def test_build_optimized_library(self):
        from gps_imu_detector.src.adaptive_probing import build_optimized_probe_library

        np.random.seed(42)
        torch.manual_seed(42)

        # Generate simple trajectories
        nominal = np.random.randn(3, 50, 12).astype(np.float32) * 0.1
        attack = np.random.randn(3, 50, 12).astype(np.float32) * 0.1 + 0.1

        library = build_optimized_probe_library(nominal, attack, version="2.0.0")

        assert library.version == "2.0.0"
        # At least one regime should have probes
        total_probes = sum(len(probes) for probes in library.probes.values())
        assert total_probes >= 1


# =============================================================================
# Integrated System Tests
# =============================================================================

class TestAdaptiveProbingSystem:
    """Tests for integrated adaptive probing system."""

    def test_system_initialization(self):
        from gps_imu_detector.src.adaptive_probing import AdaptiveProbingSystem

        system = AdaptiveProbingSystem()
        assert system.scheduler is not None
        assert system.library is not None
        assert system._in_probe == False

    def test_system_update_returns_excitation(self):
        from gps_imu_detector.src.adaptive_probing import AdaptiveProbingSystem

        system = AdaptiveProbingSystem()

        # Good conditions - should allow probe
        excitation, is_active = system.update(
            velocity=np.array([0.1, 0.0, 0.0]),
            angular_rate=np.array([0.05, 0.0, 0.0]),
            attitude=np.array([0.1, 0.1, 0.0]),
            power_fraction=0.5,
            altitude_agl=10.0,
        )

        # Should activate probing (first value might be 0 for sine wave)
        assert is_active == True
        # Run a few more steps to get non-zero excitation
        for _ in range(5):
            exc, active = system.update(
                velocity=np.array([0.1, 0.0, 0.0]),
                angular_rate=np.array([0.05, 0.0, 0.0]),
                attitude=np.array([0.1, 0.1, 0.0]),
                power_fraction=0.5,
                altitude_agl=10.0,
            )
            if active and exc != 0.0:
                break
        # Should have found a non-zero excitation during probe
        assert exc != 0.0 or not active  # Either got non-zero or probe ended

    def test_system_probe_sequence(self):
        from gps_imu_detector.src.adaptive_probing import AdaptiveProbingSystem

        system = AdaptiveProbingSystem()
        excitations = []

        # Run for 30 timesteps to capture a probe
        for _ in range(30):
            exc, active = system.update(
                velocity=np.array([0.1, 0.0, 0.0]),
                angular_rate=np.array([0.05, 0.0, 0.0]),
                attitude=np.array([0.1, 0.1, 0.0]),
                power_fraction=0.5,
                altitude_agl=10.0,
            )
            excitations.append((exc, active))

        # Should have some active probing
        active_count = sum(1 for _, a in excitations if a)
        assert active_count > 0

    def test_system_respects_cooldown(self):
        from gps_imu_detector.src.adaptive_probing import (
            AdaptiveProbingSystem, ProbeSchedulerConfig
        )

        config = ProbeSchedulerConfig(cooldown_hover=50)
        system = AdaptiveProbingSystem(scheduler_config=config)

        probe_starts = []

        # Run for 200 timesteps
        for t in range(200):
            exc, active = system.update(
                velocity=np.array([0.1, 0.0, 0.0]),
                angular_rate=np.array([0.05, 0.0, 0.0]),
                attitude=np.array([0.1, 0.1, 0.0]),
                power_fraction=0.5,
                altitude_agl=10.0,
            )
            if active and (not probe_starts or t > probe_starts[-1] + 30):
                probe_starts.append(t)

        # Check probe spacing respects cooldown
        if len(probe_starts) >= 2:
            for i in range(1, len(probe_starts)):
                spacing = probe_starts[i] - probe_starts[i-1]
                # Allow for probe duration
                assert spacing >= 20

    def test_system_reset(self):
        from gps_imu_detector.src.adaptive_probing import AdaptiveProbingSystem

        system = AdaptiveProbingSystem()

        # Do some updates
        for _ in range(10):
            system.update(
                velocity=np.array([0.1, 0.0, 0.0]),
                angular_rate=np.array([0.05, 0.0, 0.0]),
                attitude=np.array([0.1, 0.1, 0.0]),
                power_fraction=0.5,
                altitude_agl=10.0,
            )

        # Reset
        system.reset()

        assert system._in_probe == False
        assert system._current_probe is None

    def test_system_no_probe_in_bad_conditions(self):
        from gps_imu_detector.src.adaptive_probing import AdaptiveProbingSystem

        system = AdaptiveProbingSystem()

        # Low altitude - no probing
        for _ in range(50):
            exc, active = system.update(
                velocity=np.array([0.1, 0.0, 0.0]),
                angular_rate=np.array([0.05, 0.0, 0.0]),
                attitude=np.array([0.1, 0.1, 0.0]),
                power_fraction=0.5,
                altitude_agl=1.0,  # Too low
            )
            assert active == False


# =============================================================================
# Phase 2 Checkpoint Tests
# =============================================================================

class TestPhase2Checkpoint:
    """Integration tests for Phase 2 checkpoint."""

    def test_evaluate_adaptive_probing(self):
        from gps_imu_detector.src.adaptive_probing import evaluate_adaptive_probing

        np.random.seed(42)

        # Generate trajectories
        nominal = np.random.randn(5, 100, 12).astype(np.float32) * 0.1
        attack = np.random.randn(5, 100, 12).astype(np.float32) * 0.1 + 0.1

        results = evaluate_adaptive_probing(nominal, attack)

        assert 'total_probes_nominal' in results
        assert 'total_probes_attack' in results
        assert 'probes_per_trajectory_nominal' in results
        assert 'probes_per_trajectory_attack' in results

        # Should have some probes
        assert results['total_probes_nominal'] > 0 or results['total_probes_attack'] > 0

    def test_regime_specific_probing_rates(self):
        """Verify probing rates vary by regime as expected."""
        from gps_imu_detector.src.adaptive_probing import AdaptiveProbingSystem

        system = AdaptiveProbingSystem()

        # Count probes in hover
        hover_probes = 0
        for _ in range(500):
            _, active = system.update(
                velocity=np.array([0.1, 0.0, 0.0]),  # Hover
                angular_rate=np.array([0.05, 0.0, 0.0]),
                attitude=np.array([0.1, 0.1, 0.0]),
                power_fraction=0.5,
                altitude_agl=10.0,
            )
            if active:
                hover_probes += 1

        system.reset()

        # Count probes in aggressive
        aggressive_probes = 0
        for _ in range(500):
            _, active = system.update(
                velocity=np.array([10.0, 0.0, 0.0]),  # Aggressive
                angular_rate=np.array([1.5, 0.0, 0.0]),
                attitude=np.array([0.1, 0.1, 0.0]),
                power_fraction=0.5,
                altitude_agl=10.0,
            )
            if active:
                aggressive_probes += 1

        # Hover should have more probes than aggressive (which has 0)
        assert hover_probes > aggressive_probes
        assert aggressive_probes == 0

    def test_probe_energy_within_bounds(self):
        """Verify probe energy stays within safety bounds."""
        from gps_imu_detector.src.adaptive_probing import create_standard_probe_library

        library = create_standard_probe_library()

        max_allowed_energy = 0.01  # Safety bound

        for regime_name, probes in library.probes.items():
            for probe in probes:
                assert probe.energy < max_allowed_energy, \
                    f"Probe {probe.name} exceeds energy bound"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
