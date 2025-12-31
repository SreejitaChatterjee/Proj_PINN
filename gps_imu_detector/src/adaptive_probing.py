"""
Adaptive Probing Module (Phase 2)

Implements:
2.1 Regime-adaptive probe scheduler
    - Gates probes on stability, attitude, power margins
    - Cooldowns and quotas per regime
    - Pure policy (no learning)

2.2 PINN-optimized probe library
    - Offline bilevel optimization for probe design
    - 3-5 short, safe probes per regime
    - Distilled into fixed waveforms
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from .regime_detection import FlightRegime, RegimeClassifier


# =============================================================================
# Phase 2.1: Regime-Adaptive Probe Scheduler
# =============================================================================

@dataclass
class ProbeSchedulerConfig:
    """Configuration for probe scheduling."""
    # Stability margins
    min_attitude_margin: float = 0.3  # rad from limits
    min_power_margin: float = 0.2     # fraction of max power
    min_altitude: float = 2.0         # meters AGL

    # Cooldowns (in timesteps at 200 Hz)
    cooldown_hover: int = 100         # 0.5 seconds
    cooldown_forward: int = 200       # 1.0 seconds
    cooldown_aggressive: int = 1000   # Never probe
    cooldown_gusty: int = 1000        # Never probe

    # Quotas (max probes per window)
    quota_window: int = 2000          # 10 seconds
    max_probes_hover: int = 20
    max_probes_forward: int = 10
    max_probes_aggressive: int = 0
    max_probes_gusty: int = 0


@dataclass
class ProbeDecision:
    """Decision from probe scheduler."""
    allow_probe: bool
    reason: str
    regime: FlightRegime
    cooldown_remaining: int
    quota_remaining: int


class ProbeScheduler:
    """
    Regime-adaptive probe scheduler.

    Pure policy - no learning. Gates probes based on:
    - Current flight regime
    - Stability margins (attitude, power, altitude)
    - Cooldown timers
    - Per-window quotas
    """

    def __init__(self, config: Optional[ProbeSchedulerConfig] = None):
        self.config = config or ProbeSchedulerConfig()
        self.classifier = RegimeClassifier()

        # State
        self._timestep = 0
        self._last_probe_time: Dict[str, int] = {}
        self._probe_counts: Dict[str, int] = {}
        self._window_start = 0

    def decide(
        self,
        velocity: np.ndarray,
        angular_rate: np.ndarray,
        attitude: np.ndarray,
        power_fraction: float,
        altitude_agl: float,
        acceleration: Optional[np.ndarray] = None,
    ) -> ProbeDecision:
        """
        Decide whether to allow probing.

        Args:
            velocity: [3] velocity vector (m/s)
            angular_rate: [3] angular rate (rad/s)
            attitude: [3] roll, pitch, yaw (rad)
            power_fraction: Current power as fraction of max (0-1)
            altitude_agl: Altitude above ground level (m)
            acceleration: [3] acceleration (optional)

        Returns:
            ProbeDecision with allow/deny and reason
        """
        self._timestep += 1

        # Reset quotas at window boundaries
        if self._timestep - self._window_start >= self.config.quota_window:
            self._window_start = self._timestep
            self._probe_counts.clear()

        # Classify regime
        regime = self.classifier.classify(velocity, angular_rate, acceleration)
        regime_name = regime.name

        # Initialize tracking for this regime
        if regime_name not in self._last_probe_time:
            self._last_probe_time[regime_name] = -float('inf')
        if regime_name not in self._probe_counts:
            self._probe_counts[regime_name] = 0

        # Check regime restrictions
        if regime in (FlightRegime.AGGRESSIVE, FlightRegime.GUSTY):
            return ProbeDecision(
                allow_probe=False,
                reason=f"Probing disabled in {regime_name} regime",
                regime=regime,
                cooldown_remaining=0,
                quota_remaining=0,
            )

        # Check stability margins
        attitude_margin = self._compute_attitude_margin(attitude)
        if attitude_margin < self.config.min_attitude_margin:
            return ProbeDecision(
                allow_probe=False,
                reason=f"Insufficient attitude margin: {attitude_margin:.2f}",
                regime=regime,
                cooldown_remaining=self._get_cooldown_remaining(regime_name),
                quota_remaining=self._get_quota_remaining(regime_name),
            )

        if power_fraction > (1.0 - self.config.min_power_margin):
            return ProbeDecision(
                allow_probe=False,
                reason=f"Insufficient power margin: {1.0 - power_fraction:.2f}",
                regime=regime,
                cooldown_remaining=self._get_cooldown_remaining(regime_name),
                quota_remaining=self._get_quota_remaining(regime_name),
            )

        if altitude_agl < self.config.min_altitude:
            return ProbeDecision(
                allow_probe=False,
                reason=f"Altitude too low: {altitude_agl:.1f}m",
                regime=regime,
                cooldown_remaining=self._get_cooldown_remaining(regime_name),
                quota_remaining=self._get_quota_remaining(regime_name),
            )

        # Check cooldown
        cooldown = self._get_cooldown(regime)
        time_since_last = self._timestep - self._last_probe_time[regime_name]
        if time_since_last < cooldown:
            return ProbeDecision(
                allow_probe=False,
                reason=f"Cooldown active: {cooldown - time_since_last} steps remaining",
                regime=regime,
                cooldown_remaining=cooldown - time_since_last,
                quota_remaining=self._get_quota_remaining(regime_name),
            )

        # Check quota
        max_probes = self._get_max_probes(regime)
        if self._probe_counts[regime_name] >= max_probes:
            return ProbeDecision(
                allow_probe=False,
                reason=f"Quota exhausted: {self._probe_counts[regime_name]}/{max_probes}",
                regime=regime,
                cooldown_remaining=0,
                quota_remaining=0,
            )

        # All checks passed
        return ProbeDecision(
            allow_probe=True,
            reason="All conditions met",
            regime=regime,
            cooldown_remaining=0,
            quota_remaining=max_probes - self._probe_counts[regime_name],
        )

    def record_probe(self, regime: FlightRegime):
        """Record that a probe was sent."""
        regime_name = regime.name
        self._last_probe_time[regime_name] = self._timestep
        self._probe_counts[regime_name] = self._probe_counts.get(regime_name, 0) + 1

    def reset(self):
        """Reset scheduler state."""
        self._timestep = 0
        self._last_probe_time.clear()
        self._probe_counts.clear()
        self._window_start = 0

    def _compute_attitude_margin(self, attitude: np.ndarray) -> float:
        """Compute margin from attitude limits."""
        roll, pitch, _ = attitude
        max_roll = np.pi / 4  # 45 degrees
        max_pitch = np.pi / 4
        roll_margin = max_roll - abs(roll)
        pitch_margin = max_pitch - abs(pitch)
        return min(roll_margin, pitch_margin)

    def _get_cooldown(self, regime: FlightRegime) -> int:
        """Get cooldown for regime."""
        cooldowns = {
            FlightRegime.HOVER: self.config.cooldown_hover,
            FlightRegime.FORWARD: self.config.cooldown_forward,
            FlightRegime.AGGRESSIVE: self.config.cooldown_aggressive,
            FlightRegime.GUSTY: self.config.cooldown_gusty,
            FlightRegime.UNKNOWN: self.config.cooldown_forward,
        }
        return cooldowns.get(regime, self.config.cooldown_forward)

    def _get_max_probes(self, regime: FlightRegime) -> int:
        """Get max probes for regime."""
        quotas = {
            FlightRegime.HOVER: self.config.max_probes_hover,
            FlightRegime.FORWARD: self.config.max_probes_forward,
            FlightRegime.AGGRESSIVE: self.config.max_probes_aggressive,
            FlightRegime.GUSTY: self.config.max_probes_gusty,
            FlightRegime.UNKNOWN: self.config.max_probes_forward,
        }
        return quotas.get(regime, self.config.max_probes_forward)

    def _get_cooldown_remaining(self, regime_name: str) -> int:
        """Get remaining cooldown time."""
        if regime_name not in self._last_probe_time:
            return 0
        regime = FlightRegime[regime_name]
        cooldown = self._get_cooldown(regime)
        elapsed = self._timestep - self._last_probe_time[regime_name]
        return max(0, cooldown - elapsed)

    def _get_quota_remaining(self, regime_name: str) -> int:
        """Get remaining quota."""
        if regime_name not in self._probe_counts:
            regime = FlightRegime[regime_name]
            return self._get_max_probes(regime)
        regime = FlightRegime[regime_name]
        return max(0, self._get_max_probes(regime) - self._probe_counts[regime_name])


# =============================================================================
# Phase 2.2: PINN-Optimized Probe Library
# =============================================================================

@dataclass
class ProbeWaveform:
    """A single optimized probe waveform."""
    name: str
    regime: FlightRegime
    duration: int  # timesteps at 200 Hz
    amplitude: float  # max amplitude
    waveform: np.ndarray  # [duration] signal values
    energy: float  # total energy (sum of squares)
    expected_response_gain: float  # expected output/input ratio


@dataclass
class ProbeLibrary:
    """
    Library of optimized probe waveforms per regime.

    Precomputed offline - no runtime optimization.
    """
    probes: Dict[str, List[ProbeWaveform]]  # regime_name -> list of probes
    version: str
    created_at: str

    def get_probes(self, regime: FlightRegime) -> List[ProbeWaveform]:
        """Get available probes for regime."""
        return self.probes.get(regime.name, [])

    def get_best_probe(self, regime: FlightRegime) -> Optional[ProbeWaveform]:
        """Get best probe for regime (highest response gain)."""
        probes = self.get_probes(regime)
        if not probes:
            return None
        return max(probes, key=lambda p: p.expected_response_gain)


class ProbeOptimizer:
    """
    Offline bilevel optimizer for probe design.

    Outer loop: Maximize detection separability
    Inner loop: PINN predicts system response

    Outputs fixed waveforms per regime.
    """

    def __init__(
        self,
        state_dim: int = 12,
        max_amplitude: float = 0.02,
        max_duration: int = 20,  # 100ms at 200 Hz
        device: str = 'cpu',
    ):
        self.state_dim = state_dim
        self.max_amplitude = max_amplitude
        self.max_duration = max_duration
        self.device = device

        # Response predictor (inner model)
        self.response_model = nn.Sequential(
            nn.Linear(max_duration + state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, max_duration),
        ).to(device)

    def optimize_probe(
        self,
        regime: FlightRegime,
        nominal_states: np.ndarray,
        attack_states: np.ndarray,
        n_iterations: int = 100,
        lr: float = 0.01,
    ) -> ProbeWaveform:
        """
        Optimize a probe waveform for given regime.

        Args:
            regime: Target flight regime
            nominal_states: [N, state_dim] nominal state samples
            attack_states: [M, state_dim] attack state samples
            n_iterations: Optimization iterations
            lr: Learning rate

        Returns:
            Optimized ProbeWaveform
        """
        # Initialize probe as learnable parameter
        probe = torch.zeros(self.max_duration, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([probe], lr=lr)

        nominal_t = torch.tensor(nominal_states, dtype=torch.float32, device=self.device)
        attack_t = torch.tensor(attack_states, dtype=torch.float32, device=self.device)

        for iteration in range(n_iterations):
            optimizer.zero_grad()

            # Clamp probe to valid range
            clamped_probe = torch.tanh(probe) * self.max_amplitude

            # Predict responses for nominal and attack
            nom_responses = self._predict_responses(clamped_probe, nominal_t)
            att_responses = self._predict_responses(clamped_probe, attack_t)

            # Objective: maximize separation between nominal and attack responses
            nom_energy = torch.mean(torch.sum(nom_responses ** 2, dim=1))
            att_energy = torch.mean(torch.sum(att_responses ** 2, dim=1))

            # Separation loss (attack should have different response)
            separation = torch.abs(nom_energy - att_energy)

            # Energy constraint (prefer low-energy probes)
            probe_energy = torch.sum(clamped_probe ** 2)

            # Combined loss
            loss = -separation + 0.01 * probe_energy

            loss.backward()
            optimizer.step()

        # Extract final waveform
        with torch.no_grad():
            final_probe = torch.tanh(probe) * self.max_amplitude
            waveform = final_probe.cpu().numpy()

        return ProbeWaveform(
            name=f"optimized_{regime.name.lower()}",
            regime=regime,
            duration=self.max_duration,
            amplitude=self.max_amplitude,
            waveform=waveform,
            energy=float(np.sum(waveform ** 2)),
            expected_response_gain=1.0,  # Computed separately
        )

    def _predict_responses(
        self,
        probe: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Predict system responses to probe."""
        batch_size = states.shape[0]

        # Expand probe for batch
        probe_batch = probe.unsqueeze(0).expand(batch_size, -1)

        # Concatenate probe and state
        inputs = torch.cat([probe_batch, states], dim=1)

        # Predict response
        return self.response_model(inputs)


def create_standard_probe_library() -> ProbeLibrary:
    """
    Create standard probe library with hand-designed waveforms.

    Used as baseline before PINN optimization.
    """
    from datetime import datetime

    probes = {}

    # HOVER probes (can use longer, gentler probes)
    probes['HOVER'] = [
        ProbeWaveform(
            name="hover_chirp",
            regime=FlightRegime.HOVER,
            duration=20,
            amplitude=0.015,
            waveform=np.sin(np.linspace(0, 4 * np.pi, 20)) * 0.015,
            energy=0.00225,
            expected_response_gain=0.95,
        ),
        ProbeWaveform(
            name="hover_step",
            regime=FlightRegime.HOVER,
            duration=10,
            amplitude=0.01,
            waveform=np.concatenate([np.ones(5) * 0.01, np.zeros(5)]),
            energy=0.0005,
            expected_response_gain=0.90,
        ),
    ]

    # FORWARD probes (shorter, more aggressive)
    probes['FORWARD'] = [
        ProbeWaveform(
            name="forward_chirp",
            regime=FlightRegime.FORWARD,
            duration=15,
            amplitude=0.02,
            waveform=np.sin(np.linspace(0, 3 * np.pi, 15)) * 0.02,
            energy=0.003,
            expected_response_gain=0.85,
        ),
        ProbeWaveform(
            name="forward_doublet",
            regime=FlightRegime.FORWARD,
            duration=10,
            amplitude=0.015,
            waveform=np.concatenate([np.ones(5) * 0.015, -np.ones(5) * 0.015]),
            energy=0.00225,
            expected_response_gain=0.80,
        ),
    ]

    return ProbeLibrary(
        probes=probes,
        version="1.0.0",
        created_at=datetime.now().isoformat(),
    )


def build_optimized_probe_library(
    nominal_trajectories: np.ndarray,
    attack_trajectories: np.ndarray,
    version: str = "1.0.0",
) -> ProbeLibrary:
    """
    Build optimized probe library using PINN.

    Args:
        nominal_trajectories: [N, T, state_dim] nominal data
        attack_trajectories: [M, T, state_dim] attack data
        version: Version string

    Returns:
        Optimized ProbeLibrary
    """
    from datetime import datetime

    optimizer = ProbeOptimizer()
    classifier = RegimeClassifier()

    # Collect states per regime
    regime_nominal: Dict[str, List[np.ndarray]] = {}
    regime_attack: Dict[str, List[np.ndarray]] = {}

    for traj in nominal_trajectories:
        for t in range(len(traj)):
            state = traj[t]
            regime = classifier.classify(state[3:6], state[9:12])
            if regime.name not in regime_nominal:
                regime_nominal[regime.name] = []
            regime_nominal[regime.name].append(state)

    for traj in attack_trajectories:
        for t in range(len(traj)):
            state = traj[t]
            regime = classifier.classify(state[3:6], state[9:12])
            if regime.name not in regime_attack:
                regime_attack[regime.name] = []
            regime_attack[regime.name].append(state)

    # Optimize probes per regime
    probes = {}

    for regime in [FlightRegime.HOVER, FlightRegime.FORWARD]:
        nom_states = regime_nominal.get(regime.name, [])
        att_states = regime_attack.get(regime.name, [])

        if len(nom_states) < 10 or len(att_states) < 10:
            continue

        nom_array = np.array(nom_states[:100])
        att_array = np.array(att_states[:100])

        probe = optimizer.optimize_probe(regime, nom_array, att_array)
        probes[regime.name] = [probe]

    return ProbeLibrary(
        probes=probes,
        version=version,
        created_at=datetime.now().isoformat(),
    )


# =============================================================================
# Integrated Adaptive Probing System
# =============================================================================

class AdaptiveProbingSystem:
    """
    Full adaptive probing system combining scheduler and library.

    Workflow:
    1. Check if probing is allowed (scheduler)
    2. Select optimal probe for current regime
    3. Execute probe and analyze response
    4. Update scheduler state
    """

    def __init__(
        self,
        scheduler_config: Optional[ProbeSchedulerConfig] = None,
        probe_library: Optional[ProbeLibrary] = None,
    ):
        self.scheduler = ProbeScheduler(scheduler_config)
        self.library = probe_library or create_standard_probe_library()

        self._current_probe: Optional[ProbeWaveform] = None
        self._probe_index = 0
        self._in_probe = False

    def update(
        self,
        velocity: np.ndarray,
        angular_rate: np.ndarray,
        attitude: np.ndarray,
        power_fraction: float,
        altitude_agl: float,
        acceleration: Optional[np.ndarray] = None,
    ) -> Tuple[float, bool]:
        """
        Update probing system and get current excitation.

        Args:
            velocity, angular_rate, attitude, power_fraction, altitude_agl

        Returns:
            (excitation_value, is_probing_active)
        """
        # If currently in a probe, continue it
        if self._in_probe and self._current_probe is not None:
            if self._probe_index < self._current_probe.duration:
                value = self._current_probe.waveform[self._probe_index]
                self._probe_index += 1
                return value, True
            else:
                # Probe complete
                self._in_probe = False
                self._current_probe = None
                self._probe_index = 0

        # Check if we should start a new probe
        decision = self.scheduler.decide(
            velocity, angular_rate, attitude,
            power_fraction, altitude_agl, acceleration
        )

        if decision.allow_probe:
            # Get best probe for current regime
            probe = self.library.get_best_probe(decision.regime)
            if probe is not None:
                self._current_probe = probe
                self._probe_index = 1
                self._in_probe = True
                self.scheduler.record_probe(decision.regime)
                return probe.waveform[0], True

        return 0.0, False

    def get_current_regime(self) -> Optional[FlightRegime]:
        """Get the regime of current probe."""
        if self._current_probe:
            return self._current_probe.regime
        return None

    def reset(self):
        """Reset system state."""
        self.scheduler.reset()
        self._current_probe = None
        self._probe_index = 0
        self._in_probe = False


def evaluate_adaptive_probing(
    nominal_trajectories: np.ndarray,
    attack_trajectories: np.ndarray,
) -> Dict:
    """
    Evaluate adaptive probing system.

    Args:
        nominal_trajectories: [N, T, 12] nominal data
        attack_trajectories: [M, T, 12] attack data

    Returns:
        Evaluation metrics
    """
    from sklearn.metrics import roc_auc_score

    system = AdaptiveProbingSystem()

    # Simulate probing on both sets
    nominal_detections = 0
    attack_detections = 0
    total_probes_nominal = 0
    total_probes_attack = 0

    # This is simplified - real evaluation would compare responses
    for traj in nominal_trajectories:
        for t in range(len(traj)):
            state = traj[t]
            exc, active = system.update(
                velocity=state[3:6],
                angular_rate=state[9:12],
                attitude=state[6:9],
                power_fraction=0.5,
                altitude_agl=10.0,
            )
            if active:
                total_probes_nominal += 1
        system.reset()

    for traj in attack_trajectories:
        for t in range(len(traj)):
            state = traj[t]
            exc, active = system.update(
                velocity=state[3:6],
                angular_rate=state[9:12],
                attitude=state[6:9],
                power_fraction=0.5,
                altitude_agl=10.0,
            )
            if active:
                total_probes_attack += 1
        system.reset()

    return {
        'total_probes_nominal': total_probes_nominal,
        'total_probes_attack': total_probes_attack,
        'probes_per_trajectory_nominal': total_probes_nominal / len(nominal_trajectories),
        'probes_per_trajectory_attack': total_probes_attack / len(attack_trajectories),
    }
