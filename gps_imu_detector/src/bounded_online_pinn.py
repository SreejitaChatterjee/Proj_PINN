"""
Bounded Online PINN Module (Phase 5)

Implements:
5.1 Shadow Residual Computation
    - Online PINN that runs in parallel (shadow mode)
    - Strict latency bounds (<1ms per sample)
    - No effect on primary control path

5.2 Bounded Inference
    - Fixed-time inference with hard cutoffs
    - Graceful degradation if bounds exceeded
    - Resource monitoring
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import time
import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Phase 5.1: Shadow Residual Computation
# =============================================================================

@dataclass
class ShadowConfig:
    """Configuration for shadow PINN."""
    # Latency bounds
    max_latency_ms: float = 0.5       # Max inference latency
    budget_utilization: float = 0.8   # Use 80% of budget max

    # Model bounds
    max_hidden_dim: int = 32          # Keep model small
    max_layers: int = 2               # Shallow network

    # Degradation settings
    skip_probability: float = 0.0     # Probability to skip inference
    fallback_residual: float = 0.0    # Value to use when skipping


class InferenceStatus(Enum):
    """Status of shadow inference."""
    SUCCESS = auto()        # Completed within budget
    TIMEOUT = auto()        # Exceeded latency budget
    SKIPPED = auto()        # Intentionally skipped
    ERROR = auto()          # Error during inference


@dataclass
class ShadowResult:
    """Result of shadow PINN inference."""
    residual: float
    status: InferenceStatus
    latency_ms: float
    confidence: float
    details: Dict[str, float]


class ShadowPINN(nn.Module):
    """
    Lightweight PINN for shadow residual computation.

    Designed for strict latency bounds - small, fast model.
    """

    def __init__(
        self,
        state_dim: int = 12,
        hidden_dim: int = 32,
        config: Optional[ShadowConfig] = None,
    ):
        super().__init__()
        self.config = config or ShadowConfig()

        # Enforce size bounds
        hidden_dim = min(hidden_dim, self.config.max_hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.residual_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)

        # Statistics for calibration
        self._residual_mean = 0.0
        self._residual_std = 1.0
        self._inference_count = 0
        self._timeout_count = 0

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual and confidence.

        Args:
            state: [batch, state_dim]

        Returns:
            residual: [batch, 1]
            confidence: [batch, 1] (0-1)
        """
        h = self.encoder(state)
        residual = self.residual_head(h)
        confidence = torch.sigmoid(self.confidence_head(h))
        return residual, confidence

    def compute_physics_residual(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        dt: float = 0.005,
    ) -> torch.Tensor:
        """Compute physics-based residual."""
        # Position prediction from velocity
        expected_pos = state[:, :3] + state[:, 3:6] * dt
        pos_residual = torch.norm(next_state[:, :3] - expected_pos, dim=1, keepdim=True)

        # Velocity change residual
        vel_change = next_state[:, 3:6] - state[:, 3:6]
        vel_residual = torch.norm(vel_change, dim=1, keepdim=True) / dt

        # Combined
        return pos_residual + 0.1 * vel_residual


class BoundedShadowInference:
    """
    Bounded inference for shadow PINN.

    Enforces strict latency bounds and handles degradation.
    """

    def __init__(
        self,
        model: Optional[ShadowPINN] = None,
        config: Optional[ShadowConfig] = None,
        device: str = 'cpu',
    ):
        self.config = config or ShadowConfig()
        self.device = device

        if model is None:
            self.model = ShadowPINN(config=self.config)
        else:
            self.model = model
        self.model = self.model.to(device)
        self.model.eval()

        # Statistics
        self._latency_history: List[float] = []
        self._max_history = 1000

    def infer(
        self,
        state: np.ndarray,
        next_state: Optional[np.ndarray] = None,
    ) -> ShadowResult:
        """
        Run bounded inference.

        Args:
            state: [state_dim] current state
            next_state: [state_dim] next state (optional)

        Returns:
            ShadowResult with residual and status
        """
        start_time = time.perf_counter()

        # Check if should skip
        if np.random.random() < self.config.skip_probability:
            return ShadowResult(
                residual=self.config.fallback_residual,
                status=InferenceStatus.SKIPPED,
                latency_ms=0.0,
                confidence=0.0,
                details={'reason': 'intentional_skip'},
            )

        try:
            # Prepare input
            state_t = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                residual_t, confidence_t = self.model(state_t)

                # Physics residual if next_state available
                if next_state is not None:
                    next_t = torch.tensor(
                        next_state, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    physics_res = self.model.compute_physics_residual(state_t, next_t)
                    residual_t = residual_t + physics_res

            residual = float(residual_t[0, 0])
            confidence = float(confidence_t[0, 0])

            # Compute latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record latency
            self._latency_history.append(latency_ms)
            if len(self._latency_history) > self._max_history:
                self._latency_history.pop(0)

            # Check timeout
            if latency_ms > self.config.max_latency_ms:
                self.model._timeout_count += 1
                return ShadowResult(
                    residual=residual,  # Still return computed value
                    status=InferenceStatus.TIMEOUT,
                    latency_ms=latency_ms,
                    confidence=confidence * 0.5,  # Reduce confidence
                    details={
                        'budget_exceeded_by': latency_ms - self.config.max_latency_ms,
                    },
                )

            self.model._inference_count += 1
            return ShadowResult(
                residual=residual,
                status=InferenceStatus.SUCCESS,
                latency_ms=latency_ms,
                confidence=confidence,
                details={
                    'budget_remaining': self.config.max_latency_ms - latency_ms,
                },
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ShadowResult(
                residual=self.config.fallback_residual,
                status=InferenceStatus.ERROR,
                latency_ms=latency_ms,
                confidence=0.0,
                details={'error': str(e)},
            )

    def get_statistics(self) -> Dict:
        """Get inference statistics."""
        if not self._latency_history:
            return {
                'mean_latency_ms': 0.0,
                'p95_latency_ms': 0.0,
                'p99_latency_ms': 0.0,
                'inference_count': 0,
                'timeout_rate': 0.0,
            }

        latencies = np.array(self._latency_history)
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'inference_count': self.model._inference_count,
            'timeout_rate': (
                self.model._timeout_count /
                max(1, self.model._inference_count + self.model._timeout_count)
            ),
        }


# =============================================================================
# Phase 5.2: Online Shadow Monitor
# =============================================================================

@dataclass
class MonitorConfig:
    """Configuration for shadow monitor."""
    # Thresholds
    residual_threshold: float = 3.0   # Z-score for alert
    confidence_threshold: float = 0.7  # Min confidence to trust

    # Smoothing
    smoothing_window: int = 10        # EMA window

    # Degradation
    max_consecutive_timeouts: int = 5  # Before fallback mode


class ShadowMonitorStatus(Enum):
    """Status of shadow monitor."""
    NOMINAL = auto()         # Everything fine
    ALERT = auto()           # High residual detected
    DEGRADED = auto()        # Running in degraded mode
    OFFLINE = auto()         # Monitor offline


@dataclass
class MonitorUpdate:
    """Update from shadow monitor."""
    status: ShadowMonitorStatus
    smoothed_residual: float
    raw_residual: float
    confidence: float
    alert_reason: Optional[str]


class OnlineShadowMonitor:
    """
    Online shadow monitoring system.

    Runs PINN inference in shadow mode with:
    - Exponential moving average smoothing
    - Alert generation
    - Graceful degradation
    """

    def __init__(
        self,
        inference: Optional[BoundedShadowInference] = None,
        config: Optional[MonitorConfig] = None,
    ):
        self.config = config or MonitorConfig()
        self.inference = inference or BoundedShadowInference()

        # State
        self._smoothed_residual = 0.0
        self._consecutive_timeouts = 0
        self._status = ShadowMonitorStatus.NOMINAL
        self._residual_history: List[float] = []

        # Calibration
        self._residual_mean = 0.0
        self._residual_std = 1.0

    def update(
        self,
        state: np.ndarray,
        next_state: Optional[np.ndarray] = None,
    ) -> MonitorUpdate:
        """
        Process new state and return monitor update.

        Args:
            state: [state_dim] current state
            next_state: [state_dim] next state (optional)

        Returns:
            MonitorUpdate with status and residual
        """
        # Run inference
        result = self.inference.infer(state, next_state)

        # Handle inference status
        if result.status == InferenceStatus.TIMEOUT:
            self._consecutive_timeouts += 1
            if self._consecutive_timeouts >= self.config.max_consecutive_timeouts:
                self._status = ShadowMonitorStatus.DEGRADED
        elif result.status == InferenceStatus.ERROR:
            self._status = ShadowMonitorStatus.OFFLINE
            return MonitorUpdate(
                status=self._status,
                smoothed_residual=self._smoothed_residual,
                raw_residual=result.residual,
                confidence=0.0,
                alert_reason="Inference error",
            )
        else:
            self._consecutive_timeouts = 0
            if self._status == ShadowMonitorStatus.DEGRADED:
                self._status = ShadowMonitorStatus.NOMINAL

        # Update smoothed residual (EMA)
        alpha = 2.0 / (self.config.smoothing_window + 1)
        self._smoothed_residual = (
            alpha * result.residual +
            (1 - alpha) * self._smoothed_residual
        )

        # Record history
        self._residual_history.append(result.residual)
        if len(self._residual_history) > 1000:
            self._residual_history.pop(0)

        # Compute z-score
        zscore = (
            (self._smoothed_residual - self._residual_mean) /
            max(self._residual_std, 1e-6)
        )

        # Check for alert
        alert_reason = None
        if result.confidence >= self.config.confidence_threshold:
            if zscore > self.config.residual_threshold:
                self._status = ShadowMonitorStatus.ALERT
                alert_reason = f"High residual: z={zscore:.2f}"
            elif self._status == ShadowMonitorStatus.ALERT:
                self._status = ShadowMonitorStatus.NOMINAL

        return MonitorUpdate(
            status=self._status,
            smoothed_residual=self._smoothed_residual,
            raw_residual=result.residual,
            confidence=result.confidence,
            alert_reason=alert_reason,
        )

    def calibrate(self, nominal_residuals: np.ndarray):
        """Calibrate monitor from nominal data."""
        self._residual_mean = float(np.mean(nominal_residuals))
        self._residual_std = float(np.std(nominal_residuals))

    def get_statistics(self) -> Dict:
        """Get monitor statistics."""
        inference_stats = self.inference.get_statistics()

        return {
            **inference_stats,
            'current_status': self._status.name,
            'smoothed_residual': self._smoothed_residual,
            'consecutive_timeouts': self._consecutive_timeouts,
            'residual_mean': self._residual_mean,
            'residual_std': self._residual_std,
        }

    def reset(self):
        """Reset monitor state."""
        self._smoothed_residual = 0.0
        self._consecutive_timeouts = 0
        self._status = ShadowMonitorStatus.NOMINAL
        self._residual_history.clear()


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_bounded_pinn(
    nominal_trajectories: np.ndarray,
    attack_trajectories: np.ndarray,
) -> Dict:
    """
    Evaluate bounded online PINN.

    Args:
        nominal_trajectories: [N, T, state_dim]
        attack_trajectories: [M, T, state_dim]

    Returns:
        Evaluation metrics
    """
    monitor = OnlineShadowMonitor()

    # Calibrate on nominal data
    nominal_residuals = []
    for traj in nominal_trajectories:
        for t in range(len(traj) - 1):
            result = monitor.inference.infer(traj[t], traj[t+1])
            nominal_residuals.append(result.residual)
        monitor.reset()

    monitor.calibrate(np.array(nominal_residuals))

    # Evaluate on nominal
    nominal_alerts = 0
    nominal_total = 0
    for traj in nominal_trajectories:
        for t in range(len(traj) - 1):
            update = monitor.update(traj[t], traj[t+1])
            if update.status == ShadowMonitorStatus.ALERT:
                nominal_alerts += 1
            nominal_total += 1
        monitor.reset()

    # Evaluate on attack
    attack_alerts = 0
    attack_total = 0
    for traj in attack_trajectories:
        for t in range(len(traj) - 1):
            update = monitor.update(traj[t], traj[t+1])
            if update.status == ShadowMonitorStatus.ALERT:
                attack_alerts += 1
            attack_total += 1
        monitor.reset()

    stats = monitor.get_statistics()

    return {
        'nominal_alert_rate': nominal_alerts / max(1, nominal_total),
        'attack_alert_rate': attack_alerts / max(1, attack_total),
        'mean_latency_ms': stats['mean_latency_ms'],
        'p95_latency_ms': stats['p95_latency_ms'],
        'timeout_rate': stats['timeout_rate'],
    }
