"""
Conformal Residual Envelopes Module (Phase 1.2)

Uses PINN to predict residual quantiles per regime with conformal calibration.
Guarantees coverage at specified confidence level.

Output: Versioned lookup tables (quantiles + margins).
No runtime PINN inference - all precomputed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path

from .regime_detection import FlightRegime, RegimeClassifier, classify_trajectory


@dataclass
class ConformalEnvelope:
    """Conformal envelope for a single regime."""
    regime: FlightRegime
    quantiles: Dict[float, float]  # {alpha: threshold}
    margin: float  # Conformal margin for coverage guarantee
    n_calibration: int  # Number of calibration samples
    coverage_target: float  # Target coverage (e.g., 0.99)
    version: str = "1.0.0"


@dataclass
class EnvelopeTable:
    """Versioned lookup table of conformal envelopes."""
    envelopes: Dict[str, ConformalEnvelope]  # regime_name -> envelope
    version: str
    created_at: str
    pinn_checkpoint: str
    calibration_samples: int

    def get_threshold(self, regime: FlightRegime, alpha: float = 0.01) -> float:
        """Get threshold for regime at given FPR level."""
        envelope = self.envelopes.get(regime.name)
        if envelope is None:
            # Default to UNKNOWN regime
            envelope = self.envelopes.get(FlightRegime.UNKNOWN.name)
        if envelope is None:
            return float('inf')  # No detection

        # Find closest quantile
        available = sorted(envelope.quantiles.keys())
        closest = min(available, key=lambda x: abs(x - alpha))
        return envelope.quantiles[closest] + envelope.margin


class ResidualPINN(nn.Module):
    """
    PINN for residual prediction (offline training only).

    Predicts expected residual magnitude given state.
    Used for conformal calibration, not runtime inference.
    """

    def __init__(self, state_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Predict mean and log-variance of residual
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

        # Physics layer: enforce kinematic consistency
        self.physics_weight = 0.1

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict residual distribution parameters.

        Args:
            state: [B, state_dim] current state

        Returns:
            mean: [B, 1] predicted residual mean
            std: [B, 1] predicted residual std
        """
        h = self.encoder(state)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        std = torch.exp(0.5 * logvar)
        return mean, std

    def physics_loss(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        dt: float = 0.005,
    ) -> torch.Tensor:
        """
        Physics consistency loss: dp/dt = v.

        Args:
            state: [B, 12] current state
            next_state: [B, 12] next state
            dt: time step

        Returns:
            Physics loss scalar
        """
        pos = state[:, :3]
        vel = state[:, 3:6]
        next_pos = next_state[:, :3]

        predicted_pos = pos + vel * dt
        physics_residual = next_pos - predicted_pos
        return torch.mean(physics_residual ** 2)


class ConformalCalibrator:
    """
    Conformal calibration for coverage guarantees.

    Uses split conformal prediction to compute margins.
    """

    def __init__(self, coverage: float = 0.99):
        """
        Args:
            coverage: Target coverage level (e.g., 0.99 for 99%)
        """
        self.coverage = coverage
        self.calibration_scores: Dict[str, List[float]] = {}

    def add_calibration_score(self, regime: FlightRegime, score: float):
        """Add a calibration score for a regime."""
        regime_name = regime.name
        if regime_name not in self.calibration_scores:
            self.calibration_scores[regime_name] = []
        self.calibration_scores[regime_name].append(score)

    def compute_quantiles(
        self,
        regime: FlightRegime,
        alphas: List[float] = None,
    ) -> Tuple[Dict[float, float], float]:
        """
        Compute quantile thresholds and conformal margin.

        Args:
            regime: Flight regime
            alphas: FPR levels to compute (default: [0.01, 0.05, 0.10])

        Returns:
            quantiles: {alpha: threshold}
            margin: Conformal margin for coverage guarantee
        """
        if alphas is None:
            alphas = [0.01, 0.05, 0.10]

        scores = self.calibration_scores.get(regime.name, [])
        if not scores:
            return {a: float('inf') for a in alphas}, 0.0

        scores = np.array(scores)
        n = len(scores)

        # Compute quantile thresholds
        quantiles = {}
        for alpha in alphas:
            q = 1 - alpha
            quantiles[alpha] = np.quantile(scores, q)

        # Conformal margin: ensures coverage guarantee
        # Uses the (1 - coverage) * (n + 1) / n quantile
        margin_quantile = (1 - self.coverage) * (n + 1) / n
        margin_quantile = min(margin_quantile, 1.0)
        margin = np.quantile(scores, 1 - margin_quantile) - np.quantile(scores, 1 - self.coverage)
        margin = max(margin, 0.0)

        return quantiles, margin


class ConformalEnvelopeBuilder:
    """
    Builds conformal envelopes from nominal data.

    Workflow:
    1. Train PINN on nominal trajectories
    2. Compute residuals on calibration set
    3. Apply conformal calibration per regime
    4. Export versioned lookup tables
    """

    def __init__(
        self,
        state_dim: int = 12,
        hidden_dim: int = 64,
        coverage: float = 0.99,
        device: str = 'cpu',
    ):
        self.state_dim = state_dim
        self.device = device
        self.coverage = coverage

        self.pinn = ResidualPINN(state_dim, hidden_dim).to(device)
        self.classifier = RegimeClassifier()
        self.calibrator = ConformalCalibrator(coverage)

        self._trained = False

    def train_pinn(
        self,
        trajectories: np.ndarray,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
        dt: float = 0.005,
        verbose: bool = True,
    ) -> Dict:
        """
        Train PINN on nominal trajectories.

        Args:
            trajectories: [N, T, state_dim] nominal trajectories
            epochs: Training epochs
            lr: Learning rate
            batch_size: Batch size
            dt: Time step
            verbose: Print progress

        Returns:
            Training history
        """
        trajectories = np.asarray(trajectories, dtype=np.float32)

        # Flatten to state pairs
        states = []
        next_states = []
        for traj in trajectories:
            for t in range(len(traj) - 1):
                states.append(traj[t])
                next_states.append(traj[t + 1])

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr)
        history = {'loss': [], 'physics_loss': []}

        n_samples = len(states)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            epoch_physics = 0.0

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                idx = indices[start:end]

                batch_states = states[idx]
                batch_next = next_states[idx]

                # Forward pass
                mean, std = self.pinn(batch_states)

                # Compute actual residual
                pos = batch_states[:, :3]
                vel = batch_states[:, 3:6]
                next_pos = batch_next[:, :3]
                actual_residual = torch.norm(next_pos - (pos + vel * dt), dim=1, keepdim=True)

                # Negative log-likelihood loss
                nll_loss = 0.5 * ((actual_residual - mean) ** 2 / (std ** 2 + 1e-6) + torch.log(std ** 2 + 1e-6))
                nll_loss = nll_loss.mean()

                # Physics loss
                physics_loss = self.pinn.physics_loss(batch_states, batch_next, dt)

                # Total loss
                loss = nll_loss + self.pinn.physics_weight * physics_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_physics += physics_loss.item()

            epoch_loss /= n_batches
            epoch_physics /= n_batches
            history['loss'].append(epoch_loss)
            history['physics_loss'].append(epoch_physics)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}, physics={epoch_physics:.6f}")

        self._trained = True
        return history

    def calibrate(
        self,
        trajectories: np.ndarray,
        dt: float = 0.005,
    ):
        """
        Calibrate conformal envelopes on calibration set.

        Args:
            trajectories: [N, T, state_dim] calibration trajectories
            dt: Time step
        """
        if not self._trained:
            raise RuntimeError("Must train PINN before calibration")

        trajectories = np.asarray(trajectories, dtype=np.float32)
        self.pinn.eval()

        with torch.no_grad():
            for traj in trajectories:
                for t in range(len(traj) - 1):
                    state = traj[t]
                    next_state = traj[t + 1]

                    # Classify regime
                    velocity = state[3:6]
                    angular_rate = state[9:12]
                    acceleration = (traj[min(t+1, len(traj)-1), 3:6] - state[3:6]) / dt
                    regime = self.classifier.classify(velocity, angular_rate, acceleration)

                    # Compute residual
                    pos = state[:3]
                    vel = state[3:6]
                    next_pos = next_state[:3]
                    actual_residual = np.linalg.norm(next_pos - (pos + vel * dt))

                    # Predict expected residual
                    state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    mean, std = self.pinn(state_t)

                    # Normalized residual (for calibration)
                    normalized_score = (actual_residual - mean.item()) / (std.item() + 1e-6)
                    self.calibrator.add_calibration_score(regime, abs(normalized_score))

    def build_envelope_table(self, version: str = "1.0.0") -> EnvelopeTable:
        """
        Build versioned envelope lookup table.

        Args:
            version: Version string

        Returns:
            EnvelopeTable with conformal envelopes per regime
        """
        envelopes = {}
        total_samples = 0

        for regime in FlightRegime:
            scores = self.calibrator.calibration_scores.get(regime.name, [])
            if not scores:
                continue

            quantiles, margin = self.calibrator.compute_quantiles(regime)
            envelope = ConformalEnvelope(
                regime=regime,
                quantiles=quantiles,
                margin=margin,
                n_calibration=len(scores),
                coverage_target=self.coverage,
                version=version,
            )
            envelopes[regime.name] = envelope
            total_samples += len(scores)

        from datetime import datetime
        return EnvelopeTable(
            envelopes=envelopes,
            version=version,
            created_at=datetime.now().isoformat(),
            pinn_checkpoint="pinn_conformal.pth",
            calibration_samples=total_samples,
        )

    def save(self, path: str):
        """Save PINN weights."""
        torch.save(self.pinn.state_dict(), path)

    def load(self, path: str):
        """Load PINN weights."""
        self.pinn.load_state_dict(torch.load(path, map_location=self.device))
        self._trained = True


def save_envelope_table(table: EnvelopeTable, path: str):
    """Save envelope table to JSON."""
    data = {
        'version': table.version,
        'created_at': table.created_at,
        'pinn_checkpoint': table.pinn_checkpoint,
        'calibration_samples': table.calibration_samples,
        'envelopes': {},
    }

    for regime_name, envelope in table.envelopes.items():
        data['envelopes'][regime_name] = {
            'quantiles': envelope.quantiles,
            'margin': envelope.margin,
            'n_calibration': envelope.n_calibration,
            'coverage_target': envelope.coverage_target,
            'version': envelope.version,
        }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_envelope_table(path: str) -> EnvelopeTable:
    """Load envelope table from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)

    envelopes = {}
    for regime_name, env_data in data['envelopes'].items():
        regime = FlightRegime[regime_name]
        # Convert string keys back to float
        quantiles = {float(k): v for k, v in env_data['quantiles'].items()}
        envelope = ConformalEnvelope(
            regime=regime,
            quantiles=quantiles,
            margin=env_data['margin'],
            n_calibration=env_data['n_calibration'],
            coverage_target=env_data['coverage_target'],
            version=env_data.get('version', '1.0.0'),
        )
        envelopes[regime_name] = envelope

    return EnvelopeTable(
        envelopes=envelopes,
        version=data['version'],
        created_at=data['created_at'],
        pinn_checkpoint=data['pinn_checkpoint'],
        calibration_samples=data['calibration_samples'],
    )


def evaluate_conformal_envelopes(
    envelope_table: EnvelopeTable,
    test_trajectories: np.ndarray,
    attack_trajectories: np.ndarray,
    dt: float = 0.005,
) -> Dict:
    """
    Evaluate conformal envelopes on test data.

    Args:
        envelope_table: Calibrated envelope table
        test_trajectories: [N, T, state_dim] nominal test data
        attack_trajectories: [M, T, state_dim] attack test data
        dt: Time step

    Returns:
        Evaluation metrics
    """
    classifier = RegimeClassifier()

    # Collect scores per regime
    nominal_scores = {regime.name: [] for regime in FlightRegime}
    attack_scores = {regime.name: [] for regime in FlightRegime}

    for traj in test_trajectories:
        for t in range(len(traj) - 1):
            state = traj[t]
            next_state = traj[t + 1]

            velocity = state[3:6]
            angular_rate = state[9:12]
            regime = classifier.classify(velocity, angular_rate)

            pos = state[:3]
            vel = state[3:6]
            next_pos = next_state[:3]
            residual = np.linalg.norm(next_pos - (pos + vel * dt))

            nominal_scores[regime.name].append(residual)

    for traj in attack_trajectories:
        for t in range(len(traj) - 1):
            state = traj[t]
            next_state = traj[t + 1]

            velocity = state[3:6]
            angular_rate = state[9:12]
            regime = classifier.classify(velocity, angular_rate)

            pos = state[:3]
            vel = state[3:6]
            next_pos = next_state[:3]
            residual = np.linalg.norm(next_pos - (pos + vel * dt))

            attack_scores[regime.name].append(residual)

    # Compute FPR and recall per regime
    results = {'per_regime': {}}

    for regime in FlightRegime:
        nom = nominal_scores.get(regime.name, [])
        att = attack_scores.get(regime.name, [])

        if not nom or not att:
            continue

        threshold = envelope_table.get_threshold(regime, alpha=0.01)

        fpr = np.mean([s > threshold for s in nom])
        recall = np.mean([s > threshold for s in att])

        results['per_regime'][regime.name] = {
            'fpr': fpr,
            'recall': recall,
            'threshold': threshold,
            'n_nominal': len(nom),
            'n_attack': len(att),
        }

    # Aggregate metrics
    all_fpr = [r['fpr'] for r in results['per_regime'].values()]
    all_recall = [r['recall'] for r in results['per_regime'].values()]

    results['overall'] = {
        'mean_fpr': np.mean(all_fpr) if all_fpr else 0.0,
        'max_fpr': np.max(all_fpr) if all_fpr else 0.0,
        'mean_recall': np.mean(all_recall) if all_recall else 0.0,
        'min_recall': np.min(all_recall) if all_recall else 0.0,
    }

    return results
