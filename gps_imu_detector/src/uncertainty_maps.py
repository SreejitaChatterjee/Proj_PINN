"""
Uncertainty Maps Module (Phase 1.3)

Extends offline PINN to estimate regime-conditioned uncertainty.
Converts to static uncertainty map for runtime use.

Policy: Widen envelopes or defer probing when uncertainty is high.
No runtime PINN inference - all precomputed lookup tables.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path

from .regime_detection import FlightRegime, RegimeClassifier


@dataclass
class UncertaintyCell:
    """Single cell in uncertainty map."""
    regime: FlightRegime
    velocity_bin: Tuple[float, float]  # (min, max) m/s
    angular_rate_bin: Tuple[float, float]  # (min, max) rad/s
    mean_uncertainty: float
    p90_uncertainty: float
    n_samples: int
    envelope_multiplier: float  # How much to widen envelope
    abstain_probing: bool  # Whether to defer probing


@dataclass
class UncertaintyMap:
    """
    Static uncertainty map for runtime lookup.

    Discretizes state space into bins with precomputed uncertainty.
    """
    cells: Dict[str, UncertaintyCell]  # key = f"{regime}_{vel_bin}_{ang_bin}"
    version: str
    created_at: str
    velocity_bins: List[float]
    angular_rate_bins: List[float]

    def lookup(
        self,
        regime: FlightRegime,
        velocity_norm: float,
        angular_rate_norm: float,
    ) -> Optional[UncertaintyCell]:
        """
        Look up uncertainty for given state.

        Args:
            regime: Current flight regime
            velocity_norm: Velocity magnitude (m/s)
            angular_rate_norm: Angular rate magnitude (rad/s)

        Returns:
            UncertaintyCell or None if not found
        """
        vel_bin = self._find_bin(velocity_norm, self.velocity_bins)
        ang_bin = self._find_bin(angular_rate_norm, self.angular_rate_bins)

        key = f"{regime.name}_{vel_bin}_{ang_bin}"
        return self.cells.get(key)

    def _find_bin(self, value: float, bins: List[float]) -> int:
        """Find bin index for value."""
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                return i
        return len(bins) - 2  # Last bin

    def get_envelope_multiplier(
        self,
        regime: FlightRegime,
        velocity_norm: float,
        angular_rate_norm: float,
    ) -> float:
        """Get envelope widening multiplier."""
        cell = self.lookup(regime, velocity_norm, angular_rate_norm)
        if cell is None:
            return 1.5  # Conservative default
        return cell.envelope_multiplier

    def should_abstain_probing(
        self,
        regime: FlightRegime,
        velocity_norm: float,
        angular_rate_norm: float,
    ) -> bool:
        """Check if probing should be deferred."""
        cell = self.lookup(regime, velocity_norm, angular_rate_norm)
        if cell is None:
            return True  # Conservative default
        return cell.abstain_probing


class UncertaintyPINN(nn.Module):
    """
    PINN for uncertainty estimation (offline training only).

    Predicts epistemic uncertainty given state.
    Uses ensemble or MC dropout for uncertainty quantification.
    """

    def __init__(
        self,
        state_dim: int = 12,
        hidden_dim: int = 64,
        n_heads: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_heads = n_heads

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Multiple prediction heads (for ensemble uncertainty)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_heads)
        ])

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict residual with uncertainty.

        Args:
            state: [B, state_dim] current state

        Returns:
            mean: [B, 1] mean prediction
            uncertainty: [B, 1] epistemic uncertainty (std of heads)
        """
        h = self.encoder(state)

        predictions = torch.stack([head(h) for head in self.heads], dim=-1)
        mean = predictions.mean(dim=-1)
        uncertainty = predictions.std(dim=-1)

        return mean, uncertainty

    def mc_uncertainty(
        self,
        state: torch.Tensor,
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty via MC dropout.

        Args:
            state: [B, state_dim] current state
            n_samples: Number of MC samples

        Returns:
            mean: [B, 1] mean prediction
            uncertainty: [B, 1] MC uncertainty
        """
        self.train()  # Enable dropout
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                h = self.encoder(state)
                pred = self.heads[0](h)  # Use first head
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=-1)
        mean = predictions.mean(dim=-1)
        uncertainty = predictions.std(dim=-1)

        self.eval()
        return mean, uncertainty


class UncertaintyMapBuilder:
    """
    Builds static uncertainty maps from nominal data.

    Workflow:
    1. Train uncertainty PINN on nominal trajectories
    2. Discretize state space into bins
    3. Compute uncertainty statistics per bin
    4. Derive envelope multipliers and abstention policy
    5. Export versioned lookup tables
    """

    def __init__(
        self,
        state_dim: int = 12,
        hidden_dim: int = 64,
        velocity_bins: List[float] = None,
        angular_rate_bins: List[float] = None,
        device: str = 'cpu',
    ):
        self.state_dim = state_dim
        self.device = device

        self.velocity_bins = velocity_bins or [0, 0.5, 2.0, 5.0, 10.0, float('inf')]
        self.angular_rate_bins = angular_rate_bins or [0, 0.1, 0.5, 1.0, 2.0, float('inf')]

        self.pinn = UncertaintyPINN(state_dim, hidden_dim).to(device)
        self.classifier = RegimeClassifier()

        self._trained = False
        self._bin_data: Dict[str, List[float]] = {}

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
        Train uncertainty PINN on nominal trajectories.

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
        residuals = []
        for traj in trajectories:
            for t in range(len(traj) - 1):
                state = traj[t]
                next_state = traj[t + 1]

                pos = state[:3]
                vel = state[3:6]
                next_pos = next_state[:3]
                residual = np.linalg.norm(next_pos - (pos + vel * dt))

                states.append(state)
                residuals.append(residual)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        residuals = torch.tensor(np.array(residuals), dtype=torch.float32, device=self.device).unsqueeze(1)

        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr)
        history = {'loss': []}

        n_samples = len(states)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                idx = indices[start:end]

                batch_states = states[idx]
                batch_residuals = residuals[idx]

                # Forward pass
                mean, uncertainty = self.pinn(batch_states)

                # MSE loss with uncertainty penalty
                mse_loss = torch.mean((batch_residuals - mean) ** 2)
                # Encourage calibrated uncertainty
                uncertainty_loss = torch.mean((torch.abs(batch_residuals - mean) - uncertainty) ** 2)

                loss = mse_loss + 0.1 * uncertainty_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= n_batches
            history['loss'].append(epoch_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}")

        self._trained = True
        return history

    def collect_bin_statistics(
        self,
        trajectories: np.ndarray,
        dt: float = 0.005,
    ):
        """
        Collect uncertainty statistics per bin.

        Args:
            trajectories: [N, T, state_dim] calibration trajectories
            dt: Time step
        """
        if not self._trained:
            raise RuntimeError("Must train PINN first")

        trajectories = np.asarray(trajectories, dtype=np.float32)
        self.pinn.eval()

        with torch.no_grad():
            for traj in trajectories:
                for t in range(len(traj) - 1):
                    state = traj[t]

                    velocity = state[3:6]
                    angular_rate = state[9:12]
                    vel_norm = np.linalg.norm(velocity)
                    ang_norm = np.linalg.norm(angular_rate)

                    regime = self.classifier.classify(velocity, angular_rate)

                    # Get uncertainty
                    state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    _, uncertainty = self.pinn(state_t)

                    # Find bin
                    vel_bin = self._find_bin(vel_norm, self.velocity_bins)
                    ang_bin = self._find_bin(ang_norm, self.angular_rate_bins)

                    key = f"{regime.name}_{vel_bin}_{ang_bin}"
                    if key not in self._bin_data:
                        self._bin_data[key] = []
                    self._bin_data[key].append(uncertainty.item())

    def _find_bin(self, value: float, bins: List[float]) -> int:
        """Find bin index for value."""
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                return i
        return len(bins) - 2

    def build_uncertainty_map(
        self,
        version: str = "1.0.0",
        high_uncertainty_threshold: float = 0.5,
        abstain_threshold: float = 0.8,
    ) -> UncertaintyMap:
        """
        Build versioned uncertainty map.

        Args:
            version: Version string
            high_uncertainty_threshold: Threshold for widening envelopes
            abstain_threshold: Threshold for abstaining from probing

        Returns:
            UncertaintyMap with precomputed cells
        """
        cells = {}

        for key, uncertainties in self._bin_data.items():
            if not uncertainties:
                continue

            # Parse key
            parts = key.split('_')
            regime = FlightRegime[parts[0]]
            vel_bin = int(parts[1])
            ang_bin = int(parts[2])

            mean_unc = np.mean(uncertainties)
            p90_unc = np.percentile(uncertainties, 90)

            # Derive envelope multiplier (higher uncertainty = wider envelope)
            if mean_unc > high_uncertainty_threshold:
                envelope_mult = 1.0 + (mean_unc - high_uncertainty_threshold)
            else:
                envelope_mult = 1.0

            # Derive abstention policy
            abstain = p90_unc > abstain_threshold

            cell = UncertaintyCell(
                regime=regime,
                velocity_bin=(self.velocity_bins[vel_bin], self.velocity_bins[vel_bin + 1]),
                angular_rate_bin=(self.angular_rate_bins[ang_bin], self.angular_rate_bins[ang_bin + 1]),
                mean_uncertainty=mean_unc,
                p90_uncertainty=p90_unc,
                n_samples=len(uncertainties),
                envelope_multiplier=envelope_mult,
                abstain_probing=abstain,
            )
            cells[key] = cell

        from datetime import datetime
        return UncertaintyMap(
            cells=cells,
            version=version,
            created_at=datetime.now().isoformat(),
            velocity_bins=self.velocity_bins,
            angular_rate_bins=self.angular_rate_bins,
        )

    def save(self, path: str):
        """Save PINN weights."""
        torch.save(self.pinn.state_dict(), path)

    def load(self, path: str):
        """Load PINN weights."""
        self.pinn.load_state_dict(torch.load(path, map_location=self.device))
        self._trained = True


def save_uncertainty_map(umap: UncertaintyMap, path: str):
    """Save uncertainty map to JSON."""
    data = {
        'version': umap.version,
        'created_at': umap.created_at,
        'velocity_bins': umap.velocity_bins,
        'angular_rate_bins': umap.angular_rate_bins,
        'cells': {},
    }

    for key, cell in umap.cells.items():
        data['cells'][key] = {
            'regime': cell.regime.name,
            'velocity_bin': cell.velocity_bin,
            'angular_rate_bin': cell.angular_rate_bin,
            'mean_uncertainty': cell.mean_uncertainty,
            'p90_uncertainty': cell.p90_uncertainty,
            'n_samples': cell.n_samples,
            'envelope_multiplier': cell.envelope_multiplier,
            'abstain_probing': cell.abstain_probing,
        }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_uncertainty_map(path: str) -> UncertaintyMap:
    """Load uncertainty map from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)

    cells = {}
    for key, cell_data in data['cells'].items():
        cell = UncertaintyCell(
            regime=FlightRegime[cell_data['regime']],
            velocity_bin=tuple(cell_data['velocity_bin']),
            angular_rate_bin=tuple(cell_data['angular_rate_bin']),
            mean_uncertainty=cell_data['mean_uncertainty'],
            p90_uncertainty=cell_data['p90_uncertainty'],
            n_samples=cell_data['n_samples'],
            envelope_multiplier=cell_data['envelope_multiplier'],
            abstain_probing=cell_data['abstain_probing'],
        )
        cells[key] = cell

    return UncertaintyMap(
        cells=cells,
        version=data['version'],
        created_at=data['created_at'],
        velocity_bins=data['velocity_bins'],
        angular_rate_bins=data['angular_rate_bins'],
    )


class AbstentionPolicy:
    """
    Runtime abstention policy using uncertainty map.

    Decides when to:
    - Widen detection envelopes
    - Defer active probing
    - Flag for manual review
    """

    def __init__(self, uncertainty_map: UncertaintyMap):
        self.umap = uncertainty_map
        self.classifier = RegimeClassifier()

    def get_envelope_adjustment(
        self,
        velocity: np.ndarray,
        angular_rate: np.ndarray,
    ) -> float:
        """
        Get envelope widening factor.

        Args:
            velocity: [3] velocity vector
            angular_rate: [3] angular rate vector

        Returns:
            Multiplier for envelope thresholds
        """
        vel_norm = np.linalg.norm(velocity)
        ang_norm = np.linalg.norm(angular_rate)
        regime = self.classifier.classify(velocity, angular_rate)

        return self.umap.get_envelope_multiplier(regime, vel_norm, ang_norm)

    def should_probe(
        self,
        velocity: np.ndarray,
        angular_rate: np.ndarray,
    ) -> bool:
        """
        Decide if probing is allowed.

        Args:
            velocity: [3] velocity vector
            angular_rate: [3] angular rate vector

        Returns:
            True if probing is allowed
        """
        vel_norm = np.linalg.norm(velocity)
        ang_norm = np.linalg.norm(angular_rate)
        regime = self.classifier.classify(velocity, angular_rate)

        return not self.umap.should_abstain_probing(regime, vel_norm, ang_norm)

    def get_decision(
        self,
        velocity: np.ndarray,
        angular_rate: np.ndarray,
    ) -> Dict:
        """
        Get full abstention decision.

        Returns:
            Dict with all policy decisions
        """
        vel_norm = np.linalg.norm(velocity)
        ang_norm = np.linalg.norm(angular_rate)
        regime = self.classifier.classify(velocity, angular_rate)

        cell = self.umap.lookup(regime, vel_norm, ang_norm)

        return {
            'regime': regime.name,
            'envelope_multiplier': cell.envelope_multiplier if cell else 1.5,
            'allow_probing': not (cell.abstain_probing if cell else True),
            'mean_uncertainty': cell.mean_uncertainty if cell else float('inf'),
            'high_uncertainty': (cell.p90_uncertainty if cell else float('inf')) > 0.5,
        }
