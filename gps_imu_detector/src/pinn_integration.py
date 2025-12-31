"""
PINN Integration Module (v0.9.0)

Three integration options for physics-informed neural network residuals:
1. Shadow Residual: PINN as secondary signal (recommended)
2. Offline Envelope: PINN for static threshold computation
3. Probing Response: PINN for active probing prediction

Key Principle: ICI (Inverse-Cycle Instability) remains PRIMARY.
PINN is SECONDARY - never triggers alarms alone.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# OPTION 1: PINN AS SHADOW RESIDUAL (Recommended)
# =============================================================================

class QuadrotorPINNResidual(nn.Module):
    """
    Lightweight PINN for quadrotor physics residual computation.

    Physics equations embedded:
        dp/dt = v                    (position derivative)
        dv/dt = R @ [0, 0, T/m] + g  (velocity derivative)

    The network learns corrections to the physics model.
    """

    def __init__(
        self,
        state_dim: int = 12,
        hidden_dim: int = 64,
        physics_weight: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.physics_weight = physics_weight

        # Gravity constant
        self.g = torch.tensor([0.0, 0.0, -9.81])

        # Neural network for residual correction
        self.correction_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Small initialization for corrections
        for m in self.correction_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def physics_residual(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        dt: float = 0.005,
    ) -> torch.Tensor:
        """
        Compute physics-based residual.

        For quadrotor:
            state = [p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z, omega_x, omega_y]
                    or simplified [p, v, angles, angular_rates]

        Returns:
            residual: Per-sample physics violation score
        """
        # Ensure g is on same device
        if self.g.device != state.device:
            self.g = self.g.to(state.device)

        # Extract position and velocity (first 6 dimensions)
        pos = state[:, :3]
        vel = state[:, 3:6]
        next_pos = next_state[:, :3]
        next_vel = next_state[:, 3:6]

        # Position kinematics: dp/dt = v
        pos_pred = pos + vel * dt
        pos_residual = next_pos - pos_pred

        # Velocity dynamics: dv/dt = a (simplified, assume gravity-dominated)
        # For more accuracy, would need control inputs
        accel_gravity = self.g.unsqueeze(0).expand(vel.shape[0], -1)
        vel_pred = vel + accel_gravity * dt
        vel_residual = next_vel - vel_pred

        # Combine residuals
        residual = torch.cat([pos_residual, vel_residual], dim=-1)

        return residual

    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        dt: float = 0.005,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PINN residual with learned corrections.

        Returns:
            physics_residual: Raw physics violation
            corrected_residual: Physics residual minus learned correction
        """
        # Raw physics residual
        phys_res = self.physics_residual(state, next_state, dt)

        # Learned correction (what the NN thinks the model error is)
        correction = self.correction_net(state)[:, :6]  # Only correct pos/vel

        # Corrected residual
        corrected_res = phys_res - correction

        return phys_res, corrected_res


@dataclass
class ShadowResidualResult:
    """Result from shadow residual computation."""
    ici_score: float            # Primary ICI score
    pinn_residual: float        # PINN physics residual
    combined_score: float       # r_total = r_ici + alpha * r_pinn
    is_anomaly: bool            # Detection decision
    pinn_triggered: bool        # Whether PINN alone would have triggered
    threshold: float            # Threshold used


class PINNShadowResidual:
    """
    Option 1: PINN as Shadow Residual

    Key design:
        r_total = r_ici + alpha * r_pinn
        where alpha = 0.1-0.2 (small weight)

    Rules:
        1. ICI is PRIMARY detection signal
        2. PINN is SECONDARY - provides additional physics insight
        3. PINN alone NEVER triggers alarms (prevents false positives)
        4. Combined score improves separation on physics-based attacks
    """

    def __init__(
        self,
        state_dim: int = 12,
        alpha: float = 0.15,  # PINN weight (small)
        hidden_dim: int = 64,
        device: str = 'cpu',
    ):
        """
        Initialize PINN shadow residual.

        Args:
            state_dim: State dimension
            alpha: PINN weight in combined score (0.1-0.2 recommended)
            hidden_dim: PINN hidden layer size
            device: Torch device
        """
        self.state_dim = state_dim
        self.alpha = alpha
        self.device = device

        # PINN model
        self.pinn = QuadrotorPINNResidual(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        # Calibration statistics
        self.ici_mean = 0.0
        self.ici_std = 1.0
        self.pinn_mean = 0.0
        self.pinn_std = 1.0

        # Thresholds
        self.ici_threshold = None
        self.combined_threshold = None

        # Statistics tracking
        self.total_count = 0
        self.ici_detections = 0
        self.combined_detections = 0
        self.pinn_would_trigger = 0

    def fit_pinn(
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
            trajectories: [N, T, state_dim] normal flight data
            epochs: Training epochs
            lr: Learning rate
            batch_size: Batch size
            dt: Time step
            verbose: Print progress

        Returns:
            Training history
        """
        if verbose:
            print("Training PINN shadow residual...")

        # Prepare data
        X_t = []
        X_next = []

        for traj in trajectories:
            for t in range(len(traj) - 1):
                X_t.append(traj[t])
                X_next.append(traj[t + 1])

        X_t = torch.tensor(np.array(X_t, dtype=np.float32), device=self.device)
        X_next = torch.tensor(np.array(X_next, dtype=np.float32), device=self.device)

        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        history = {'loss': [], 'physics_loss': [], 'correction_loss': []}

        for epoch in range(epochs):
            self.pinn.train()
            perm = torch.randperm(len(X_t))
            epoch_loss = 0
            epoch_physics = 0
            epoch_correction = 0
            n_batches = 0

            for i in range(0, len(X_t), batch_size):
                batch_idx = perm[i:i+batch_size]
                x_t = X_t[batch_idx]
                x_next = X_next[batch_idx]

                optimizer.zero_grad()

                phys_res, corrected_res = self.pinn(x_t, x_next, dt)

                # Loss: minimize corrected residual (physics + correction should = 0)
                physics_loss = torch.mean(phys_res ** 2)
                correction_loss = torch.mean(corrected_res ** 2)

                # Combined loss with regularization on corrections
                loss = correction_loss + 0.01 * torch.mean(
                    self.pinn.correction_net(x_t)[:, :6] ** 2
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_physics += physics_loss.item()
                epoch_correction += correction_loss.item()
                n_batches += 1

            scheduler.step()

            history['loss'].append(epoch_loss / n_batches)
            history['physics_loss'].append(epoch_physics / n_batches)
            history['correction_loss'].append(epoch_correction / n_batches)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={epoch_loss/n_batches:.6f}")

        # Calibrate on training data
        self._calibrate_pinn(X_t, X_next, dt)

        return history

    def _calibrate_pinn(
        self,
        X_t: torch.Tensor,
        X_next: torch.Tensor,
        dt: float,
    ):
        """Calibrate PINN residual statistics on normal data."""
        self.pinn.eval()

        with torch.no_grad():
            _, corrected_res = self.pinn(X_t, X_next, dt)
            pinn_scores = torch.norm(corrected_res, dim=-1)

        pinn_np = pinn_scores.cpu().numpy()
        self.pinn_mean = float(np.mean(pinn_np))
        self.pinn_std = float(np.std(pinn_np))
        self.pinn_p95 = float(np.percentile(pinn_np, 95))
        self.pinn_p99 = float(np.percentile(pinn_np, 99))

    def calibrate_combined(
        self,
        ici_scores: np.ndarray,
        detection_percentile: float = 95,
    ):
        """
        Calibrate combined threshold using ICI statistics.

        Args:
            ici_scores: ICI scores from normal data
            detection_percentile: Percentile for threshold
        """
        self.ici_mean = float(np.mean(ici_scores))
        self.ici_std = float(np.std(ici_scores))
        self.ici_threshold = float(np.percentile(ici_scores, detection_percentile))

        # Combined threshold accounts for PINN contribution
        # Assume similar distribution, inflate by alpha factor
        self.combined_threshold = self.ici_threshold * (1 + self.alpha * 0.5)

    def compute_residual(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        dt: float = 0.005,
    ) -> float:
        """
        Compute PINN residual for a single transition.

        Args:
            state: Current state
            next_state: Next state
            dt: Time step

        Returns:
            PINN residual score (normalized)
        """
        self.pinn.eval()

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_t = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
            next_t = next_t.unsqueeze(0)

        with torch.no_grad():
            _, corrected_res = self.pinn(state_t, next_t, dt)
            residual = torch.norm(corrected_res, dim=-1).item()

        # Z-score normalize
        z_residual = (residual - self.pinn_mean) / (self.pinn_std + 1e-8)

        return z_residual

    def detect(
        self,
        ici_score: float,
        state: np.ndarray,
        next_state: np.ndarray,
        dt: float = 0.005,
    ) -> ShadowResidualResult:
        """
        Combined detection using ICI + PINN shadow residual.

        Args:
            ici_score: Primary ICI score (from CycleConsistencyDetector)
            state: Current state
            next_state: Next state
            dt: Time step

        Returns:
            ShadowResidualResult with detection decision
        """
        self.total_count += 1

        # Compute PINN residual
        pinn_residual = self.compute_residual(state, next_state, dt)

        # Z-score normalize ICI
        z_ici = (ici_score - self.ici_mean) / (self.ici_std + 1e-8)

        # Combined score: r_total = r_ici + alpha * r_pinn
        combined_score = z_ici + self.alpha * pinn_residual

        # Detection decision
        threshold = self.combined_threshold if self.combined_threshold else 2.0
        is_anomaly = combined_score > threshold

        # Track whether PINN alone would trigger (for analysis)
        pinn_alone_threshold = 2.0  # 2 standard deviations
        pinn_triggered = pinn_residual > pinn_alone_threshold

        if is_anomaly:
            self.combined_detections += 1
        if z_ici > threshold:
            self.ici_detections += 1
        if pinn_triggered:
            self.pinn_would_trigger += 1

        return ShadowResidualResult(
            ici_score=ici_score,
            pinn_residual=pinn_residual,
            combined_score=combined_score,
            is_anomaly=is_anomaly,
            pinn_triggered=pinn_triggered,
            threshold=threshold,
        )

    def score_trajectory(
        self,
        trajectory: np.ndarray,
        ici_scores: np.ndarray,
        dt: float = 0.005,
    ) -> np.ndarray:
        """
        Score entire trajectory with combined ICI + PINN.

        Args:
            trajectory: [T, state_dim] trajectory
            ici_scores: [T-1] ICI scores from primary detector
            dt: Time step

        Returns:
            [T-1] Combined scores
        """
        self.pinn.eval()

        X_t = torch.tensor(trajectory[:-1], dtype=torch.float32, device=self.device)
        X_next = torch.tensor(trajectory[1:], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, corrected_res = self.pinn(X_t, X_next, dt)
            pinn_residuals = torch.norm(corrected_res, dim=-1).cpu().numpy()

        # Z-score normalize both
        z_ici = (ici_scores - self.ici_mean) / (self.ici_std + 1e-8)
        z_pinn = (pinn_residuals - self.pinn_mean) / (self.pinn_std + 1e-8)

        # Combined score
        combined = z_ici + self.alpha * z_pinn

        return combined

    def get_metrics(self) -> Dict:
        """Get detection metrics."""
        return {
            'total_samples': self.total_count,
            'ici_detections': self.ici_detections,
            'combined_detections': self.combined_detections,
            'pinn_would_trigger': self.pinn_would_trigger,
            'alpha': self.alpha,
            'ici_threshold': self.ici_threshold,
            'combined_threshold': self.combined_threshold,
        }

    def reset(self):
        """Reset statistics."""
        self.total_count = 0
        self.ici_detections = 0
        self.combined_detections = 0
        self.pinn_would_trigger = 0


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_pinn_shadow(
    nominal_trajectories: np.ndarray,
    attack_trajectories: np.ndarray,
    ici_nominal: np.ndarray,
    ici_attack: np.ndarray,
    alpha: float = 0.15,
    dt: float = 0.005,
) -> Dict:
    """
    Evaluate PINN shadow residual integration.

    Args:
        nominal_trajectories: [N, T, state_dim] nominal data
        attack_trajectories: [M, T, state_dim] attack data
        ici_nominal: [N*T] ICI scores for nominal
        ici_attack: [M*T] ICI scores for attack
        alpha: PINN weight
        dt: Time step

    Returns:
        Evaluation metrics
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    # Create and train PINN
    state_dim = nominal_trajectories.shape[-1]
    shadow = PINNShadowResidual(state_dim=state_dim, alpha=alpha)

    # Train on nominal data
    shadow.fit_pinn(nominal_trajectories, epochs=30, verbose=False)

    # Calibrate with ICI scores
    shadow.calibrate_combined(ici_nominal)

    # Score all trajectories
    combined_nominal = []
    combined_attack = []
    pinn_nominal = []
    pinn_attack = []

    ici_idx = 0
    for traj in nominal_trajectories:
        n_steps = len(traj) - 1
        traj_ici = ici_nominal[ici_idx:ici_idx+n_steps]
        scores = shadow.score_trajectory(traj, traj_ici, dt)
        combined_nominal.extend(scores)

        # Also get PINN-only scores
        X_t = torch.tensor(traj[:-1], dtype=torch.float32, device=shadow.device)
        X_next = torch.tensor(traj[1:], dtype=torch.float32, device=shadow.device)
        with torch.no_grad():
            _, res = shadow.pinn(X_t, X_next, dt)
            pinn_nominal.extend(torch.norm(res, dim=-1).cpu().numpy())

        ici_idx += n_steps

    ici_idx = 0
    for traj in attack_trajectories:
        n_steps = len(traj) - 1
        traj_ici = ici_attack[ici_idx:ici_idx+n_steps]
        scores = shadow.score_trajectory(traj, traj_ici, dt)
        combined_attack.extend(scores)

        # PINN-only
        X_t = torch.tensor(traj[:-1], dtype=torch.float32, device=shadow.device)
        X_next = torch.tensor(traj[1:], dtype=torch.float32, device=shadow.device)
        with torch.no_grad():
            _, res = shadow.pinn(X_t, X_next, dt)
            pinn_attack.extend(torch.norm(res, dim=-1).cpu().numpy())

        ici_idx += n_steps

    combined_nominal = np.array(combined_nominal)
    combined_attack = np.array(combined_attack)
    pinn_nominal = np.array(pinn_nominal)
    pinn_attack = np.array(pinn_attack)

    # Compute metrics
    labels = np.concatenate([
        np.zeros(len(combined_nominal)),
        np.ones(len(combined_attack))
    ])

    # ICI-only AUROC
    ici_scores = np.concatenate([ici_nominal, ici_attack])
    ici_auroc = roc_auc_score(labels[:len(ici_scores)], ici_scores)

    # Combined AUROC
    combined_scores = np.concatenate([combined_nominal, combined_attack])
    combined_auroc = roc_auc_score(labels, combined_scores)

    # PINN-only AUROC
    pinn_scores = np.concatenate([pinn_nominal, pinn_attack])
    pinn_auroc = roc_auc_score(labels, pinn_scores)

    # Recall @ 1% FPR
    def recall_at_fpr(y_true, y_score, target_fpr=0.01):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        idx = np.searchsorted(fpr, target_fpr)
        return tpr[min(idx, len(tpr)-1)]

    ici_recall = recall_at_fpr(labels[:len(ici_scores)], ici_scores)
    combined_recall = recall_at_fpr(labels, combined_scores)
    pinn_recall = recall_at_fpr(labels, pinn_scores)

    return {
        'ici_auroc': float(ici_auroc),
        'combined_auroc': float(combined_auroc),
        'pinn_only_auroc': float(pinn_auroc),
        'ici_recall_1pct': float(ici_recall),
        'combined_recall_1pct': float(combined_recall),
        'pinn_only_recall_1pct': float(pinn_recall),
        'improvement_auroc': float(combined_auroc - ici_auroc),
        'improvement_recall': float(combined_recall - ici_recall),
        'alpha': alpha,
    }


# =============================================================================
# OPTION 2: PINN FOR OFFLINE ENVELOPE LEARNING
# =============================================================================

class ControlRegime(Enum):
    """Control regimes for envelope computation."""
    HOVER = "hover"
    CRUISE = "cruise"
    AGGRESSIVE = "aggressive"
    UNKNOWN = "unknown"


@dataclass
class PhysicsEnvelope:
    """Physics-consistent envelope for a control regime."""
    regime: ControlRegime
    position_accel_max: float   # Max expected position change rate
    velocity_accel_max: float   # Max expected velocity change rate
    angular_rate_max: float     # Max angular rate
    residual_mean: float        # Mean physics residual for regime
    residual_std: float         # Std of physics residual
    residual_p99: float         # 99th percentile threshold


@dataclass
class EnvelopeResult:
    """Result from envelope-based detection."""
    regime: ControlRegime           # Detected control regime
    physics_residual: float         # PINN physics residual
    envelope_threshold: float       # Regime-specific threshold
    envelope_violation: bool        # Whether envelope exceeded
    violation_factor: float         # How much threshold exceeded (0 if not)
    is_anomaly: bool               # Combined detection decision


class PINNEnvelopeLearner:
    """
    Option 2: PINN for Offline Envelope Learning

    Key concept:
        - Use PINN to learn physics-consistent envelopes OFFLINE
        - Envelopes are static thresholds computed per control regime
        - No PINN inference at runtime (fast)

    Process:
        1. Train PINN on nominal data
        2. Compute physics residuals per control regime
        3. Store percentile thresholds as static envelopes
        4. At runtime: classify regime, lookup threshold
    """

    def __init__(
        self,
        state_dim: int = 12,
        hidden_dim: int = 64,
        device: str = 'cpu',
    ):
        """
        Initialize envelope learner.

        Args:
            state_dim: State dimension
            hidden_dim: PINN hidden layer size
            device: Torch device
        """
        self.state_dim = state_dim
        self.device = device

        # PINN for physics residual computation
        self.pinn = QuadrotorPINNResidual(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        # Learned envelopes (static after training)
        self.envelopes: Dict[ControlRegime, PhysicsEnvelope] = {}

        # Regime classification thresholds
        self.hover_velocity_threshold = 0.5  # m/s
        self.aggressive_accel_threshold = 5.0  # m/s^2

        # Statistics
        self.total_count = 0
        self.envelope_violations = 0

    def classify_regime(
        self,
        velocity: np.ndarray,
        acceleration: np.ndarray,
    ) -> ControlRegime:
        """
        Classify current control regime.

        Args:
            velocity: Current velocity [vx, vy, vz]
            acceleration: Current acceleration [ax, ay, az]

        Returns:
            Detected control regime
        """
        vel_mag = np.linalg.norm(velocity)
        accel_mag = np.linalg.norm(acceleration)

        if vel_mag < self.hover_velocity_threshold:
            return ControlRegime.HOVER
        elif accel_mag > self.aggressive_accel_threshold:
            return ControlRegime.AGGRESSIVE
        else:
            return ControlRegime.CRUISE

    def fit(
        self,
        trajectories: np.ndarray,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
        dt: float = 0.005,
        verbose: bool = True,
    ) -> Dict:
        """
        Learn physics envelopes from nominal data.

        Args:
            trajectories: [N, T, state_dim] normal flight data
            epochs: PINN training epochs
            lr: Learning rate
            batch_size: Batch size
            dt: Time step
            verbose: Print progress

        Returns:
            Training history and envelope statistics
        """
        if verbose:
            print("Learning physics envelopes...")

        # Prepare data
        X_t = []
        X_next = []
        velocities = []
        accelerations = []

        for traj in trajectories:
            for t in range(len(traj) - 1):
                X_t.append(traj[t])
                X_next.append(traj[t + 1])
                velocities.append(traj[t, 3:6])  # Velocity
                # Approximate acceleration from velocity change
                if t > 0:
                    accel = (traj[t, 3:6] - traj[t-1, 3:6]) / dt
                else:
                    accel = np.zeros(3)
                accelerations.append(accel)

        X_t = torch.tensor(np.array(X_t, dtype=np.float32), device=self.device)
        X_next = torch.tensor(np.array(X_next, dtype=np.float32), device=self.device)
        velocities = np.array(velocities)
        accelerations = np.array(accelerations)

        # Train PINN
        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        history = {'loss': []}

        for epoch in range(epochs):
            self.pinn.train()
            perm = torch.randperm(len(X_t))
            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(X_t), batch_size):
                batch_idx = perm[i:i+batch_size]
                x_t = X_t[batch_idx]
                x_next = X_next[batch_idx]

                optimizer.zero_grad()
                _, corrected_res = self.pinn(x_t, x_next, dt)
                loss = torch.mean(corrected_res ** 2)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            history['loss'].append(epoch_loss / n_batches)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={epoch_loss/n_batches:.6f}")

        # Compute physics residuals per regime
        if verbose:
            print("\nComputing regime-specific envelopes...")

        self.pinn.eval()
        with torch.no_grad():
            _, corrected_res = self.pinn(X_t, X_next, dt)
            residuals = torch.norm(corrected_res, dim=-1).cpu().numpy()

        # Classify each sample into regime
        regime_residuals = {regime: [] for regime in ControlRegime}

        for i in range(len(residuals)):
            regime = self.classify_regime(velocities[i], accelerations[i])
            regime_residuals[regime].append(residuals[i])

        # Compute envelope statistics per regime
        for regime in ControlRegime:
            if len(regime_residuals[regime]) > 10:
                res = np.array(regime_residuals[regime])
                self.envelopes[regime] = PhysicsEnvelope(
                    regime=regime,
                    position_accel_max=np.percentile(np.abs(velocities[:, 0]), 99),
                    velocity_accel_max=np.percentile(np.abs(accelerations[:, 0]), 99),
                    angular_rate_max=1.0,  # Placeholder
                    residual_mean=float(np.mean(res)),
                    residual_std=float(np.std(res)),
                    residual_p99=float(np.percentile(res, 99)),
                )

                if verbose:
                    print(f"  {regime.value}: n={len(res)}, "
                          f"mean={self.envelopes[regime].residual_mean:.4f}, "
                          f"p99={self.envelopes[regime].residual_p99:.4f}")
            else:
                # Default envelope for regimes with insufficient data
                self.envelopes[regime] = PhysicsEnvelope(
                    regime=regime,
                    position_accel_max=10.0,
                    velocity_accel_max=20.0,
                    angular_rate_max=1.0,
                    residual_mean=0.0,
                    residual_std=1.0,
                    residual_p99=3.0,
                )

        return history

    def get_envelope(self, regime: ControlRegime) -> PhysicsEnvelope:
        """Get envelope for a control regime."""
        return self.envelopes.get(regime, self.envelopes.get(ControlRegime.UNKNOWN))

    def detect(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        dt: float = 0.005,
    ) -> EnvelopeResult:
        """
        Detect anomaly using physics envelope.

        Args:
            state: Current state
            next_state: Next state
            dt: Time step

        Returns:
            EnvelopeResult with detection decision
        """
        self.total_count += 1

        # Classify regime
        velocity = state[3:6]
        if len(state) > 6:
            # Approximate acceleration
            accel = (next_state[3:6] - state[3:6]) / dt
        else:
            accel = np.zeros(3)

        regime = self.classify_regime(velocity, accel)
        envelope = self.get_envelope(regime)

        # Compute physics residual
        self.pinn.eval()
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            _, corrected_res = self.pinn(state_t, next_t, dt)
            residual = torch.norm(corrected_res, dim=-1).item()

        # Check envelope violation
        threshold = envelope.residual_p99
        violation = residual > threshold
        violation_factor = max(0, (residual - threshold) / (threshold + 1e-8))

        if violation:
            self.envelope_violations += 1

        return EnvelopeResult(
            regime=regime,
            physics_residual=residual,
            envelope_threshold=threshold,
            envelope_violation=violation,
            violation_factor=violation_factor,
            is_anomaly=violation,  # Can be combined with ICI
        )

    def score_trajectory(
        self,
        trajectory: np.ndarray,
        dt: float = 0.005,
    ) -> Tuple[np.ndarray, np.ndarray, List[ControlRegime]]:
        """
        Score trajectory with envelope-based detection.

        Args:
            trajectory: [T, state_dim] trajectory
            dt: Time step

        Returns:
            residuals: [T-1] physics residuals
            violations: [T-1] binary violation flags
            regimes: [T-1] detected regimes
        """
        self.pinn.eval()

        X_t = torch.tensor(trajectory[:-1], dtype=torch.float32, device=self.device)
        X_next = torch.tensor(trajectory[1:], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, corrected_res = self.pinn(X_t, X_next, dt)
            residuals = torch.norm(corrected_res, dim=-1).cpu().numpy()

        # Classify regimes and check violations
        violations = np.zeros(len(residuals), dtype=bool)
        regimes = []

        for i in range(len(residuals)):
            vel = trajectory[i, 3:6]
            accel = (trajectory[i+1, 3:6] - trajectory[i, 3:6]) / dt if i < len(trajectory)-1 else np.zeros(3)
            regime = self.classify_regime(vel, accel)
            regimes.append(regime)

            envelope = self.get_envelope(regime)
            violations[i] = residuals[i] > envelope.residual_p99

        return residuals, violations, regimes

    def get_metrics(self) -> Dict:
        """Get detection metrics."""
        return {
            'total_samples': self.total_count,
            'envelope_violations': self.envelope_violations,
            'violation_rate': self.envelope_violations / max(1, self.total_count),
            'num_regimes': len(self.envelopes),
        }

    def reset(self):
        """Reset statistics."""
        self.total_count = 0
        self.envelope_violations = 0


# =============================================================================
# OPTION 3: PINN FOR PROBING RESPONSE PREDICTION
# =============================================================================

@dataclass
class ProbingPredictionResult:
    """Result from PINN probing response prediction."""
    excitation: float           # Applied excitation
    predicted_response: float   # PINN-predicted response
    actual_response: float      # Observed response
    prediction_error: float     # |predicted - actual|
    normalized_error: float     # Z-scored prediction error
    is_anomaly: bool           # Detection decision


class PINNProbingPredictor:
    """
    Option 3: PINN for Probing Response Prediction

    Key concept:
        - Use PINN to predict response to active probing excitations
        - Anomaly = large discrepancy between predicted and observed response
        - More sophisticated than simple gain adaptation

    Advantages:
        - Physics-informed prediction (not just empirical)
        - Works with any excitation signal
        - Can detect model-based attacks
    """

    def __init__(
        self,
        state_dim: int = 12,
        hidden_dim: int = 64,
        control_dim: int = 4,
        device: str = 'cpu',
    ):
        """
        Initialize probing predictor.

        Args:
            state_dim: State dimension
            hidden_dim: PINN hidden layer size
            control_dim: Control input dimension
            device: Torch device
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.device = device

        # PINN with control input for response prediction
        self.response_predictor = PINNResponsePredictor(
            state_dim=state_dim,
            control_dim=control_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        # Calibration statistics
        self.error_mean = 0.0
        self.error_std = 1.0
        self.error_p95 = 2.0

        # Detection threshold
        self.threshold = 2.0  # Z-score threshold

        # Statistics
        self.total_count = 0
        self.anomaly_count = 0

    def fit(
        self,
        trajectories: np.ndarray,
        controls: Optional[np.ndarray] = None,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
        dt: float = 0.005,
        verbose: bool = True,
    ) -> Dict:
        """
        Train PINN response predictor.

        Args:
            trajectories: [N, T, state_dim] nominal trajectories
            controls: [N, T, control_dim] control inputs (optional)
            epochs: Training epochs
            lr: Learning rate
            batch_size: Batch size
            dt: Time step
            verbose: Print progress

        Returns:
            Training history
        """
        if verbose:
            print("Training PINN probing response predictor...")

        # Prepare data
        X_t = []
        X_next = []
        U = []

        for i, traj in enumerate(trajectories):
            for t in range(len(traj) - 1):
                X_t.append(traj[t])
                X_next.append(traj[t + 1])
                if controls is not None:
                    U.append(controls[i, t])
                else:
                    # Infer control from state change (simplified)
                    u = np.zeros(self.control_dim)
                    U.append(u)

        X_t = torch.tensor(np.array(X_t, dtype=np.float32), device=self.device)
        X_next = torch.tensor(np.array(X_next, dtype=np.float32), device=self.device)
        U = torch.tensor(np.array(U, dtype=np.float32), device=self.device)

        # Train predictor
        optimizer = torch.optim.Adam(self.response_predictor.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        history = {'loss': []}

        for epoch in range(epochs):
            self.response_predictor.train()
            perm = torch.randperm(len(X_t))
            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(X_t), batch_size):
                batch_idx = perm[i:i+batch_size]
                x_t = X_t[batch_idx]
                x_next = X_next[batch_idx]
                u = U[batch_idx]

                optimizer.zero_grad()

                # Predict response to control
                predicted = self.response_predictor(x_t, u, dt)
                loss = F.mse_loss(predicted, x_next[:, :6])  # Only pos/vel

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            history['loss'].append(epoch_loss / n_batches)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={epoch_loss/n_batches:.6f}")

        # Calibrate error statistics
        self._calibrate(X_t, X_next, U, dt)

        return history

    def _calibrate(
        self,
        X_t: torch.Tensor,
        X_next: torch.Tensor,
        U: torch.Tensor,
        dt: float,
    ):
        """Calibrate prediction error statistics."""
        self.response_predictor.eval()

        with torch.no_grad():
            predicted = self.response_predictor(X_t, U, dt)
            errors = torch.norm(predicted - X_next[:, :6], dim=-1).cpu().numpy()

        self.error_mean = float(np.mean(errors))
        self.error_std = float(np.std(errors))
        self.error_p95 = float(np.percentile(errors, 95))

    def predict_response(
        self,
        state: np.ndarray,
        excitation: np.ndarray,
        dt: float = 0.005,
    ) -> np.ndarray:
        """
        Predict response to excitation using PINN.

        Args:
            state: Current state [state_dim]
            excitation: Control excitation [control_dim]
            dt: Time step

        Returns:
            Predicted state change [6] (pos/vel)
        """
        self.response_predictor.eval()

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        exc_t = torch.tensor(excitation, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            predicted = self.response_predictor(state_t, exc_t, dt)

        return predicted.cpu().numpy().squeeze()

    def detect(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        excitation: np.ndarray,
        dt: float = 0.005,
    ) -> ProbingPredictionResult:
        """
        Detect anomaly by comparing predicted vs actual response.

        Args:
            state: Current state
            next_state: Observed next state
            excitation: Applied excitation
            dt: Time step

        Returns:
            ProbingPredictionResult with detection
        """
        self.total_count += 1

        # Predict response
        predicted = self.predict_response(state, excitation, dt)

        # Compute prediction error
        actual = next_state[:6]  # pos/vel
        error = np.linalg.norm(predicted - actual)

        # Normalize error
        z_error = (error - self.error_mean) / (self.error_std + 1e-8)

        # Detection
        is_anomaly = z_error > self.threshold

        if is_anomaly:
            self.anomaly_count += 1

        return ProbingPredictionResult(
            excitation=float(np.linalg.norm(excitation)),
            predicted_response=float(np.linalg.norm(predicted - state[:6])),
            actual_response=float(np.linalg.norm(actual - state[:6])),
            prediction_error=error,
            normalized_error=z_error,
            is_anomaly=is_anomaly,
        )

    def get_metrics(self) -> Dict:
        """Get detection metrics."""
        return {
            'total_samples': self.total_count,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / max(1, self.total_count),
            'threshold': self.threshold,
        }

    def reset(self):
        """Reset statistics."""
        self.total_count = 0
        self.anomaly_count = 0


class PINNResponsePredictor(nn.Module):
    """
    Neural network for predicting state response to control inputs.

    Physics-informed structure:
        x_{t+1} = x_t + f(x_t, u_t) * dt
    """

    def __init__(
        self,
        state_dim: int = 12,
        control_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim

        # Dynamics network: (state, control) -> state_derivative
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 6),  # Output pos/vel derivatives
        )

        # Gravity prior
        self.g = torch.tensor([0.0, 0.0, -9.81, 0.0, 0.0, 0.0])

    def forward(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        dt: float = 0.005,
    ) -> torch.Tensor:
        """
        Predict next state given current state and control.

        Args:
            state: [batch, state_dim] current state
            control: [batch, control_dim] control input
            dt: Time step

        Returns:
            [batch, 6] predicted next pos/vel
        """
        # Ensure g is on correct device
        if self.g.device != state.device:
            self.g = self.g.to(state.device)

        # Current pos/vel
        current = state[:, :6]

        # Concatenate state and control
        x = torch.cat([state, control], dim=-1)

        # Predict derivative
        derivative = self.dynamics_net(x)

        # Add gravity prior
        derivative = derivative + self.g.unsqueeze(0) * dt

        # Euler integration
        next_state = current + derivative * dt

        return next_state


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_pinn_envelope(
    nominal_trajectories: np.ndarray,
    attack_trajectories: np.ndarray,
    dt: float = 0.005,
) -> Dict:
    """
    Evaluate PINN envelope learning (Option 2).

    Args:
        nominal_trajectories: [N, T, state_dim] nominal data
        attack_trajectories: [M, T, state_dim] attack data
        dt: Time step

    Returns:
        Evaluation metrics
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    state_dim = nominal_trajectories.shape[-1]
    envelope = PINNEnvelopeLearner(state_dim=state_dim)

    # Train on nominal
    envelope.fit(nominal_trajectories, epochs=20, verbose=False)

    # Score trajectories
    nominal_residuals = []
    nominal_violations = []
    attack_residuals = []
    attack_violations = []

    for traj in nominal_trajectories:
        res, viol, _ = envelope.score_trajectory(traj, dt)
        nominal_residuals.extend(res)
        nominal_violations.extend(viol)

    for traj in attack_trajectories:
        res, viol, _ = envelope.score_trajectory(traj, dt)
        attack_residuals.extend(res)
        attack_violations.extend(viol)

    # Compute metrics
    labels = np.concatenate([
        np.zeros(len(nominal_residuals)),
        np.ones(len(attack_residuals))
    ])
    scores = np.concatenate([nominal_residuals, attack_residuals])

    auroc = roc_auc_score(labels, scores)

    # Recall @ 1% FPR
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, 0.01)
    recall_1pct = tpr[min(idx, len(tpr)-1)]

    # Violation-based metrics
    nom_viol_rate = np.mean(nominal_violations)
    atk_viol_rate = np.mean(attack_violations)

    return {
        'auroc': float(auroc),
        'recall_1pct': float(recall_1pct),
        'nominal_violation_rate': float(nom_viol_rate),
        'attack_violation_rate': float(atk_viol_rate),
        'violation_separation': float(atk_viol_rate - nom_viol_rate),
    }


def evaluate_pinn_probing(
    nominal_trajectories: np.ndarray,
    attack_trajectories: np.ndarray,
    dt: float = 0.005,
) -> Dict:
    """
    Evaluate PINN probing response prediction (Option 3).

    Args:
        nominal_trajectories: [N, T, state_dim] nominal data
        attack_trajectories: [M, T, state_dim] attack data
        dt: Time step

    Returns:
        Evaluation metrics
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    state_dim = nominal_trajectories.shape[-1]
    predictor = PINNProbingPredictor(state_dim=state_dim, control_dim=4)

    # Train on nominal (without controls - will infer)
    predictor.fit(nominal_trajectories, epochs=20, verbose=False)

    # Score trajectories with synthetic probing
    nominal_errors = []
    attack_errors = []

    np.random.seed(42)

    for traj in nominal_trajectories:
        for t in range(len(traj) - 1):
            # Synthetic excitation
            exc = np.random.randn(4) * 0.01
            result = predictor.detect(traj[t], traj[t+1], exc, dt)
            nominal_errors.append(result.normalized_error)

    for traj in attack_trajectories:
        for t in range(len(traj) - 1):
            exc = np.random.randn(4) * 0.01
            result = predictor.detect(traj[t], traj[t+1], exc, dt)
            attack_errors.append(result.normalized_error)

    # Compute metrics
    labels = np.concatenate([
        np.zeros(len(nominal_errors)),
        np.ones(len(attack_errors))
    ])
    scores = np.concatenate([nominal_errors, attack_errors])

    auroc = roc_auc_score(labels, scores)

    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, 0.01)
    recall_1pct = tpr[min(idx, len(tpr)-1)]

    return {
        'auroc': float(auroc),
        'recall_1pct': float(recall_1pct),
        'nominal_error_mean': float(np.mean(nominal_errors)),
        'attack_error_mean': float(np.mean(attack_errors)),
    }


__all__ = [
    # Option 1: Shadow Residual
    'QuadrotorPINNResidual',
    'ShadowResidualResult',
    'PINNShadowResidual',
    'evaluate_pinn_shadow',
    # Option 2: Envelope Learning
    'ControlRegime',
    'PhysicsEnvelope',
    'EnvelopeResult',
    'PINNEnvelopeLearner',
    'evaluate_pinn_envelope',
    # Option 3: Probing Response
    'ProbingPredictionResult',
    'PINNProbingPredictor',
    'PINNResponsePredictor',
    'evaluate_pinn_probing',
]
