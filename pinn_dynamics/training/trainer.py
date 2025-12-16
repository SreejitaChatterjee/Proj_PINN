"""
Training infrastructure for PINN models.

Provides the Trainer class for training physics-informed neural networks
with multiple loss terms and scheduled sampling for autoregressive stability.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for Physics-Informed Neural Networks.

    Supports multiple loss terms:
        - Data loss (MSE between prediction and ground truth)
        - Physics loss (from model's physics_loss method)
        - Temporal smoothness loss (penalize unrealistic change rates)
        - Stability loss (prevent divergence)
        - Regularization loss (penalize parameter deviation)
        - Energy conservation loss (enforce energy balance)

    Args:
        model: A DynamicsPINN model instance
        device: Device to train on ('cpu', 'cuda', or torch.device)
        lr: Learning rate (default: 0.001)
        weight_decay: L2 regularization (default: 0)

    Example:
        from pinn_dynamics import QuadrotorPINN, Trainer

        model = QuadrotorPINN()
        trainer = Trainer(model, device='cuda', lr=0.001)

        trainer.fit(
            train_loader,
            val_loader,
            epochs=100,
            weights={'physics': 10.0, 'temporal': 5.0}
        )
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
        lr: float = 0.001,
        weight_decay: float = 0,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=20
        )
        self.criterion = torch.nn.MSELoss()
        self.history = {
            "train": [],
            "val": [],
            "physics": [],
            "temporal": [],
            "stability": [],
            "reg": [],
            "energy": [],
        }

    def train_epoch(
        self,
        loader: DataLoader,
        weights: Optional[Dict[str, float]] = None,
        scheduled_sampling_prob: float = 0.0,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            loader: DataLoader with (input, target) pairs
            weights: Loss weights dict, e.g. {'physics': 10.0, 'temporal': 5.0}
            scheduled_sampling_prob: Probability of using model's prediction as next input

        Returns:
            Dict of average losses for this epoch
        """
        weights = weights or {
            "physics": 10.0,
            "temporal": 12.0,
            "stability": 5.0,
            "reg": 1.0,
            "energy": 5.0,
        }

        self.model.train()
        losses = {
            "total": 0,
            "data": 0,
            "physics": 0,
            "temporal": 0,
            "stability": 0,
            "reg": 0,
            "energy": 0,
        }

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Scheduled sampling: sometimes use model's own prediction
            if scheduled_sampling_prob > 0 and torch.rand(1).item() < scheduled_sampling_prob:
                with torch.no_grad():
                    pred = self.model(data)
                    # Replace states with predictions, keep controls
                    state_dim = self.model.state_dim
                    data = torch.cat([pred[:, :state_dim].detach(), data[:, state_dim:]], dim=1)

            output = self.model(data)

            # Data loss (always computed)
            data_loss = self.criterion(output, target)

            # Optional physics-based losses
            physics_loss = torch.tensor(0.0, device=self.device)
            temporal_loss = torch.tensor(0.0, device=self.device)
            stability_loss = torch.tensor(0.0, device=self.device)
            reg_loss = torch.tensor(0.0, device=self.device)
            energy_loss = torch.tensor(0.0, device=self.device)

            if weights.get("physics", 0) > 0 and hasattr(self.model, "physics_loss"):
                physics_loss = self.model.physics_loss(data, output)

            if weights.get("temporal", 0) > 0 and hasattr(self.model, "temporal_smoothness_loss"):
                temporal_loss = self.model.temporal_smoothness_loss(data, output)

            if weights.get("stability", 0) > 0 and hasattr(self.model, "stability_loss"):
                stability_loss = self.model.stability_loss(data, output)

            if weights.get("reg", 0) > 0 and hasattr(self.model, "regularization_loss"):
                reg_loss = self.model.regularization_loss()

            if weights.get("energy", 0) > 0 and hasattr(self.model, "energy_conservation_loss"):
                energy_loss = self.model.energy_conservation_loss(data, output)

            # Total loss
            loss = (
                data_loss
                + weights.get("physics", 0) * physics_loss
                + weights.get("temporal", 0) * temporal_loss
                + weights.get("stability", 0) * stability_loss
                + weights.get("reg", 0) * reg_loss
                + weights.get("energy", 0) * energy_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Constrain physics parameters
            if hasattr(self.model, "constrain_parameters"):
                self.model.constrain_parameters()

            # Track losses
            losses["total"] += loss.item()
            losses["data"] += data_loss.item()
            losses["physics"] += physics_loss.item()
            losses["temporal"] += temporal_loss.item()
            losses["stability"] += stability_loss.item()
            losses["reg"] += reg_loss.item()
            losses["energy"] += energy_loss.item()

        return {k: v / len(loader) for k, v in losses.items()}

    def validate(self, loader: DataLoader) -> float:
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                total_loss += self.criterion(self.model(data), target).item()
        return total_loss / len(loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 250,
        weights: Optional[Dict[str, float]] = None,
        scheduled_sampling_final: float = 0.3,
        verbose: bool = True,
        callback: Optional[Callable] = None,
    ) -> Dict[str, list]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            weights: Loss weight dict
            scheduled_sampling_final: Final scheduled sampling probability (ramps from 0)
            verbose: Print progress
            callback: Optional callback(epoch, losses) called each epoch

        Returns:
            Training history dict
        """
        weights = weights or {
            "physics": 10.0,
            "temporal": 12.0,
            "stability": 5.0,
            "reg": 1.0,
            "energy": 5.0,
        }

        if verbose:
            logger.info(f"Training for {epochs} epochs")
            logger.info(f"  Loss weights: {weights}")
            logger.info(f"  Scheduled sampling: 0% -> {scheduled_sampling_final*100:.0f}%")

        for epoch in range(epochs):
            # Ramp scheduled sampling
            ss_prob = scheduled_sampling_final * (epoch / epochs)

            # Train
            losses = self.train_epoch(train_loader, weights, ss_prob)
            val_loss = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Track history
            self.history["train"].append(losses["total"])
            self.history["val"].append(val_loss)
            self.history["physics"].append(losses["physics"])
            self.history["temporal"].append(losses["temporal"])
            self.history["stability"].append(losses["stability"])
            self.history["reg"].append(losses["reg"])
            self.history["energy"].append(losses["energy"])

            # Logging
            if verbose and epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:03d}: Train={losses['total']:.4f}, Val={val_loss:.6f}, "
                    f"Physics={losses['physics']:.4f}, SS={ss_prob:.2f}"
                )

            # Callback
            if callback:
                callback(epoch, losses, val_loss)

        return self.history

    def save(self, path: str):
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
