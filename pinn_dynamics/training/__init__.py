"""
Training infrastructure for PINN models.

This module provides the Trainer class and loss functions for training
physics-informed neural networks.

Classes:
    - Trainer: Main training loop with multi-loss support
    - RealDataTrainer: Training on real sensor data (no control inputs)

Loss Functions:
    - physics_loss: Governed by model's physics_loss() method
    - temporal_loss: Penalize unrealistic state change rates
    - stability_loss: Prevent state divergence
"""

from .trainer import Trainer
from .losses import physics_loss, temporal_loss, stability_loss

__all__ = [
    "Trainer",
    "physics_loss",
    "temporal_loss",
    "stability_loss",
]
