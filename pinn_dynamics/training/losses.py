"""
Loss functions for physics-informed training.

These are standalone loss functions that can be used with any model.
For model-specific physics losses, see the model's physics_loss() method.
"""

import torch
from typing import Dict


def physics_loss(model, inputs: torch.Tensor, outputs: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
    """
    Compute physics loss using model's physics_loss method.

    Args:
        model: A DynamicsPINN model with physics_loss method
        inputs: [batch, state_dim + control_dim] inputs
        outputs: [batch, state_dim] predicted outputs
        dt: Timestep

    Returns:
        Scalar physics violation loss
    """
    if hasattr(model, "physics_loss"):
        return model.physics_loss(inputs, outputs, dt)
    return torch.tensor(0.0)


def temporal_loss(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    state_dim: int,
    dt: float = 0.001,
    limits: Dict[str, float] = None,
) -> torch.Tensor:
    """
    Penalize unrealistic state change rates.

    Args:
        inputs: [batch, state_dim + control_dim] inputs
        outputs: [batch, state_dim] outputs
        state_dim: Number of state dimensions
        dt: Timestep
        limits: Dict of max change rates per state

    Returns:
        Scalar temporal smoothness loss
    """
    limits = limits or {}
    default_limit = 10.0

    current_states = inputs[:, :state_dim]
    next_states = outputs[:, :state_dim]

    # Compute change rates
    rates = (next_states - current_states) / dt

    # Apply soft constraints
    loss = torch.tensor(0.0, device=inputs.device)
    for i in range(state_dim):
        limit = limits.get(f"state_{i}", default_limit)
        loss += torch.relu(torch.abs(rates[:, i]) - limit).pow(2).mean()

    return loss


def stability_loss(
    outputs: torch.Tensor,
    bounds: Dict[int, tuple] = None,
) -> torch.Tensor:
    """
    Prevent state divergence by penalizing out-of-bounds predictions.

    Args:
        outputs: [batch, state_dim] predicted outputs
        bounds: Dict mapping state index to (min, max) tuple

    Returns:
        Scalar stability loss
    """
    bounds = bounds or {}
    loss = torch.tensor(0.0, device=outputs.device)

    for state_idx, (lo, hi) in bounds.items():
        if state_idx < outputs.shape[1]:
            state = outputs[:, state_idx]
            loss += torch.relu(lo - state).pow(2).mean()  # Below min
            loss += torch.relu(state - hi).pow(2).mean()  # Above max

    return loss


def regularization_loss(
    model,
    target_params: Dict[str, float] = None,
    weight: float = 100.0,
) -> torch.Tensor:
    """
    Penalize deviation of learnable parameters from target values.

    Args:
        model: Model with params ParameterDict
        target_params: Dict of target parameter values
        weight: Scaling factor

    Returns:
        Scalar regularization loss
    """
    if not hasattr(model, "params") or not target_params:
        return torch.tensor(0.0)

    loss = torch.tensor(0.0)
    for name, target in target_params.items():
        if name in model.params:
            param = model.params[name]
            loss += ((param - target) / target) ** 2

    return weight * loss


def energy_loss(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    mass: float,
    gravity: float = 9.81,
    dt: float = 0.001,
) -> torch.Tensor:
    """
    Simple energy conservation loss for point mass systems.

    For more complex energy losses, implement in the model class.

    Args:
        inputs: [batch, state_dim + control_dim]
        outputs: [batch, state_dim]
        mass: System mass
        gravity: Gravity constant
        dt: Timestep

    Returns:
        Scalar energy balance loss
    """
    # Assumes states are [position..., velocity...]
    # This is a simplified version - override in model for complex dynamics

    # Placeholder - specific models should implement their own
    return torch.tensor(0.0, device=inputs.device)
