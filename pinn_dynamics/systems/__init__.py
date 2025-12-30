"""
Dynamical system implementations.

This module provides the base class and concrete implementations for
physics-informed neural networks.

Available Systems:
    - DynamicsPINN: Abstract base class for any dynamical system
    - QuadrotorPINN: 6-DOF quadrotor dynamics (12 states, 4 controls)
    - PendulumPINN: Simple pendulum (2 states, 1 control)
    - CartPolePINN: Cart-pole / inverted pendulum (4 states, 1 control)
    - SequencePINN: Sequence-based PINN for temporal pattern detection
    - SequenceAnomalyDetector: Specialized detector using SequencePINN
"""

from .base import DynamicsPINN
from .quadrotor import QuadrotorPINN
from .pendulum import PendulumPINN
from .cartpole import CartPolePINN
from .sequence import SequencePINN, SequenceAnomalyDetector

__all__ = [
    "DynamicsPINN",
    "QuadrotorPINN",
    "PendulumPINN",
    "CartPolePINN",
    "SequencePINN",
    "SequenceAnomalyDetector",
]
