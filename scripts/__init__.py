"""
PINN Dynamics Framework

Physics-Informed Neural Networks for learning dynamical systems.

Quick start:
    from scripts import QuadrotorPINN, PendulumPINN, CartPolePINN

    model = PendulumPINN()
    print(model.summary())
"""

from .analysis.pinn_base import CartPolePINN, DynamicsPINN, PendulumPINN
from .analysis.pinn_model import QuadrotorPINN

__all__ = [
    "DynamicsPINN",
    "PendulumPINN",
    "CartPolePINN",
    "QuadrotorPINN",
]

__version__ = "0.1.0"
