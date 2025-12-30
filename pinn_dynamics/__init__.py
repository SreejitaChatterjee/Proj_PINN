"""
PINN Dynamics - Physics-Informed Neural Networks for Dynamics Learning

A framework for learning dynamics models from data with optional physics constraints.

Features:
    - Easy System Definition: Define any dynamical system in <20 lines
    - Real Data Training: Train on sensor data without control inputs
    - Multi-Step Prediction: Autoregressive rollout with uncertainty
    - Deployment Ready: Export to ONNX for embedded systems
    - UAV Fault Detection: Experimental anomaly detection (see research/security/)

Quick Start:
    from pinn_dynamics import QuadrotorPINN, Trainer, Predictor

    # Load pre-trained model
    model = QuadrotorPINN()
    model.load_state_dict(torch.load('model.pth'))

    # Make predictions
    predictor = Predictor(model)
    trajectory = predictor.rollout(initial_state, controls, steps=100)

Example - Define Custom System:
    from pinn_dynamics import DynamicsPINN

    class MySystem(DynamicsPINN):
        def __init__(self):
            super().__init__(state_dim=4, control_dim=2)

        def physics_loss(self, inputs, outputs, dt=0.01):
            # Your physics equations here
            ...

Example - Fault Detection:
    from pinn_dynamics.security import AnomalyDetector

    detector = AnomalyDetector(predictor, threshold=0.1707)
    result = detector.detect(state, control, next_state_measured)
    if result.is_anomaly:
        print(f"FAULT DETECTED! Score={result.score:.3f}")
"""

__version__ = "1.0.0"

# Core system classes
from .systems.base import DynamicsPINN
from .systems.quadrotor import QuadrotorPINN
from .systems.pendulum import PendulumPINN
from .systems.cartpole import CartPolePINN
from .systems.sequence import SequencePINN, SequenceAnomalyDetector

# Training
from .training.trainer import Trainer
from .training.losses import physics_loss, temporal_loss, stability_loss

# Inference
from .inference.predictor import Predictor
from .inference.export import export_onnx, export_torchscript

# Security/Fault Detection (NEW)
try:
    from .security.anomaly_detector import AnomalyDetector
    from .security.evaluation import DetectionMetrics, DetectionEvaluator
    _SECURITY_AVAILABLE = True
except ImportError:
    _SECURITY_AVAILABLE = False
    # Security module not installed - this is OK for basic dynamics usage

# Data loading (optional - not required for basic usage)
# from .data.loaders import load_csv, prepare_data
# from .data.euroc import load_euroc

# Utilities
# from .utils.config import load_config

__all__ = [
    # Version
    "__version__",
    # Systems
    "DynamicsPINN",
    "QuadrotorPINN",
    "PendulumPINN",
    "CartPolePINN",
    "SequencePINN",
    "SequenceAnomalyDetector",
    # Training
    "Trainer",
    "physics_loss",
    "temporal_loss",
    "stability_loss",
    # Inference
    "Predictor",
    "export_onnx",
    "export_torchscript",
]

# Add security classes if available
if _SECURITY_AVAILABLE:
    __all__.extend([
        "AnomalyDetector",
        "DetectionMetrics",
        "DetectionEvaluator",
    ])
