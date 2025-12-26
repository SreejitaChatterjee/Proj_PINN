"""
Security module for PINN-based anomaly detection in cyber-physical systems.

This module provides tools for detecting sensor spoofing attacks on UAVs
using physics-informed neural networks.

Main components:
    - AnomalyDetector: Real-time attack detection using PINN predictions
    - AttackDataLoader: Load and preprocess attack datasets
    - Evaluation tools: ROC curves, confusion matrices, TTD metrics

Example:
    from pinn_dynamics.security import AnomalyDetector
    from pinn_dynamics import QuadrotorPINN, Predictor

    # Train PINN on clean data
    model = QuadrotorPINN()
    # ... training ...

    # Create detector
    predictor = Predictor(model)
    detector = AnomalyDetector(predictor, threshold=3.0)

    # Detect attacks
    detector.calibrate(clean_states, clean_controls, clean_next_states)
    result = detector.detect(state, control, measured_next_state)

    if result.is_anomaly:
        print(f"Attack detected! Score: {result.total_score}")
"""

from .anomaly_detector import AnomalyDetector, AnomalyScore

__all__ = ["AnomalyDetector", "AnomalyScore"]
