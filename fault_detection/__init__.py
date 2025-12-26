"""
Motor Fault Detection Package
=============================

A comprehensive fault detection system for UAV motors using the PADRE dataset.

Features:
    - Multiple feature extraction methods (FFT, Wavelet, Statistical)
    - Advanced architectures (CNN, Transformer, TCN, Ensemble)
    - Uncertainty quantification (MC Dropout, Conformal Prediction)
    - Explainability (GradCAM-1D, Attention visualization)
    - Deployment (ONNX, Streaming inference, Quantization)

Dataset:
    AeroLab UAV Measurement Data (Poznan University of Technology)
    https://github.com/AeroLabPUT/UAV_measurement_data

Usage:
    from fault_detection import PADREDataset, EnsembleDetector, StreamingInference

    # Load data with advanced features
    dataset = PADREDataset(data_dir, feature_type='fft+wavelet+stats')

    # Train ensemble model
    model = EnsembleDetector(n_classes=3)

    # Deploy with streaming inference
    streamer = StreamingInference(model, window_size=256)
"""

__version__ = "1.0.0"

from .features import (
    FFTFeatureExtractor,
    WaveletFeatureExtractor,
    StatisticalFeatureExtractor,
    CrossMotorFeatureExtractor,
    CombinedFeatureExtractor
)

from .models import (
    MotorFaultCNN,
    TransformerDetector,
    MultiScaleCNN,
    TCNDetector,
    EnsembleDetector
)

from .uncertainty import (
    MCDropoutWrapper,
    TemperatureScaler,
    OODDetector,
    ConformalPredictor
)

from .advanced_tasks import (
    SeverityRegressor,
    AnomalyDetector,
    PerMotorClassifier
)

from .deployment import (
    export_onnx,
    StreamingInference,
    QuantizedModel
)

from .explainability import (
    GradCAM1D,
    SensorImportance,
    AttentionVisualizer
)

from .data import PADREDataset, DataAugmentation

__all__ = [
    # Features
    "FFTFeatureExtractor",
    "WaveletFeatureExtractor",
    "StatisticalFeatureExtractor",
    "CrossMotorFeatureExtractor",
    "CombinedFeatureExtractor",
    # Models
    "MotorFaultCNN",
    "TransformerDetector",
    "MultiScaleCNN",
    "TCNDetector",
    "EnsembleDetector",
    # Uncertainty
    "MCDropoutWrapper",
    "TemperatureScaler",
    "OODDetector",
    "ConformalPredictor",
    # Advanced Tasks
    "SeverityRegressor",
    "AnomalyDetector",
    "PerMotorClassifier",
    # Deployment
    "export_onnx",
    "StreamingInference",
    "QuantizedModel",
    # Explainability
    "GradCAM1D",
    "SensorImportance",
    "AttentionVisualizer",
    # Data
    "PADREDataset",
    "DataAugmentation",
]
