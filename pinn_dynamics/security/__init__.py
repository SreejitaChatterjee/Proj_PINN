"""
Security module for PINN-based anomaly detection in cyber-physical systems.

This module provides tools for detecting sensor spoofing attacks on UAVs
using physics-informed neural networks and hybrid multi-modal detection.

Architecture Overview (CPU-Friendly):
    Layer 1: Emulated Sensors
        - BarometerEmulator: IIR filter + drift + noise
        - MagnetometerEmulator: Attitude rotation + hard/soft iron

    Layer 2: State Estimation & Integrity
        - IntegrityEKF: 15-state error-state EKF with NIS monitoring
        - Provides position, velocity, attitude, bias estimates

    Layer 3: Attack Classification
        - RuleBasedClassifier: Physics-based attack type detection
        - HybridClassifier: Rule-based + temporal smoothing

    Layer 4: Score Fusion
        - HybridFusion: Weighted combination of detector scores
        - MultiModalDetector: Full detection pipeline

Main components:
    - AnomalyDetector: Real-time attack detection using PINN predictions
    - IntegrityEKF: EKF with NIS-based integrity monitoring
    - HybridFusion: Multi-modal score fusion
    - SensorEmulationPipeline: Emulate baro/mag from IMU+position

Example:
    from pinn_dynamics.security import (
        AnomalyDetector, IntegrityEKF, HybridFusion,
        SensorEmulationPipeline, HybridClassifier
    )

    # Emulate missing sensors
    pipeline = SensorEmulationPipeline()
    emulated = pipeline.emulate(df)

    # Run EKF with integrity monitoring
    ekf = IntegrityEKF()
    ekf.predict(acc_data, gyro_data)
    ekf.update_position(pos)
    integrity = ekf.get_integrity_score()

    # Fuse multiple detector outputs
    fusion = HybridFusion()
    score, level = fusion.fuse(detector_scores)
"""

from .anomaly_detector import AnomalyDetector, AnomalyScore

# Emulated sensors (L1)
from .emulated_sensors import (
    BarometerEmulator,
    MagnetometerEmulator,
    SensorEmulationPipeline,
    inject_sensor_attack,
)

# Integrity EKF (L2)
from .integrity_ekf import IntegrityEKF, IntegrityLevel, EKFConfig

# Attack classification (L3)
from .attack_classifier import (
    AttackCategory,
    AttackType,
    RuleBasedClassifier,
    HybridClassifier,
    FeatureExtractor,
    run_attack_classifier,
)

# Hybrid fusion (L4)
from .hybrid_fusion import (
    HybridFusion,
    FusionWeights,
    FusionConfig,
    DetectorScores,
    DetectionLevel,
    MultiModalDetector,
    run_hybrid_detection,
)

# Hardened detector (L5) - Defense against mathematical attacks
from .hardened_detector import (
    HardenedDetector,
    HardenedConfig,
    MultiRateJerkChecker,
    CUSUMDriftMonitor,
    VelocityAugmentedEKF,
    RandomizedThresholds,
    RobustNormalizer,
    SPRTDetector,
    SpectralMonitor,
)

# Physics checks
from .physics_checks import (
    PhysicsAnomalyDetector,
    JerkChecker,
    EnergyChecker,
    KinematicTriadChecker,
    PhysicsLimits,
)

# Hybrid detector (data-driven routing)
from .hybrid_detector import HybridAttackDetector, RoutingClassifier

# Enhanced detector
from .enhanced_detector import EnhancedAttackDetector, MultiScaleFeatureExtractor

__all__ = [
    # Original
    "AnomalyDetector",
    "AnomalyScore",
    # L1: Sensors
    "BarometerEmulator",
    "MagnetometerEmulator",
    "SensorEmulationPipeline",
    "inject_sensor_attack",
    # L2: EKF
    "IntegrityEKF",
    "IntegrityLevel",
    "EKFConfig",
    # L3: Classification
    "AttackCategory",
    "AttackType",
    "RuleBasedClassifier",
    "HybridClassifier",
    "FeatureExtractor",
    "run_attack_classifier",
    # L4: Fusion
    "HybridFusion",
    "FusionWeights",
    "FusionConfig",
    "DetectorScores",
    "DetectionLevel",
    "MultiModalDetector",
    "run_hybrid_detection",
    # L5: Hardened Detector
    "HardenedDetector",
    "HardenedConfig",
    "MultiRateJerkChecker",
    "CUSUMDriftMonitor",
    "VelocityAugmentedEKF",
    "RandomizedThresholds",
    "RobustNormalizer",
    "SPRTDetector",
    "SpectralMonitor",
    # Physics Checks
    "PhysicsAnomalyDetector",
    "JerkChecker",
    "EnergyChecker",
    "KinematicTriadChecker",
    "PhysicsLimits",
]
