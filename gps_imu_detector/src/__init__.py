"""
GPS-IMU Anomaly Detector

A physics-first, multi-scale unsupervised fusion detector
for real-time GPS-IMU anomaly detection at 200 Hz.
"""

__version__ = "0.9.0"

# Core detection
from .inverse_model import CycleConsistencyDetector, InverseModel, ForwardModel

# Improvements (v0.2.0)
from .temporal_ici import TemporalICIAggregator, TemporalICIConfig, ConsensusAggregator
from .conditional_fusion import ConditionalHybridFusion, ConditionalFusionConfig
from .iasp_v2 import IASPv2Healer, IASPv2Config, AdaptiveIASP

# Coordinated spoofing defense (v0.3.0)
from .coordinated_defense import (
    CoordinatedDefenseSystem,
    CoordinatedDefenseConfig,
    MultiScaleAggregator,
    TimingCoherenceAnalyzer,
    OverConsistencyDetector,
    PersistenceLogic,
)

# Actuator fault observability (v0.4.0)
from .actuator_observability import (
    ControlEffortChecker,
    ControlEffortMetrics,
    DualTimescaleDetector,
    DualScaleResult,
    ResidualEnvelopeNormalizer,
    SplitFaultHead,
    PhaseConsistencyChecker,
    EnhancedActuatorDetector,
    compute_proper_metrics,
    extract_motor_features,
    extract_actuator_features,
)

# Advanced detection (v0.5.0)
from .advanced_detection import (
    # Improvement A: Lag drift tracking
    LagDriftTracker,
    LagDriftResult,
    # Improvement B: Second-order consistency
    SecondOrderConsistency,
    SecondOrderResult,
    # Improvement C: Control regime envelopes
    ControlRegime,
    ControlRegimeEnvelopes,
    RegimeEnvelope,
    # Improvement D: Fault attribution
    FaultType,
    FaultAttributor,
    FaultAttribution,
    # Improvement E: Prediction-retrodiction asymmetry
    PredictionRetrodictionChecker,
    AsymmetryResult,
    # Improvement F: Randomized subspace sampling
    RandomizedSubspaceSampler,
    RandomizedResult,
    # Integrated detector
    AdvancedDetector,
    AdvancedDetectionResult,
)

# Final improvements (v0.5.1)
from .final_improvements import (
    # Persistence scoring
    FaultPersistenceScorer,
    PersistenceResult,
    # Cost-aware thresholds
    FaultClass,
    AsymmetricThresholds,
    CostAwareThresholder,
    CostAwareResult,
    # TTD metrics
    TTDMetrics,
    TTDAnalyzer,
    # Controller prediction
    ControllerPredictor,
    ControllerResidualResult,
    # Cross-axis coupling
    CrossAxisCouplingChecker,
    CrossAxisResult,
    # Combined detector
    FinalDetector,
    FinalDetectionResult,
    # Evaluation
    evaluate_with_final_improvements,
)

# Industry-aligned detection (v0.6.0)
from .industry_aligned import (
    # Two-stage decision
    TwoStageDecisionLogic,
    TwoStageResult,
    SuspicionState,
    # Risk-weighted
    HazardClass,
    HazardThresholds,
    RiskWeightedDetector,
    RiskWeightedResult,
    FAULT_HAZARD_MAPPING,
    # Integrity
    ProtectionLevels,
    AlertLimits,
    IntegrityMonitor,
    IntegrityResult,
    # Combined
    IndustryAlignedDetector,
    IndustryAlignedResult,
    # Evaluation
    evaluate_industry_aligned,
)

# Analytical redundancy (v0.7.0) - BREAKS THE CLAO CEILING
from .analytical_redundancy import (
    # EKF estimator
    EKFState,
    NonlinearEKF,
    # Complementary estimator
    LinearComplementaryEstimator,
    # Disagreement detection
    DisagreementResult,
    EstimatorDisagreementDetector,
    # Combined system
    AnalyticalRedundancyResult,
    AnalyticalRedundancySystem,
    # Evaluation
    evaluate_analytical_redundancy,
)

# Active probing (v0.8.0) - BREAKS STEALTH CEILING
from .active_probing import (
    # Excitation types
    ExcitationType,
    ExcitationSignal,
    # Generators
    MicroChirpGenerator,
    StepGenerator,
    PRBSGenerator,
    CompositeExcitationGenerator,
    # Controller
    ProbingController,
    ProbingState,
    # Analyzer
    ResponseAnalyzer,
    ResponseAnalysisResult,
    # System
    ActiveProbingSystem,
    ActiveProbingResult,
    # Evaluation
    evaluate_active_probing,
)

# PINN integration (v0.9.0) - PHYSICS-INFORMED ENHANCEMENT
from .pinn_integration import (
    # Option 1: Shadow Residual (recommended)
    QuadrotorPINNResidual,
    ShadowResidualResult,
    PINNShadowResidual,
    evaluate_pinn_shadow,
    # Option 2: Envelope Learning
    ControlRegime,
    PhysicsEnvelope,
    EnvelopeResult,
    PINNEnvelopeLearner,
    evaluate_pinn_envelope,
    # Option 3: Probing Response
    ProbingPredictionResult,
    PINNProbingPredictor,
    PINNResponsePredictor,
    evaluate_pinn_probing,
)

__all__ = [
    # Core
    "CycleConsistencyDetector",
    "InverseModel",
    "ForwardModel",
    # Temporal aggregation
    "TemporalICIAggregator",
    "TemporalICIConfig",
    "ConsensusAggregator",
    # Conditional fusion
    "ConditionalHybridFusion",
    "ConditionalFusionConfig",
    # IASP v2
    "IASPv2Healer",
    "IASPv2Config",
    "AdaptiveIASP",
    # Coordinated defense (v0.3.0)
    "CoordinatedDefenseSystem",
    "CoordinatedDefenseConfig",
    "MultiScaleAggregator",
    "TimingCoherenceAnalyzer",
    "OverConsistencyDetector",
    "PersistenceLogic",
    # Actuator observability (v0.4.0)
    "ControlEffortChecker",
    "ControlEffortMetrics",
    "DualTimescaleDetector",
    "DualScaleResult",
    "ResidualEnvelopeNormalizer",
    "SplitFaultHead",
    "PhaseConsistencyChecker",
    "EnhancedActuatorDetector",
    "compute_proper_metrics",
    "extract_motor_features",
    "extract_actuator_features",
    # Advanced detection (v0.5.0)
    "LagDriftTracker",
    "LagDriftResult",
    "SecondOrderConsistency",
    "SecondOrderResult",
    "ControlRegime",
    "ControlRegimeEnvelopes",
    "RegimeEnvelope",
    "FaultType",
    "FaultAttributor",
    "FaultAttribution",
    "PredictionRetrodictionChecker",
    "AsymmetryResult",
    "RandomizedSubspaceSampler",
    "RandomizedResult",
    "AdvancedDetector",
    "AdvancedDetectionResult",
    # Final improvements (v0.5.1)
    "FaultPersistenceScorer",
    "PersistenceResult",
    "FaultClass",
    "AsymmetricThresholds",
    "CostAwareThresholder",
    "CostAwareResult",
    "TTDMetrics",
    "TTDAnalyzer",
    "ControllerPredictor",
    "ControllerResidualResult",
    "CrossAxisCouplingChecker",
    "CrossAxisResult",
    "FinalDetector",
    "FinalDetectionResult",
    "evaluate_with_final_improvements",
    # Industry-aligned (v0.6.0)
    "TwoStageDecisionLogic",
    "TwoStageResult",
    "SuspicionState",
    "HazardClass",
    "HazardThresholds",
    "RiskWeightedDetector",
    "RiskWeightedResult",
    "FAULT_HAZARD_MAPPING",
    "ProtectionLevels",
    "AlertLimits",
    "IntegrityMonitor",
    "IntegrityResult",
    "IndustryAlignedDetector",
    "IndustryAlignedResult",
    "evaluate_industry_aligned",
    # Analytical redundancy (v0.7.0)
    "EKFState",
    "NonlinearEKF",
    "LinearComplementaryEstimator",
    "DisagreementResult",
    "EstimatorDisagreementDetector",
    "AnalyticalRedundancyResult",
    "AnalyticalRedundancySystem",
    "evaluate_analytical_redundancy",
    # Active probing (v0.8.0)
    "ExcitationType",
    "ExcitationSignal",
    "MicroChirpGenerator",
    "StepGenerator",
    "PRBSGenerator",
    "CompositeExcitationGenerator",
    "ProbingController",
    "ProbingState",
    "ResponseAnalyzer",
    "ResponseAnalysisResult",
    "ActiveProbingSystem",
    "ActiveProbingResult",
    "evaluate_active_probing",
    # PINN integration (v0.9.0)
    "QuadrotorPINNResidual",
    "ShadowResidualResult",
    "PINNShadowResidual",
    "evaluate_pinn_shadow",
    "ControlRegime",
    "PhysicsEnvelope",
    "EnvelopeResult",
    "PINNEnvelopeLearner",
    "evaluate_pinn_envelope",
    "ProbingPredictionResult",
    "PINNProbingPredictor",
    "PINNResponsePredictor",
    "evaluate_pinn_probing",
]
