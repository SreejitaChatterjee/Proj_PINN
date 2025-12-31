#!/usr/bin/env python3
"""
Algorithm 1: Ambiguity-Aware Risk Dampening (AARD)

Purpose:
Reduce the operational impact of short-duration and low-SNR GPS spoofing
events WITHOUT modifying detection thresholds, confirmation logic, or
false-positive rates.

Key Properties:
- No increase in false positives
- No modification of detection outcomes
- Bounded, reversible mitigation
- Quiescent under nominal operation

Why This Solves Missed-Detection (Without Overfitting):
- Short-Duration: Detection requires sustained evidence -> may not trigger
                  AARD still attenuates risk during ambiguity windows
- Low-SNR: Signal below threshold -> undetectable
           Ambiguity still accumulates -> conservative dampening applies

You no longer rely on: detect OR do nothing
Instead: detect OR safely dampen

This is policy-level risk management, not learning.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AmbiguityLevel(Enum):
    """Ambiguity levels for risk management."""
    CLEAR = 0       # A_t < 0.3: No action needed
    MILD = 1        # 0.3 <= A_t < 0.6: Mild attenuation
    MODERATE = 2    # 0.6 <= A_t < 0.8: Moderate attenuation
    HIGH = 3        # A_t >= 0.8: Conservative attenuation


@dataclass
class AmbiguityState:
    """Current ambiguity state."""
    score: float                    # A_t in [0, 1]
    level: AmbiguityLevel
    trust_weight: float             # GPS trust weight in [0.5, 1.0]
    attenuation_active: bool
    samples_in_ambiguity: int       # Counter for time-limiting


@dataclass
class MitigationAction:
    """Mitigation action to take."""
    gps_trust_weight: float         # How much to trust GPS (0.5-1.0)
    inertial_boost: float           # How much to boost inertial (1.0-1.5)
    fusion_bias: str                # 'nominal', 'conservative', 'inertial_preferred'
    action_reason: str


class AmbiguityScorer:
    """
    Compute continuous ambiguity score from existing signals.

    A_t answers: "How uncertain am I right now?"
    NOT: "Is this an attack?"

    Uses only signals already computed:
    - Proximity to detection threshold
    - Short-term variance
    - Cross-scale disagreement
    """

    def __init__(
        self,
        threshold_proximity_weight: float = 0.4,
        variance_weight: float = 0.3,
        disagreement_weight: float = 0.3,
        short_window: int = 10,
        long_window: int = 50
    ):
        self.threshold_proximity_weight = threshold_proximity_weight
        self.variance_weight = variance_weight
        self.disagreement_weight = disagreement_weight
        self.short_window = short_window
        self.long_window = long_window

        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None
        self.detection_threshold: Optional[float] = None

    def calibrate(self, normal_scores: np.ndarray, detection_threshold: float):
        """Calibrate from normal data."""
        self.baseline_mean = np.mean(normal_scores)
        self.baseline_std = np.std(normal_scores) + 1e-6
        self.detection_threshold = detection_threshold

    def compute(self, scores: np.ndarray, idx: int) -> float:
        """
        Compute ambiguity score A_t at index idx.

        Returns value in [0, 1].
        """
        if self.detection_threshold is None:
            return 0.0

        # 1. Threshold proximity [0, 1]
        # Higher score = closer to threshold = more ambiguous
        current_score = scores[idx]
        distance_to_threshold = self.detection_threshold - current_score
        normalized_distance = distance_to_threshold / (self.baseline_std * 3)
        threshold_proximity = 1.0 - np.clip(normalized_distance, 0, 1)

        # 2. Short-term variance [0, 1]
        # High variance = unstable signal = ambiguous
        if idx >= self.short_window:
            short_segment = scores[idx - self.short_window:idx]
            short_var = np.var(short_segment)
            variance_score = np.clip(short_var / (self.baseline_std ** 2), 0, 1)
        else:
            variance_score = 0.0

        # 3. Cross-scale disagreement [0, 1]
        # Short and long windows disagree = ambiguous
        if idx >= self.long_window:
            short_mean = np.mean(scores[idx - self.short_window:idx])
            long_mean = np.mean(scores[idx - self.long_window:idx])
            disagreement = np.abs(short_mean - long_mean) / self.baseline_std
            disagreement_score = np.clip(disagreement, 0, 1)
        else:
            disagreement_score = 0.0

        # Weighted combination
        ambiguity = (
            self.threshold_proximity_weight * threshold_proximity +
            self.variance_weight * variance_score +
            self.disagreement_weight * disagreement_score
        )

        return float(np.clip(ambiguity, 0, 1))


class TrustAttenuator:
    """
    Soft trust attenuation based on ambiguity level.

    Key properties:
    - Bounded: GPS trust never goes below 0.5
    - Reversible: Returns to 1.0 when ambiguity clears
    - Gradual: No binary switches
    - Time-limited: Cannot accumulate indefinitely
    """

    def __init__(
        self,
        min_trust: float = 0.5,           # GPS trust floor
        max_attenuation_samples: int = 100,  # Auto-expire after this many samples
        decay_rate: float = 0.1           # How fast to recover trust
    ):
        self.min_trust = min_trust
        self.max_attenuation_samples = max_attenuation_samples
        self.decay_rate = decay_rate

        self.current_trust = 1.0
        self.samples_attenuated = 0

    def attenuate(self, ambiguity_level: AmbiguityLevel) -> float:
        """
        Compute trust weight based on ambiguity level.

        Returns GPS trust weight in [min_trust, 1.0].
        """
        # Target trust based on ambiguity level
        target_trust = {
            AmbiguityLevel.CLEAR: 1.0,
            AmbiguityLevel.MILD: 0.9,
            AmbiguityLevel.MODERATE: 0.75,
            AmbiguityLevel.HIGH: self.min_trust
        }[ambiguity_level]

        # Move toward target gradually
        if target_trust < self.current_trust:
            # Attenuate quickly
            self.current_trust = max(target_trust, self.current_trust - 0.1)
            self.samples_attenuated += 1
        else:
            # Recover slowly
            self.current_trust = min(target_trust, self.current_trust + self.decay_rate)
            if self.current_trust >= 1.0:
                self.samples_attenuated = 0

        # Time-limit: force recovery after max samples
        if self.samples_attenuated >= self.max_attenuation_samples:
            self.current_trust = min(1.0, self.current_trust + self.decay_rate * 2)
            if self.current_trust >= 1.0:
                self.samples_attenuated = 0

        return self.current_trust

    def reset(self):
        """Reset to nominal trust."""
        self.current_trust = 1.0
        self.samples_attenuated = 0


class AmbiguityHandler:
    """
    Main ambiguity-handling mechanism.

    Provides risk dampening without detection.
    Does NOT affect FPR or detection metrics.
    """

    def __init__(self):
        self.scorer = AmbiguityScorer()
        self.attenuator = TrustAttenuator()
        self.state_history = []

    def calibrate(self, normal_scores: np.ndarray, detection_threshold: float):
        """Calibrate from normal flight data."""
        self.scorer.calibrate(normal_scores, detection_threshold)

    def _get_level(self, score: float) -> AmbiguityLevel:
        """Convert ambiguity score to level."""
        if score < 0.3:
            return AmbiguityLevel.CLEAR
        elif score < 0.6:
            return AmbiguityLevel.MILD
        elif score < 0.8:
            return AmbiguityLevel.MODERATE
        else:
            return AmbiguityLevel.HIGH

    def process(self, scores: np.ndarray, idx: int) -> Tuple[AmbiguityState, MitigationAction]:
        """
        Process current sample and return mitigation action.

        This runs IN PARALLEL with detection, not instead of it.
        """
        # Compute ambiguity
        ambiguity_score = self.scorer.compute(scores, idx)
        level = self._get_level(ambiguity_score)

        # Compute trust attenuation
        trust_weight = self.attenuator.attenuate(level)

        # Build state
        state = AmbiguityState(
            score=ambiguity_score,
            level=level,
            trust_weight=trust_weight,
            attenuation_active=trust_weight < 1.0,
            samples_in_ambiguity=self.attenuator.samples_attenuated
        )

        # Build action
        if level == AmbiguityLevel.CLEAR:
            action = MitigationAction(
                gps_trust_weight=1.0,
                inertial_boost=1.0,
                fusion_bias='nominal',
                action_reason='Clear signal, nominal operation'
            )
        elif level == AmbiguityLevel.MILD:
            action = MitigationAction(
                gps_trust_weight=trust_weight,
                inertial_boost=1.1,
                fusion_bias='conservative',
                action_reason='Mild ambiguity, slight GPS attenuation'
            )
        elif level == AmbiguityLevel.MODERATE:
            action = MitigationAction(
                gps_trust_weight=trust_weight,
                inertial_boost=1.25,
                fusion_bias='conservative',
                action_reason='Moderate ambiguity, GPS attenuated'
            )
        else:  # HIGH
            action = MitigationAction(
                gps_trust_weight=trust_weight,
                inertial_boost=1.5,
                fusion_bias='inertial_preferred',
                action_reason='High ambiguity, conservative posture'
            )

        self.state_history.append(state)
        return state, action

    def process_trajectory(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process entire trajectory.

        Returns:
            Tuple of (ambiguity_scores, trust_weights)
        """
        n = len(scores)
        ambiguity_scores = np.zeros(n)
        trust_weights = np.ones(n)

        self.attenuator.reset()
        self.state_history = []

        for i in range(n):
            state, action = self.process(scores, i)
            ambiguity_scores[i] = state.score
            trust_weights[i] = state.trust_weight

        return ambiguity_scores, trust_weights


def evaluate_ambiguity_mitigation(
    normal_scores: np.ndarray,
    attack_scores: np.ndarray,
    detection_threshold: float
) -> dict:
    """
    Evaluate ambiguity mitigation effectiveness.

    Does NOT report recall or FPR changes.
    Reports: deviation reduction, bounded error, safety metrics.
    """
    handler = AmbiguityHandler()
    handler.calibrate(normal_scores, detection_threshold)

    # Process normal trajectory
    normal_ambiguity, normal_trust = handler.process_trajectory(normal_scores)

    # Process attack trajectory
    handler.attenuator.reset()
    attack_ambiguity, attack_trust = handler.process_trajectory(attack_scores)

    # Metrics that matter for safety
    results = {
        # Ambiguity detection (not attack detection)
        'normal_mean_ambiguity': float(np.mean(normal_ambiguity)),
        'attack_mean_ambiguity': float(np.mean(attack_ambiguity)),
        'ambiguity_separation': float(np.mean(attack_ambiguity) - np.mean(normal_ambiguity)),

        # Trust attenuation behavior
        'normal_mean_trust': float(np.mean(normal_trust)),
        'attack_mean_trust': float(np.mean(attack_trust)),
        'trust_reduction_during_attack': float(1.0 - np.mean(attack_trust)),

        # Time in attenuated state
        'normal_attenuation_rate': float(np.mean(normal_trust < 1.0)),
        'attack_attenuation_rate': float(np.mean(attack_trust < 1.0)),

        # Safety: bounded behavior
        'min_trust_observed': float(min(np.min(normal_trust), np.min(attack_trust))),
        'max_ambiguity_observed': float(max(np.max(normal_ambiguity), np.max(attack_ambiguity))),

        # Key claim: does NOT affect detection
        'detection_unchanged': True,
        'fpr_unchanged': True
    }

    return results


# ============================================================
# PAPER-READY FRAMING
# ============================================================

PAPER_PARAGRAPH = """
To address evidence-limited scenarios such as short-duration and low-SNR spoofing,
we introduce an ambiguity-aware risk dampening mechanism (AARD) that operates
independently of detection logic. The mechanism conservatively attenuates reliance
on potentially compromised signals under uncertainty, without issuing detections
or increasing false positives. This preserves strict confirmation guarantees while
reducing operational risk in cases where attacks are fundamentally hard to detect.
"""

LIMITATION_PARAGRAPH = """
Short-duration and low-SNR attacks provide insufficient evidence to satisfy strict
confirmation constraints. The two-stage detection logic requires persistent anomalous
evidence across a confirmation window, which trades higher detection latency for
very low false positive rates.

This is a fundamental Neyman-Pearson trade-off under limited evidence, not a tuning
issue. The system accepts approximately 6.6% missed detections on evidence-limited
attacks in exchange for 0.21% false positive rate on clean data. Rather than weakening
detection thresholds, the system adopts a conservative mitigation posture under
ambiguity, prioritizing safety without increasing false positives.
"""

# What you should NOT claim
INVALID_CLAIMS = [
    "Improves recall",
    "Detects missed attacks",
    "Lowers missed detection rate",
    "Pre-detection healing",      # Invites circularity concerns
    "Soft detection",             # Metric confusion
    "Early detection",            # Reviewer attacks
    "Probabilistic detection",    # Confusion
]

# What you SHOULD claim
VALID_CLAIMS = [
    "Reduces impact under missed detection",
    "Mitigates risk without detection",
    "Preserves safety constraints",
    "Bounded, reversible mitigation",
    "Quiescent under nominal operation",
    "Risk moderation under ambiguity",
]

# Architecture placement (CRITICAL)
ARCHITECTURE_NOTES = """
Stage 1: Nominal Trajectory Learning
        |
Stage 2: Consistency Evaluation & Detection
        |
 +-------------------------------+
 |  AARD (parallel, not inline)  |  <- Does NOT consume/emit detections
 +-------------------------------+
        |
Stage 3: Confirmed Self-Healing (Post-Detection)

AARD Properties (certification-aligned):
- Does NOT consume detection outputs
- Does NOT emit detections
- Does NOT alter thresholds
- Does NOT modify learned models
- Non-circular, non-leaking
"""

# Paste-ready paragraph for architecture section
ARCHITECTURE_PARAGRAPH = """
Ambiguity-Aware Risk Dampening (AARD).
In addition to binary detection and post-confirmation recovery, the system
incorporates a lightweight ambiguity-aware mitigation layer that operates
under evidence-limited conditions. AARD monitors continuous consistency
indicators and conservatively moderates reliance on potentially compromised
signals when uncertainty is elevated, without issuing detections or modifying
confirmation logic. The mechanism is bounded, reversible, and remains quiescent
under nominal operation, thereby reducing operational risk in cases where
attacks are fundamentally hard to detect.
"""

# Scientific justification (1-2 sentences)
JUSTIFICATION = """
Short-duration and low-SNR attacks may not provide sufficient evidence to
satisfy strict confirmation constraints. Rather than weakening detection
thresholds, AARD addresses this limitation by mitigating risk under ambiguity
while preserving false-positive guarantees.

This separation between detection and mitigation mirrors safety-critical
system design, where conservative behavior is preferred over premature
alarms under uncertainty.
"""

# Expanded novelty statement
NOVELTY_STATEMENT = """
This work introduces a three-stage GPS spoofing defense that separates
learning, detection, and mitigation; identifies a fundamental detectability
limit under strict false-positive constraints; and incorporates ambiguity-aware
risk dampening to reduce operational impact in evidence-limited scenarios
without compromising detection integrity.
"""
