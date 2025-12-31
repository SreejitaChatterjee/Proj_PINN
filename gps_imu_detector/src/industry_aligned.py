"""
Industry-Aligned Detection (v0.6.0)

Implements three industry-standard techniques:

A. Two-Stage Decision Logic (DO-178C alignment)
   - Stage 1: High-sensitivity suspicion trigger
   - Stage 2: Temporal confirmation window (K of M)
   - Alarm only after confirmation

B. Risk-Weighted Fault Acceptance (DO-229 / MIL-STD-882E)
   - Hazard classes: Catastrophic, Hazardous, Major, Minor
   - Per-class thresholds based on risk
   - Aggressive detection for catastrophic, conservative for minor

C. Integrity-Based Detection (DO-229 GPS/GNSS)
   - Protection Level (HPL/VPL) computation
   - Alert when bounds exceed safety limits
   - Converts stealth failures to integrity violations

References:
- DO-178C: Software Considerations in Airborne Systems
- DO-229: MOPS for GPS/WAAS
- ARP4754A: Guidelines for Development of Civil Aircraft
- MIL-STD-882E: System Safety
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple
import numpy as np


# =============================================================================
# A. Two-Stage Decision Logic
# =============================================================================

@dataclass
class SuspicionState:
    """State of suspicion tracking."""
    is_suspicious: bool = False
    suspicion_count: int = 0
    confirmation_count: int = 0
    window_start: int = 0


@dataclass
class TwoStageResult:
    """Result from two-stage decision logic."""
    raw_score: float
    is_suspicious: bool  # Stage 1 triggered
    is_confirmed: bool   # Stage 2 confirmed
    is_alarm: bool       # Final alarm decision
    suspicion_count: int
    confirmation_ratio: float
    time_in_suspicion: int


class TwoStageDecisionLogic:
    """
    Two-stage decision logic for FPR reduction.

    Stage 1: Suspicion trigger (high sensitivity)
    Stage 2: Confirmation window (high specificity)

    Industry pattern:
        Detector -> Suspicion -> Confirmation -> Alarm

    Parameters:
        suspicion_threshold: Threshold to enter suspicion state
        confirmation_threshold: Threshold for confirmation votes
        confirmation_window_K: Number of samples in confirmation window
        confirmation_required_M: Minimum confirmations to alarm
        cooldown_samples: Samples to wait after alarm before re-arming
    """

    def __init__(
        self,
        suspicion_threshold: float = 0.4,
        confirmation_threshold: float = 0.5,
        confirmation_window_K: int = 40,  # 200ms at 200Hz
        confirmation_required_M: int = 25,  # ~62.5% of window
        cooldown_samples: int = 100,
    ):
        self.suspicion_threshold = suspicion_threshold
        self.confirmation_threshold = confirmation_threshold
        self.confirmation_window_K = confirmation_window_K
        self.confirmation_required_M = confirmation_required_M
        self.cooldown_samples = cooldown_samples

        # State
        self.state = SuspicionState()
        self.confirmation_buffer: List[bool] = []
        self.sample_count = 0
        self.cooldown_remaining = 0
        self.history: List[TwoStageResult] = []

    def update(self, score: float) -> TwoStageResult:
        """
        Process a new anomaly score through two-stage logic.

        Args:
            score: Raw anomaly score from detector

        Returns:
            TwoStageResult with alarm decision
        """
        self.sample_count += 1

        # Check cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return TwoStageResult(
                raw_score=score,
                is_suspicious=False,
                is_confirmed=False,
                is_alarm=False,
                suspicion_count=0,
                confirmation_ratio=0.0,
                time_in_suspicion=0,
            )

        # Stage 1: Suspicion check
        is_suspicious = score >= self.suspicion_threshold

        if is_suspicious and not self.state.is_suspicious:
            # Enter suspicion state
            self.state.is_suspicious = True
            self.state.window_start = self.sample_count
            self.confirmation_buffer = []

        if self.state.is_suspicious:
            self.state.suspicion_count += 1

            # Stage 2: Confirmation accumulation
            is_confirmation = score >= self.confirmation_threshold
            self.confirmation_buffer.append(is_confirmation)

            # Keep only last K samples
            if len(self.confirmation_buffer) > self.confirmation_window_K:
                self.confirmation_buffer.pop(0)

            confirmation_count = sum(self.confirmation_buffer)
            confirmation_ratio = confirmation_count / len(self.confirmation_buffer)

            # Check for alarm condition
            is_confirmed = confirmation_count >= self.confirmation_required_M
            is_alarm = is_confirmed and len(self.confirmation_buffer) >= self.confirmation_window_K // 2

            if is_alarm:
                # Alarm triggered - enter cooldown
                self.cooldown_remaining = self.cooldown_samples
                time_in_suspicion = self.sample_count - self.state.window_start

                result = TwoStageResult(
                    raw_score=score,
                    is_suspicious=True,
                    is_confirmed=True,
                    is_alarm=True,
                    suspicion_count=self.state.suspicion_count,
                    confirmation_ratio=confirmation_ratio,
                    time_in_suspicion=time_in_suspicion,
                )

                # Reset state
                self.state = SuspicionState()
                self.confirmation_buffer = []
                self.history.append(result)
                return result

            # Check for suspicion timeout (no confirmation after full window)
            if len(self.confirmation_buffer) >= self.confirmation_window_K and not is_confirmed:
                # Exit suspicion without alarm
                self.state = SuspicionState()
                self.confirmation_buffer = []

            result = TwoStageResult(
                raw_score=score,
                is_suspicious=True,
                is_confirmed=is_confirmed,
                is_alarm=False,
                suspicion_count=self.state.suspicion_count,
                confirmation_ratio=confirmation_ratio,
                time_in_suspicion=self.sample_count - self.state.window_start,
            )
        else:
            result = TwoStageResult(
                raw_score=score,
                is_suspicious=False,
                is_confirmed=False,
                is_alarm=False,
                suspicion_count=0,
                confirmation_ratio=0.0,
                time_in_suspicion=0,
            )

        self.history.append(result)
        return result

    def reset(self):
        """Reset all state."""
        self.state = SuspicionState()
        self.confirmation_buffer = []
        self.sample_count = 0
        self.cooldown_remaining = 0
        self.history = []

    def get_alarm_rate(self) -> float:
        """Get fraction of samples that triggered alarms."""
        if not self.history:
            return 0.0
        return sum(1 for r in self.history if r.is_alarm) / len(self.history)


# =============================================================================
# B. Risk-Weighted Fault Acceptance (Hazard Classes)
# =============================================================================

class HazardClass(Enum):
    """
    MIL-STD-882E / ARP4754A Hazard Classification.

    Catastrophic: Could cause death or system loss
    Hazardous: Large reduction in safety margins
    Major: Significant reduction in safety margins
    Minor: Nuisance, no significant safety impact
    """
    CATASTROPHIC = auto()  # DAL A
    HAZARDOUS = auto()     # DAL B
    MAJOR = auto()         # DAL C
    MINOR = auto()         # DAL D


@dataclass
class HazardThresholds:
    """Per-hazard-class detection thresholds."""
    # Lower threshold = more aggressive detection
    catastrophic: float = 0.20  # Very aggressive
    hazardous: float = 0.35
    major: float = 0.50
    minor: float = 0.70  # Conservative

    def get(self, hazard_class: HazardClass) -> float:
        """Get threshold for hazard class."""
        mapping = {
            HazardClass.CATASTROPHIC: self.catastrophic,
            HazardClass.HAZARDOUS: self.hazardous,
            HazardClass.MAJOR: self.major,
            HazardClass.MINOR: self.minor,
        }
        return mapping.get(hazard_class, self.major)


@dataclass
class RiskWeightedResult:
    """Result from risk-weighted detection."""
    score: float
    hazard_class: HazardClass
    threshold: float
    is_detected: bool
    risk_priority: int  # 1=highest, 4=lowest


# Fault type to hazard class mapping
FAULT_HAZARD_MAPPING: Dict[str, HazardClass] = {
    # Catastrophic - loss of control
    "actuator_stuck": HazardClass.CATASTROPHIC,
    "actuator_degraded": HazardClass.CATASTROPHIC,
    "motor_failure": HazardClass.CATASTROPHIC,
    "total_motor_loss": HazardClass.CATASTROPHIC,

    # Hazardous - major capability loss
    "gps_spoofing": HazardClass.HAZARDOUS,
    "coordinated_attack": HazardClass.HAZARDOUS,
    "imu_bias": HazardClass.HAZARDOUS,

    # Major - degraded operation
    "gps_drift": HazardClass.MAJOR,
    "sensor_noise": HazardClass.MAJOR,
    "time_delay": HazardClass.MAJOR,

    # Minor - nuisance
    "minor_offset": HazardClass.MINOR,
    "transient_glitch": HazardClass.MINOR,
}


class RiskWeightedDetector:
    """
    Risk-weighted fault acceptance based on hazard classification.

    Implements DO-229 / MIL-STD-882E principles:
    - Aggressive thresholds for catastrophic faults
    - Conservative thresholds for minor faults
    - Explicit hazard class assignment

    Parameters:
        thresholds: HazardThresholds configuration
        fault_mapping: Dict mapping fault types to hazard classes
    """

    def __init__(
        self,
        thresholds: Optional[HazardThresholds] = None,
        fault_mapping: Optional[Dict[str, HazardClass]] = None,
    ):
        self.thresholds = thresholds or HazardThresholds()
        self.fault_mapping = fault_mapping or FAULT_HAZARD_MAPPING
        self.detection_counts: Dict[HazardClass, int] = {h: 0 for h in HazardClass}
        self.total_counts: Dict[HazardClass, int] = {h: 0 for h in HazardClass}

    def classify_fault(self, fault_type: str) -> HazardClass:
        """
        Classify a fault type into hazard class.

        Args:
            fault_type: String identifier of fault type

        Returns:
            HazardClass enum value
        """
        # Normalize fault type
        fault_lower = fault_type.lower().replace(" ", "_").replace("-", "_")

        # Direct lookup
        if fault_lower in self.fault_mapping:
            return self.fault_mapping[fault_lower]

        # Pattern matching
        if "actuator" in fault_lower or "motor" in fault_lower:
            return HazardClass.CATASTROPHIC
        if "spoof" in fault_lower or "attack" in fault_lower:
            return HazardClass.HAZARDOUS
        if "drift" in fault_lower or "bias" in fault_lower:
            return HazardClass.MAJOR

        # Default to Major
        return HazardClass.MAJOR

    def detect(
        self,
        score: float,
        fault_type: str,
        hazard_override: Optional[HazardClass] = None,
    ) -> RiskWeightedResult:
        """
        Apply risk-weighted detection.

        Args:
            score: Anomaly score
            fault_type: Type of fault being checked
            hazard_override: Optional explicit hazard class

        Returns:
            RiskWeightedResult with detection decision
        """
        hazard_class = hazard_override or self.classify_fault(fault_type)
        threshold = self.thresholds.get(hazard_class)
        is_detected = score >= threshold

        # Track statistics
        self.total_counts[hazard_class] += 1
        if is_detected:
            self.detection_counts[hazard_class] += 1

        risk_priority = {
            HazardClass.CATASTROPHIC: 1,
            HazardClass.HAZARDOUS: 2,
            HazardClass.MAJOR: 3,
            HazardClass.MINOR: 4,
        }[hazard_class]

        return RiskWeightedResult(
            score=score,
            hazard_class=hazard_class,
            threshold=threshold,
            is_detected=is_detected,
            risk_priority=risk_priority,
        )

    def get_recall_by_class(self) -> Dict[HazardClass, float]:
        """Get detection recall per hazard class."""
        recalls = {}
        for hc in HazardClass:
            if self.total_counts[hc] > 0:
                recalls[hc] = self.detection_counts[hc] / self.total_counts[hc]
            else:
                recalls[hc] = 0.0
        return recalls

    def reset(self):
        """Reset detection counts."""
        self.detection_counts = {h: 0 for h in HazardClass}
        self.total_counts = {h: 0 for h in HazardClass}


# =============================================================================
# C. Integrity-Based Detection (Protection Levels)
# =============================================================================

@dataclass
class ProtectionLevels:
    """
    GPS/GNSS Protection Levels (DO-229 style).

    HPL: Horizontal Protection Level
    VPL: Vertical Protection Level

    Alert when protection level exceeds alert limit.
    """
    hpl: float = 0.0  # Horizontal protection level (meters)
    vpl: float = 0.0  # Vertical protection level (meters)
    hpl_variance: float = 0.0
    vpl_variance: float = 0.0


@dataclass
class AlertLimits:
    """Alert limits for different flight phases."""
    # DO-229 Cat I precision approach
    hal_approach: float = 40.0  # meters
    val_approach: float = 35.0  # meters

    # En-route
    hal_enroute: float = 556.0  # 0.3 NM
    val_enroute: float = 556.0

    # Terminal
    hal_terminal: float = 185.0  # 0.1 NM
    val_terminal: float = 185.0


@dataclass
class IntegrityResult:
    """Result from integrity-based detection."""
    hpl: float
    vpl: float
    hal: float  # Horizontal alert limit
    val: float  # Vertical alert limit
    horizontal_integrity: bool  # HPL < HAL
    vertical_integrity: bool    # VPL < VAL
    overall_integrity: bool     # Both satisfied
    integrity_margin_h: float   # HAL - HPL
    integrity_margin_v: float   # VAL - VPL
    is_alert: bool              # Integrity violation


class IntegrityMonitor:
    """
    Integrity-based detection using protection levels.

    DO-229 approach: Instead of asking "is this anomalous?",
    ask "is integrity still guaranteed?"

    This converts stealth failures into integrity violations
    by computing protection levels that bound position error.

    Parameters:
        alert_limits: AlertLimits configuration
        k_factor: Confidence multiplier for protection levels (typically 5.33 for 10^-7)
        position_noise_std: Expected position noise standard deviation
    """

    def __init__(
        self,
        alert_limits: Optional[AlertLimits] = None,
        k_factor: float = 5.33,  # 10^-7 integrity risk
        position_noise_std: float = 1.0,  # meters
    ):
        self.alert_limits = alert_limits or AlertLimits()
        self.k_factor = k_factor
        self.position_noise_std = position_noise_std

        # State for protection level computation
        self.position_history: List[np.ndarray] = []
        self.velocity_history: List[np.ndarray] = []
        self.innovation_history: List[float] = []
        self.window_size = 100

        # Statistics
        self.alert_count = 0
        self.total_count = 0

    def compute_protection_levels(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        innovation: Optional[float] = None,
        covariance: Optional[np.ndarray] = None,
    ) -> ProtectionLevels:
        """
        Compute protection levels from current state.

        Args:
            position: Current position estimate [x, y, z]
            velocity: Current velocity estimate [vx, vy, vz]
            innovation: Filter innovation (residual) if available
            covariance: Position covariance matrix if available

        Returns:
            ProtectionLevels with HPL/VPL
        """
        # Update history
        self.position_history.append(position.copy())
        self.velocity_history.append(velocity.copy())
        if innovation is not None:
            self.innovation_history.append(innovation)

        # Trim to window
        if len(self.position_history) > self.window_size:
            self.position_history.pop(0)
        if len(self.velocity_history) > self.window_size:
            self.velocity_history.pop(0)
        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)

        if covariance is not None:
            # Use provided covariance
            h_var = covariance[0, 0] + covariance[1, 1]
            v_var = covariance[2, 2]
        else:
            # Estimate from history
            if len(self.position_history) >= 10:
                positions = np.array(self.position_history)
                h_var = np.var(positions[:, 0]) + np.var(positions[:, 1])
                v_var = np.var(positions[:, 2])
            else:
                h_var = self.position_noise_std ** 2
                v_var = self.position_noise_std ** 2

        # Add innovation-based inflation
        if len(self.innovation_history) >= 5:
            innovation_std = np.std(self.innovation_history)
            inflation = 1.0 + innovation_std
            h_var *= inflation
            v_var *= inflation

        # Compute protection levels
        hpl = self.k_factor * np.sqrt(h_var)
        vpl = self.k_factor * np.sqrt(v_var)

        return ProtectionLevels(
            hpl=hpl,
            vpl=vpl,
            hpl_variance=h_var,
            vpl_variance=v_var,
        )

    def check_integrity(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        flight_phase: str = "enroute",
        innovation: Optional[float] = None,
        covariance: Optional[np.ndarray] = None,
    ) -> IntegrityResult:
        """
        Check integrity using protection levels.

        Args:
            position: Current position estimate
            velocity: Current velocity estimate
            flight_phase: One of "approach", "terminal", "enroute"
            innovation: Filter innovation if available
            covariance: Position covariance if available

        Returns:
            IntegrityResult with alert decision
        """
        self.total_count += 1

        # Compute protection levels
        pl = self.compute_protection_levels(position, velocity, innovation, covariance)

        # Get alert limits for flight phase
        if flight_phase == "approach":
            hal = self.alert_limits.hal_approach
            val = self.alert_limits.val_approach
        elif flight_phase == "terminal":
            hal = self.alert_limits.hal_terminal
            val = self.alert_limits.val_terminal
        else:
            hal = self.alert_limits.hal_enroute
            val = self.alert_limits.val_enroute

        # Check integrity
        h_ok = pl.hpl < hal
        v_ok = pl.vpl < val
        overall_ok = h_ok and v_ok

        is_alert = not overall_ok
        if is_alert:
            self.alert_count += 1

        return IntegrityResult(
            hpl=pl.hpl,
            vpl=pl.vpl,
            hal=hal,
            val=val,
            horizontal_integrity=h_ok,
            vertical_integrity=v_ok,
            overall_integrity=overall_ok,
            integrity_margin_h=hal - pl.hpl,
            integrity_margin_v=val - pl.vpl,
            is_alert=is_alert,
        )

    def get_alert_rate(self) -> float:
        """Get fraction of samples with integrity alerts."""
        if self.total_count == 0:
            return 0.0
        return self.alert_count / self.total_count

    def reset(self):
        """Reset all state."""
        self.position_history = []
        self.velocity_history = []
        self.innovation_history = []
        self.alert_count = 0
        self.total_count = 0


# =============================================================================
# Combined Industry-Aligned Detector
# =============================================================================

@dataclass
class IndustryAlignedResult:
    """Combined result from industry-aligned detection."""
    # Raw inputs
    anomaly_score: float
    fault_type: str

    # Two-stage decision
    two_stage: TwoStageResult

    # Risk-weighted
    risk_weighted: RiskWeightedResult

    # Integrity (if applicable)
    integrity: Optional[IntegrityResult]

    # Final decision
    final_alarm: bool
    alarm_source: str  # "two_stage", "risk_weighted", "integrity", "none"
    confidence: float


class IndustryAlignedDetector:
    """
    Combined industry-aligned detector.

    Integrates:
    - Two-stage decision logic (DO-178C FPR reduction)
    - Risk-weighted thresholds (MIL-STD-882E)
    - Integrity monitoring (DO-229)

    Final alarm logic:
    - Catastrophic faults: Risk-weighted OR two-stage
    - Other faults: Two-stage AND (risk-weighted OR integrity)

    Parameters:
        two_stage_config: Dict of TwoStageDecisionLogic params
        hazard_thresholds: HazardThresholds configuration
        alert_limits: AlertLimits for integrity monitoring
        enable_integrity: Whether to use integrity-based detection
    """

    def __init__(
        self,
        two_stage_config: Optional[Dict] = None,
        hazard_thresholds: Optional[HazardThresholds] = None,
        alert_limits: Optional[AlertLimits] = None,
        enable_integrity: bool = True,
    ):
        two_stage_config = two_stage_config or {}
        self.two_stage = TwoStageDecisionLogic(**two_stage_config)
        self.risk_weighted = RiskWeightedDetector(thresholds=hazard_thresholds)
        self.integrity = IntegrityMonitor(alert_limits=alert_limits) if enable_integrity else None
        self.enable_integrity = enable_integrity

        # Statistics
        self.alarm_count = 0
        self.total_count = 0
        self.alarm_sources: Dict[str, int] = {
            "two_stage": 0,
            "risk_weighted": 0,
            "integrity": 0,
            "combined": 0,
        }

    def detect(
        self,
        anomaly_score: float,
        fault_type: str,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        flight_phase: str = "enroute",
        innovation: Optional[float] = None,
    ) -> IndustryAlignedResult:
        """
        Run industry-aligned detection.

        Args:
            anomaly_score: Raw anomaly score from base detector
            fault_type: Type of fault being checked
            position: Current position (for integrity monitoring)
            velocity: Current velocity (for integrity monitoring)
            flight_phase: Flight phase for alert limits
            innovation: Filter innovation (for integrity)

        Returns:
            IndustryAlignedResult with final alarm decision
        """
        self.total_count += 1

        # Two-stage decision
        ts_result = self.two_stage.update(anomaly_score)

        # Risk-weighted decision
        rw_result = self.risk_weighted.detect(anomaly_score, fault_type)

        # Integrity check (if enabled and position available)
        int_result = None
        if self.enable_integrity and position is not None and velocity is not None:
            int_result = self.integrity.check_integrity(
                position, velocity, flight_phase, innovation
            )

        # Final decision logic
        hazard = rw_result.hazard_class

        if hazard == HazardClass.CATASTROPHIC:
            # Catastrophic: aggressive - either trigger is enough
            final_alarm = ts_result.is_alarm or rw_result.is_detected
            if ts_result.is_alarm and rw_result.is_detected:
                alarm_source = "combined"
            elif ts_result.is_alarm:
                alarm_source = "two_stage"
            elif rw_result.is_detected:
                alarm_source = "risk_weighted"
            else:
                alarm_source = "none"
        else:
            # Non-catastrophic: require two-stage confirmation
            # Plus either risk-weighted or integrity
            secondary = rw_result.is_detected
            if int_result is not None:
                secondary = secondary or int_result.is_alert

            final_alarm = ts_result.is_alarm and secondary

            if final_alarm:
                if int_result is not None and int_result.is_alert:
                    alarm_source = "integrity"
                else:
                    alarm_source = "combined"
            else:
                alarm_source = "none"

        # Update statistics
        if final_alarm:
            self.alarm_count += 1
            self.alarm_sources[alarm_source] += 1

        # Compute confidence
        confidence = anomaly_score
        if ts_result.is_confirmed:
            confidence = max(confidence, ts_result.confirmation_ratio)
        if rw_result.is_detected:
            confidence = max(confidence, anomaly_score / rw_result.threshold)

        return IndustryAlignedResult(
            anomaly_score=anomaly_score,
            fault_type=fault_type,
            two_stage=ts_result,
            risk_weighted=rw_result,
            integrity=int_result,
            final_alarm=final_alarm,
            alarm_source=alarm_source,
            confidence=min(1.0, confidence),
        )

    def get_metrics(self) -> Dict[str, float]:
        """Get detection metrics."""
        return {
            "alarm_rate": self.alarm_count / max(1, self.total_count),
            "two_stage_alarms": self.alarm_sources["two_stage"],
            "risk_weighted_alarms": self.alarm_sources["risk_weighted"],
            "integrity_alarms": self.alarm_sources["integrity"],
            "combined_alarms": self.alarm_sources["combined"],
            "total_alarms": self.alarm_count,
            "total_samples": self.total_count,
        }

    def reset(self):
        """Reset all state."""
        self.two_stage.reset()
        self.risk_weighted.reset()
        if self.integrity:
            self.integrity.reset()
        self.alarm_count = 0
        self.total_count = 0
        self.alarm_sources = {k: 0 for k in self.alarm_sources}


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_industry_aligned(
    scores: np.ndarray,
    labels: np.ndarray,
    fault_types: List[str],
    positions: Optional[np.ndarray] = None,
    velocities: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate industry-aligned detector.

    Args:
        scores: Anomaly scores array
        labels: Ground truth labels (0=nominal, 1=fault)
        fault_types: List of fault type strings per sample
        positions: Optional position array [N, 3]
        velocities: Optional velocity array [N, 3]

    Returns:
        Dict with recall, FPR, and per-hazard metrics
    """
    detector = IndustryAlignedDetector()

    predictions = []
    for i in range(len(scores)):
        pos = positions[i] if positions is not None else None
        vel = velocities[i] if velocities is not None else None

        result = detector.detect(
            anomaly_score=float(scores[i]),
            fault_type=fault_types[i] if i < len(fault_types) else "unknown",
            position=pos,
            velocity=vel,
        )
        predictions.append(result.final_alarm)

    predictions = np.array(predictions)

    # Compute metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    precision = tp / max(1, tp + fp)

    # Per-hazard class recall
    hazard_recalls = detector.risk_weighted.get_recall_by_class()

    return {
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "f1": 2 * precision * recall / max(0.001, precision + recall),
        "catastrophic_recall": hazard_recalls.get(HazardClass.CATASTROPHIC, 0.0),
        "hazardous_recall": hazard_recalls.get(HazardClass.HAZARDOUS, 0.0),
        "major_recall": hazard_recalls.get(HazardClass.MAJOR, 0.0),
        "minor_recall": hazard_recalls.get(HazardClass.MINOR, 0.0),
        **detector.get_metrics(),
    }
