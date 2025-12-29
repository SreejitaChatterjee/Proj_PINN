"""
Explainable Alarms: Per-Alarm Attribution

Maps each alarm to the contributing feature groups:
- PINN residuals (physics violations)
- EKF NIS (filter inconsistencies)
- ML detector (learned patterns)
- Temporal features (statistical anomalies)

Provides interpretable explanations for operators.

Usage:
    explainer = AlarmExplainer(weights)
    explanation = explainer.explain(component_scores)
    print(explanation.summary())
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class AlarmSource(Enum):
    """Sources that can trigger alarms."""
    PINN = "pinn_residual"
    EKF = "ekf_nis"
    ML = "ml_detector"
    TEMPORAL = "temporal_stats"
    MULTI = "multiple_sources"


@dataclass
class AlarmExplanation:
    """Explanation for a single alarm."""
    timestamp: int
    is_alarm: bool
    fused_score: float
    threshold: float

    # Per-component contributions
    pinn_score: float
    ekf_score: float
    ml_score: float
    temporal_score: float

    # Weighted contributions (after calibration)
    pinn_contribution: float
    ekf_contribution: float
    ml_contribution: float
    temporal_contribution: float

    # Primary source
    primary_source: AlarmSource
    confidence: float

    # Additional context
    anomaly_type_guess: Optional[str] = None
    sensor_group_guess: Optional[str] = None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append(f"{'ALARM' if self.is_alarm else 'NORMAL'} at t={self.timestamp}")
        lines.append(f"  Fused score: {self.fused_score:.3f} (threshold: {self.threshold:.3f})")
        lines.append(f"  Primary source: {self.primary_source.value}")
        lines.append(f"  Confidence: {self.confidence:.1%}")
        lines.append(f"  Contributions:")
        lines.append(f"    PINN:     {self.pinn_contribution:.1%}")
        lines.append(f"    EKF:      {self.ekf_contribution:.1%}")
        lines.append(f"    ML:       {self.ml_contribution:.1%}")
        lines.append(f"    Temporal: {self.temporal_contribution:.1%}")

        if self.anomaly_type_guess:
            lines.append(f"  Likely attack: {self.anomaly_type_guess}")
        if self.sensor_group_guess:
            lines.append(f"  Likely sensor: {self.sensor_group_guess}")

        return "\n".join(lines)


@dataclass
class BatchExplanation:
    """Explanations for a batch of samples."""
    explanations: List[AlarmExplanation]
    n_alarms: int
    n_samples: int

    # Aggregate statistics
    source_distribution: Dict[str, int] = field(default_factory=dict)
    mean_contributions: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate batch summary."""
        lines = []
        lines.append("=" * 50)
        lines.append("ALARM EXPLANATION SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Total samples: {self.n_samples}")
        lines.append(f"Total alarms:  {self.n_alarms}")
        lines.append(f"Alarm rate:    {self.n_alarms/self.n_samples:.2%}")

        lines.append("\nAlarm source distribution:")
        for source, count in sorted(self.source_distribution.items(),
                                   key=lambda x: -x[1]):
            pct = count / max(self.n_alarms, 1)
            lines.append(f"  {source}: {count} ({pct:.1%})")

        lines.append("\nMean contributions (for alarms):")
        for component, contrib in self.mean_contributions.items():
            lines.append(f"  {component}: {contrib:.1%}")

        lines.append("=" * 50)
        return "\n".join(lines)


class AlarmExplainer:
    """
    Explain alarms by attributing to component sources.
    """

    def __init__(
        self,
        weights: np.ndarray,
        threshold: float,
        component_names: List[str] = None
    ):
        """
        Initialize explainer.

        Args:
            weights: Calibrated fusion weights [w_pinn, w_ekf, w_ml, w_temporal]
            threshold: Detection threshold
            component_names: Names of components
        """
        self.weights = weights / weights.sum()  # Normalize
        self.threshold = threshold
        self.component_names = component_names or ['pinn', 'ekf', 'ml', 'temporal']

    def explain_single(
        self,
        component_scores: np.ndarray,
        timestamp: int = 0
    ) -> AlarmExplanation:
        """
        Explain a single sample.

        Args:
            component_scores: [n_components] array of raw scores
            timestamp: Sample timestamp

        Returns:
            AlarmExplanation
        """
        # Compute fused score
        fused = np.dot(component_scores, self.weights)
        is_alarm = fused > self.threshold

        # Compute contributions (how much each component contributes to fused)
        weighted_scores = component_scores * self.weights
        total_weighted = weighted_scores.sum()

        if total_weighted > 0:
            contributions = weighted_scores / total_weighted
        else:
            contributions = self.weights.copy()

        # Identify primary source
        primary_idx = np.argmax(weighted_scores)
        primary_source = self._get_source_enum(primary_idx)

        # Check if multiple sources are significant
        significant_mask = contributions > 0.25
        if significant_mask.sum() > 1:
            primary_source = AlarmSource.MULTI

        # Confidence based on margin above threshold
        if is_alarm:
            margin = (fused - self.threshold) / self.threshold
            confidence = min(1.0, 0.5 + margin)
        else:
            margin = (self.threshold - fused) / self.threshold
            confidence = min(1.0, 0.5 + margin)

        # Guess anomaly type based on score patterns
        anomaly_guess = self._guess_anomaly_type(component_scores)
        sensor_guess = self._guess_sensor_group(component_scores)

        return AlarmExplanation(
            timestamp=timestamp,
            is_alarm=is_alarm,
            fused_score=float(fused),
            threshold=self.threshold,
            pinn_score=float(component_scores[0]),
            ekf_score=float(component_scores[1]),
            ml_score=float(component_scores[2]),
            temporal_score=float(component_scores[3]),
            pinn_contribution=float(contributions[0]),
            ekf_contribution=float(contributions[1]),
            ml_contribution=float(contributions[2]),
            temporal_contribution=float(contributions[3]),
            primary_source=primary_source,
            confidence=float(confidence),
            anomaly_type_guess=anomaly_guess if is_alarm else None,
            sensor_group_guess=sensor_guess if is_alarm else None
        )

    def explain_batch(
        self,
        component_scores: np.ndarray,
        start_timestamp: int = 0
    ) -> BatchExplanation:
        """
        Explain a batch of samples.

        Args:
            component_scores: [n_samples, n_components] array
            start_timestamp: Starting timestamp

        Returns:
            BatchExplanation
        """
        explanations = []
        for i, scores in enumerate(component_scores):
            exp = self.explain_single(scores, timestamp=start_timestamp + i)
            explanations.append(exp)

        # Compute aggregate statistics
        alarms = [e for e in explanations if e.is_alarm]
        n_alarms = len(alarms)

        source_dist = {}
        for e in alarms:
            source = e.primary_source.value
            source_dist[source] = source_dist.get(source, 0) + 1

        mean_contribs = {}
        if alarms:
            mean_contribs['pinn'] = np.mean([e.pinn_contribution for e in alarms])
            mean_contribs['ekf'] = np.mean([e.ekf_contribution for e in alarms])
            mean_contribs['ml'] = np.mean([e.ml_contribution for e in alarms])
            mean_contribs['temporal'] = np.mean([e.temporal_contribution for e in alarms])

        return BatchExplanation(
            explanations=explanations,
            n_alarms=n_alarms,
            n_samples=len(explanations),
            source_distribution=source_dist,
            mean_contributions=mean_contribs
        )

    def _get_source_enum(self, idx: int) -> AlarmSource:
        """Map index to AlarmSource."""
        mapping = {
            0: AlarmSource.PINN,
            1: AlarmSource.EKF,
            2: AlarmSource.ML,
            3: AlarmSource.TEMPORAL
        }
        return mapping.get(idx, AlarmSource.ML)

    def _guess_anomaly_type(self, scores: np.ndarray) -> str:
        """
        Guess anomaly type based on score pattern.

        Heuristics:
        - High PINN + low EKF → Physics violation (bias, drift)
        - High EKF + low PINN → Filter inconsistency (noise injection)
        - High temporal only → Statistical anomaly
        - All high → Large magnitude attack
        """
        pinn, ekf, ml, temporal = scores

        if pinn > ekf * 1.5 and pinn > ml:
            return "physics_violation (likely bias/drift)"
        elif ekf > pinn * 1.5:
            return "filter_inconsistency (likely noise)"
        elif temporal > (pinn + ekf + ml) / 3 * 1.5:
            return "statistical_anomaly (likely subtle)"
        elif min(scores) > np.mean(scores) * 0.7:
            return "large_magnitude_attack"
        else:
            return "unknown_pattern"

    def _guess_sensor_group(self, scores: np.ndarray) -> str:
        """
        Guess affected sensor group.

        This is a simplified heuristic - real implementation would
        need per-sensor scores.
        """
        pinn, ekf, ml, temporal = scores

        # PINN typically catches position/velocity anomalies
        # EKF catches attitude/angular rate issues
        if pinn > ekf:
            return "position/velocity sensors"
        elif ekf > pinn:
            return "attitude/angular_rate sensors"
        else:
            return "multiple sensor groups"


class RuleFusionExplainer:
    """
    Rule-based fusion for improved attribution accuracy.

    Uses interpretable rules to combine component signals:
    - Rule 1: Physics + EKF → High confidence bias/drift
    - Rule 2: EKF only → Possible noise injection
    - Rule 3: ML only → Learned pattern (possibly spurious)
    - Rule 4: Temporal only → Statistical anomaly
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.rules = self._define_rules()

    def _define_rules(self) -> List[Dict]:
        """Define interpretable fusion rules."""
        return [
            {
                'name': 'physics_bias',
                'condition': lambda p, e, m, t: p > 0.7 and e > 0.5,
                'priority': 1,
                'explanation': 'Physics constraint violated with filter inconsistency → likely bias/drift attack',
                'confidence': 0.9
            },
            {
                'name': 'noise_injection',
                'condition': lambda p, e, m, t: e > 0.7 and p < 0.3,
                'priority': 2,
                'explanation': 'Filter inconsistency without physics violation → likely noise injection',
                'confidence': 0.8
            },
            {
                'name': 'coordinated_attack',
                'condition': lambda p, e, m, t: p > 0.5 and e > 0.5 and m > 0.5,
                'priority': 1,
                'explanation': 'Multiple signals elevated → likely coordinated attack',
                'confidence': 0.85
            },
            {
                'name': 'subtle_drift',
                'condition': lambda p, e, m, t: t > 0.6 and p < 0.4 and e < 0.4,
                'priority': 3,
                'explanation': 'Statistical anomaly without strong physics signal → likely subtle drift',
                'confidence': 0.6
            },
            {
                'name': 'learned_pattern',
                'condition': lambda p, e, m, t: m > 0.7 and p < 0.3 and e < 0.3,
                'priority': 4,
                'explanation': 'ML detection without physics signal → verify with domain knowledge',
                'confidence': 0.5
            }
        ]

    def apply_rules(
        self,
        component_scores: np.ndarray
    ) -> Tuple[str, str, float]:
        """
        Apply rules to get explanation.

        Args:
            component_scores: [pinn, ekf, ml, temporal] normalized to [0, 1]

        Returns:
            (rule_name, explanation, confidence)
        """
        pinn, ekf, ml, temporal = component_scores

        # Check rules in priority order
        matching_rules = []
        for rule in self.rules:
            if rule['condition'](pinn, ekf, ml, temporal):
                matching_rules.append(rule)

        if matching_rules:
            # Return highest priority (lowest number)
            best = min(matching_rules, key=lambda r: r['priority'])
            return best['name'], best['explanation'], best['confidence']

        return 'unknown', 'No matching rule - manual investigation needed', 0.3


def generate_explanation_report(
    explanations: BatchExplanation,
    save_path: Optional[str] = None
) -> str:
    """
    Generate detailed explanation report.

    Args:
        explanations: BatchExplanation object
        save_path: Optional path to save report

    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("GPS-IMU ANOMALY DETECTION - EXPLANATION REPORT")
    lines.append("=" * 70)

    # Summary
    lines.append("\n" + explanations.summary())

    # Top alarms by confidence
    alarms = [e for e in explanations.explanations if e.is_alarm]
    alarms_sorted = sorted(alarms, key=lambda x: -x.confidence)[:10]

    if alarms_sorted:
        lines.append("\nTop 10 Highest Confidence Alarms:")
        lines.append("-" * 50)
        for i, alarm in enumerate(alarms_sorted, 1):
            lines.append(f"\n[{i}] {alarm.summary()}")

    report = "\n".join(lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {save_path}")

    return report


if __name__ == "__main__":
    # Demo
    np.random.seed(42)

    # Calibrated weights
    weights = np.array([0.3, 0.25, 0.25, 0.2])
    threshold = 0.5

    explainer = AlarmExplainer(weights, threshold)

    # Generate synthetic scores
    n_samples = 100

    # Normal samples (low scores)
    normal_scores = np.random.rand(70, 4) * 0.4

    # Attack samples (high scores, different patterns)
    attack1 = np.array([[0.8, 0.6, 0.5, 0.4]])  # PINN-dominated
    attack2 = np.array([[0.3, 0.8, 0.4, 0.3]])  # EKF-dominated
    attack3 = np.array([[0.7, 0.7, 0.7, 0.6]])  # Multi-source
    attack4 = np.array([[0.3, 0.3, 0.7, 0.3]])  # ML-dominated

    # Repeat attacks
    attack_scores = np.vstack([
        np.tile(attack1, (10, 1)) + np.random.randn(10, 4) * 0.1,
        np.tile(attack2, (7, 1)) + np.random.randn(7, 4) * 0.1,
        np.tile(attack3, (8, 1)) + np.random.randn(8, 4) * 0.1,
        np.tile(attack4, (5, 1)) + np.random.randn(5, 4) * 0.1,
    ])

    all_scores = np.vstack([normal_scores, attack_scores])

    # Get explanations
    batch_exp = explainer.explain_batch(all_scores)

    # Print summary
    print(batch_exp.summary())

    # Print a few individual explanations
    print("\n\nSample Alarm Explanations:")
    for exp in batch_exp.explanations:
        if exp.is_alarm:
            print("-" * 40)
            print(exp.summary())
            break

    # Test rule fusion
    print("\n\nRule Fusion Explainer Demo:")
    rule_explainer = RuleFusionExplainer()

    test_patterns = [
        np.array([0.8, 0.6, 0.4, 0.3]),
        np.array([0.2, 0.8, 0.3, 0.2]),
        np.array([0.7, 0.7, 0.7, 0.6]),
        np.array([0.2, 0.2, 0.8, 0.3]),
    ]

    for pattern in test_patterns:
        name, explanation, conf = rule_explainer.apply_rules(pattern)
        print(f"Pattern {pattern} → {name} (conf={conf:.1%})")
        print(f"  {explanation}")
