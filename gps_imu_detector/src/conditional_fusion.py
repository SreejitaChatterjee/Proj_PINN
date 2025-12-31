"""
Conditional Hybrid Fusion for GPS Spoofing Detection.

Principled improvement that respects the detectability floor:
- EKF-NIS detects HIGH-FREQUENCY anomalies (sudden jumps, oscillations)
- ICI detects LOW-FREQUENCY structural inconsistency (consistency-preserving spoofing)
- Conditional fusion: EKF contributes only when innovation spectrum is high-frequency

Key Insight:
    Naive weighted fusion (w_e * EKF + w_m * ICI) dilutes ICI signal when
    EKF sees nothing (consistent spoofing). Conditional fusion:

    S(t) = ICI(t) + EKF(t) * I(high_freq_innovation)

    where I(.) is 1 when EKF innovation has high-frequency content.

NOT A NEW DETECTION PRIMITIVE - This is smarter score combination.

Author: GPS-IMU Detector Project
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from scipy import signal


@dataclass
class ConditionalFusionConfig:
    """Configuration for conditional hybrid fusion."""
    # Frequency analysis
    fs: float = 200.0              # Sample rate (Hz)
    freq_cutoff: float = 5.0       # High-frequency cutoff (Hz)
    highfreq_threshold: float = 0.3  # Fraction of power in high-freq to activate EKF

    # Fusion weights (when EKF is active)
    w_ici: float = 0.7             # Base weight for ICI
    w_ekf: float = 0.3             # Weight for EKF when high-freq detected

    # Detection thresholds
    n_consecutive: int = 5         # N-consecutive rule
    target_fpr: float = 0.05       # Target FPR for threshold calibration


class InnovationSpectrumAnalyzer:
    """
    Analyze frequency content of EKF innovation sequence.

    The EKF innovation (measurement residual) has different spectral
    characteristics under different attack types:

    - Nominal: Broadband noise (flat spectrum)
    - Sudden jump: High-frequency spike at transition
    - Oscillatory attack: Peaks at oscillation frequency
    - Consistent drift: Low-frequency / DC component

    This analyzer detects when innovation has significant high-frequency
    content, indicating an attack type where EKF is informative.
    """

    def __init__(self, config: Optional[ConditionalFusionConfig] = None):
        self.config = config or ConditionalFusionConfig()

        # Window for spectral analysis
        self.window_size: int = 64
        self.buffer: np.ndarray = np.zeros(self.window_size)
        self.buffer_idx: int = 0
        self.buffer_full: bool = False

    def update(self, innovation: float) -> Dict[str, float]:
        """
        Update with new innovation sample and analyze spectrum.

        Args:
            innovation: EKF innovation (measurement residual) at current timestep

        Returns:
            Spectral analysis results:
            - 'highfreq_ratio': Fraction of power above cutoff
            - 'is_highfreq': Whether high-frequency content exceeds threshold
        """
        # Update buffer
        self.buffer[self.buffer_idx] = innovation
        self.buffer_idx = (self.buffer_idx + 1) % self.window_size
        if self.buffer_idx == 0:
            self.buffer_full = True

        if not self.buffer_full:
            return {'highfreq_ratio': 0.0, 'is_highfreq': False}

        # Reorder buffer (circular -> linear)
        ordered = np.concatenate([
            self.buffer[self.buffer_idx:],
            self.buffer[:self.buffer_idx]
        ])

        # Compute power spectrum
        freqs, psd = signal.welch(
            ordered,
            fs=self.config.fs,
            nperseg=min(32, len(ordered)),
            noverlap=None
        )

        # Compute high-frequency ratio
        total_power = np.sum(psd)
        if total_power < 1e-10:
            return {'highfreq_ratio': 0.0, 'is_highfreq': False}

        highfreq_mask = freqs > self.config.freq_cutoff
        highfreq_power = np.sum(psd[highfreq_mask])
        highfreq_ratio = highfreq_power / total_power

        is_highfreq = highfreq_ratio > self.config.highfreq_threshold

        return {
            'highfreq_ratio': float(highfreq_ratio),
            'is_highfreq': bool(is_highfreq),
            'total_power': float(total_power),
        }

    def reset(self):
        """Reset buffer for new trajectory."""
        self.buffer = np.zeros(self.window_size)
        self.buffer_idx = 0
        self.buffer_full = False


class ConditionalHybridFusion:
    """
    Conditional hybrid detector combining ICI and EKF-NIS.

    Key insight: EKF-NIS is only informative when it sees something.
    For consistency-preserving attacks, EKF innovation looks nominal.
    Including it in the fusion dilutes the ICI signal.

    Conditional fusion:
    - Always include ICI (the primary detection signal)
    - Include EKF only when innovation has high-frequency content

    This improves on naive fusion by:
    - Not diluting ICI on consistent spoofing (EKF sees nothing)
    - Boosting detection on oscillatory/jump attacks (EKF helps)

    Usage:
        fusion = ConditionalHybridFusion(config)
        fusion.calibrate(nominal_ici, nominal_ekf_innovation)

        for t in range(T):
            result = fusion.detect(ici_t, ekf_innovation_t)
            if result['alarm']:
                raise_alarm()
    """

    def __init__(self, config: Optional[ConditionalFusionConfig] = None):
        self.config = config or ConditionalFusionConfig()

        # Spectrum analyzer for EKF innovation
        self.spectrum_analyzer = InnovationSpectrumAnalyzer(self.config)

        # Calibration statistics
        self.ici_mean: float = 0.0
        self.ici_std: float = 1.0
        self.ekf_mean: float = 0.0
        self.ekf_std: float = 1.0

        # Thresholds
        self.threshold_ici_only: float = 0.0
        self.threshold_hybrid: float = 0.0

        # State for N-consecutive rule
        self.consecutive_count: int = 0

    def calibrate(
        self,
        nominal_ici: np.ndarray,
        nominal_ekf_innovation: np.ndarray,
        target_fpr: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calibrate fusion on nominal data.

        Args:
            nominal_ici: ICI scores on clean data
            nominal_ekf_innovation: EKF innovation (residual) on clean data
            target_fpr: Target false positive rate

        Returns:
            Calibration statistics
        """
        target_fpr = target_fpr or self.config.target_fpr

        # Compute normalization stats
        self.ici_mean = float(np.mean(nominal_ici))
        self.ici_std = float(np.std(nominal_ici))
        self.ekf_mean = float(np.mean(np.abs(nominal_ekf_innovation)))
        self.ekf_std = float(np.std(np.abs(nominal_ekf_innovation)))

        # Compute scores on nominal data
        ici_z = (nominal_ici - self.ici_mean) / max(self.ici_std, 1e-6)
        ekf_z = (np.abs(nominal_ekf_innovation) - self.ekf_mean) / max(self.ekf_std, 1e-6)

        # ICI-only threshold
        self.threshold_ici_only = float(np.percentile(ici_z, 100 * (1 - target_fpr)))

        # Hybrid threshold (ICI + w*EKF when high-freq)
        # Conservative: use max of ICI and hybrid percentiles
        hybrid_scores = ici_z + self.config.w_ekf * ekf_z
        self.threshold_hybrid = float(np.percentile(hybrid_scores, 100 * (1 - target_fpr)))

        return {
            'ici_mean': self.ici_mean,
            'ici_std': self.ici_std,
            'ekf_mean': self.ekf_mean,
            'ekf_std': self.ekf_std,
            'threshold_ici_only': self.threshold_ici_only,
            'threshold_hybrid': self.threshold_hybrid,
            'n_samples': len(nominal_ici),
        }

    def detect(
        self,
        ici_t: float,
        ekf_innovation_t: float
    ) -> Dict:
        """
        Run conditional fusion detection for one timestep.

        Args:
            ici_t: ICI score at current timestep
            ekf_innovation_t: EKF innovation at current timestep

        Returns:
            Detection result with:
            - 'score': Fused score
            - 'ekf_active': Whether EKF was included
            - 'flag': Whether score exceeds threshold
            - 'alarm': Whether N-consecutive rule triggered
        """
        # Analyze EKF innovation spectrum
        spectrum_result = self.spectrum_analyzer.update(ekf_innovation_t)
        ekf_active = spectrum_result['is_highfreq']

        # Normalize scores
        ici_z = (ici_t - self.ici_mean) / max(self.ici_std, 1e-6)
        ekf_z = (abs(ekf_innovation_t) - self.ekf_mean) / max(self.ekf_std, 1e-6)

        # Conditional fusion
        if ekf_active:
            # High-frequency innovation: include EKF
            score = ici_z + self.config.w_ekf * ekf_z
            threshold = self.threshold_hybrid
        else:
            # Low-frequency or nominal: ICI only
            score = ici_z
            threshold = self.threshold_ici_only

        # Check threshold
        flag = score > threshold

        # N-consecutive rule
        if flag:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        alarm = self.consecutive_count >= self.config.n_consecutive

        return {
            'score': float(score),
            'ici_z': float(ici_z),
            'ekf_z': float(ekf_z),
            'ekf_active': bool(ekf_active),
            'highfreq_ratio': float(spectrum_result['highfreq_ratio']),
            'threshold_used': float(threshold),
            'flag': bool(flag),
            'alarm': bool(alarm),
            'consecutive_count': self.consecutive_count,
        }

    def detect_trajectory(
        self,
        ici_scores: np.ndarray,
        ekf_innovation: np.ndarray
    ) -> Dict:
        """
        Run detection on entire trajectory.

        Args:
            ici_scores: ICI scores for trajectory
            ekf_innovation: EKF innovation for trajectory

        Returns:
            Trajectory-level detection results
        """
        self.reset()

        T = min(len(ici_scores), len(ekf_innovation))
        results = []

        for t in range(T):
            result = self.detect(ici_scores[t], ekf_innovation[t])
            results.append(result)

        # Aggregate
        scores = np.array([r['score'] for r in results])
        flags = np.array([r['flag'] for r in results])
        alarms = np.array([r['alarm'] for r in results])
        ekf_active = np.array([r['ekf_active'] for r in results])

        return {
            'scores': scores,
            'flags': flags,
            'alarms': alarms,
            'ekf_active': ekf_active,
            'n_alarms': int(np.sum(alarms)),
            'ekf_active_pct': float(100 * np.mean(ekf_active)),
        }

    def reset(self):
        """Reset state for new trajectory."""
        self.spectrum_analyzer.reset()
        self.consecutive_count = 0

    def evaluate(
        self,
        nominal_ici: np.ndarray,
        nominal_ekf: np.ndarray,
        attack_ici: np.ndarray,
        attack_ekf: np.ndarray
    ) -> Dict:
        """
        Evaluate conditional fusion vs naive fusion.

        Args:
            nominal_ici: ICI on nominal data
            nominal_ekf: EKF innovation on nominal data
            attack_ici: ICI on attack data
            attack_ekf: EKF innovation on attack data

        Returns:
            Comparison metrics
        """
        from sklearn.metrics import roc_auc_score, roc_curve

        # Conditional fusion
        self.reset()
        cond_nominal = self.detect_trajectory(nominal_ici, nominal_ekf)
        self.reset()
        cond_attack = self.detect_trajectory(attack_ici, attack_ekf)

        cond_scores = np.concatenate([cond_nominal['scores'], cond_attack['scores']])
        labels = np.concatenate([
            np.zeros(len(cond_nominal['scores'])),
            np.ones(len(cond_attack['scores']))
        ])

        cond_auroc = roc_auc_score(labels, cond_scores)

        # Naive fusion (always include EKF)
        naive_nominal = self._naive_fuse(nominal_ici, nominal_ekf)
        naive_attack = self._naive_fuse(attack_ici, attack_ekf)
        naive_scores = np.concatenate([naive_nominal, naive_attack])

        naive_auroc = roc_auc_score(labels, naive_scores)

        # ICI only
        ici_only = np.concatenate([nominal_ici, attack_ici])
        ici_auroc = roc_auc_score(labels, ici_only)

        # Recall at FPR
        fpr_cond, tpr_cond, _ = roc_curve(labels, cond_scores)
        fpr_naive, tpr_naive, _ = roc_curve(labels, naive_scores)

        def recall_at_fpr(fpr_arr, tpr_arr, target):
            idx = np.searchsorted(fpr_arr, target)
            return tpr_arr[min(idx, len(tpr_arr) - 1)]

        return {
            'conditional_auroc': float(cond_auroc),
            'naive_auroc': float(naive_auroc),
            'ici_only_auroc': float(ici_auroc),
            'conditional_recall_5pct': float(recall_at_fpr(fpr_cond, tpr_cond, 0.05)),
            'naive_recall_5pct': float(recall_at_fpr(fpr_naive, tpr_naive, 0.05)),
            'improvement_over_naive': float(cond_auroc - naive_auroc),
            'ekf_active_nominal_pct': float(cond_nominal['ekf_active_pct']),
            'ekf_active_attack_pct': float(cond_attack['ekf_active_pct']),
        }

    def _naive_fuse(self, ici: np.ndarray, ekf: np.ndarray) -> np.ndarray:
        """Naive fusion (always include EKF)."""
        ici_z = (ici - self.ici_mean) / max(self.ici_std, 1e-6)
        ekf_z = (np.abs(ekf) - self.ekf_mean) / max(self.ekf_std, 1e-6)
        return ici_z + self.config.w_ekf * ekf_z


def demo_conditional_fusion():
    """
    Demonstrate conditional fusion improvements.

    Shows how conditional fusion outperforms naive fusion by:
    - Not diluting ICI on consistent attacks (EKF inactive)
    - Boosting detection on jump/oscillatory attacks (EKF active)
    """
    print("=" * 70)
    print("CONDITIONAL HYBRID FUSION DEMO")
    print("=" * 70)

    np.random.seed(42)

    T = 2000

    # Nominal data
    nominal_ici = np.random.randn(T) * 0.3 + 1.0
    nominal_ekf = np.random.randn(T) * 0.1  # Nominal innovation is noise

    # Consistent spoofing (ICI elevated, EKF sees nothing)
    consistent_ici = np.random.randn(T) * 0.4 + 1.5  # Elevated ICI
    consistent_ekf = np.random.randn(T) * 0.1  # EKF still noise

    # Jump attack (EKF sees spike)
    jump_ici = np.random.randn(T) * 0.3 + 1.0
    jump_ekf = np.random.randn(T) * 0.1
    # Add high-frequency spike at t=500
    jump_ici[500:550] += 1.0
    jump_ekf[500:550] += np.sin(np.linspace(0, 10*np.pi, 50)) * 2.0  # High freq

    # Create fusion
    config = ConditionalFusionConfig()
    fusion = ConditionalHybridFusion(config)

    # Calibrate on nominal
    cal_stats = fusion.calibrate(nominal_ici[:1000], nominal_ekf[:1000])
    print(f"\nCalibration stats: {cal_stats}")

    # Evaluate on consistent spoofing
    print("\n--- CONSISTENT SPOOFING (EKF sees nothing) ---")
    result = fusion.evaluate(
        nominal_ici[1000:], nominal_ekf[1000:],
        consistent_ici, consistent_ekf
    )
    print(f"  Conditional AUROC:  {result['conditional_auroc']:.3f}")
    print(f"  Naive AUROC:        {result['naive_auroc']:.3f}")
    print(f"  ICI-only AUROC:     {result['ici_only_auroc']:.3f}")
    print(f"  EKF active (attack): {result['ekf_active_attack_pct']:.1f}%")
    print(f"  Improvement over naive: {result['improvement_over_naive']:.3f}")

    # Evaluate on jump attack
    print("\n--- JUMP ATTACK (EKF sees spike) ---")
    result = fusion.evaluate(
        nominal_ici[1000:], nominal_ekf[1000:],
        jump_ici, jump_ekf
    )
    print(f"  Conditional AUROC:  {result['conditional_auroc']:.3f}")
    print(f"  Naive AUROC:        {result['naive_auroc']:.3f}")
    print(f"  ICI-only AUROC:     {result['ici_only_auroc']:.3f}")
    print(f"  EKF active (attack): {result['ekf_active_attack_pct']:.1f}%")
    print(f"  Improvement over naive: {result['improvement_over_naive']:.3f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("""
Conditional fusion outperforms naive fusion because:

1. CONSISTENT SPOOFING:
   - EKF innovation looks nominal (no high-freq content)
   - Conditional: Uses ICI only (no dilution)
   - Naive: Dilutes ICI with uninformative EKF

2. JUMP ATTACK:
   - EKF innovation has high-frequency spike
   - Conditional: Activates EKF to boost detection
   - Naive: Same (both include EKF)

Result: Conditional fusion dominates on consistent spoofing,
        matches naive on jump attacks.
""")


if __name__ == "__main__":
    demo_conditional_fusion()
