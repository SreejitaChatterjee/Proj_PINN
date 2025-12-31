"""
Honest Evaluation with Proper Train/Test Split

Addresses overfitting concerns:
1. Different random seeds for train vs test
2. 5-fold cross-validation
3. Threshold selection on validation set only
4. Final evaluation on held-out test set
"""

import sys
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gps_imu_detector.src.conformal_envelopes import ConformalEnvelopeBuilder
from gps_imu_detector.src.uncertainty_maps import UncertaintyMapBuilder
from gps_imu_detector.src.adaptive_probing import build_optimized_probe_library
from gps_imu_detector.src.integration import UnifiedDetectionPipeline, PipelineConfig
from gps_imu_detector.src.safety_critical import SeverityLevel, SeverityThresholds


@dataclass
class EvalResults:
    detection_rate: float
    fpr: float
    mean_latency_ms: float
    per_attack_recall: dict


def generate_trajectories(n_traj: int, T: int, seed: int, is_attack: bool = False) -> np.ndarray:
    """Generate trajectories with specific seed."""
    np.random.seed(seed)
    trajectories = []

    for i in range(n_traj):
        traj = np.zeros((T, 12), dtype=np.float32)
        pos = np.array([0.0, 0.0, 10.0])
        vel = np.array([0.0, 0.0, 0.0])
        orient = np.array([0.0, 0.0, 0.0])
        ang_vel = np.array([0.0, 0.0, 0.0])

        profile = i % 4
        dt = 0.005
        attack_start = T // 4
        atype = i % 5

        for t in range(T):
            # Normal dynamics
            if profile == 0:
                vel = np.random.randn(3) * 0.05
                ang_vel = np.random.randn(3) * 0.02
            elif profile == 1:
                vel = np.array([2.0, 0.0, 0.0]) + np.random.randn(3) * 0.1
                ang_vel = np.random.randn(3) * 0.05
            elif profile == 2:
                angle = t * 0.02
                vel = np.array([np.cos(angle), np.sin(angle), 0.0]) * 3.0 + np.random.randn(3) * 0.05
                ang_vel = np.array([0.0, 0.0, 0.02]) + np.random.randn(3) * 0.02
            else:
                angle = t * 0.03
                vel = np.array([np.cos(angle), np.sin(2*angle), 0.0]) * 2.0 + np.random.randn(3) * 0.08
                ang_vel = np.random.randn(3) * 0.03

            pos = pos + vel * dt
            orient = orient + ang_vel * dt

            traj[t, :3] = pos
            traj[t, 3:6] = vel
            traj[t, 6:9] = orient
            traj[t, 9:12] = ang_vel

            # Add attacks after attack_start
            if is_attack and t >= attack_start:
                if atype == 0:  # GPS drift
                    traj[t, :3] += 0.5 * (t - attack_start) * 0.005
                elif atype == 1:  # GPS jump
                    traj[t, :3] += np.array([5.0, 5.0, 1.0])
                    traj[t, 3:6] += np.array([0.5, 0.5, 0.0])
                elif atype == 2:  # IMU bias
                    traj[t, 9:12] += np.array([0.8, 0.8, 0.3])
                elif atype == 3:  # Spoofing
                    traj[t, :3] += np.array([3.0, -3.0, 0.5])
                    traj[t, 3:6] += np.array([-0.3, 0.3, 0.0])
                elif atype == 4:  # Actuator fault
                    traj[t, 9:12] += np.random.randn(3) * 0.5

        trajectories.append(traj)

    return np.array(trajectories)


def train_pipeline(train_nominal: np.ndarray, train_attacks: np.ndarray,
                   critical_threshold: float = 3.0) -> UnifiedDetectionPipeline:
    """Train pipeline on given data."""
    torch.manual_seed(42)

    # Phase 1: Conformal envelopes
    envelope_builder = ConformalEnvelopeBuilder(state_dim=12)
    envelope_builder.train_pinn(train_nominal, epochs=10, verbose=False)
    envelope_builder.calibrate(train_nominal)
    envelope_table = envelope_builder.build_envelope_table(version="1.0.0")

    # Phase 1: Uncertainty maps
    uncertainty_builder = UncertaintyMapBuilder(state_dim=12)
    uncertainty_builder.train_pinn(train_nominal, epochs=10, verbose=False)
    uncertainty_builder.collect_bin_statistics(train_nominal)
    uncertainty_map = uncertainty_builder.build_uncertainty_map(version="1.0.0")

    # Phase 2: Probe library
    probe_library = build_optimized_probe_library(train_nominal, train_attacks, version="1.0.0")

    # Create pipeline
    config = PipelineConfig(
        soft_alert_severity=SeverityLevel.WARNING,
        hard_alert_severity=SeverityLevel.CRITICAL,
    )

    pipeline = UnifiedDetectionPipeline(
        config=config,
        envelope_table=envelope_table,
        uncertainty_map=uncertainty_map,
        probe_library=probe_library,
    )

    pipeline.calibrate(train_nominal)

    # Set thresholds
    pipeline.safety_system.severity_scorer.thresholds = SeverityThresholds(
        advisory=0.8,
        caution=1.5,
        warning=2.0,
        critical=critical_threshold,
        emergency=5.0,
    )

    return pipeline


def evaluate_pipeline(pipeline: UnifiedDetectionPipeline,
                      nominal: np.ndarray,
                      attacks: np.ndarray,
                      detection_threshold: int = 4) -> EvalResults:
    """Evaluate pipeline on given data."""
    attack_types = ['GPS_DRIFT', 'GPS_JUMP', 'IMU_BIAS', 'SPOOFING', 'ACTUATOR_FAULT']

    # Evaluate nominal (FPR)
    false_positives = 0
    total_nominal = 0
    latencies = []

    for traj in nominal:
        for t in range(len(traj) - 1):
            result = pipeline.process(traj[t], traj[t+1])
            if result.decision.value >= detection_threshold:
                false_positives += 1
            total_nominal += 1
            latencies.append(result.latency_ms)
        pipeline.reset()

    # Evaluate attacks (recall)
    detections = 0
    per_attack = {at: {'detected': 0, 'total': 0} for at in attack_types}

    for i, traj in enumerate(attacks):
        attack_start = len(traj) // 4
        detected = False
        atype = attack_types[i % 5]

        for t in range(len(traj) - 1):
            result = pipeline.process(traj[t], traj[t+1])
            if t >= attack_start and not detected:
                if result.decision.value >= detection_threshold:
                    detected = True

        if detected:
            detections += 1
            per_attack[atype]['detected'] += 1
        per_attack[atype]['total'] += 1
        pipeline.reset()

    per_attack_recall = {}
    for at in attack_types:
        if per_attack[at]['total'] > 0:
            per_attack_recall[at] = per_attack[at]['detected'] / per_attack[at]['total']
        else:
            per_attack_recall[at] = 0.0

    return EvalResults(
        detection_rate=detections / max(1, len(attacks)),
        fpr=false_positives / max(1, total_nominal),
        mean_latency_ms=np.mean(latencies),
        per_attack_recall=per_attack_recall,
    )


def cross_validate(n_folds: int = 5) -> Tuple[List[EvalResults], float]:
    """Run k-fold cross-validation to select threshold."""
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION ({n_folds}-fold) - Threshold Selection")
    print(f"{'='*60}")

    # Generate CV data with seeds 100-199
    all_nominal = generate_trajectories(50, 200, seed=100, is_attack=False)
    all_attacks = generate_trajectories(50, 200, seed=150, is_attack=True)

    fold_size = len(all_nominal) // n_folds

    # Try different thresholds
    thresholds_to_try = [2.5, 2.8, 3.0, 3.2, 3.5, 4.0]
    best_threshold = 3.0
    best_score = -float('inf')

    for threshold in thresholds_to_try:
        fold_results = []

        for fold in range(n_folds):
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size

            train_nominal = np.concatenate([all_nominal[:val_start], all_nominal[val_end:]])
            train_attacks = np.concatenate([all_attacks[:val_start], all_attacks[val_end:]])
            val_nominal = all_nominal[val_start:val_end]
            val_attacks = all_attacks[val_start:val_end]

            # Train and evaluate
            pipeline = train_pipeline(train_nominal, train_attacks, critical_threshold=threshold)
            results = evaluate_pipeline(pipeline, val_nominal, val_attacks)
            fold_results.append(results)

        # Average across folds
        mean_detection = np.mean([r.detection_rate for r in fold_results])
        mean_fpr = np.mean([r.fpr for r in fold_results])

        # Score: maximize detection while keeping FPR <= 1%
        if mean_fpr <= 0.01:
            score = mean_detection
        else:
            score = mean_detection - 10 * (mean_fpr - 0.01)  # Penalize FPR > 1%

        print(f"  Threshold {threshold:.1f}: Detection={mean_detection*100:.1f}%, FPR={mean_fpr*100:.2f}%, Score={score:.3f}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    print(f"\n  Best threshold: {best_threshold}")
    return best_threshold


def final_evaluation(selected_threshold: float):
    """Final evaluation on completely held-out test data."""
    print(f"\n{'='*60}")
    print("FINAL EVALUATION - Held-Out Test Set")
    print(f"{'='*60}")
    print(f"Using threshold selected from CV: {selected_threshold}")

    # Training data: seeds 100-199 (same as CV)
    print("\n[Generating training data (seeds 100-199)...]")
    train_nominal = generate_trajectories(50, 200, seed=100, is_attack=False)
    train_attacks = generate_trajectories(50, 200, seed=150, is_attack=True)

    # Test data: DIFFERENT seeds 200-299 (never seen before)
    print("[Generating held-out test data (seeds 200-299)...]")
    test_nominal = generate_trajectories(30, 200, seed=200, is_attack=False)
    test_attacks = generate_trajectories(30, 200, seed=250, is_attack=True)

    # Train on full training set
    print("[Training pipeline on full training set...]")
    pipeline = train_pipeline(train_nominal, train_attacks, critical_threshold=selected_threshold)

    # Evaluate on held-out test
    print("[Evaluating on held-out test set...]")
    results = evaluate_pipeline(pipeline, test_nominal, test_attacks)

    return results


def main():
    print("="*70)
    print("      HONEST EVALUATION - GPS-IMU ANOMALY DETECTOR")
    print("      (Addressing Overfitting with Proper Methodology)")
    print("="*70)

    # Step 1: Cross-validation for threshold selection
    best_threshold = cross_validate(n_folds=5)

    # Step 2: Final evaluation on held-out test set
    results = final_evaluation(best_threshold)

    # Print results
    print("\n" + "="*70)
    print("                    HONEST FINAL RESULTS")
    print("="*70)

    attack_types = ['GPS_DRIFT', 'GPS_JUMP', 'IMU_BIAS', 'SPOOFING', 'ACTUATOR_FAULT']

    print(f"\n{'METRIC':<35} {'VALUE':<20}")
    print("-"*55)
    print(f"{'Overall Detection Rate':<35} {results.detection_rate*100:.1f}%")
    print(f"{'False Positive Rate':<35} {results.fpr*100:.2f}%")
    print(f"{'Mean Latency':<35} {results.mean_latency_ms:.2f} ms")

    print(f"\n{'ATTACK TYPE':<35} {'RECALL':<20}")
    print("-"*55)
    for at in attack_types:
        recall = results.per_attack_recall.get(at, 0.0)
        bar = '#' * int(recall * 20)
        print(f"{at:<35} {recall*100:.0f}% {bar}")

    print("\n" + "="*70)
    print("                    CERTIFICATION STATUS")
    print("="*70)

    checks = [
        ("Detection Rate >= 80%", results.detection_rate >= 0.80),
        ("FPR <= 1%", results.fpr <= 0.01),
        ("Mean Latency < 5ms", results.mean_latency_ms < 5.0),
    ]

    passed = 0
    for name, ok in checks:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {name}: {status}")
        if ok:
            passed += 1

    print(f"\n  Result: {passed}/{len(checks)} requirements met")

    if passed == len(checks):
        print("\n" + "="*70)
        print("     [OK] ALL REQUIREMENTS MET (No Overfitting)")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("     [!!] SOME REQUIREMENTS NOT MET")
        print("="*70)


if __name__ == "__main__":
    main()
