#!/usr/bin/env python3
"""
Robustness Fixes Evaluation

Tests the 5 principled fixes:
1. Temporal Contrast
2. Multi-resolution Agreement
3. Conditional Normalization
4. Control-Conditioned Features
5. Gap-Tolerant Accumulation

Philosophy: Add INVARIANCE, not SENSITIVITY.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple

from robustness_fixes import (
    TemporalContrastDetector,
    MultiResolutionAgreement,
    ConditionalNormalizer,
    ControlConditionedFeatures,
    GapTolerantAccumulator,
    RobustDetector,
    evaluate_temporal_reliance,
    evaluate_domain_robustness
)

# Configuration
SEED = 42
TRAIN_SEED = 100
TEST_SEED = 100
OOD_SEED = 200
TRAJ_LENGTH = 2000
N_TRAJECTORIES = 5

np.random.seed(SEED)

print("=" * 70)
print("ROBUSTNESS FIXES EVALUATION")
print("=" * 70)
print("Philosophy: Add INVARIANCE, not SENSITIVITY")
print()


def generate_trajectory(n_samples: int, seed: int) -> np.ndarray:
    """Generate trajectory."""
    np.random.seed(seed)
    trajectory = np.zeros((n_samples, 15), dtype=np.float32)
    pos = np.array([0.0, 0.0, 10.0])
    vel = np.array([0.0, 0.0, 0.0])
    att = np.array([0.0, 0.0, 0.0])
    dt = 0.005

    for t in range(n_samples):
        accel = np.random.randn(3) * 0.1
        vel = vel + accel * dt + np.random.randn(3) * 0.01
        pos = pos + vel * dt
        ang_rate = np.random.randn(3) * 0.05
        att = att + ang_rate * dt

        trajectory[t, 0:3] = pos
        trajectory[t, 3:6] = vel
        trajectory[t, 6:9] = att
        trajectory[t, 9:12] = ang_rate
        trajectory[t, 12:15] = accel

    return trajectory


def generate_attack(nominal: np.ndarray, attack_type: str, seed: int) -> np.ndarray:
    """Generate attacked trajectory."""
    np.random.seed(seed)
    attacked = nominal.copy()
    n = len(nominal)

    if attack_type == 'drift':
        drift = np.linspace(0, 5, n).reshape(-1, 1)
        attacked[:, :3] += drift
    elif attack_type == 'bias':
        attacked[:, :3] += np.random.randn(3) * 2
    elif attack_type == 'noise':
        attacked[:, :6] += np.random.randn(n, 6) * 0.5
    elif attack_type == 'intermittent':
        for i in range(0, n, 100):
            if np.random.rand() > 0.5:
                attacked[i:i+20, :3] += np.random.randn(3) * 3

    return attacked


# ============================================================
# GENERATE DATA
# ============================================================
print("Generating data...")

train_trajectories = []
for i in range(N_TRAJECTORIES):
    traj = generate_trajectory(TRAJ_LENGTH, seed=TRAIN_SEED + i)
    train_trajectories.append(traj)
train_normal = np.vstack(train_trajectories)

test_trajectories = []
for i in range(N_TRAJECTORIES // 2):
    traj = generate_trajectory(TRAJ_LENGTH, seed=TEST_SEED + i)
    test_trajectories.append(traj)
test_normal = np.vstack(test_trajectories)

ood_trajectories = []
for i in range(N_TRAJECTORIES // 2):
    traj = generate_trajectory(TRAJ_LENGTH, seed=OOD_SEED + i)
    ood_trajectories.append(traj)
ood_normal = np.vstack(ood_trajectories)

print(f"  Train: {len(train_normal)} samples")
print(f"  Test (ID): {len(test_normal)} samples")
print(f"  Test (OOD): {len(ood_normal)} samples")

results = {}


# ============================================================
# TEST 1: TEMPORAL CONTRAST
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: Temporal Contrast (forward vs reversed)")
print("=" * 60)

temporal = TemporalContrastDetector(window_size=50)
temporal.fit(train_trajectories)

# Test on normal
normal_scores = temporal.score(test_normal[:TRAJ_LENGTH])

# Test on attack
attack = generate_attack(test_normal[:TRAJ_LENGTH], 'drift', TEST_SEED)
attack_scores = temporal.score(attack)

# AUROC
labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(attack_scores))])
scores = np.concatenate([normal_scores, attack_scores])
auroc = roc_auc_score(labels, scores)

# Temporal reliance test
reliance = evaluate_temporal_reliance(temporal, test_normal[:TRAJ_LENGTH])

results['temporal_contrast'] = {
    'auroc': float(auroc),
    'degradation_on_shuffle': reliance['degradation'],
    'relies_on_structure': reliance['relies_on_structure']
}

print(f"  AUROC on drift attack: {auroc*100:.1f}%")
print(f"  Degradation on shuffle: {reliance['degradation']*100:.1f}%")
print(f"  Verdict: {'[OK] Uses temporal structure' if reliance['relies_on_structure'] else '[!] Limited temporal reliance'}")


# ============================================================
# TEST 2: MULTI-RESOLUTION AGREEMENT
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Multi-Resolution Agreement")
print("=" * 60)

multi_res = MultiResolutionAgreement(short_window=10, medium_window=50, long_window=200)
multi_res.fit(normal_scores)

# Compute agreement on normal vs attack
normal_agreement, normal_disagree = multi_res.compute_agreement(normal_scores)
attack_agreement, attack_disagree = multi_res.compute_agreement(attack_scores)

# Disagreement should be higher for attacks
normal_disagree_mean = np.mean(normal_disagree[200:])
attack_disagree_mean = np.mean(attack_disagree[200:])

results['multi_resolution'] = {
    'normal_disagreement': float(normal_disagree_mean),
    'attack_disagreement': float(attack_disagree_mean),
    'ratio': float(attack_disagree_mean / (normal_disagree_mean + 1e-6)),
    'effective': attack_disagree_mean > normal_disagree_mean * 1.5
}

print(f"  Normal disagreement: {normal_disagree_mean:.3f}")
print(f"  Attack disagreement: {attack_disagree_mean:.3f}")
print(f"  Ratio: {attack_disagree_mean / (normal_disagree_mean + 1e-6):.2f}x")
print(f"  Verdict: {'[OK] Cross-scale disagreement detects attacks' if results['multi_resolution']['effective'] else '[!] Limited effect'}")


# ============================================================
# TEST 3: CONDITIONAL NORMALIZATION
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Conditional Normalization (per-flight)")
print("=" * 60)

cond_norm = ConditionalNormalizer(warmup_samples=100)

# Compare ID vs OOD with and without normalization
id_scores = temporal.score(test_normal[:TRAJ_LENGTH])
ood_scores = temporal.score(ood_normal[:TRAJ_LENGTH])

# Without normalization
id_mean_raw = np.mean(id_scores)
ood_mean_raw = np.mean(ood_scores)
shift_raw = np.abs(id_mean_raw - ood_mean_raw)

# With normalization
id_norm = cond_norm.normalize(id_scores)
ood_norm = cond_norm.normalize(ood_scores)
id_mean_norm = np.mean(id_norm[100:])
ood_mean_norm = np.mean(ood_norm[100:])
shift_norm = np.abs(id_mean_norm - ood_mean_norm)

results['conditional_norm'] = {
    'shift_raw': float(shift_raw),
    'shift_normalized': float(shift_norm),
    'reduction': float(1 - shift_norm / (shift_raw + 1e-6)),
    'effective': shift_norm < shift_raw
}

print(f"  Domain shift (raw): {shift_raw:.3f}")
print(f"  Domain shift (normalized): {shift_norm:.3f}")
print(f"  Reduction: {(1 - shift_norm / (shift_raw + 1e-6))*100:.1f}%")
print(f"  Verdict: {'[OK] Per-flight normalization reduces shift' if shift_norm < shift_raw else '[!] Limited effect'}")


# ============================================================
# TEST 4: CONTROL-CONDITIONED FEATURES
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Control-Conditioned Features")
print("=" * 60)

control_feat = ControlConditionedFeatures()

# Compute features
normal_feat = control_feat.compute_all(test_normal[:TRAJ_LENGTH])
attack_feat = control_feat.compute_all(attack)

# Compare separability
normal_max = np.max(np.abs(normal_feat), axis=1)
attack_max = np.max(np.abs(attack_feat), axis=1)

labels = np.concatenate([np.zeros(len(normal_max)), np.ones(len(attack_max))])
scores = np.concatenate([normal_max, attack_max])
auroc_control = roc_auc_score(labels, scores)

results['control_conditioned'] = {
    'auroc': float(auroc_control),
    'effective': auroc_control > 0.70
}

print(f"  AUROC with control-conditioned features: {auroc_control*100:.1f}%")
print(f"  Verdict: {'[OK] Control normalization improves detection' if auroc_control > 0.70 else '[!] Limited effect'}")


# ============================================================
# TEST 5: GAP-TOLERANT ACCUMULATION
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: Gap-Tolerant Accumulation")
print("=" * 60)

gap_accum = GapTolerantAccumulator(window_size=50, required_count=5)

# Generate intermittent attack
intermittent = generate_attack(test_normal[:TRAJ_LENGTH], 'intermittent', TEST_SEED)
intermittent_scores = temporal.score(intermittent)

# Detect with standard consecutive logic
threshold = np.percentile(normal_scores, 95)
standard_detections = intermittent_scores > threshold
standard_detected = np.any(standard_detections)

# Detect with gap-tolerant logic
gap_detections = gap_accum.detect(standard_detections)
gap_detected = np.any(gap_detections)

# Count detections
standard_count = np.sum(standard_detections)
gap_count = np.sum(gap_detections)

results['gap_tolerant'] = {
    'standard_detections': int(standard_count),
    'gap_detections': int(gap_count),
    'intermittent_caught_standard': bool(standard_detected),
    'intermittent_caught_gap': bool(gap_detected),
    'effective': gap_detected or gap_count > 0
}

print(f"  Standard detections: {standard_count}")
print(f"  Gap-tolerant detections: {gap_count}")
print(f"  Intermittent attack caught (standard): {standard_detected}")
print(f"  Intermittent attack caught (gap-tolerant): {gap_detected}")
print(f"  Verdict: {'[OK] Gap tolerance helps intermittent detection' if gap_detected else '[!] Limited effect'}")


# ============================================================
# TEST 6: COMBINED ROBUST DETECTOR
# ============================================================
print("\n" + "=" * 60)
print("TEST 6: Combined Robust Detector")
print("=" * 60)

robust = RobustDetector()
robust.fit(train_trajectories)

# Test on multiple attack types
attack_types = ['drift', 'bias', 'noise', 'intermittent']
combined_results = {}

for attack_type in attack_types:
    attack = generate_attack(test_normal[:TRAJ_LENGTH], attack_type, TEST_SEED)

    normal_scores = robust.score(test_normal[:TRAJ_LENGTH])
    attack_scores = robust.score(attack)

    labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(attack_scores))])
    scores = np.concatenate([normal_scores, attack_scores])

    auroc = roc_auc_score(labels, scores)
    combined_results[attack_type] = float(auroc)

    print(f"  {attack_type}: AUROC = {auroc*100:.1f}%")

results['combined'] = combined_results
results['combined_mean'] = float(np.mean(list(combined_results.values())))

print(f"\n  Mean AUROC: {results['combined_mean']*100:.1f}%")


# ============================================================
# TEST 7: TEMPORAL RELIANCE (SHUFFLING TEST)
# ============================================================
print("\n" + "=" * 60)
print("TEST 7: Temporal Reliance (Shuffling Test)")
print("=" * 60)

# Test robust detector's temporal reliance
reliance_robust = evaluate_temporal_reliance(robust, test_normal[:TRAJ_LENGTH])

results['temporal_reliance'] = reliance_robust

print(f"  Original score mean: {reliance_robust['original_mean']:.3f}")
print(f"  Shuffled score mean: {reliance_robust['shuffled_mean']:.3f}")
print(f"  Degradation: {reliance_robust['degradation']*100:.1f}%")
print(f"  Verdict: {'[OK] Relies on temporal structure' if reliance_robust['relies_on_structure'] else '[!] Uses marginal statistics'}")


# ============================================================
# TEST 8: DOMAIN ROBUSTNESS (OOD TEST)
# ============================================================
print("\n" + "=" * 60)
print("TEST 8: Domain Robustness (OOD Test)")
print("=" * 60)

domain_result = evaluate_domain_robustness(robust, test_normal[:TRAJ_LENGTH], ood_normal[:TRAJ_LENGTH])

results['domain_robustness'] = domain_result

print(f"  ID score mean: {domain_result['id_mean']:.3f}")
print(f"  OOD score mean: {domain_result['ood_mean']:.3f}")
print(f"  KS statistic: {domain_result['ks_statistic']:.3f}")
print(f"  KS p-value: {domain_result['ks_pvalue']:.3f}")
print(f"  Verdict: {'[OK] Domain robust' if domain_result['domain_robust'] else '[!] Domain shift detected'}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: ROBUSTNESS FIXES EVALUATION")
print("=" * 70)

print("""
    Fix                          | Result           | Verdict
    -----------------------------|------------------|------------------
    1. Temporal Contrast         | {tc_deg:.1f}% degradation | {tc_v}
    2. Multi-Resolution          | {mr_ratio:.1f}x disagreement | {mr_v}
    3. Conditional Normalization | {cn_red:.1f}% reduction   | {cn_v}
    4. Control-Conditioned       | {cc_auc:.1f}% AUROC       | {cc_v}
    5. Gap-Tolerant              | {gt_det} detections    | {gt_v}
    6. Combined Detector         | {comb:.1f}% mean AUROC  | {comb_v}
    7. Temporal Reliance         | {tr_deg:.1f}% degradation | {tr_v}
    8. Domain Robustness         | p={dr_p:.3f}           | {dr_v}
""".format(
    tc_deg=results['temporal_contrast']['degradation_on_shuffle'] * 100,
    tc_v="[OK]" if results['temporal_contrast']['relies_on_structure'] else "[!]",
    mr_ratio=results['multi_resolution']['ratio'],
    mr_v="[OK]" if results['multi_resolution']['effective'] else "[!]",
    cn_red=results['conditional_norm']['reduction'] * 100,
    cn_v="[OK]" if results['conditional_norm']['effective'] else "[!]",
    cc_auc=results['control_conditioned']['auroc'] * 100,
    cc_v="[OK]" if results['control_conditioned']['effective'] else "[!]",
    gt_det=results['gap_tolerant']['gap_detections'],
    gt_v="[OK]" if results['gap_tolerant']['effective'] else "[!]",
    comb=results['combined_mean'] * 100,
    comb_v="[OK]" if results['combined_mean'] > 0.80 else "[!]",
    tr_deg=results['temporal_reliance']['degradation'] * 100,
    tr_v="[OK]" if results['temporal_reliance']['relies_on_structure'] else "[!]",
    dr_p=results['domain_robustness']['ks_pvalue'],
    dr_v="[OK]" if results['domain_robustness']['domain_robust'] else "[!]"
))

# Count passes
passes = sum([
    results['temporal_contrast']['relies_on_structure'],
    results['multi_resolution']['effective'],
    results['conditional_norm']['effective'],
    results['control_conditioned']['effective'],
    results['gap_tolerant']['effective'],
    results['combined_mean'] > 0.80,
    results['temporal_reliance']['relies_on_structure'],
    results['domain_robustness']['domain_robust']
])

print(f"  Overall: {passes}/8 tests passed")

if passes >= 5:
    print("  Verdict: [OK] Robustness fixes are effective")
else:
    print("  Verdict: [!] Some fixes need refinement")

# Save results
results['timestamp'] = datetime.now().isoformat()
results['methodology'] = 'Robustness fixes evaluation - adding invariance not sensitivity'
results['passes'] = passes

output_path = Path('results/robustness_fixes_evaluation.json')
output_path.parent.mkdir(exist_ok=True)
def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

with open(output_path, 'w') as f:
    json.dump(convert_to_json_serializable(results), f, indent=2)

print(f"\n  Results saved to: {output_path}")
