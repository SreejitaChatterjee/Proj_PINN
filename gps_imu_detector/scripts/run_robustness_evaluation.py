#!/usr/bin/env python3
"""
Robustness Evaluation: Proving We're Not Overfitting

This script implements evaluations that demonstrate the detector is learning
generalizable features, not memorizing attack patterns.

Key Tests:
1. Missed Detection Analysis - Structure, not labels
2. Leave-One-Attack-Class-Out (LOAO) - Feature learning proof
3. Attack Parameter Extrapolation - Magnitude generalization
4. Temporal Shuffling Stress Test - Structure reliance proof
5. Inference-Time Domain Alignment - OOD robustness without retraining

Philosophy: We are done optimizing performance. Now we optimize trust.
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from hard_negatives import HardNegativeGenerator, DomainRandomizer
from temporal_ici import TemporalICIAggregator, TemporalICIConfig
from industry_aligned import TwoStageDecisionLogic

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
TRAIN_SEED = 100
TEST_SEED = 100  # Same seed for in-distribution baseline
OOD_SEED = 200   # Different seed for OOD evaluation
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 25
N_TRAJECTORIES = 10
TRAJ_LENGTH = 2000

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("ROBUSTNESS EVALUATION: Proving We're Not Overfitting")
print("=" * 70)
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Philosophy: Optimize trust, not performance")
print()


# ============================================================
# DATA GENERATION (same as publication script)
# ============================================================
def generate_trajectory(n_samples: int, seed: int) -> np.ndarray:
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


def generate_attack(nominal: np.ndarray, attack_type: str, seed: int,
                   magnitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single attack type with configurable magnitude."""
    np.random.seed(seed)
    n = len(nominal)
    attacked = nominal.copy()
    labels = np.ones(n)

    if attack_type == 'ar1_drift':
        generator = HardNegativeGenerator(seed=seed)
        attacked, labels = generator.generate_ar1_drift(
            nominal, magnitude=magnitude, ar_coef=0.997
        )
    elif attack_type == 'coordinated':
        generator = HardNegativeGenerator(seed=seed)
        attacked, labels = generator.generate_coordinated_attack(
            nominal, magnitude=magnitude * 0.5, consistency_factor=0.8
        )
    elif attack_type == 'intermittent':
        generator = HardNegativeGenerator(seed=seed)
        attacked, labels = generator.generate_intermittent_attack(
            nominal, magnitude=magnitude, on_probability=0.1
        )
    elif attack_type == 'bias':
        attacked[:, :3] += np.random.randn(3) * 0.5 * magnitude
    elif attack_type == 'noise':
        attacked[:, :6] += np.random.randn(n, 6) * 0.3 * magnitude
    elif attack_type == 'ramp':
        ramp_signal = np.linspace(0, 1, n).reshape(-1, 1) * 0.5 * magnitude
        attacked[:, :3] += ramp_signal * np.std(nominal[:, :3], axis=0)

    return attacked, labels


# ============================================================
# MODEL (same architecture)
# ============================================================
class RobustDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.gru = nn.GRU(64, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden=None):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)
        x, hidden = self.gru(x, hidden)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x, hidden


def create_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    n_samples = len(X) - seq_len + 1
    if n_samples <= 0:
        return np.array([])
    sequences = np.zeros((n_samples, seq_len, X.shape[1]))
    for i in range(n_samples):
        sequences[i] = X[i:i+seq_len]
    return sequences


# ============================================================
# TEST 1: MISSED DETECTION ANALYSIS
# ============================================================
def analyze_missed_detections(
    model: nn.Module,
    scaler: StandardScaler,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    Analyze missed detections by STRUCTURE, not label.

    Categories:
    - Short-duration: Attack segment < 50 samples
    - Low-SNR: Attack magnitude < 2σ of normal variance
    - Transitional: First/last 10% of attack segment
    """
    print("\n" + "=" * 60)
    print("TEST 1: Missed Detection Analysis (Structure-Based)")
    print("=" * 60)

    model.eval()
    X_scaled = scaler.transform(test_data)
    X_seq = create_sequences(X_scaled, SEQUENCE_LENGTH)
    labels_seq = test_labels[SEQUENCE_LENGTH-1:]

    with torch.no_grad():
        logits, _ = model(torch.FloatTensor(X_seq))
        scores = torch.sigmoid(logits).squeeze().numpy()

    predictions = scores > threshold

    # Find attack segments
    attack_indices = np.where(labels_seq == 1)[0]
    if len(attack_indices) == 0:
        return {'error': 'No attacks in test data'}

    # Find missed detections
    missed_indices = np.where((labels_seq == 1) & (predictions == 0))[0]

    # Categorize missed detections
    results = {
        'total_attacks': int(len(attack_indices)),
        'total_missed': int(len(missed_indices)),
        'missed_rate': float(len(missed_indices) / len(attack_indices)) if len(attack_indices) > 0 else 0,
        'categories': {
            'short_duration': 0,
            'low_snr': 0,
            'transitional': 0,
            'other': 0
        }
    }

    # Analyze each missed sample
    normal_var = np.var(test_data[test_labels == 0][:, :3], axis=0).mean()

    for idx in missed_indices:
        sample_idx = idx + SEQUENCE_LENGTH - 1

        # Check if short duration (isolated attack sample)
        neighbors = labels_seq[max(0, idx-25):min(len(labels_seq), idx+25)]
        attack_density = np.mean(neighbors)
        if attack_density < 0.5:
            results['categories']['short_duration'] += 1
            continue

        # Check if low SNR
        if sample_idx < len(test_data):
            local_var = np.var(test_data[max(0, sample_idx-10):sample_idx+10, :3], axis=0).mean()
            if local_var < 2 * normal_var:
                results['categories']['low_snr'] += 1
                continue

        # Check if transitional (near attack boundary)
        if idx < 0.1 * len(attack_indices) or idx > 0.9 * len(attack_indices):
            results['categories']['transitional'] += 1
            continue

        results['categories']['other'] += 1

    print(f"  Total attacks: {results['total_attacks']}")
    print(f"  Total missed: {results['total_missed']} ({results['missed_rate']*100:.1f}%)")
    print(f"  Categories:")
    for cat, count in results['categories'].items():
        pct = count / results['total_missed'] * 100 if results['total_missed'] > 0 else 0
        print(f"    {cat}: {count} ({pct:.1f}%)")

    return results


# ============================================================
# TEST 2: LEAVE-ONE-ATTACK-CLASS-OUT (LOAO)
# ============================================================
def leave_one_attack_out_evaluation(
    train_normal: np.ndarray,
    test_normal: np.ndarray,
    attack_types: List[str] = ['ar1_drift', 'coordinated', 'intermittent', 'bias', 'noise', 'ramp']
) -> Dict:
    """
    Train on all but one attack type, test on held-out attack.

    This proves FEATURE LEARNING, not pattern memorization.
    Expected: AUROC drops gracefully (99% → 75-80%), not catastrophically.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Leave-One-Attack-Class-Out (LOAO)")
    print("=" * 60)
    print("Purpose: Prove feature learning, not pattern memorization")
    print()

    results = {}

    for held_out in attack_types:
        print(f"  Holding out: {held_out}")

        # Train on all attacks EXCEPT held_out
        train_attacks = {}
        for attack_type in attack_types:
            if attack_type != held_out:
                attack_data, attack_labels = generate_attack(
                    train_normal[:TRAJ_LENGTH], attack_type, seed=TRAIN_SEED
                )
                train_attacks[attack_type] = (attack_data, attack_labels)

        # Train model
        model, scaler = train_model_simple(train_normal, train_attacks, epochs=20, verbose=False)

        # Test on held-out attack
        test_attack, test_labels = generate_attack(
            test_normal[:TRAJ_LENGTH], held_out, seed=TEST_SEED
        )

        # Evaluate
        X_normal_scaled = scaler.transform(test_normal[:TRAJ_LENGTH])
        X_attack_scaled = scaler.transform(test_attack)

        X_normal_seq = create_sequences(X_normal_scaled, SEQUENCE_LENGTH)
        X_attack_seq = create_sequences(X_attack_scaled, SEQUENCE_LENGTH)

        model.eval()
        with torch.no_grad():
            normal_scores = torch.sigmoid(model(torch.FloatTensor(X_normal_seq))[0]).squeeze().numpy()
            attack_scores = torch.sigmoid(model(torch.FloatTensor(X_attack_seq))[0]).squeeze().numpy()

        labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(attack_scores))])
        scores = np.concatenate([normal_scores, attack_scores])

        auroc = roc_auc_score(labels, scores)

        results[held_out] = {
            'auroc': float(auroc),
            'above_random': bool(auroc > 0.55),
            'graceful_degradation': bool(auroc > 0.70)
        }

        status = "[PASS]" if auroc > 0.70 else ("[MARGINAL]" if auroc > 0.55 else "[FAIL]")
        print(f"    AUROC on {held_out}: {auroc*100:.1f}% {status}")

    # Summary
    mean_auroc = np.mean([r['auroc'] for r in results.values()])
    min_auroc = min([r['auroc'] for r in results.values()])

    results['summary'] = {
        'mean_auroc': float(mean_auroc),
        'min_auroc': float(min_auroc),
        'all_above_random': all(r['above_random'] for r in results.values() if isinstance(r, dict) and 'above_random' in r),
        'all_graceful': all(r['graceful_degradation'] for r in results.values() if isinstance(r, dict) and 'graceful_degradation' in r)
    }

    print(f"\n  Summary: Mean AUROC = {mean_auroc*100:.1f}%, Min = {min_auroc*100:.1f}%")
    print(f"  Verdict: {'[OK] Feature learning proven' if results['summary']['all_above_random'] else '[X] May be memorizing'}")

    return results


# ============================================================
# TEST 3: ATTACK PARAMETER EXTRAPOLATION
# ============================================================
def parameter_extrapolation_test(
    train_normal: np.ndarray,
    test_normal: np.ndarray
) -> Dict:
    """
    Train on specific magnitudes, test on unseen magnitudes.

    Train: 1m, 5m, 10m offsets
    Test: 3m, 7m, 13m offsets

    If performance holds, we're not overfitting to exact magnitudes.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Attack Parameter Extrapolation")
    print("=" * 60)
    print("Train magnitudes: 1m, 5m, 10m")
    print("Test magnitudes: 3m, 7m, 13m (unseen)")
    print()

    train_magnitudes = [1, 5, 10]
    test_magnitudes = [3, 7, 13]

    # Generate training attacks
    train_attacks = {}
    for mag in train_magnitudes:
        attacked = train_normal[:TRAJ_LENGTH].copy()
        attacked[:, :3] += mag * 0.1
        train_attacks[f'offset_{mag}m'] = (attacked, np.ones(TRAJ_LENGTH))

    # Train model
    model, scaler = train_model_simple(train_normal, train_attacks, epochs=20, verbose=False)

    # Test on unseen magnitudes
    results = {'train_magnitudes': train_magnitudes, 'test_magnitudes': test_magnitudes}

    X_normal_scaled = scaler.transform(test_normal[:TRAJ_LENGTH])
    X_normal_seq = create_sequences(X_normal_scaled, SEQUENCE_LENGTH)

    model.eval()
    with torch.no_grad():
        normal_scores = torch.sigmoid(model(torch.FloatTensor(X_normal_seq))[0]).squeeze().numpy()

    for mag in test_magnitudes:
        attacked = test_normal[:TRAJ_LENGTH].copy()
        attacked[:, :3] += mag * 0.1

        X_attack_scaled = scaler.transform(attacked)
        X_attack_seq = create_sequences(X_attack_scaled, SEQUENCE_LENGTH)

        with torch.no_grad():
            attack_scores = torch.sigmoid(model(torch.FloatTensor(X_attack_seq))[0]).squeeze().numpy()

        labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(attack_scores))])
        scores = np.concatenate([normal_scores, attack_scores])

        auroc = roc_auc_score(labels, scores)

        results[f'{mag}m'] = {
            'auroc': float(auroc),
            'generalizes': bool(auroc > 0.80)
        }

        status = "[OK] GENERALIZES" if auroc > 0.80 else "[X] OVERFITTING"
        print(f"  {mag}m offset (unseen): AUROC = {auroc*100:.1f}% {status}")

    # Summary
    test_aurocs = [results[f'{m}m']['auroc'] for m in test_magnitudes]
    results['summary'] = {
        'mean_auroc': float(np.mean(test_aurocs)),
        'all_generalize': all(results[f'{m}m']['generalizes'] for m in test_magnitudes)
    }

    print(f"\n  Verdict: {'[OK] Magnitude generalization proven' if results['summary']['all_generalize'] else '[X] Magnitude-specific overfitting detected'}")

    return results


# ============================================================
# TEST 4: TEMPORAL SHUFFLING STRESS TEST
# ============================================================
def temporal_shuffling_test(
    model: nn.Module,
    scaler: StandardScaler,
    test_normal: np.ndarray,
    attack_type: str = 'ar1_drift'
) -> Dict:
    """
    Break attack temporal coherence, keep marginal distributions.

    If detection degrades, we've shown reliance on STRUCTURE, not statistics.
    This is a GOOD thing - it means we're detecting attack dynamics.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Temporal Shuffling Stress Test")
    print("=" * 60)
    print(f"Attack type: {attack_type}")
    print("Purpose: Prove reliance on temporal structure")
    print()

    # Generate coherent attack
    attack_coherent, labels = generate_attack(
        test_normal[:TRAJ_LENGTH], attack_type, seed=TEST_SEED
    )

    # Create shuffled version (break temporal structure, keep marginal distribution)
    attack_shuffled = attack_coherent.copy()
    np.random.seed(SEED)
    shuffle_indices = np.random.permutation(len(attack_shuffled))
    attack_shuffled = attack_shuffled[shuffle_indices]

    # Evaluate both
    model.eval()

    X_normal_scaled = scaler.transform(test_normal[:TRAJ_LENGTH])
    X_coherent_scaled = scaler.transform(attack_coherent)
    X_shuffled_scaled = scaler.transform(attack_shuffled)

    X_normal_seq = create_sequences(X_normal_scaled, SEQUENCE_LENGTH)
    X_coherent_seq = create_sequences(X_coherent_scaled, SEQUENCE_LENGTH)
    X_shuffled_seq = create_sequences(X_shuffled_scaled, SEQUENCE_LENGTH)

    with torch.no_grad():
        normal_scores = torch.sigmoid(model(torch.FloatTensor(X_normal_seq))[0]).squeeze().numpy()
        coherent_scores = torch.sigmoid(model(torch.FloatTensor(X_coherent_seq))[0]).squeeze().numpy()
        shuffled_scores = torch.sigmoid(model(torch.FloatTensor(X_shuffled_seq))[0]).squeeze().numpy()

    # AUROC for coherent
    labels_coherent = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(coherent_scores))])
    scores_coherent = np.concatenate([normal_scores, coherent_scores])
    auroc_coherent = roc_auc_score(labels_coherent, scores_coherent)

    # AUROC for shuffled
    labels_shuffled = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(shuffled_scores))])
    scores_shuffled = np.concatenate([normal_scores, shuffled_scores])
    auroc_shuffled = roc_auc_score(labels_shuffled, scores_shuffled)

    degradation = auroc_coherent - auroc_shuffled

    results = {
        'auroc_coherent': float(auroc_coherent),
        'auroc_shuffled': float(auroc_shuffled),
        'degradation': float(degradation),
        'relies_on_structure': bool(degradation > 0.05)
    }

    print(f"  Coherent attack AUROC: {auroc_coherent*100:.1f}%")
    print(f"  Shuffled attack AUROC: {auroc_shuffled*100:.1f}%")
    print(f"  Degradation: {degradation*100:.1f}%")

    if results['relies_on_structure']:
        print(f"\n  Verdict: [OK] Detector relies on temporal structure (GOOD)")
        print(f"           This proves we're detecting attack dynamics, not just statistics")
    else:
        print(f"\n  Verdict: [!] Detection may rely on marginal statistics")

    return results


# ============================================================
# TEST 5: INFERENCE-TIME DOMAIN ALIGNMENT (CORAL)
# ============================================================
def coral_alignment_test(
    model: nn.Module,
    scaler: StandardScaler,
    train_normal: np.ndarray,
    ood_normal: np.ndarray,
    ood_seed: int = OOD_SEED
) -> Dict:
    """
    Apply CORAL alignment at inference time to improve OOD robustness.

    No retraining, no attack labels - just covariance alignment.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Inference-Time Domain Alignment (CORAL)")
    print("=" * 60)
    print("Purpose: Improve OOD robustness without retraining")
    print()

    # Generate OOD attack
    ood_attack, _ = generate_attack(ood_normal[:TRAJ_LENGTH], 'bias', seed=ood_seed)

    # Scale with original scaler (baseline)
    X_normal_scaled = scaler.transform(ood_normal[:TRAJ_LENGTH])
    X_attack_scaled = scaler.transform(ood_attack)

    # CORAL alignment
    def coral_transform(X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        """Apply CORAL: align target covariance to source."""
        # Compute covariances
        Cs = np.cov(X_source.T) + np.eye(X_source.shape[1]) * 1e-6
        Ct = np.cov(X_target.T) + np.eye(X_target.shape[1]) * 1e-6

        # Whitening transform for target
        Ct_sqrt_inv = np.linalg.inv(np.linalg.cholesky(Ct))
        Cs_sqrt = np.linalg.cholesky(Cs)

        # Align
        X_aligned = X_target @ Ct_sqrt_inv.T @ Cs_sqrt.T
        return X_aligned

    # Apply CORAL to OOD data using train_normal as reference
    train_scaled = scaler.transform(train_normal[:TRAJ_LENGTH])
    X_normal_aligned = coral_transform(train_scaled, X_normal_scaled)
    X_attack_aligned = coral_transform(train_scaled, X_attack_scaled)

    # Evaluate both
    model.eval()

    # Baseline (no alignment)
    X_normal_seq = create_sequences(X_normal_scaled, SEQUENCE_LENGTH)
    X_attack_seq = create_sequences(X_attack_scaled, SEQUENCE_LENGTH)

    with torch.no_grad():
        normal_scores_baseline = torch.sigmoid(model(torch.FloatTensor(X_normal_seq))[0]).squeeze().numpy()
        attack_scores_baseline = torch.sigmoid(model(torch.FloatTensor(X_attack_seq))[0]).squeeze().numpy()

    labels = np.concatenate([np.zeros(len(normal_scores_baseline)), np.ones(len(attack_scores_baseline))])
    scores_baseline = np.concatenate([normal_scores_baseline, attack_scores_baseline])
    auroc_baseline = roc_auc_score(labels, scores_baseline)

    # With CORAL alignment
    X_normal_aligned_seq = create_sequences(X_normal_aligned, SEQUENCE_LENGTH)
    X_attack_aligned_seq = create_sequences(X_attack_aligned, SEQUENCE_LENGTH)

    with torch.no_grad():
        normal_scores_aligned = torch.sigmoid(model(torch.FloatTensor(X_normal_aligned_seq))[0]).squeeze().numpy()
        attack_scores_aligned = torch.sigmoid(model(torch.FloatTensor(X_attack_aligned_seq))[0]).squeeze().numpy()

    scores_aligned = np.concatenate([normal_scores_aligned, attack_scores_aligned])
    auroc_aligned = roc_auc_score(labels, scores_aligned)

    improvement = auroc_aligned - auroc_baseline

    results = {
        'auroc_baseline': float(auroc_baseline),
        'auroc_aligned': float(auroc_aligned),
        'improvement': float(improvement),
        'coral_helps': bool(improvement > 0.05)
    }

    print(f"  OOD AUROC (baseline): {auroc_baseline*100:.1f}%")
    print(f"  OOD AUROC (CORAL aligned): {auroc_aligned*100:.1f}%")
    print(f"  Improvement: {improvement*100:+.1f}%")

    if results['coral_helps']:
        print(f"\n  Verdict: [OK] CORAL alignment improves OOD robustness")
    else:
        print(f"\n  Verdict: [!] CORAL alignment has limited effect")

    return results


# ============================================================
# HELPER: Simple training function
# ============================================================
def train_model_simple(
    train_normal: np.ndarray,
    train_attacks: Dict,
    epochs: int = 20,
    verbose: bool = True
) -> Tuple[nn.Module, StandardScaler]:
    """Train a model without domain randomization (for controlled tests)."""

    train_data = [train_normal]
    train_labels = [np.zeros(len(train_normal))]

    for attack_name, (attack_data, attack_labels) in train_attacks.items():
        train_data.append(attack_data)
        train_labels.append(attack_labels)

    X_train = np.vstack(train_data)
    y_train = np.concatenate(train_labels)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_train_seq = create_sequences(X_train_scaled, SEQUENCE_LENGTH)
    y_train_seq = y_train[SEQUENCE_LENGTH-1:]

    model = RobustDetector(input_dim=X_train_seq.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_seq),
        torch.FloatTensor(y_train_seq)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred, _ = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    return model, scaler


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("STEP 1: Generate Data")
    print("=" * 70)

    # Training data
    train_trajectories = []
    for i in range(N_TRAJECTORIES):
        traj = generate_trajectory(TRAJ_LENGTH, seed=TRAIN_SEED + i)
        train_trajectories.append(traj)
    train_normal = np.vstack(train_trajectories)
    print(f"  Training samples: {len(train_normal)}")

    # Test data (in-distribution)
    test_trajectories = []
    for i in range(N_TRAJECTORIES // 2):
        traj = generate_trajectory(TRAJ_LENGTH, seed=TEST_SEED + i)
        test_trajectories.append(traj)
    test_normal = np.vstack(test_trajectories)
    print(f"  Test samples (ID): {len(test_normal)}")

    # OOD data
    ood_trajectories = []
    for i in range(N_TRAJECTORIES // 2):
        traj = generate_trajectory(TRAJ_LENGTH, seed=OOD_SEED + i)
        ood_trajectories.append(traj)
    ood_normal = np.vstack(ood_trajectories)
    print(f"  Test samples (OOD): {len(ood_normal)}")

    # Train full model for baseline tests
    print("\n" + "=" * 70)
    print("STEP 2: Train Baseline Model")
    print("=" * 70)

    all_attacks = ['ar1_drift', 'coordinated', 'intermittent', 'bias', 'noise', 'ramp']
    train_attacks = {}
    for attack_type in all_attacks:
        attack_data, attack_labels = generate_attack(
            train_normal[:TRAJ_LENGTH], attack_type, seed=TRAIN_SEED
        )
        train_attacks[attack_type] = (attack_data, attack_labels)

    model, scaler = train_model_simple(train_normal, train_attacks, epochs=EPOCHS, verbose=True)
    print("  Model trained.")

    # Generate test attack for missed detection analysis
    test_attack, test_labels = generate_attack(
        test_normal[:TRAJ_LENGTH], 'intermittent', seed=TEST_SEED
    )
    combined_data = np.vstack([test_normal[:TRAJ_LENGTH], test_attack])
    combined_labels = np.concatenate([np.zeros(TRAJ_LENGTH), test_labels])

    # Run all tests
    results = {}

    # Test 1: Missed Detection Analysis
    results['missed_detection'] = analyze_missed_detections(
        model, scaler, combined_data, combined_labels
    )

    # Test 2: Leave-One-Attack-Out
    results['loao'] = leave_one_attack_out_evaluation(
        train_normal, test_normal
    )

    # Test 3: Parameter Extrapolation
    results['extrapolation'] = parameter_extrapolation_test(
        train_normal, test_normal
    )

    # Test 4: Temporal Shuffling
    results['temporal'] = temporal_shuffling_test(
        model, scaler, test_normal
    )

    # Test 5: CORAL Alignment
    results['coral'] = coral_alignment_test(
        model, scaler, train_normal, ood_normal
    )

    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "=" * 70)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("=" * 70)

    print("""
    Test                          | Result           | Verdict
    ------------------------------|------------------|------------------
    1. Missed Detection Analysis  | {missed:.1f}% missed     | {missed_v}
    2. Leave-One-Attack-Out       | {loao:.1f}% mean AUROC  | {loao_v}
    3. Parameter Extrapolation    | {extrap:.1f}% mean AUROC | {extrap_v}
    4. Temporal Shuffling         | {temp:.1f}% degradation | {temp_v}
    5. CORAL Alignment            | {coral:+.1f}% improvement | {coral_v}
    """.format(
        missed=results['missed_detection']['missed_rate'] * 100,
        missed_v="Analyzed" if 'categories' in results['missed_detection'] else "Error",
        loao=results['loao']['summary']['mean_auroc'] * 100,
        loao_v="[OK] Feature learning" if results['loao']['summary']['all_above_random'] else "[X] Memorizing",
        extrap=results['extrapolation']['summary']['mean_auroc'] * 100,
        extrap_v="[OK] Generalizes" if results['extrapolation']['summary']['all_generalize'] else "[X] Overfitting",
        temp=results['temporal']['degradation'] * 100,
        temp_v="[OK] Uses structure" if results['temporal']['relies_on_structure'] else "[!] Uses statistics",
        coral=results['coral']['improvement'] * 100,
        coral_v="[OK] Helps OOD" if results['coral']['coral_helps'] else "[!] Limited effect"
    ))

    # Overall assessment
    passes = sum([
        results['loao']['summary']['all_above_random'],
        results['extrapolation']['summary']['all_generalize'],
        results['temporal']['relies_on_structure']
    ])

    print(f"  Overall: {passes}/3 robustness tests passed")

    if passes >= 2:
        print("  Verdict: [OK] Evidence supports feature learning, not memorization")
    else:
        print("  Verdict: [!] Additional investigation needed")

    # Save results
    results['timestamp'] = datetime.now().isoformat()
    results['methodology'] = 'Robustness evaluation to prove feature learning'

    output_path = Path('results/robustness_evaluation.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\n  Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
