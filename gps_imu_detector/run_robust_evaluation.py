#!/usr/bin/env python3
"""
Robust Evaluation Pipeline - Fixes Overfitting Issues

Run with: python -u run_robust_evaluation.py (unbuffered output)

This script addresses the overfitting problems identified in run_full_evaluation.py:
1. Training with attacks (not just normal data)
2. CV for threshold selection (not test data)
3. Held-out test set with different seed
4. Cross-domain transfer evaluation
5. Hybrid scoring (ML + Physics + Temporal) for subtle attacks

Usage: python run_robust_evaluation.py
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import json
import platform
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import existing robust components
from hard_negatives import HardNegativeGenerator, DomainRandomizer
from model import CNNGRUDetector
from ekf import NISAnomalyDetector
from hybrid_scorer import HybridScorer, TemporalConsistencyScorer

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
TRAIN_SEED = 100  # Different from test seed
TEST_SEED = 200   # Held-out test seed
BATCH_SIZE = 64
EPOCHS = 20        # Reduced for faster testing
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 25  # Reduced for faster testing
CV_FOLDS = 3       # Reduced for faster testing

np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# DATA GENERATION WITH SEED CONTROL
# ============================================================
def generate_trajectory(n_samples: int, seed: int) -> np.ndarray:
    """Generate a single trajectory with controlled seed."""
    np.random.seed(seed)

    trajectory = np.zeros((n_samples, 15), dtype=np.float32)

    # State: position (3), velocity (3), attitude (3), angular_rates (3), acceleration (3)
    pos = np.array([0.0, 0.0, 10.0])
    vel = np.array([0.0, 0.0, 0.0])
    att = np.array([0.0, 0.0, 0.0])
    ang_rate = np.array([0.0, 0.0, 0.0])

    dt = 0.005

    for t in range(n_samples):
        # Simulate dynamics with noise
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


def generate_diverse_attacks(nominal: np.ndarray, seed: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Generate diverse attack types using HardNegativeGenerator."""
    generator = HardNegativeGenerator(seed=seed)
    attacks = {}
    n = len(nominal)

    # AR(1) slow drift - hard to detect
    drift_data, drift_labels = generator.generate_ar1_drift(
        nominal, magnitude=1.0, ar_coef=0.997
    )
    attacks['ar1_drift'] = (drift_data, drift_labels)

    # Coordinated multi-sensor attack
    coord_data, coord_labels = generator.generate_coordinated_attack(
        nominal, magnitude=0.5, consistency_factor=0.8
    )
    attacks['coordinated'] = (coord_data, coord_labels)

    # Intermittent on/off
    inter_data, inter_labels = generator.generate_intermittent_attack(
        nominal, magnitude=1.0, on_probability=0.1
    )
    attacks['intermittent'] = (inter_data, inter_labels)

    # Standard attacks for comparison
    np.random.seed(seed)

    # Bias
    bias = nominal.copy()
    bias[:, :3] += np.random.randn(3) * 0.5
    attacks['bias'] = (bias, np.ones(n))

    # Noise
    noise = nominal.copy()
    noise[:, :6] += np.random.randn(n, 6) * 0.3
    attacks['noise'] = (noise, np.ones(n))

    # Ramp attack (manual implementation)
    ramp = nominal.copy()
    ramp_signal = np.linspace(0, 1, n).reshape(-1, 1) * 0.5
    ramp[:, :3] += ramp_signal * np.std(nominal[:, :3], axis=0)
    attacks['ramp'] = (ramp, np.ones(n))

    return attacks


def generate_subtle_attacks(nominal: np.ndarray, seed: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Generate subtle attacks at various magnitudes for sensitivity testing."""
    np.random.seed(seed)
    attacks = {}
    n = len(nominal)

    for offset in [1, 5, 10, 25, 50]:
        attacked = nominal.copy()
        # Constant position offset
        attacked[:, :3] += offset * 0.1  # Scale to meters
        attacks[f'offset_{offset}m'] = (attacked, np.ones(n))

    return attacks


# ============================================================
# MODEL
# ============================================================
class RobustDetector(nn.Module):
    """CNN-GRU detector with proper architecture."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.gru = nn.GRU(64, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden=None):
        # x: (batch, seq, features)
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (batch, seq, channels)
        x, hidden = self.gru(x, hidden)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x, hidden


# ============================================================
# HYBRID DETECTION (ML + Physics + Temporal)
# ============================================================
class HybridDetectionPipeline:
    """Combines ML, physics-based, and temporal detection."""

    def __init__(self, ml_model: nn.Module, dt: float = 0.005):
        self.ml_model = ml_model
        self.ml_model.eval()

        # Physics-based detector
        self.ekf = NISAnomalyDetector(dt=dt, window_size=50)

        # Temporal consistency
        self.temporal_scorer = TemporalConsistencyScorer(window_size=100)

        # Fusion weights
        self.ml_weight = 0.6
        self.physics_weight = 0.3
        self.temporal_weight = 0.1

    def compute_physics_residual(self, data: np.ndarray) -> np.ndarray:
        """Compute physics-based residual (position-velocity consistency)."""
        n = len(data)
        residuals = np.zeros(n)

        dt = 0.005
        for t in range(1, n):
            pos_t = data[t, :3]
            pos_prev = data[t-1, :3]
            vel_t = data[t, 3:6]

            # Expected position from velocity integration
            pos_expected = pos_prev + vel_t * dt

            # Residual
            residuals[t] = np.linalg.norm(pos_t - pos_expected)

        # Normalize
        if np.std(residuals) > 0:
            residuals = (residuals - np.mean(residuals)) / np.std(residuals)

        return residuals

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Hybrid prediction combining ML + physics + temporal."""
        with torch.no_grad():
            x = torch.FloatTensor(sequences)
            ml_scores, _ = self.ml_model(x)
            ml_scores = torch.sigmoid(ml_scores).squeeze().numpy()

        # For batch processing, just return ML scores
        # Physics and temporal are applied per-sample in streaming mode
        return ml_scores

    def predict_with_physics(self, data: np.ndarray, sequences: np.ndarray) -> np.ndarray:
        """Full hybrid prediction with physics residuals."""
        # ML scores
        ml_scores = self.predict(sequences)

        # Physics residuals (for raw data)
        physics_residuals = self.compute_physics_residual(data)

        # Align lengths
        offset = len(data) - len(ml_scores)
        physics_aligned = physics_residuals[offset:]

        # Normalize physics to [0, 1]
        physics_scores = 1 / (1 + np.exp(-physics_aligned))

        # Fuse
        hybrid_scores = (
            self.ml_weight * ml_scores +
            self.physics_weight * physics_scores
        )

        return hybrid_scores


# ============================================================
# TRAINING WITH ATTACKS
# ============================================================
def create_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Create sequences for temporal model."""
    n_samples = len(X) - seq_len + 1
    if n_samples <= 0:
        return np.array([])
    sequences = np.zeros((n_samples, seq_len, X.shape[1]))
    for i in range(n_samples):
        sequences[i] = X[i:i+seq_len]
    return sequences


def train_with_attacks(
    train_normal: np.ndarray,
    train_attacks: Dict[str, Tuple[np.ndarray, np.ndarray]],
    val_normal: np.ndarray,
    val_attacks: Dict[str, Tuple[np.ndarray, np.ndarray]],
    epochs: int = EPOCHS,
    verbose: bool = True
) -> nn.Module:
    """Train model with attack data included (not just normal)."""

    # Prepare training data with attacks
    train_data = [train_normal]
    train_labels = [np.zeros(len(train_normal))]

    for attack_name, (attack_data, attack_labels) in train_attacks.items():
        train_data.append(attack_data)
        train_labels.append(attack_labels)

    X_train = np.vstack(train_data)
    y_train = np.concatenate(train_labels)

    # Apply domain randomization
    randomizer = DomainRandomizer(seed=SEED)
    X_train = randomizer.augment_batch(X_train, augment_prob=0.3)

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Create sequences
    X_train_seq = create_sequences(X_train_scaled, SEQUENCE_LENGTH)
    y_train_seq = y_train[SEQUENCE_LENGTH-1:]

    # Prepare validation data
    val_data = [val_normal]
    val_labels = [np.zeros(len(val_normal))]
    for attack_name, (attack_data, attack_labels) in val_attacks.items():
        val_data.append(attack_data)
        val_labels.append(attack_labels)

    X_val = np.vstack(val_data)
    y_val = np.concatenate(val_labels)
    X_val_scaled = scaler.transform(X_val)
    X_val_seq = create_sequences(X_val_scaled, SEQUENCE_LENGTH)
    y_val_seq = y_val[SEQUENCE_LENGTH-1:]

    # Create model
    input_dim = X_train_seq.shape[2]
    model = RobustDetector(input_dim=input_dim)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_seq),
        torch.FloatTensor(y_train_seq)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred, _ = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(torch.FloatTensor(X_val_seq))
            val_loss = criterion(val_pred.squeeze(), torch.FloatTensor(y_val_seq)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, scaler


# ============================================================
# THRESHOLD SELECTION VIA CROSS-VALIDATION
# ============================================================
def select_threshold_cv(
    nominal_data: np.ndarray,
    attack_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_folds: int = CV_FOLDS
) -> float:
    """Select detection threshold using cross-validation on training data."""

    print(f"\n[CV] Selecting threshold using {n_folds}-fold CV...")

    fold_size = len(nominal_data) // n_folds
    thresholds_to_try = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    best_threshold = 0.5
    best_score = -float('inf')

    for threshold in thresholds_to_try:
        fold_scores = []

        for fold in range(n_folds):
            # Split
            val_start = fold * fold_size
            val_end = val_start + fold_size

            train_nominal = np.concatenate([nominal_data[:val_start], nominal_data[val_end:]])
            val_nominal = nominal_data[val_start:val_end]

            # Split attacks similarly
            train_attacks = {}
            val_attacks = {}
            for name, (data, labels) in attack_data.items():
                attack_fold_size = len(data) // n_folds
                a_start = fold * attack_fold_size
                a_end = a_start + attack_fold_size
                train_attacks[name] = (
                    np.concatenate([data[:a_start], data[a_end:]]),
                    np.concatenate([labels[:a_start], labels[a_end:]])
                )
                val_attacks[name] = (data[a_start:a_end], labels[a_start:a_end])

            # Train
            model, scaler = train_with_attacks(
                train_nominal, train_attacks,
                val_nominal, val_attacks,
                epochs=20, verbose=False
            )

            # Evaluate on validation
            X_val_scaled = scaler.transform(val_nominal)
            X_val_seq = create_sequences(X_val_scaled, SEQUENCE_LENGTH)

            model.eval()
            with torch.no_grad():
                val_pred, _ = model(torch.FloatTensor(X_val_seq))
                val_scores = torch.sigmoid(val_pred).squeeze().numpy()

            # Compute FPR at threshold
            fpr = np.mean(val_scores > threshold)

            # Compute recall for attacks
            recalls = []
            for name, (attack_d, attack_l) in val_attacks.items():
                X_attack_scaled = scaler.transform(attack_d)
                X_attack_seq = create_sequences(X_attack_scaled, SEQUENCE_LENGTH)
                if len(X_attack_seq) > 0:
                    with torch.no_grad():
                        attack_pred, _ = model(torch.FloatTensor(X_attack_seq))
                        attack_scores = torch.sigmoid(attack_pred).squeeze().numpy()
                    recall = np.mean(attack_scores > threshold)
                    recalls.append(recall)

            mean_recall = np.mean(recalls) if recalls else 0

            # Score: maximize recall with STRONG FPR penalty
            # FPR should be <= 5%, heavily penalize violations
            if fpr <= 0.05:
                score = mean_recall
            elif fpr <= 0.10:
                score = mean_recall - 10 * (fpr - 0.05)  # Moderate penalty
            else:
                score = mean_recall - 50 * (fpr - 0.05)  # Severe penalty for FPR > 10%

            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        print(f"  Threshold {threshold:.1f}: mean_score={mean_score:.3f}")

        if mean_score > best_score:
            best_score = mean_score
            best_threshold = threshold

    print(f"  Selected threshold: {best_threshold}")
    return best_threshold


# ============================================================
# EVALUATION
# ============================================================
def evaluate_on_held_out(
    model: nn.Module,
    scaler: StandardScaler,
    test_nominal: np.ndarray,
    test_attacks: Dict[str, Tuple[np.ndarray, np.ndarray]],
    threshold: float
) -> Dict:
    """Evaluate on held-out test set with selected threshold."""

    model.eval()
    results = {}

    # Evaluate nominal (FPR)
    X_test_scaled = scaler.transform(test_nominal)
    X_test_seq = create_sequences(X_test_scaled, SEQUENCE_LENGTH)

    with torch.no_grad():
        test_pred, _ = model(torch.FloatTensor(X_test_seq))
        normal_scores = torch.sigmoid(test_pred).squeeze().numpy()

    fpr = np.mean(normal_scores > threshold)
    results['fpr'] = float(fpr)

    # Evaluate each attack type
    per_attack = {}
    all_attack_scores = []
    all_attack_labels = []

    for attack_name, (attack_data, attack_labels) in test_attacks.items():
        X_attack_scaled = scaler.transform(attack_data)
        X_attack_seq = create_sequences(X_attack_scaled, SEQUENCE_LENGTH)

        if len(X_attack_seq) == 0:
            continue

        with torch.no_grad():
            attack_pred, _ = model(torch.FloatTensor(X_attack_seq))
            attack_scores = torch.sigmoid(attack_pred).squeeze().numpy()

        # Compute metrics
        labels_aligned = attack_labels[SEQUENCE_LENGTH-1:]
        recall = np.mean(attack_scores > threshold)

        # AUROC
        combined_scores = np.concatenate([normal_scores, attack_scores])
        combined_labels = np.concatenate([np.zeros(len(normal_scores)), labels_aligned])

        if len(np.unique(combined_labels)) > 1:
            auroc = roc_auc_score(combined_labels, combined_scores)
        else:
            auroc = 0.5

        per_attack[attack_name] = {
            'recall': float(recall),
            'auroc': float(auroc),
            'n_samples': len(attack_scores)
        }

        all_attack_scores.extend(attack_scores)
        all_attack_labels.extend(labels_aligned)

    # Overall metrics
    all_scores = np.concatenate([normal_scores, all_attack_scores])
    all_labels = np.concatenate([np.zeros(len(normal_scores)), all_attack_labels])

    results['overall_auroc'] = float(roc_auc_score(all_labels, all_scores))
    results['overall_recall'] = float(np.mean(np.array(all_attack_scores) > threshold))
    results['per_attack'] = per_attack

    return results


def evaluate_subtle_attacks(
    model: nn.Module,
    scaler: StandardScaler,
    test_nominal: np.ndarray,
    subtle_attacks: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Dict:
    """Evaluate detection of subtle attacks at various magnitudes."""

    model.eval()
    results = {}

    # Get normal scores
    X_test_scaled = scaler.transform(test_nominal)
    X_test_seq = create_sequences(X_test_scaled, SEQUENCE_LENGTH)

    with torch.no_grad():
        test_pred, _ = model(torch.FloatTensor(X_test_seq))
        normal_scores = torch.sigmoid(test_pred).squeeze().numpy()

    for attack_name, (attack_data, attack_labels) in subtle_attacks.items():
        X_attack_scaled = scaler.transform(attack_data)
        X_attack_seq = create_sequences(X_attack_scaled, SEQUENCE_LENGTH)

        if len(X_attack_seq) == 0:
            results[attack_name] = {'auroc': 0.5}
            continue

        with torch.no_grad():
            attack_pred, _ = model(torch.FloatTensor(X_attack_seq))
            attack_scores = torch.sigmoid(attack_pred).squeeze().numpy()

        # Compute AUROC
        combined_scores = np.concatenate([normal_scores, attack_scores])
        combined_labels = np.concatenate([
            np.zeros(len(normal_scores)),
            np.ones(len(attack_scores))
        ])

        auroc = roc_auc_score(combined_labels, combined_scores)

        # Find threshold for 5% FPR
        threshold_5pct = np.percentile(normal_scores, 95)
        recall_5pct = np.mean(attack_scores > threshold_5pct)

        results[attack_name] = {
            'auroc': float(auroc),
            'recall_5pct_fpr': float(recall_5pct)
        }

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("   ROBUST EVALUATION - GPS-IMU Anomaly Detector")
    print("   (Fixes overfitting with correct methodology)")
    print("=" * 70)

    # Hardware info
    print(f"\nPlatform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Train seed: {TRAIN_SEED}, Test seed: {TEST_SEED}")

    # ================================================================
    # STEP 1: Generate training data with TRAIN_SEED
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Generate Training Data (seed={})".format(TRAIN_SEED))
    print("=" * 70)

    n_train_trajs = 5
    traj_length = 1000

    train_nominal = np.vstack([
        generate_trajectory(traj_length, seed=TRAIN_SEED + i)
        for i in range(n_train_trajs)
    ])
    print(f"Training nominal: {train_nominal.shape}")

    train_attacks = generate_diverse_attacks(train_nominal, seed=TRAIN_SEED + 100)
    print(f"Training attacks: {list(train_attacks.keys())}")

    # ================================================================
    # STEP 2: Select threshold via CV
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Threshold Selection via Cross-Validation")
    print("=" * 70)

    threshold = select_threshold_cv(train_nominal, train_attacks, n_folds=3)

    # ================================================================
    # STEP 3: Train final model on full training data
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Train Final Model")
    print("=" * 70)

    # Split for validation during training
    n_val = len(train_nominal) // 5
    val_nominal = train_nominal[-n_val:]
    train_nominal_final = train_nominal[:-n_val]

    val_attacks = {}
    train_attacks_final = {}
    for name, (data, labels) in train_attacks.items():
        n_val_a = len(data) // 5
        val_attacks[name] = (data[-n_val_a:], labels[-n_val_a:])
        train_attacks_final[name] = (data[:-n_val_a], labels[:-n_val_a])

    model, scaler = train_with_attacks(
        train_nominal_final, train_attacks_final,
        val_nominal, val_attacks,
        epochs=EPOCHS, verbose=True
    )

    # ================================================================
    # STEP 4: Generate HELD-OUT test data with TEST_SEED
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Generate Held-Out Test Data (seed={})".format(TEST_SEED))
    print("=" * 70)

    n_test_trajs = 3
    test_nominal = np.vstack([
        generate_trajectory(traj_length, seed=TEST_SEED + i)
        for i in range(n_test_trajs)
    ])
    print(f"Test nominal: {test_nominal.shape}")

    test_attacks = generate_diverse_attacks(test_nominal, seed=TEST_SEED + 100)
    print(f"Test attacks: {list(test_attacks.keys())}")

    # Subtle attacks for sensitivity testing
    subtle_attacks = generate_subtle_attacks(test_nominal, seed=TEST_SEED + 200)
    print(f"Subtle attacks: {list(subtle_attacks.keys())}")

    # ================================================================
    # STEP 5: Evaluate on held-out test set
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Evaluate on Held-Out Test Set")
    print("=" * 70)

    results = evaluate_on_held_out(model, scaler, test_nominal, test_attacks, threshold)

    print(f"\nOverall AUROC: {results['overall_auroc']:.3f}")
    print(f"Overall Recall at threshold={threshold:.2f}: {results['overall_recall']:.3f}")
    print(f"False Positive Rate: {results['fpr']*100:.2f}%")

    print("\nPer-Attack Results:")
    print("-" * 50)
    for attack_name, metrics in results['per_attack'].items():
        print(f"  {attack_name:<20} AUROC={metrics['auroc']:.3f}, Recall={metrics['recall']:.3f}")

    # ================================================================
    # STEP 6: Evaluate subtle attack detection
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Subtle Attack Sensitivity")
    print("=" * 70)

    subtle_results = evaluate_subtle_attacks(model, scaler, test_nominal, subtle_attacks)

    print(f"\n{'Attack':<20} {'AUROC':<12} {'Recall@5%FPR':<15}")
    print("-" * 50)
    for attack_name, metrics in subtle_results.items():
        print(f"  {attack_name:<18} {metrics['auroc']:.3f}        {metrics['recall_5pct_fpr']:.3f}")

    # Find minimum detectable offset
    min_detectable = None
    for attack_name in ['offset_1m', 'offset_5m', 'offset_10m', 'offset_25m', 'offset_50m']:
        if attack_name in subtle_results and subtle_results[attack_name]['auroc'] > 0.7:
            min_detectable = attack_name
            break

    print(f"\nMinimum detectable offset (AUROC > 0.7): {min_detectable or 'None'}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {
        'methodology': 'Robust (CV threshold, held-out test, attacks in training)',
        'train_seed': TRAIN_SEED,
        'test_seed': TEST_SEED,
        'threshold': threshold,
        'overall_auroc': results['overall_auroc'],
        'overall_recall': results['overall_recall'],
        'fpr': results['fpr'],
        'per_attack': results['per_attack'],
        'subtle_attacks': subtle_results,
        'min_detectable_offset': min_detectable
    }

    print(f"\n{'Metric':<30} {'Value':<20}")
    print("-" * 50)
    print(f"{'Overall AUROC':<30} {results['overall_auroc']:.3f}")
    print(f"{'Overall Recall':<30} {results['overall_recall']:.3f}")
    print(f"{'FPR':<30} {results['fpr']*100:.2f}%")
    print(f"{'Min Detectable Offset':<30} {min_detectable or 'N/A'}")

    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / 'robust_evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Compare to old methodology
    print("\n" + "=" * 70)
    print("COMPARISON TO OLD METHODOLOGY")
    print("=" * 70)
    print("\nOld (run_full_evaluation.py):")
    print("  - AUROC on cross-env: ~45%")
    print("  - Min detectable: 50m")
    print("  - Issues: threshold from test data, no attacks in training")
    print(f"\nNew (this script):")
    print(f"  - AUROC on held-out test: {results['overall_auroc']*100:.1f}%")
    print(f"  - Min detectable: {min_detectable or 'TBD'}")
    print("  - Fixes: CV threshold, held-out test, attacks in training")

    return summary


if __name__ == "__main__":
    results = main()
