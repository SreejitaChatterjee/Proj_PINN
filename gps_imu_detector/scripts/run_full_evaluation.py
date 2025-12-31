#!/usr/bin/env python3
"""
Full Training and Evaluation Pipeline
Generates validated results with proper documentation.

Run: python run_full_evaluation.py
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import json
import platform
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 100
CV_FOLDS = 5

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ============================================================
# HARDWARE INFO
# ============================================================
def get_hardware_info() -> Dict:
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
        'timestamp': datetime.now().isoformat(),
        'seed': SEED
    }

# ============================================================
# DATA LOADING
# ============================================================
def load_euroc_data(data_path: str) -> pd.DataFrame:
    """Load EuRoC dataset."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    print(f"Sequences: {df['sequence'].unique().tolist()}")
    return df

def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Extract features - NO circular sensors."""
    feature_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'ax', 'ay', 'az']
    X = df[feature_cols].values
    return X, feature_cols

def create_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Create sequences for temporal model."""
    n_samples = len(X) - seq_len + 1
    sequences = np.zeros((n_samples, seq_len, X.shape[1]))
    for i in range(n_samples):
        sequences[i] = X[i:i+seq_len]
    return sequences

# ============================================================
# ATTACK GENERATION
# ============================================================
def generate_attacks(X: np.ndarray, seed: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Generate attack datasets with fixed seeds."""
    np.random.seed(seed)
    attacks = {}
    n = len(X)

    # Bias attack
    bias = X.copy()
    bias[:, :3] += np.random.randn(3) * 0.5  # Position bias
    labels = np.ones(n)
    attacks['bias'] = (bias, labels)

    # Drift attack (AR1)
    drift = X.copy()
    drift_signal = np.zeros((n, 3))
    for i in range(1, n):
        drift_signal[i] = 0.995 * drift_signal[i-1] + np.random.randn(3) * 0.01
    drift[:, :3] += drift_signal
    attacks['drift'] = (drift, labels.copy())

    # Noise attack
    noise = X.copy()
    noise[:, :6] += np.random.randn(n, 6) * 0.3
    attacks['noise'] = (noise, labels.copy())

    # Coordinated attack
    coord = X.copy()
    coord[:, :3] += 0.3  # Position
    coord[:, 3:6] += 0.1  # Velocity (consistent)
    attacks['coordinated'] = (coord, labels.copy())

    # Intermittent attack
    intermittent = X.copy()
    attack_mask = np.random.rand(n) < 0.1  # 10% of time
    intermittent[attack_mask, :3] += np.random.randn(attack_mask.sum(), 3) * 1.0
    labels_int = attack_mask.astype(float)
    attacks['intermittent'] = (intermittent, labels_int)

    return attacks

# ============================================================
# MODEL
# ============================================================
class CNNGRUDetector(nn.Module):
    def __init__(self, input_dim: int, conv_channels: int = 32, gru_hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1)
        self.gru = nn.GRU(conv_channels * 2, gru_hidden, batch_first=True)
        self.fc = nn.Linear(gru_hidden, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden=None):
        # x: (batch, seq, features)
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (batch, seq, channels)
        x, hidden = self.gru(x, hidden)
        x = self.dropout(x[:, -1, :])  # Last timestep
        x = torch.sigmoid(self.fc(x))
        return x, hidden

# ============================================================
# TRAINING
# ============================================================
def train_model(model, train_loader, val_loader, epochs, lr):
    """Train with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred, _ = model(X_batch)
                val_loss += criterion(y_pred.squeeze(), y_batch).item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")

    return model

# ============================================================
# EVALUATION
# ============================================================
def compute_recall_at_fpr(labels, scores, target_fpr):
    """Compute recall at specific FPR."""
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, target_fpr)
    if idx >= len(tpr):
        idx = len(tpr) - 1
    return float(tpr[idx])

def evaluate_model(model, X_test, y_test) -> Dict:
    """Evaluate model on test data."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        y_pred, _ = model(X_tensor)
        scores = y_pred.squeeze().numpy()

    # Handle edge cases
    if len(np.unique(y_test)) < 2:
        return {'auroc': 0.5, 'aupr': 0.5, 'recall_1pct': 0.0, 'recall_5pct': 0.0}

    auroc = roc_auc_score(y_test, scores)
    aupr = average_precision_score(y_test, scores)
    recall_1pct = compute_recall_at_fpr(y_test, scores, 0.01)
    recall_5pct = compute_recall_at_fpr(y_test, scores, 0.05)

    return {
        'auroc': float(auroc),
        'aupr': float(aupr),
        'recall_1pct': float(recall_1pct),
        'recall_5pct': float(recall_5pct)
    }

def benchmark_latency(model, input_shape, n_warmup=50, n_iterations=500) -> Dict:
    """Benchmark inference latency."""
    model.eval()
    X = torch.randn(*input_shape)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(X)

    # Benchmark
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            model(X)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'n_iterations': n_iterations
    }

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("GPS-IMU ANOMALY DETECTOR - FULL EVALUATION")
    print("=" * 70)

    # Hardware info
    hw_info = get_hardware_info()
    print(f"\nHardware: {hw_info['processor']}")
    print(f"Platform: {hw_info['platform']}")
    print(f"PyTorch: {hw_info['pytorch_version']}")
    print(f"Seed: {hw_info['seed']}")

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    # Use absolute path or relative from script location
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / 'data' / 'euroc' / 'all_sequences.csv'
    df = load_euroc_data(str(data_path))
    X, feature_cols = prepare_features(df)
    sequences = df['sequence'].values
    print(f"Features: {feature_cols}")
    print(f"Shape: {X.shape}")

    # Generate attacks
    print("\n" + "=" * 70)
    print("GENERATING ATTACKS")
    print("=" * 70)
    attacks = generate_attacks(X, seed=SEED)
    for name in attacks:
        print(f"  {name}: {len(attacks[name][0])} samples")

    # Sequence-wise CV
    print("\n" + "=" * 70)
    print(f"SEQUENCE-WISE {CV_FOLDS}-FOLD CROSS-VALIDATION")
    print("=" * 70)

    unique_sequences = df['sequence'].unique()
    print(f"Sequences: {unique_sequences.tolist()}")

    all_results = []
    fold_results = []

    for fold, test_seq in enumerate(unique_sequences):
        print(f"\nFold {fold+1}/{len(unique_sequences)}: Test on {test_seq}")

        # Split by sequence
        train_mask = sequences != test_seq
        test_mask = sequences == test_seq

        X_train_raw = X[train_mask]
        X_test_raw = X[test_mask]

        # Fit scaler on TRAINING ONLY
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # Create sequences
        X_train_seq = create_sequences(X_train_scaled, SEQUENCE_LENGTH)
        X_test_seq = create_sequences(X_test_scaled, SEQUENCE_LENGTH)

        # Create normal labels (0 = normal)
        y_train = np.zeros(len(X_train_seq))
        y_test_normal = np.zeros(len(X_test_seq))

        # Train/val split
        n_train = int(len(X_train_seq) * 0.8)
        X_train_final = X_train_seq[:n_train]
        y_train_final = y_train[:n_train]
        X_val = X_train_seq[n_train:]
        y_val = y_train[n_train:]

        # DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train_final), torch.FloatTensor(y_train_final))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Train model
        input_dim = X_train_seq.shape[2]
        model = CNNGRUDetector(input_dim=input_dim)
        print(f"  Training model (input_dim={input_dim})...")
        model = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE)

        # Evaluate on each attack type
        fold_attack_results = {}
        for attack_name, (X_attack_raw, y_attack_labels) in attacks.items():
            # Scale attack data with TRAINING scaler
            X_attack_scaled = scaler.transform(X_attack_raw[test_mask])
            X_attack_seq = create_sequences(X_attack_scaled, SEQUENCE_LENGTH)
            y_attack = y_attack_labels[test_mask][SEQUENCE_LENGTH-1:]

            # Combine normal + attack for evaluation
            X_eval = np.vstack([X_test_seq, X_attack_seq])
            y_eval = np.concatenate([y_test_normal, y_attack])

            metrics = evaluate_model(model, X_eval, y_eval)
            fold_attack_results[attack_name] = metrics
            print(f"    {attack_name}: AUROC={metrics['auroc']:.3f}, R@5%={metrics['recall_5pct']:.3f}")

        fold_results.append({
            'fold': fold + 1,
            'test_sequence': test_seq,
            'attack_results': fold_attack_results
        })

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    attack_metrics = {}
    for attack_name in attacks.keys():
        aurocs = [f['attack_results'][attack_name]['auroc'] for f in fold_results]
        recall_5pcts = [f['attack_results'][attack_name]['recall_5pct'] for f in fold_results]
        attack_metrics[attack_name] = {
            'auroc_mean': float(np.mean(aurocs)),
            'auroc_std': float(np.std(aurocs)),
            'recall_5pct_mean': float(np.mean(recall_5pcts)),
            'recall_5pct_std': float(np.std(recall_5pcts))
        }
        print(f"{attack_name}: AUROC={np.mean(aurocs):.3f}±{np.std(aurocs):.3f}, "
              f"R@5%={np.mean(recall_5pcts):.3f}±{np.std(recall_5pcts):.3f}")

    # Worst-case recall
    worst_case_recall = min(m['recall_5pct_mean'] for m in attack_metrics.values())
    worst_case_attack = min(attack_metrics.keys(), key=lambda k: attack_metrics[k]['recall_5pct_mean'])
    print(f"\nWorst-case Recall@5%FPR: {worst_case_recall:.3f} ({worst_case_attack})")

    # Latency benchmark
    print("\n" + "=" * 70)
    print("LATENCY BENCHMARK")
    print("=" * 70)
    model = CNNGRUDetector(input_dim=len(feature_cols))
    latency = benchmark_latency(model, (1, SEQUENCE_LENGTH, len(feature_cols)))
    print(f"Mean: {latency['mean_ms']:.2f} ms")
    print(f"P50:  {latency['p50_ms']:.2f} ms")
    print(f"P95:  {latency['p95_ms']:.2f} ms")
    print(f"P99:  {latency['p99_ms']:.2f} ms")

    # Model size
    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = model_params * 4 / (1024 * 1024)  # FP32
    print(f"\nModel: {model_params:,} parameters ({model_size_mb:.2f} MB)")

    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    final_results = {
        'hardware': hw_info,
        'config': {
            'seed': SEED,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'sequence_length': SEQUENCE_LENGTH,
            'cv_folds': CV_FOLDS,
            'features': feature_cols
        },
        'per_attack_metrics': attack_metrics,
        'overall': {
            'mean_auroc': float(np.mean([m['auroc_mean'] for m in attack_metrics.values()])),
            'worst_case_recall_5pct': worst_case_recall,
            'worst_case_attack': worst_case_attack
        },
        'latency': latency,
        'model': {
            'parameters': model_params,
            'size_mb': model_size_mb
        },
        'fold_results': fold_results
    }

    results_path = results_dir / 'validated_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return final_results

if __name__ == "__main__":
    results = main()
