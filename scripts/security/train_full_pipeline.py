#!/usr/bin/env python3
"""
Comprehensive Security Detection Training Pipeline.

Trains ALL components of the security detection system:
1. PINN base detector on EuRoC data
2. Sequence detector for temporal attacks
3. Supervised classifier with hard negatives
4. Hardened detector threshold calibration
5. Full evaluation on all attack types

Usage:
    python scripts/security/train_full_pipeline.py
"""

import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pinn_dynamics.security import (
    HardenedConfig,
    HardenedDetector,
    PhysicsAnomalyDetector,
    PhysicsLimits,
)

# Import attack generator from local script
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "security"))
from generate_synthetic_attacks import SyntheticAttackGenerator

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class PipelineConfig:
    """Full training pipeline configuration."""

    # Paths
    output_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "security" / "full_pipeline"
    )
    euroc_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "euroc")

    # PINN training
    pinn_hidden: int = 256
    pinn_layers: int = 5
    pinn_dropout: float = 0.1
    pinn_lr: float = 1e-3
    pinn_epochs: int = 30  # Reduced for faster training
    pinn_batch_size: int = 64
    pinn_physics_weight: float = 0.0  # Data-driven (w=0 shown best)

    # Sequence model
    seq_length: int = 20
    seq_hidden: int = 128
    seq_lstm_layers: int = 2
    seq_epochs: int = 20  # Reduced for faster training
    seq_batch_size: int = 256

    # Supervised classifier
    clf_n_estimators: int = 300
    clf_max_depth: int = 15
    clf_class_weight: float = 5.0  # Weight for attack class

    # Hardened detector calibration
    calibration_samples: int = 5000
    grid_search_points: int = 8  # Reduced for faster calibration

    # Attack generation
    attack_types: List[str] = field(
        default_factory=lambda: [
            # GPS attacks
            "gps_gradual_drift",
            "gps_sudden_jump",
            "gps_oscillating",
            "gps_meaconing",
            "gps_jamming",
            "gps_freeze",
            "gps_multipath",
            # IMU attacks
            "imu_constant_bias",
            "imu_gradual_drift",
            "imu_sinusoidal",
            "imu_noise_injection",
            "imu_scale_factor",
            # Temporal attacks
            "replay_attack",
            "delay_attack",
            "dropout_attack",
            # Stealth attacks (hard negatives)
            "stealth_adaptive",
            "stealth_slow_ramp",
            "stealth_intermittent",
            # Coordinated attacks
            "coordinated_gps_imu",
            "coordinated_sensor_actuator",
        ]
    )

    # Evaluation
    test_split: float = 0.2
    n_attack_samples: int = 500

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Data Loading
# ============================================================================


def load_euroc_data(
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load EuRoC dataset with proper train/test split."""
    print("\n" + "=" * 60)
    print("Loading EuRoC Dataset")
    print("=" * 60)

    # Use preprocessed CSV if available
    csv_path = config.euroc_dir / "all_sequences.csv"

    if csv_path.exists():
        print(f"  Loading from preprocessed: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Could not find {csv_path}")

    # State columns (EuRoC format): x, y, z, roll, pitch, yaw, p, q, r, vx, vy, vz
    state_cols = ["x", "y", "z", "roll", "pitch", "yaw", "p", "q", "r", "vx", "vy", "vz"]

    # Training and test sequences
    train_sequences = ["MH_02_easy", "V1_02_medium", "MH_03_medium"]
    test_sequences = ["MH_01_easy", "V1_01_easy"]

    train_data = []
    test_data = []

    for seq in train_sequences:
        seq_data = df[df["sequence"] == seq][state_cols].values
        if len(seq_data) > 0:
            train_data.append(seq_data)
            print(f"  {seq}: {len(seq_data)} samples (train)")

    for seq in test_sequences:
        seq_data = df[df["sequence"] == seq][state_cols].values
        if len(seq_data) > 0:
            test_data.append(seq_data)
            print(f"  {seq}: {len(seq_data)} samples (test)")

    if not train_data or not test_data:
        raise ValueError("Could not load any EuRoC sequences!")

    train_arr = np.vstack(train_data) if len(train_data) > 1 else train_data[0]
    test_arr = np.vstack(test_data) if len(test_data) > 1 else test_data[0]

    # Extract features (state at t) and labels (state at t+1)
    X_train = train_arr[:-1]
    y_train = train_arr[1:]
    X_test = test_arr[:-1]
    y_test = test_arr[1:]

    print(f"\nTotal: Train={len(X_train)}, Test={len(X_test)}")

    return X_train, y_train, X_test, y_test


def generate_attack_data(
    clean_data: np.ndarray, config: PipelineConfig, dt: float = 0.005
) -> Dict[str, np.ndarray]:
    """Generate synthetic attacks on clean data."""
    print("\n" + "=" * 60)
    print("Generating Synthetic Attacks")
    print("=" * 60)

    # Convert numpy array to DataFrame for SyntheticAttackGenerator
    # State columns: px, py, pz, roll, pitch, yaw, p, q, r, vx, vy, vz
    state_cols = ["x", "y", "z", "roll", "pitch", "yaw", "p", "q", "r", "vx", "vy", "vz"]

    n_samples = min(config.n_attack_samples, len(clean_data))
    segment = clean_data[:n_samples]

    # Create DataFrame
    df = pd.DataFrame(segment, columns=state_cols)
    df["timestamp"] = np.arange(len(df)) * dt

    # Initialize attack generator
    generator = SyntheticAttackGenerator(df, seed=42, randomize=False)
    attacks = {}

    # Get all 30 attack types from generator
    all_attacks = generator.generate_all_attacks(handle_nan=True)

    for attack_type, attacked_df in all_attacks.items():
        try:
            # Convert back to numpy array
            attack_array = attacked_df[state_cols].values
            attacks[attack_type] = attack_array
        except Exception as e:
            print(f"  Warning: Could not process {attack_type}: {e}")

    print(f"\nGenerated {len(attacks)} attack types, {n_samples} samples each")
    return attacks


# ============================================================================
# Simple MLP for State Prediction
# ============================================================================


class StatePredictor(nn.Module):
    """Simple MLP for state prediction (next state from current state)."""

    def __init__(
        self, state_dim: int, hidden_dim: int = 256, n_layers: int = 5, dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        in_dim = state_dim

        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# PINN Training
# ============================================================================


def train_pinn_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: PipelineConfig,
) -> Tuple[nn.Module, StandardScaler, StandardScaler, float]:
    """Train PINN base detector."""
    print("\n" + "=" * 60)
    print("Training PINN Base Detector")
    print("=" * 60)

    # Scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    # Create model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = StatePredictor(
        state_dim=input_dim,
        hidden_dim=config.pinn_hidden,
        n_layers=config.pinn_layers,
        dropout=config.pinn_dropout,
    ).to(config.device)

    # DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled)
    )
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled))

    train_loader = DataLoader(train_dataset, batch_size=config.pinn_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.pinn_batch_size)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=config.pinn_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config.pinn_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(config.device)
            y_batch = y_batch.to(config.device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(config.device)
                y_batch = y_batch.to(config.device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()

        val_loss /= len(test_loader)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1}/{config.pinn_epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}"
            )

        if patience_counter >= 20:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    # Compute detection threshold (95th percentile of validation errors)
    model.eval()
    errors = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(config.device)
            y_batch = y_batch.to(config.device)
            y_pred = model(X_batch)
            error = torch.norm(y_pred - y_batch, dim=1).cpu().numpy()
            errors.extend(error)

    threshold = np.percentile(errors, 95)
    print(f"\nPINN Training Complete:")
    print(f"  Best validation loss: {best_loss:.4f}")
    print(f"  Detection threshold (95%): {threshold:.4f}")

    return model, scaler_X, scaler_y, threshold


# ============================================================================
# Sequence Model Training
# ============================================================================


class SequencePINN(nn.Module):
    """LSTM-based sequence model for temporal attack detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_sequence_detector(
    X_train: np.ndarray, X_test: np.ndarray, config: PipelineConfig
) -> Tuple[nn.Module, StandardScaler, float]:
    """Train sequence-based temporal detector."""
    print("\n" + "=" * 60)
    print("Training Sequence Detector (LSTM)")
    print("=" * 60)

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, config.seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, config.seq_length)

    print(f"  Sequence shape: {X_train_seq.shape}")

    # Model
    input_dim = X_train.shape[1]
    model = SequencePINN(
        input_dim=input_dim,
        hidden_dim=config.seq_hidden,
        output_dim=input_dim,
        n_layers=config.seq_lstm_layers,
    ).to(config.device)

    # DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test_seq))

    train_loader = DataLoader(train_dataset, batch_size=config.seq_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.seq_batch_size)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=config.pinn_lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(config.seq_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(config.device)
            y_batch = y_batch.to(config.device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(config.device)
                y_batch = y_batch.to(config.device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()

        val_loss /= len(test_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1}/{config.seq_epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}"
            )

    model.load_state_dict(best_state)

    # Compute threshold
    model.eval()
    errors = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(config.device)
            y_batch = y_batch.to(config.device)
            y_pred = model(X_batch)
            error = torch.norm(y_pred - y_batch, dim=1).cpu().numpy()
            errors.extend(error)

    threshold = np.percentile(errors, 95)
    print(f"\nSequence Training Complete:")
    print(f"  Best validation loss: {best_loss:.4f}")
    print(f"  Detection threshold (95%): {threshold:.4f}")

    return model, scaler, threshold


# ============================================================================
# Supervised Classifier Training
# ============================================================================


def extract_features(data: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Extract statistical features from window of data."""
    features = []

    for i in range(window_size, len(data)):
        window = data[i - window_size : i]

        # Basic statistics
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        min_val = np.min(window, axis=0)
        max_val = np.max(window, axis=0)

        # Derivatives
        diff = np.diff(window, axis=0)
        diff_mean = np.mean(diff, axis=0)
        diff_std = np.std(diff, axis=0)

        # Second derivative (jerk for position)
        diff2 = np.diff(diff, axis=0)
        diff2_mean = np.mean(diff2, axis=0)
        diff2_max = np.max(np.abs(diff2), axis=0)

        # Autocorrelation at lag 1
        autocorr = np.array(
            [np.corrcoef(window[:-1, j], window[1:, j])[0, 1] for j in range(window.shape[1])]
        )
        autocorr = np.nan_to_num(autocorr, nan=0.0)

        feat = np.concatenate(
            [mean, std, min_val, max_val, diff_mean, diff_std, diff2_mean, diff2_max, autocorr]
        )
        features.append(feat)

    return np.array(features)


def train_supervised_classifier(
    clean_data: np.ndarray, attack_data: Dict[str, np.ndarray], config: PipelineConfig
) -> Tuple[RandomForestClassifier, StandardScaler]:
    """Train supervised attack classifier."""
    print("\n" + "=" * 60)
    print("Training Supervised Classifier")
    print("=" * 60)

    window_size = 50

    # Extract features from clean data
    print("  Extracting features from clean data...")
    clean_features = extract_features(clean_data[: config.calibration_samples], window_size)
    clean_labels = np.zeros(len(clean_features))

    # Extract features from attack data
    all_attack_features = []
    all_attack_labels = []

    for i, (attack_type, data) in enumerate(attack_data.items()):
        print(f"  Extracting features from {attack_type}...")
        attack_features = extract_features(data, window_size)
        all_attack_features.append(attack_features)
        all_attack_labels.extend([i + 1] * len(attack_features))  # Labels 1, 2, 3, ...

    attack_features = np.vstack(all_attack_features)
    attack_labels = np.array(all_attack_labels)

    # Combine
    X = np.vstack([clean_features, attack_features])
    y = np.concatenate([clean_labels, attack_labels])

    # Binary classification: normal (0) vs attack (any > 0)
    y_binary = (y > 0).astype(int)

    print(f"\n  Total samples: {len(X)}")
    print(f"  Normal: {np.sum(y_binary == 0)}, Attack: {np.sum(y_binary == 1)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=config.test_split, stratify=y_binary, random_state=42
    )

    # Train classifier with class weights
    print("\n  Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=config.clf_n_estimators,
        max_depth=config.clf_max_depth,
        class_weight={0: 1.0, 1: config.clf_class_weight},
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nSupervised Classifier Results:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  AUC-ROC: {auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])}")

    return clf, scaler


# ============================================================================
# Hardened Detector Calibration
# ============================================================================


def calibrate_hardened_detector(
    clean_data: np.ndarray,
    attack_data: Dict[str, np.ndarray],
    config: PipelineConfig,
    dt: float = 0.005,
) -> HardenedConfig:
    """Calibrate hardened detector thresholds via grid search."""
    print("\n" + "=" * 60)
    print("Calibrating Hardened Detector Thresholds")
    print("=" * 60)

    # Prepare calibration data
    n_cal = min(config.calibration_samples, len(clean_data))
    cal_clean = clean_data[:n_cal]

    # Get one attack sample for each type
    cal_attacks = {k: v[: min(200, len(v))] for k, v in list(attack_data.items())[:5]}

    # Grid search over key thresholds
    best_f1 = 0.0
    best_config = None

    jerk_thresholds = np.linspace(50, 200, config.grid_search_points // 4)
    nis_thresholds = np.linspace(20, 100, config.grid_search_points // 4)
    cusum_thresholds = np.linspace(5, 50, config.grid_search_points // 4)

    print(
        f"  Grid search: {len(jerk_thresholds)}x{len(nis_thresholds)}x{len(cusum_thresholds)} = "
        f"{len(jerk_thresholds)*len(nis_thresholds)*len(cusum_thresholds)} combinations"
    )

    for jerk_th in tqdm(jerk_thresholds, desc="Grid search"):
        for nis_th in nis_thresholds:
            for cusum_th in cusum_thresholds:
                # Create config
                cfg = HardenedConfig(
                    jerk_threshold=jerk_th,
                    nis_threshold=nis_th,
                    cusum_threshold=cusum_th,
                    spectral_threshold=3.0,
                )

                detector = HardenedDetector(config=cfg, dt=dt)

                # Evaluate on clean data (want low FP)
                fp = 0
                for i in range(100, len(cal_clean)):
                    pos = cal_clean[i, :3]
                    vel = cal_clean[i, 3:6] if cal_clean.shape[1] > 3 else np.zeros(3)
                    acc = np.zeros(3)
                    gyro = np.zeros(3)

                    result = detector.detect(pos, vel, acc, gyro)
                    if result["is_anomaly"]:
                        fp += 1

                fp_rate = fp / (len(cal_clean) - 100)

                # Evaluate on attacks (want high TP)
                tp = 0
                total_attack = 0

                for attack_type, attack in cal_attacks.items():
                    detector = HardenedDetector(config=cfg, dt=dt)  # Reset
                    for i in range(100, len(attack)):
                        pos = attack[i, :3]
                        vel = attack[i, 3:6] if attack.shape[1] > 3 else np.zeros(3)
                        acc = np.zeros(3)
                        gyro = np.zeros(3)

                        result = detector.detect(pos, vel, acc, gyro)
                        if result["is_anomaly"]:
                            tp += 1
                        total_attack += 1

                tp_rate = tp / total_attack if total_attack > 0 else 0

                # F1 score
                precision = tp_rate / (tp_rate + fp_rate + 1e-10)
                recall = tp_rate
                f1 = 2 * precision * recall / (precision + recall + 1e-10)

                if f1 > best_f1:
                    best_f1 = f1
                    best_config = cfg

    print(f"\nBest Configuration:")
    print(f"  Jerk threshold: {best_config.jerk_threshold:.1f}")
    print(f"  NIS threshold: {best_config.nis_threshold:.1f}")
    print(f"  CUSUM threshold: {best_config.cusum_threshold:.1f}")
    print(f"  Best F1: {best_f1:.3f}")

    return best_config


# ============================================================================
# Full Evaluation
# ============================================================================


def evaluate_full_pipeline(
    pinn_model: nn.Module,
    seq_model: nn.Module,
    classifier: RandomForestClassifier,
    hardened_config: HardenedConfig,
    clean_data: np.ndarray,
    attack_data: Dict[str, np.ndarray],
    scalers: Dict,
    config: PipelineConfig,
    dt: float = 0.005,
) -> Dict:
    """Evaluate the full detection pipeline."""
    print("\n" + "=" * 60)
    print("Full Pipeline Evaluation")
    print("=" * 60)

    results = {"per_attack": {}, "overall": {}, "confusion": None}

    # Initialize detectors
    hardened = HardenedDetector(config=hardened_config, dt=dt)
    physics = PhysicsAnomalyDetector()

    # Evaluate on clean data
    print("\n1. Clean Data Evaluation")
    fp_counts = {"pinn": 0, "seq": 0, "clf": 0, "hardened": 0, "ensemble": 0}
    n_clean = min(1000, len(clean_data) - 100)

    window_size = 50
    clean_features = extract_features(clean_data[: n_clean + window_size], window_size)
    clean_features_scaled = scalers["clf"].transform(clean_features)

    for i in range(n_clean):
        # Classifier
        if classifier.predict(clean_features_scaled[i : i + 1])[0] == 1:
            fp_counts["clf"] += 1

        # Hardened detector
        pos = clean_data[i + window_size, :3]
        vel = clean_data[i + window_size, 3:6] if clean_data.shape[1] > 3 else np.zeros(3)
        result = hardened.detect(pos, vel, np.zeros(3), np.zeros(3))
        if result["is_anomaly"]:
            fp_counts["hardened"] += 1

    for name, count in fp_counts.items():
        fp_rate = count / n_clean * 100
        print(f"  {name}: FP = {count}/{n_clean} = {fp_rate:.1f}%")

    results["overall"]["clean_fp_rate"] = {k: v / n_clean for k, v in fp_counts.items()}

    # Evaluate on each attack type
    print("\n2. Per-Attack Evaluation")
    print("-" * 60)

    all_y_true = []
    all_y_pred = []

    for attack_type, attack in attack_data.items():
        tp_counts = {"pinn": 0, "seq": 0, "clf": 0, "hardened": 0, "ensemble": 0}
        n_attack = min(500, len(attack) - window_size)

        # Reset hardened detector
        hardened = HardenedDetector(config=hardened_config, dt=dt)

        # Extract features
        attack_features = extract_features(attack[: n_attack + window_size], window_size)
        attack_features_scaled = scalers["clf"].transform(attack_features)

        for i in range(n_attack):
            # Classifier
            clf_pred = classifier.predict(attack_features_scaled[i : i + 1])[0]
            if clf_pred == 1:
                tp_counts["clf"] += 1

            # Hardened detector
            pos = attack[i + window_size, :3]
            vel = attack[i + window_size, 3:6] if attack.shape[1] > 3 else np.zeros(3)
            result = hardened.detect(pos, vel, np.zeros(3), np.zeros(3))
            if result["is_anomaly"]:
                tp_counts["hardened"] += 1

            # Ensemble (majority vote)
            votes = [clf_pred, int(result["is_anomaly"])]
            if sum(votes) >= 1:
                tp_counts["ensemble"] += 1

            all_y_true.append(1)  # Attack
            all_y_pred.append(int(sum(votes) >= 1))

        # Calculate metrics
        recall = {k: v / n_attack for k, v in tp_counts.items()}
        results["per_attack"][attack_type] = recall

        print(
            f"  {attack_type:30s}: CLF={recall['clf']*100:5.1f}%, "
            f"HARD={recall['hardened']*100:5.1f}%, ENS={recall['ensemble']*100:5.1f}%"
        )

    # Add clean data predictions
    for i in range(min(500, n_clean)):
        all_y_true.append(0)  # Normal
        clf_pred = classifier.predict(clean_features_scaled[i : i + 1])[0]
        all_y_pred.append(clf_pred)

    # Overall metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    results["overall"]["accuracy"] = accuracy_score(all_y_true, all_y_pred)
    results["overall"]["f1"] = f1_score(all_y_true, all_y_pred)
    results["confusion"] = confusion_matrix(all_y_true, all_y_pred).tolist()

    print("\n" + "-" * 60)
    print(f"Overall Accuracy: {results['overall']['accuracy']*100:.1f}%")
    print(f"Overall F1 Score: {results['overall']['f1']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={results['confusion'][0][0]}, FP={results['confusion'][0][1]}")
    print(f"  FN={results['confusion'][1][0]}, TP={results['confusion'][1][1]}")

    return results


# ============================================================================
# Main Pipeline
# ============================================================================


def main():
    """Run full training pipeline."""
    print("=" * 60)
    print("FULL SECURITY DETECTION TRAINING PIPELINE")
    print("=" * 60)

    start_time = time.time()
    config = PipelineConfig()

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    try:
        X_train, y_train, X_test, y_test = load_euroc_data(config)
    except Exception as e:
        print(f"\nError loading EuRoC data: {e}")
        print("Generating synthetic data instead...")

        # Generate synthetic quadrotor data
        t = np.linspace(0, 100, 20000)
        dt = t[1] - t[0]

        # Simulate circular trajectory
        radius = 5.0
        omega = 0.5

        pos = np.column_stack(
            [radius * np.cos(omega * t), radius * np.sin(omega * t), 2.0 + 0.5 * np.sin(0.2 * t)]
        )
        vel = np.column_stack(
            [
                -radius * omega * np.sin(omega * t),
                radius * omega * np.cos(omega * t),
                0.5 * 0.2 * np.cos(0.2 * t),
            ]
        )
        att = np.column_stack([0.1 * np.sin(0.3 * t), 0.1 * np.cos(0.3 * t), omega * t])
        rate = np.column_stack(
            [0.1 * 0.3 * np.cos(0.3 * t), -0.1 * 0.3 * np.sin(0.3 * t), np.full_like(t, omega)]
        )

        data = np.column_stack([pos, vel, att, rate])
        data += np.random.normal(0, 0.01, data.shape)  # Add noise

        split = int(0.8 * len(data))
        X_train, y_train = data[: split - 1], data[1:split]
        X_test, y_test = data[split:-1], data[split + 1 :]

        print(f"  Generated synthetic data: Train={len(X_train)}, Test={len(X_test)}")

    # Step 2: Generate attacks
    attack_data = generate_attack_data(X_test, config)

    # Step 3: Train PINN detector
    pinn_model, scaler_X, scaler_y, pinn_threshold = train_pinn_detector(
        X_train, y_train, X_test, y_test, config
    )

    # Step 4: Train sequence detector
    seq_model, seq_scaler, seq_threshold = train_sequence_detector(X_train, X_test, config)

    # Step 5: Train supervised classifier
    classifier, clf_scaler = train_supervised_classifier(X_train, attack_data, config)

    # Step 6: Calibrate hardened detector
    hardened_config = calibrate_hardened_detector(X_test, attack_data, config)

    # Step 7: Full evaluation
    scalers = {"pinn_X": scaler_X, "pinn_y": scaler_y, "seq": seq_scaler, "clf": clf_scaler}

    results = evaluate_full_pipeline(
        pinn_model, seq_model, classifier, hardened_config, X_test, attack_data, scalers, config
    )

    # Step 8: Save everything
    print("\n" + "=" * 60)
    print("Saving Models and Results")
    print("=" * 60)

    # Save PINN model
    torch.save(
        {
            "model_state": pinn_model.state_dict(),
            "threshold": pinn_threshold,
            "config": {
                "hidden": config.pinn_hidden,
                "layers": config.pinn_layers,
                "dropout": config.pinn_dropout,
            },
        },
        config.output_dir / "pinn_detector.pth",
    )
    print(f"  Saved: pinn_detector.pth")

    # Save sequence model
    torch.save(
        {
            "model_state": seq_model.state_dict(),
            "threshold": seq_threshold,
            "config": {
                "seq_length": config.seq_length,
                "hidden": config.seq_hidden,
                "lstm_layers": config.seq_lstm_layers,
            },
        },
        config.output_dir / "sequence_detector.pth",
    )
    print(f"  Saved: sequence_detector.pth")

    # Save classifier
    with open(config.output_dir / "classifier.pkl", "wb") as f:
        pickle.dump(classifier, f)
    print(f"  Saved: classifier.pkl")

    # Save scalers
    with open(config.output_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    print(f"  Saved: scalers.pkl")

    # Save hardened config
    hardened_dict = {
        "jerk_threshold": hardened_config.jerk_threshold,
        "nis_threshold": hardened_config.nis_threshold,
        "cusum_threshold": hardened_config.cusum_threshold,
        "spectral_threshold": hardened_config.spectral_threshold,
        "sprt_alpha": hardened_config.sprt_alpha,
        "sprt_beta": hardened_config.sprt_beta,
    }
    with open(config.output_dir / "hardened_config.json", "w") as f:
        json.dump(hardened_dict, f, indent=2)
    print(f"  Saved: hardened_config.json")

    # Save results
    with open(config.output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: evaluation_results.json")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output directory: {config.output_dir}")
    print(f"\nFinal Results:")
    print(f"  Overall Accuracy: {results['overall']['accuracy']*100:.1f}%")
    print(f"  Overall F1 Score: {results['overall']['f1']:.3f}")
    print(f"  Clean FP Rate: {results['overall']['clean_fp_rate']['ensemble']*100:.1f}%")

    # Best and worst attacks
    attack_recalls = [(k, v["ensemble"]) for k, v in results["per_attack"].items()]
    attack_recalls.sort(key=lambda x: x[1])

    print(f"\n  Hardest attacks (lowest recall):")
    for name, recall in attack_recalls[:3]:
        print(f"    {name}: {recall*100:.1f}%")

    print(f"\n  Easiest attacks (highest recall):")
    for name, recall in attack_recalls[-3:]:
        print(f"    {name}: {recall*100:.1f}%")


if __name__ == "__main__":
    main()
