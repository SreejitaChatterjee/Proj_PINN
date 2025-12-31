#!/usr/bin/env python3
"""
Comprehensive Detector Research - Testing ALL possible approaches.

Categories tested:
1. Anomaly Detection Algorithms (LOF, OCSVM, IsoForest, EllipticEnvelope)
2. Advanced Feature Engineering (Spectral, Wavelet, Multi-scale)
3. Statistical Process Control (CUSUM, EWMA, Hotelling T²)
4. Deep Learning (LSTM Autoencoder, Conv1D Autoencoder)
5. Physics-Based (Kalman Filter Residuals, Energy-based)
6. Hybrid Ensembles
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import chi2
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("=" * 70)
print("COMPREHENSIVE DETECTOR RESEARCH")
print("=" * 70)

# Load data
df = pd.read_csv(Path(__file__).parent.parent.parent / "data/euroc/all_sequences.csv")
state_cols = ["x", "y", "z", "roll", "pitch", "yaw", "p", "q", "r", "vx", "vy", "vz"]
print(f"Loaded {len(df):,} samples")

clean_data = df[state_cols].values[:15000]
test_base = df[state_cols].values[50000:50500]

# ============================================================================
# FEATURE EXTRACTION METHODS
# ============================================================================


def extract_basic_features(data, window=50):
    """Basic statistical features."""
    features = []
    for i in range(window, len(data)):
        w = data[i - window : i]
        feat = np.concatenate(
            [
                np.mean(w, axis=0),
                np.std(w, axis=0),
                np.min(w, axis=0),
                np.max(w, axis=0),
                np.percentile(w, 25, axis=0),
                np.percentile(w, 75, axis=0),
            ]
        )
        features.append(feat)
    return np.array(features)


def extract_spectral_features(data, window=64, fs=200):
    """FFT-based spectral features."""
    features = []
    for i in range(window, len(data)):
        w = data[i - window : i]
        feat_list = []
        for j in range(w.shape[1]):
            fft_vals = np.abs(fft(w[:, j]))[: window // 2]
            # Spectral features
            feat_list.extend(
                [
                    np.max(fft_vals),  # Peak magnitude
                    np.argmax(fft_vals),  # Dominant frequency
                    np.mean(fft_vals),  # Spectral mean
                    np.std(fft_vals),  # Spectral spread
                    np.sum(fft_vals**2),  # Spectral energy
                ]
            )
        features.append(feat_list)
    return np.array(features)


def extract_multiscale_features(data, windows=[10, 25, 50, 100]):
    """Multi-scale statistical features."""
    all_features = []
    for i in range(max(windows), len(data)):
        feat_list = []
        for w_size in windows:
            w = data[i - w_size : i]
            feat_list.extend(
                [
                    np.mean(w, axis=0).mean(),
                    np.std(w, axis=0).mean(),
                    np.max(np.abs(np.diff(w, axis=0))),
                ]
            )
        all_features.append(feat_list)
    return np.array(all_features)


def extract_derivative_features(data, dt=0.005):
    """Velocity, acceleration, jerk features."""
    features = []
    for i in range(4, len(data)):
        # Derivatives
        vel = (data[i] - data[i - 1]) / dt
        acc = (data[i] - 2 * data[i - 1] + data[i - 2]) / (dt**2)
        jerk = (data[i] - 3 * data[i - 1] + 3 * data[i - 2] - data[i - 3]) / (dt**3)

        feat = np.concatenate(
            [
                vel,
                np.abs(vel),
                acc,
                np.abs(acc),
                jerk,
                np.abs(jerk),
                [np.linalg.norm(vel), np.linalg.norm(acc), np.linalg.norm(jerk)],
            ]
        )
        features.append(feat)
    return np.array(features)


def extract_correlation_features(data, window=50):
    """Cross-correlation between state variables."""
    features = []
    for i in range(window, len(data)):
        w = data[i - window : i]
        # Correlation matrix (upper triangle)
        corr = np.corrcoef(w.T)
        upper_tri = corr[np.triu_indices(corr.shape[0], k=1)]
        upper_tri = np.nan_to_num(upper_tri, nan=0)

        # Eigenvalues of covariance
        cov = np.cov(w.T)
        eigvals = np.linalg.eigvalsh(cov)

        feat = np.concatenate([upper_tri, eigvals])
        features.append(feat)
    return np.array(features)


# ============================================================================
# STATISTICAL PROCESS CONTROL
# ============================================================================


class CUSUMDetector:
    """Cumulative Sum control chart."""

    def __init__(self, threshold=5.0, drift=0.5):
        self.threshold = threshold
        self.drift = drift
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8

    def predict(self, data):
        z = (data - self.mean) / self.std
        cusum_pos = np.zeros_like(z)
        cusum_neg = np.zeros_like(z)

        for i in range(1, len(z)):
            cusum_pos[i] = np.maximum(0, cusum_pos[i - 1] + z[i] - self.drift)
            cusum_neg[i] = np.maximum(0, cusum_neg[i - 1] - z[i] - self.drift)

        anomaly = np.any((cusum_pos > self.threshold) | (cusum_neg > self.threshold), axis=1)
        return anomaly.astype(int)


class EWMADetector:
    """Exponentially Weighted Moving Average detector."""

    def __init__(self, lambda_=0.2, L=3.0):
        self.lambda_ = lambda_
        self.L = L
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8

    def predict(self, data):
        z = (data - self.mean) / self.std
        ewma = np.zeros_like(z)
        ewma[0] = z[0]

        for i in range(1, len(z)):
            ewma[i] = self.lambda_ * z[i] + (1 - self.lambda_) * ewma[i - 1]

        # Control limits
        sigma_ewma = self.std * np.sqrt(self.lambda_ / (2 - self.lambda_))
        ucl = self.L * sigma_ewma

        anomaly = np.any(np.abs(ewma) > ucl, axis=1)
        return anomaly.astype(int)


class HotellingT2Detector:
    """Hotelling's T² multivariate control chart."""

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mean = None
        self.cov_inv = None
        self.threshold = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        cov = np.cov(data.T) + np.eye(data.shape[1]) * 1e-6
        self.cov_inv = np.linalg.inv(cov)
        p = data.shape[1]
        n = len(data)
        self.threshold = chi2.ppf(1 - self.alpha, p)

    def predict(self, data):
        diff = data - self.mean
        t2 = np.sum(diff @ self.cov_inv * diff, axis=1)
        return (t2 > self.threshold).astype(int)


# ============================================================================
# KALMAN FILTER RESIDUAL DETECTOR
# ============================================================================


class KalmanResidualDetector:
    """Kalman filter-based anomaly detection using innovation sequence."""

    def __init__(self, threshold_percentile=95):
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.A = None  # State transition
        self.Q = None  # Process noise
        self.R = None  # Measurement noise

    def fit(self, data, dt=0.005):
        n_states = data.shape[1]

        # Simple constant velocity model
        self.A = np.eye(n_states)

        # Estimate process and measurement noise from data
        residuals = np.diff(data, axis=0)
        self.Q = np.diag(np.var(residuals, axis=0)) * 0.1
        self.R = np.diag(np.var(data, axis=0)) * 0.01

        # Compute residuals on training data
        train_residuals = self._compute_residuals(data)
        self.threshold = np.percentile(train_residuals, self.threshold_percentile)

    def _compute_residuals(self, data):
        n = len(data)
        n_states = data.shape[1]

        # Initialize
        x = data[0].copy()
        P = np.eye(n_states) * 1.0

        residuals = []
        for i in range(1, n):
            # Predict
            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.Q

            # Innovation (measurement residual)
            y = data[i] - x_pred
            S = P_pred + self.R

            # Normalized innovation squared (NIS)
            try:
                S_inv = np.linalg.inv(S)
                nis = y @ S_inv @ y
            except:
                nis = np.sum(y**2)

            residuals.append(nis)

            # Update
            K = P_pred @ np.linalg.inv(S)
            x = x_pred + K @ y
            P = (np.eye(n_states) - K) @ P_pred

        return np.array(residuals)

    def predict(self, data):
        residuals = self._compute_residuals(data)
        # Pad to match length
        residuals = np.concatenate([[0], residuals])
        return (residuals > self.threshold).astype(int)


# ============================================================================
# DEEP LEARNING AUTOENCODER
# ============================================================================


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for sequence anomaly detection."""

    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        # Encode
        _, (h, _) = self.encoder(x)
        z = self.enc_fc(h[-1])

        # Decode
        h_dec = self.dec_fc(z).unsqueeze(0).repeat(2, 1, 1)
        c_dec = torch.zeros_like(h_dec)
        out, _ = self.decoder(x, (h_dec, c_dec))
        return out


class AutoencoderDetector:
    """Autoencoder-based anomaly detector."""

    def __init__(self, seq_len=20, hidden_dim=64, threshold_percentile=95):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.scaler = None
        self.threshold = None

    def fit(self, data, epochs=50, batch_size=64):
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data)

        # Create sequences
        sequences = []
        for i in range(len(data_scaled) - self.seq_len):
            sequences.append(data_scaled[i : i + self.seq_len])
        sequences = np.array(sequences)

        # Model
        input_dim = data.shape[1]
        self.model = LSTMAutoencoder(input_dim, self.hidden_dim)

        # Training
        dataset = TensorDataset(torch.FloatTensor(sequences))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                recon = self.model(x)
                loss = criterion(recon, x)
                loss.backward()
                optimizer.step()

        # Compute threshold
        self.model.eval()
        with torch.no_grad():
            recon = self.model(torch.FloatTensor(sequences))
            errors = torch.mean((recon - torch.FloatTensor(sequences)) ** 2, dim=(1, 2)).numpy()
        self.threshold = np.percentile(errors, self.threshold_percentile)

    def predict(self, data):
        data_scaled = self.scaler.transform(data)

        sequences = []
        for i in range(len(data_scaled) - self.seq_len):
            sequences.append(data_scaled[i : i + self.seq_len])
        sequences = np.array(sequences)

        self.model.eval()
        with torch.no_grad():
            recon = self.model(torch.FloatTensor(sequences))
            errors = torch.mean((recon - torch.FloatTensor(sequences)) ** 2, dim=(1, 2)).numpy()

        # Pad to match length
        preds = (errors > self.threshold).astype(int)
        preds = np.concatenate([np.zeros(self.seq_len), preds])
        return preds


# ============================================================================
# TEST ATTACK GENERATION
# ============================================================================


def generate_test_attacks(base_data, magnitudes=[0.25, 0.5, 1.0, 2.0, 4.0]):
    attacks = {}
    for mag in magnitudes:
        # GPS drift
        drift = np.linspace(0, 5 * mag, len(base_data)).reshape(-1, 1)
        attacks[f"gps_drift_{mag}x"] = base_data + drift * np.array(
            [1, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )

        # IMU bias
        bias = 0.05 * mag
        attacks[f"imu_bias_{mag}x"] = base_data + np.array(
            [0, 0, 0, bias, bias, 0, bias / 2, bias / 2, 0, 0, 0, 0]
        )

        # Noise
        np.random.seed(int(mag * 100))
        attacks[f"noise_{mag}x"] = base_data + np.random.normal(0, 0.1 * mag, base_data.shape)

        # Jump
        jump_data = base_data.copy()
        jump_data[len(base_data) // 2 :, :3] += 2.0 * mag
        attacks[f"jump_{mag}x"] = jump_data

        # Oscillation (new)
        t = np.arange(len(base_data)).reshape(-1, 1)
        osc = np.sin(2 * np.pi * mag * t / 100) * mag
        attacks[f"oscillation_{mag}x"] = base_data + osc * np.array(
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )

        # Ramp (new)
        ramp = (t / len(base_data)) ** 2 * 5 * mag
        attacks[f"ramp_{mag}x"] = base_data + ramp * np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    return attacks


# ============================================================================
# MAIN EVALUATION
# ============================================================================


def evaluate_detector(name, detector, train_data, test_attacks, clean_test, feature_func=None):
    """Evaluate a detector on all attacks."""

    # Extract features if needed
    if feature_func:
        train_features = feature_func(train_data)
        train_features = np.nan_to_num(train_features, nan=0, posinf=1e6, neginf=-1e6)
    else:
        train_features = train_data

    # Fit detector
    try:
        if hasattr(detector, "fit"):
            detector.fit(train_features)
    except Exception as e:
        print(f"  {name}: FAILED TO FIT - {e}")
        return None

    results = {}

    # Test on attacks
    for attack_name, attack_data in test_attacks.items():
        try:
            if feature_func:
                test_features = feature_func(attack_data)
                test_features = np.nan_to_num(test_features, nan=0, posinf=1e6, neginf=-1e6)
            else:
                test_features = attack_data

            preds = detector.predict(test_features)

            # Handle different prediction formats
            if hasattr(preds, "shape") and len(preds.shape) == 1:
                if np.min(preds) < 0:  # LOF/IsoForest style (-1 = anomaly)
                    recall = np.mean(preds == -1)
                else:
                    recall = np.mean(preds)
            else:
                recall = np.mean(preds)

            results[attack_name] = recall
        except Exception as e:
            results[attack_name] = 0.0

    # Test on clean data (FPR)
    try:
        if feature_func:
            clean_features = feature_func(clean_test)
            clean_features = np.nan_to_num(clean_features, nan=0, posinf=1e6, neginf=-1e6)
        else:
            clean_features = clean_test

        clean_preds = detector.predict(clean_features)
        if np.min(clean_preds) < 0:
            fpr = np.mean(clean_preds == -1)
        else:
            fpr = np.mean(clean_preds)
    except:
        fpr = 1.0

    avg_recall = np.mean(list(results.values()))
    return {"recall": avg_recall, "fpr": fpr, "per_attack": results}


# ============================================================================
# RUN ALL EXPERIMENTS
# ============================================================================

print("\nPreparing data...")
train_data = clean_data[:10000]
clean_test = clean_data[12000:13000]
test_attacks = generate_test_attacks(test_base)

all_results = {}

# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CATEGORY 1: ANOMALY DETECTION ALGORITHMS")
print("=" * 70)

detectors_cat1 = [
    (
        "IsolationForest (n=100)",
        IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1),
    ),
    (
        "IsolationForest (n=200)",
        IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1),
    ),
    (
        "IsolationForest (n=500)",
        IsolationForest(n_estimators=500, contamination=0.03, random_state=42, n_jobs=-1),
    ),
    (
        "LocalOutlierFactor",
        LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1),
    ),
    ("OneClassSVM (rbf)", OneClassSVM(kernel="rbf", nu=0.05, gamma="auto")),
    ("OneClassSVM (poly)", OneClassSVM(kernel="poly", nu=0.05, degree=3)),
    ("EllipticEnvelope", EllipticEnvelope(contamination=0.05, random_state=42)),
]

for name, det in detectors_cat1:
    print(f"\nTesting: {name}")
    result = evaluate_detector(
        name, det, train_data, test_attacks, clean_test, extract_basic_features
    )
    if result:
        all_results[name] = result
        print(f'  Recall: {result["recall"]*100:.1f}%, FPR: {result["fpr"]*100:.1f}%')

# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CATEGORY 2: FEATURE ENGINEERING VARIATIONS")
print("=" * 70)

feature_methods = [
    ("Basic Features", extract_basic_features),
    ("Spectral Features", extract_spectral_features),
    ("Multi-scale Features", extract_multiscale_features),
    ("Derivative Features", extract_derivative_features),
    ("Correlation Features", extract_correlation_features),
]

best_detector = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)

for feat_name, feat_func in feature_methods:
    print(f"\nTesting: IsoForest + {feat_name}")
    det = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    result = evaluate_detector(
        f"IsoForest+{feat_name}", det, train_data, test_attacks, clean_test, feat_func
    )
    if result:
        all_results[f"IsoForest+{feat_name}"] = result
        print(f'  Recall: {result["recall"]*100:.1f}%, FPR: {result["fpr"]*100:.1f}%')

# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CATEGORY 3: STATISTICAL PROCESS CONTROL")
print("=" * 70)

spc_detectors = [
    ("CUSUM (t=5)", CUSUMDetector(threshold=5.0, drift=0.5)),
    ("CUSUM (t=3)", CUSUMDetector(threshold=3.0, drift=0.3)),
    ("CUSUM (t=10)", CUSUMDetector(threshold=10.0, drift=1.0)),
    ("EWMA (l=0.1)", EWMADetector(lambda_=0.1, L=3.0)),
    ("EWMA (l=0.2)", EWMADetector(lambda_=0.2, L=3.0)),
    ("EWMA (l=0.3)", EWMADetector(lambda_=0.3, L=2.5)),
    ("Hotelling T2 (a=0.01)", HotellingT2Detector(alpha=0.01)),
    ("Hotelling T2 (a=0.05)", HotellingT2Detector(alpha=0.05)),
]

for name, det in spc_detectors:
    print(f"\nTesting: {name}")
    result = evaluate_detector(
        name, det, train_data, test_attacks, clean_test, extract_basic_features
    )
    if result:
        all_results[name] = result
        print(f'  Recall: {result["recall"]*100:.1f}%, FPR: {result["fpr"]*100:.1f}%')

# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CATEGORY 4: KALMAN FILTER RESIDUALS")
print("=" * 70)

kalman_detectors = [
    ("Kalman (95%)", KalmanResidualDetector(threshold_percentile=95)),
    ("Kalman (99%)", KalmanResidualDetector(threshold_percentile=99)),
    ("Kalman (90%)", KalmanResidualDetector(threshold_percentile=90)),
]

for name, det in kalman_detectors:
    print(f"\nTesting: {name}")
    result = evaluate_detector(name, det, train_data, test_attacks, clean_test)
    if result:
        all_results[name] = result
        print(f'  Recall: {result["recall"]*100:.1f}%, FPR: {result["fpr"]*100:.1f}%')

# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CATEGORY 5: DEEP LEARNING (LSTM AUTOENCODER)")
print("=" * 70)

print("\nTraining LSTM Autoencoder (this may take a moment)...")
try:
    ae_detector = AutoencoderDetector(seq_len=20, hidden_dim=64, threshold_percentile=95)
    result = evaluate_detector(
        "LSTM Autoencoder", ae_detector, train_data, test_attacks, clean_test
    )
    if result:
        all_results["LSTM Autoencoder"] = result
        print(f'  Recall: {result["recall"]*100:.1f}%, FPR: {result["fpr"]*100:.1f}%')
except Exception as e:
    print(f"  LSTM Autoencoder: FAILED - {e}")

# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CATEGORY 6: HYBRID ENSEMBLE")
print("=" * 70)


class HybridEnsemble:
    """Ensemble of best detectors."""

    def __init__(self, detectors, voting="any"):
        self.detectors = detectors
        self.voting = voting  # 'any', 'majority', 'all'

    def fit(self, data):
        for name, det, feat_func in self.detectors:
            if feat_func:
                features = feat_func(data)
                features = np.nan_to_num(features, nan=0, posinf=1e6, neginf=-1e6)
            else:
                features = data
            det.fit(features)

    def predict(self, data):
        votes = []
        min_len = len(data)

        for name, det, feat_func in self.detectors:
            if feat_func:
                features = feat_func(data)
                features = np.nan_to_num(features, nan=0, posinf=1e6, neginf=-1e6)
            else:
                features = data

            preds = det.predict(features)
            if np.min(preds) < 0:
                preds = (preds == -1).astype(int)

            min_len = min(min_len, len(preds))
            votes.append(preds[:min_len])

        votes = np.array([v[:min_len] for v in votes])

        if self.voting == "any":
            return (np.sum(votes, axis=0) >= 1).astype(int)
        elif self.voting == "majority":
            return (np.sum(votes, axis=0) >= len(self.detectors) // 2 + 1).astype(int)
        else:  # 'all'
            return (np.sum(votes, axis=0) == len(self.detectors)).astype(int)


# Create ensembles
ensemble_configs = [
    (
        "Ensemble (IsoForest+CUSUM+Kalman) - Any",
        [
            (
                "IsoForest",
                IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1),
                extract_basic_features,
            ),
            ("CUSUM", CUSUMDetector(threshold=5.0), extract_basic_features),
            ("Kalman", KalmanResidualDetector(threshold_percentile=95), None),
        ],
        "any",
    ),
    (
        "Ensemble (IsoForest+CUSUM+Kalman) - Majority",
        [
            (
                "IsoForest",
                IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1),
                extract_basic_features,
            ),
            ("CUSUM", CUSUMDetector(threshold=5.0), extract_basic_features),
            ("Kalman", KalmanResidualDetector(threshold_percentile=95), None),
        ],
        "majority",
    ),
    (
        "Ensemble (IsoForest+LOF+EWMA) - Any",
        [
            (
                "IsoForest",
                IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1),
                extract_basic_features,
            ),
            (
                "LOF",
                LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1),
                extract_basic_features,
            ),
            ("EWMA", EWMADetector(lambda_=0.2), extract_basic_features),
        ],
        "any",
    ),
    (
        "Ensemble (Multi-feature IsoForest) - Any",
        [
            (
                "IsoForest-Basic",
                IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1),
                extract_basic_features,
            ),
            (
                "IsoForest-Spectral",
                IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1),
                extract_spectral_features,
            ),
            (
                "IsoForest-Deriv",
                IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1),
                extract_derivative_features,
            ),
        ],
        "any",
    ),
]

for name, detectors, voting in ensemble_configs:
    print(f"\nTesting: {name}")
    ensemble = HybridEnsemble(detectors, voting=voting)
    result = evaluate_detector(name, ensemble, train_data, test_attacks, clean_test)
    if result:
        all_results[name] = result
        print(f'  Recall: {result["recall"]*100:.1f}%, FPR: {result["fpr"]*100:.1f}%')

# ============================================================================
# FINAL RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("FINAL RESULTS - SORTED BY BALANCED SCORE (Recall - 0.5*FPR)")
print("=" * 70)

# Sort by balanced score
sorted_results = sorted(
    all_results.items(), key=lambda x: x[1]["recall"] - 0.5 * x[1]["fpr"], reverse=True
)

print(f'\n{"Method":<50} | {"Recall":>8} | {"FPR":>8} | {"Score":>8} | Status')
print("-" * 90)

for name, res in sorted_results:
    recall = res["recall"] * 100
    fpr = res["fpr"] * 100
    score = recall - 0.5 * fpr

    if recall >= 80 and fpr <= 10:
        status = "EXCELLENT"
    elif recall >= 70 and fpr <= 15:
        status = "GOOD"
    elif recall >= 60 and fpr <= 25:
        status = "MODERATE"
    else:
        status = "POOR"

    print(f"{name:<50} | {recall:>7.1f}% | {fpr:>7.1f}% | {score:>7.1f} | {status}")

# Best method
best_name, best_res = sorted_results[0]
print(f'\n{"="*70}')
print(f"BEST METHOD: {best_name}")
print(f'  Recall: {best_res["recall"]*100:.1f}%')
print(f'  FPR: {best_res["fpr"]*100:.1f}%')
print(f'{"="*70}')

# Per-attack breakdown for top 3
print("\n" + "=" * 70)
print("TOP 3 METHODS - PER ATTACK BREAKDOWN")
print("=" * 70)

for name, res in sorted_results[:3]:
    print(f"\n{name}:")
    for attack, recall in sorted(res["per_attack"].items()):
        print(f"  {attack:<25}: {recall*100:5.1f}%")
