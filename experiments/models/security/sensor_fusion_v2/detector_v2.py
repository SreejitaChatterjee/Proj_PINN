"""
Sensor Fusion Attack Detector v2

Key insight from analysis:
- Pure physics: 64.6% recall
- Learned model: 29.8% recall (WORSE!)

Problem: The learned components were hurting, not helping.

Solution: Physics-first architecture with learning ONLY for hard cases.

Architecture v2:
================
1. Pure physics consistency (handles 64.6% of attacks)
2. Learned residual network (ONLY for attacks that fail physics check)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class DetectorConfigV2:
    dt: float = 0.005
    # Physics thresholds (from 99th percentile analysis)
    pos_vel_threshold: float = 0.02
    att_rate_threshold: float = 2.2
    kinematic_threshold: float = 0.00056


class PhysicsDetector:
    """
    Pure physics consistency detector.

    No learning - just physics checks.
    Achieves 64.6% recall on its own.
    """

    def __init__(self, config: DetectorConfigV2):
        self.config = config
        self.dt = config.dt

    def detect(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect anomalies using physics consistency.

        Args:
            data: [N, 16] - state(12) + control(4)

        Returns:
            dict with scores and predictions
        """
        pos = data[:, 0:3]
        att = data[:, 3:6]
        rate = data[:, 6:9]
        vel = data[:, 9:12]

        N = len(data)

        # 1. Position-Velocity Consistency
        pos_deriv = (pos[1:] - pos[:-1]) / self.dt
        pos_vel_score = np.linalg.norm(pos_deriv - vel[1:], axis=1)
        pos_vel_pred = (pos_vel_score > self.config.pos_vel_threshold).astype(int)

        # 2. Attitude-Rate Consistency
        att_deriv = (att[1:] - att[:-1]) / self.dt
        att_rate_score = np.linalg.norm(att_deriv - rate[1:], axis=1)
        att_rate_pred = (att_rate_score > self.config.att_rate_threshold).astype(int)

        # 3. Kinematic Consistency
        window = 20
        kinematic_score = np.zeros(N - 1)
        for i in range(window, N - 1):
            vel_integral = vel[i-window+1:i+1].sum(axis=0) * self.dt
            pos_change = pos[i+1] - pos[i-window+1]
            kinematic_score[i] = np.linalg.norm(vel_integral - pos_change)
        kinematic_pred = (kinematic_score > self.config.kinematic_threshold).astype(int)

        # Combined prediction (OR logic)
        combined_pred = np.maximum.reduce([pos_vel_pred, att_rate_pred, kinematic_pred])

        return {
            'pos_vel_score': pos_vel_score,
            'att_rate_score': att_rate_score,
            'kinematic_score': kinematic_score,
            'predictions': combined_pred,
            'pos_vel_pred': pos_vel_pred,
            'att_rate_pred': att_rate_pred,
            'kinematic_pred': kinematic_pred
        }


class ResidualLearner(nn.Module):
    """
    Learns to detect attacks that FAIL physics checks.

    Only activates when physics scores are low but attack might still be present.
    Focuses on: actuator attacks, stealthy attacks, time delays.
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32):
        super().__init__()

        # Temporal feature extractor
        self.temporal = nn.Sequential(
            nn.Linear(input_dim * 10, hidden_dim),  # 10-step window
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Anomaly scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.threshold = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 10, 16] - 10-step windows

        Returns:
            scores: [batch, 1]
        """
        x_flat = x.view(x.shape[0], -1)
        h = self.temporal(x_flat)
        return self.scorer(h)


class HybridDetector:
    """
    Hybrid physics + learning detector.

    Strategy:
    1. Run physics checks first (handles 64.6%)
    2. For samples that PASS physics, run learned model
    3. Combine with OR logic
    """

    def __init__(self, config: Optional[DetectorConfigV2] = None):
        self.config = config or DetectorConfigV2()
        self.physics = PhysicsDetector(self.config)
        self.learner = ResidualLearner()
        self.learner_threshold = 0.5

    def detect(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Full detection pipeline."""

        # Physics detection
        physics_out = self.physics.detect(data)
        physics_pred = physics_out['predictions']

        # Learned detection (for samples that pass physics)
        # Create windows for temporal analysis
        window_size = 10
        N = len(data) - 1  # -1 because physics reduces by 1

        learned_pred = np.zeros(N)

        if N > window_size:
            # Prepare windows
            windows = []
            for i in range(window_size, N):
                windows.append(data[i-window_size+1:i+1])
            windows = np.array(windows)

            # Run learned model
            with torch.no_grad():
                windows_t = torch.FloatTensor(windows)
                scores = self.learner(windows_t).numpy().flatten()

            # Only apply learned predictions where physics passed
            for i in range(len(scores)):
                idx = i + window_size
                if physics_pred[idx] == 0:  # Physics passed
                    learned_pred[idx] = (scores[i] > self.learner_threshold)

        # Combine
        combined = np.maximum(physics_pred, learned_pred)

        return {
            'predictions': combined,
            'physics_pred': physics_pred,
            'learned_pred': learned_pred,
            'pos_vel_score': physics_out['pos_vel_score'],
            'att_rate_score': physics_out['att_rate_score'],
            'kinematic_score': physics_out['kinematic_score']
        }

    def train_residual(self, normal_data: np.ndarray, epochs: int = 20):
        """Train the residual learner on normal data."""
        window_size = 10
        N = len(normal_data)

        # Create windows
        windows = []
        for i in range(window_size, N):
            windows.append(normal_data[i-window_size+1:i+1])
        windows = np.array(windows)

        # Train to output LOW scores for normal data
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(windows))
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(self.learner.parameters(), lr=1e-3)

        for epoch in range(epochs):
            total_loss = 0
            for (batch,) in loader:
                optimizer.zero_grad()
                scores = self.learner(batch)
                # Loss: scores should be LOW for normal data
                loss = scores.mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss = {total_loss/len(loader):.4f}")

        # Calibrate threshold
        with torch.no_grad():
            windows_t = torch.FloatTensor(windows)
            scores = self.learner(windows_t).numpy().flatten()
            self.learner_threshold = np.percentile(scores, 99)
            print(f"  Learned threshold: {self.learner_threshold:.4f}")


def evaluate_detector(detector: HybridDetector, test_df, attacks: dict) -> dict:
    """Evaluate on all attacks."""
    results = {}

    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    print(f"\n{'Attack Type':<30} {'Recall':>8} {'Precision':>10} {'F1':>8}")
    print("-" * 60)

    for attack_name, attack_data in attacks.items():
        data = attack_data[state_cols + control_cols].values
        labels = attack_data["label"].values[1:]  # Align with predictions

        out = detector.detect(data)
        preds = out['predictions']

        # Align lengths
        min_len = min(len(preds), len(labels))
        preds = preds[:min_len]
        labels = labels[:min_len]

        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[attack_name] = {'recall': recall, 'precision': precision, 'f1': f1}

        if attack_name != "clean":
            print(f"{attack_name:<30} {recall*100:>7.1f}% {precision*100:>9.1f}% {f1*100:>7.1f}%")

    # Overall
    attack_results = [v for k, v in results.items() if k != "clean"]
    avg_recall = np.mean([r['recall'] for r in attack_results])
    print("-" * 60)
    print(f"{'AVERAGE':<30} {avg_recall*100:>7.1f}%")

    return results
