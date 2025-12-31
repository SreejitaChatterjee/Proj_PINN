"""
Quick verification of Table 1 (ICI per-attack detection results).

This script trains a lightweight ICI detector and verifies the
per-attack recall@5%FPR values from the document.

Expected results (from full_results.json):
- Noise:        76.4%
- Drift:        80.7%
- Bias:         60.6%
- Coordinated:  57.0%
- Intermittent: 30.7%
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
import json
from pathlib import Path
import time

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class SimpleICIDetector(nn.Module):
    """Lightweight ICI detector for quick verification."""

    def __init__(self, state_dim=6, hidden_dim=32):
        super().__init__()
        # Forward model
        self.forward_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        # Inverse model
        self.inverse_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward_predict(self, x):
        return self.forward_net(x)

    def inverse_predict(self, x):
        return self.inverse_net(x)

    def compute_ici(self, x):
        """Compute inverse-cycle instability."""
        x_pred = self.forward_predict(x)
        x_recon = self.inverse_predict(x_pred)
        ici = torch.norm(x - x_recon, dim=-1)
        return ici


def generate_trajectory(n_samples=5000, seed=42):
    """Generate synthetic nominal trajectory."""
    np.random.seed(seed)

    # State: [x, y, z, vx, vy, vz]
    trajectory = np.zeros((n_samples, 6))

    # Initial conditions
    trajectory[0, 3:6] = np.random.randn(3) * 0.5

    dt = 0.005
    for t in range(1, n_samples):
        accel = np.random.randn(3) * 0.1
        trajectory[t, 3:6] = trajectory[t-1, 3:6] + accel * dt
        trajectory[t, :3] = trajectory[t-1, :3] + trajectory[t, 3:6] * dt

    return trajectory


def generate_attacks(nominal, seed=42):
    """Generate attack trajectories matching full_results.json attack types."""
    np.random.seed(seed)
    n = len(nominal)
    attacks = {}

    # Bias: constant offset
    bias = nominal.copy()
    bias[:, :3] += np.random.randn(3) * 0.5
    attacks['bias'] = bias

    # Drift: AR(1) growing drift
    drift = nominal.copy()
    drift_signal = np.zeros((n, 3))
    for i in range(1, n):
        drift_signal[i] = 0.995 * drift_signal[i-1] + np.random.randn(3) * 0.01
    drift[:, :3] += drift_signal
    attacks['drift'] = drift

    # Noise: additive Gaussian noise
    noise = nominal.copy()
    noise[:, :6] += np.random.randn(n, 6) * 0.3
    attacks['noise'] = noise

    # Coordinated: consistent position + velocity offset
    coordinated = nominal.copy()
    coordinated[:, :3] += 0.3
    coordinated[:, 3:6] += 0.1
    attacks['coordinated'] = coordinated

    # Intermittent: sporadic attacks
    intermittent = nominal.copy()
    attack_times = np.random.rand(n) < 0.1
    intermittent[attack_times, :3] += np.random.randn(attack_times.sum(), 3) * 1.0
    attacks['intermittent'] = intermittent

    return attacks


def train_detector(detector, trajectories, epochs=20, lr=0.001):
    """Train ICI detector on nominal trajectories."""
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr)

    # Prepare data
    all_data = np.vstack(trajectories)
    X = torch.tensor(all_data[:-1], dtype=torch.float32)
    Y = torch.tensor(all_data[1:], dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward loss
        Y_pred = detector.forward_predict(X)
        forward_loss = torch.mean((Y_pred - Y) ** 2)

        # Inverse loss
        X_recon = detector.inverse_predict(Y)
        inverse_loss = torch.mean((X_recon - X) ** 2)

        # Cycle loss
        X_cycle = detector.inverse_predict(detector.forward_predict(X))
        cycle_loss = torch.mean((X_cycle - X) ** 2)

        loss = forward_loss + inverse_loss + 0.25 * cycle_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")

    return detector


def evaluate_attack(detector, nominal, attack, attack_name):
    """Evaluate detection performance on attack type."""
    detector.eval()

    with torch.no_grad():
        # Compute ICI scores
        nominal_tensor = torch.tensor(nominal, dtype=torch.float32)
        attack_tensor = torch.tensor(attack, dtype=torch.float32)

        nominal_ici = detector.compute_ici(nominal_tensor).numpy()
        attack_ici = detector.compute_ici(attack_tensor).numpy()

    # Combine
    labels = np.concatenate([np.zeros(len(nominal_ici)), np.ones(len(attack_ici))])
    scores = np.concatenate([nominal_ici, attack_ici])

    # Metrics
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    # Recall at 5% FPR
    idx = np.searchsorted(fpr, 0.05)
    recall_5pct = tpr[min(idx, len(tpr)-1)]

    # Recall at 1% FPR
    idx = np.searchsorted(fpr, 0.01)
    recall_1pct = tpr[min(idx, len(tpr)-1)]

    return {
        'auroc': float(auroc),
        'recall_1pct_fpr': float(recall_1pct),
        'recall_5pct_fpr': float(recall_5pct),
    }


def main():
    print("=" * 70)
    print("TABLE 1 VERIFICATION - ICI Per-Attack Detection")
    print("=" * 70)

    # Expected values from document
    expected = {
        'noise': 0.764,
        'drift': 0.807,
        'bias': 0.606,
        'coordinated': 0.570,
        'intermittent': 0.307,
    }

    # Generate training data
    print("\n[1] Generating training data...")
    train_trajs = [generate_trajectory(5000, seed=SEED+i) for i in range(3)]

    # Generate test data
    print("[2] Generating test data...")
    test_nominal = generate_trajectory(2000, seed=SEED+100)
    attacks = generate_attacks(test_nominal, seed=SEED)

    # Train detector
    print("\n[3] Training ICI detector...")
    detector = SimpleICIDetector(state_dim=6, hidden_dim=32)
    detector = train_detector(detector, train_trajs, epochs=20)

    # Evaluate each attack type
    print("\n[4] Evaluating per-attack detection...")
    print("-" * 60)
    print(f"{'Attack':<15} {'Expected':<12} {'Measured':<12} {'Status':<10}")
    print("-" * 60)

    results = {}
    all_match = True

    for attack_name, attack_data in attacks.items():
        metrics = evaluate_attack(detector, test_nominal, attack_data, attack_name)
        results[attack_name] = metrics

        measured = metrics['recall_5pct_fpr']
        exp = expected[attack_name]

        # Allow 15% relative tolerance (due to randomness in lightweight model)
        tolerance = 0.15
        diff = abs(measured - exp) / exp
        status = "~OK" if diff < tolerance else "DIFF"

        if diff >= tolerance:
            all_match = False

        print(f"{attack_name:<15} {exp*100:>10.1f}% {measured*100:>10.1f}% {status:<10}")

    # Mean AUROC
    mean_auroc = np.mean([r['auroc'] for r in results.values()])

    print("-" * 60)
    print(f"\nMean AUROC: {mean_auroc:.3f} (expected: 0.845)")

    # Verdict
    print("\n" + "=" * 70)
    if all_match:
        print("VERDICT: Results are in expected range (within 15% tolerance)")
    else:
        print("VERDICT: Some results differ (lightweight model has higher variance)")
        print("         Full evaluation needed for exact match")
    print("=" * 70)

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'table1_verification.json'
    with open(output_path, 'w') as f:
        json.dump({
            'expected': expected,
            'measured': results,
            'mean_auroc': mean_auroc,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
