"""
PINN-Residual Based Anomaly Detection

Key Insight: Instead of detecting statistical anomalies in RAW DATA,
detect PHYSICS VIOLATIONS via prediction residuals.

Approach:
1. PINN predicts next_state from current_state
2. Residual = actual_next_state - predicted_next_state
3. Normal flight: residual ≈ 0 (physics satisfied)
4. Attack: residual spikes (physics violated)
5. Apply anomaly detection to RESIDUAL features

Why this should work for BIAS attacks:
- Bias shifts sensor readings but doesn't follow physics
- PINN expects angular rates to be consistent with attitude changes
- Biased readings violate this consistency -> residual spike

Why this should transfer across platforms:
- Physics is universal (Newton-Euler equations)
- Different platforms have same physics, just different parameters
- PINN learns physics, not platform-specific patterns
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EUROC_PATH = PROJECT_ROOT / "data" / "euroc" / "all_sequences.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "euroc_pinn.pth"
OUTPUT_DIR = PROJECT_ROOT / "models" / "security" / "pinn_residual"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [5, 10, 25, 50, 100]  # Shorter windows for residuals
CONTAMINATION = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EuRoCPINN(nn.Module):
    """
    PINN architecture matching the saved euroc_pinn.pth model.

    Architecture from saved model:
    - Input: 15 (12 states + 3 angular velocities)
    - Hidden: 256 with LayerNorm
    - Output: 12 (next state)
    """

    def __init__(self, input_dim=15, hidden_size=256, output_dim=12):
        super().__init__()

        # Match exact architecture from saved model
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),  # net.0
            nn.LayerNorm(hidden_size),  # net.1
            nn.Tanh(),  # net.2
            nn.Dropout(0.1),  # net.3
            nn.Linear(hidden_size, hidden_size),  # net.4
            nn.LayerNorm(hidden_size),  # net.5
            nn.Tanh(),  # net.6
            nn.Dropout(0.1),  # net.7
            nn.Linear(hidden_size, hidden_size),  # net.8
            nn.LayerNorm(hidden_size),  # net.9
            nn.Tanh(),  # net.10
            nn.Dropout(0.1),  # net.11
            nn.Linear(hidden_size, hidden_size),  # net.12
            nn.LayerNorm(hidden_size),  # net.13
            nn.Tanh(),  # net.14
            nn.Dropout(0.1),  # net.15
            nn.Linear(hidden_size, hidden_size),  # net.16
            nn.LayerNorm(hidden_size),  # net.17
            nn.Tanh(),  # net.18
            nn.Dropout(0.1),  # net.19
            nn.Linear(hidden_size, output_dim),  # net.20
        )

    def forward(self, x):
        return self.net(x)


def load_pinn_model():
    """Load trained PINN model."""
    print(f"Loading PINN model from {MODEL_PATH}...")

    # Use architecture matching saved model
    model = EuRoCPINN(input_dim=15, hidden_size=256, output_dim=12)

    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("  Model loaded successfully.")
    else:
        print(f"  WARNING: Model not found at {MODEL_PATH}")
        print("  Using untrained model (will have high residuals)")

    model.to(DEVICE)
    model.eval()
    return model


def compute_residuals(model, data, batch_size=1000):
    """
    Compute prediction residuals: actual - predicted.

    Args:
        model: Trained PINN model
        data: [N, 15] array with [states(12) + accelerations(3)]
              Format: x, y, z, roll, pitch, yaw, p, q, r, vx, vy, vz, ax, ay, az

    Returns:
        residuals: [N-1, 12] array of prediction residuals
    """
    # Ensure we have enough data
    if len(data) < 2:
        return np.array([])

    residuals = []

    with torch.no_grad():
        # Batch processing for efficiency
        for start in range(0, len(data) - 1, batch_size):
            end = min(start + batch_size, len(data) - 1)

            # Current inputs: [batch, 15]
            x_batch = torch.tensor(data[start:end], dtype=torch.float32, device=DEVICE)

            # Actual next states: [batch, 12]
            actual_next = data[start + 1 : end + 1, :12]

            # PINN prediction: [batch, 12]
            predicted_next = model(x_batch).cpu().numpy()

            # Residuals
            batch_residuals = actual_next - predicted_next
            residuals.append(batch_residuals)

    return np.concatenate(residuals, axis=0)


def extract_residual_features(residuals, windows=WINDOWS):
    """
    Extract multi-scale features from residuals.

    For residuals, we focus on:
    - Magnitude (should be near zero for normal)
    - Variance (should be stable)
    - Maximum absolute value (spikes indicate attacks)
    """
    if len(residuals) < max(windows):
        return np.array([])

    all_features = []
    max_window = max(windows)

    for i in range(max_window, len(residuals)):
        feat_list = []
        for w_size in windows:
            w = residuals[i - w_size : i]

            # Mean absolute residual (should be ~0 for normal)
            feat_list.append(np.mean(np.abs(w)))

            # Std of residuals (stability)
            feat_list.append(np.std(w))

            # Max absolute residual (spike detection)
            feat_list.append(np.max(np.abs(w)))

            # RMS residual
            feat_list.append(np.sqrt(np.mean(w**2)))

        all_features.append(feat_list)

    return np.array(all_features)


def generate_attack(clean_data, attack_type, magnitude):
    """Generate synthetic attack."""
    attacked = clean_data.copy()
    n = len(clean_data)

    if attack_type == "noise":
        noise = np.random.normal(0, magnitude * 0.1, attacked.shape)
        attacked += noise

    elif attack_type == "bias":
        # Bias on attitude sensors - this should cause physics violation!
        attacked[:, 3] += magnitude * 0.05  # roll bias
        attacked[:, 4] += magnitude * 0.05  # pitch bias
        # Angular rates don't change -> inconsistent with attitude change

    elif attack_type == "drift":
        drift = np.linspace(0, magnitude * 5.0, n)
        attacked[:, 0] += drift  # position drift
        # Velocity doesn't match position change -> physics violation

    elif attack_type == "jump":
        jump_idx = n // 2
        attacked[jump_idx:, 0] += magnitude * 2.0
        # Instant position change with no velocity -> physics violation

    elif attack_type == "oscillation":
        t = np.linspace(0, 10 * np.pi, n)
        attacked[:, 0] += magnitude * np.sin(t)

    return attacked


def run_pinn_residual_evaluation():
    """Main evaluation using PINN residual approach."""
    print("=" * 70)
    print("PINN-RESIDUAL ANOMALY DETECTION")
    print("=" * 70)

    # Load PINN model
    model = load_pinn_model()

    # Load EuRoC data
    print("\nLoading EuRoC data...")
    df = pd.read_csv(EUROC_PATH)
    # 15 input features: 12 states + 3 accelerations (matches model input)
    input_cols = [
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "p",
        "q",
        "r",
        "vx",
        "vy",
        "vz",
        "ax",
        "ay",
        "az",
    ]
    state_cols = ["x", "y", "z", "roll", "pitch", "yaw", "p", "q", "r", "vx", "vy", "vz"]
    sequences = df["sequence"].unique()

    print(f"Found {len(sequences)} sequences")

    # Leave-One-Sequence-Out CV
    all_results = []

    for test_seq in sequences:
        print(f"\n--- Testing on {test_seq} ---")

        # Train/test split
        train_df = df[df["sequence"] != test_seq]
        test_df = df[df["sequence"] == test_seq]

        # Use 15-column input format for PINN
        train_data = train_df[input_cols].values
        test_data = test_df[input_cols].values

        # Step 1: Compute residuals on training data (normal)
        print("  Computing training residuals...")
        train_residuals = compute_residuals(model, train_data[:50000])  # Limit for speed
        print(f"    Got {len(train_residuals)} residuals")

        if len(train_residuals) < max(WINDOWS) + 100:
            print("    Insufficient residuals, skipping...")
            continue

        # Step 2: Extract features from residuals
        print("  Extracting residual features...")
        train_features = extract_residual_features(train_residuals)
        print(f"    Got {len(train_features)} feature vectors")

        # Step 3: Train anomaly detector on residual features
        print("  Training IsolationForest on residual features...")
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)

        detector = IsolationForest(
            n_estimators=200, contamination=CONTAMINATION, random_state=42, n_jobs=-1
        )
        detector.fit(train_scaled)

        # Step 4: Evaluate on clean test data (FPR)
        print("  Evaluating on clean test data...")
        test_residuals = compute_residuals(model, test_data[:5000])
        test_features = extract_residual_features(test_residuals)

        if len(test_features) == 0:
            continue

        test_scaled = scaler.transform(test_features)
        clean_preds = detector.predict(test_scaled)

        fp = np.sum(clean_preds == -1)
        tn = np.sum(clean_preds == 1)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"    FPR on clean: {fpr*100:.1f}%")

        # Step 5: Evaluate on attacks
        attack_types = ["noise", "bias", "drift", "jump", "oscillation"]
        magnitudes = [0.25, 0.5, 1.0, 2.0, 4.0]

        seq_results = {"sequence": test_seq, "fpr": fpr, "attacks": {}}

        for attack_type in attack_types:
            attack_recalls = []

            for magnitude in magnitudes:
                # Generate attack on 15-column data
                test_base = test_data[:1000]
                attacked = generate_attack(test_base.copy(), attack_type, magnitude)

                # Compute residuals on attacked data
                attack_residuals = compute_residuals(model, attacked)

                if len(attack_residuals) < max(WINDOWS):
                    continue

                attack_features = extract_residual_features(attack_residuals)
                attack_scaled = scaler.transform(attack_features)

                # Predict
                attack_preds = detector.predict(attack_scaled)

                # Recall = fraction detected as anomaly
                recall = np.sum(attack_preds == -1) / len(attack_preds)
                attack_recalls.append(recall)

            avg_recall = np.mean(attack_recalls) if attack_recalls else 0
            seq_results["attacks"][attack_type] = avg_recall
            print(f"    {attack_type}: {avg_recall*100:.1f}% recall")

        all_results.append(seq_results)

    # Summary
    print("\n" + "=" * 70)
    print("PINN-RESIDUAL DETECTION SUMMARY")
    print("=" * 70)

    if not all_results:
        print("No results collected!")
        return

    # Aggregate across sequences
    avg_fpr = np.mean([r["fpr"] for r in all_results])
    print(f"\nAverage FPR: {avg_fpr*100:.1f}%")

    print("\nPer-Attack Recall (averaged across sequences):")
    attack_types = ["noise", "bias", "drift", "jump", "oscillation"]
    for attack_type in attack_types:
        recalls = [r["attacks"].get(attack_type, 0) for r in all_results]
        avg = np.mean(recalls)
        print(f"  {attack_type:15s}: {avg*100:.1f}%")

    # Overall average
    all_recalls = []
    for r in all_results:
        for attack, recall in r["attacks"].items():
            all_recalls.append(recall)

    overall_avg = np.mean(all_recalls) if all_recalls else 0
    print(f"\nOverall Average Recall: {overall_avg*100:.1f}%")

    # Compare with raw feature approach
    print("\n" + "-" * 70)
    print("COMPARISON: Raw Features vs PINN Residuals")
    print("-" * 70)
    print("Raw multi-scale features (from rigorous_evaluation.py):")
    print("  Overall: 22.7% recall, 6.5% FPR")
    print("  Bias attacks: 3.1% recall")
    print("")
    print("PINN-residual features:")
    print(f"  Overall: {overall_avg*100:.1f}% recall, {avg_fpr*100:.1f}% FPR")

    bias_recall = np.mean([r["attacks"].get("bias", 0) for r in all_results])
    print(f"  Bias attacks: {bias_recall*100:.1f}% recall")

    if bias_recall > 0.031:
        print("\n  *** IMPROVEMENT on bias attacks! ***")
    else:
        print("\n  No improvement on bias attacks.")

    # Save results
    report = f"""
================================================================================
PINN-RESIDUAL ANOMALY DETECTION RESULTS
================================================================================

METHODOLOGY:
1. Load trained PINN dynamics model (models/euroc_pinn.pth)
2. Compute residuals: actual_next - pinn_predicted_next
3. Extract multi-scale features from RESIDUALS (not raw data)
4. Train IsolationForest on residual features
5. Detect anomalies as physics violations

HYPOTHESIS:
- Attacks cause PHYSICS VIOLATIONS, not just statistical anomalies
- PINN predicts expected physics -> attacks cause residual spikes
- Bias attacks should be detectable (violate dynamics consistency)

RESULTS (LOSO-CV):
- Average FPR: {avg_fpr*100:.1f}%
- Overall Recall: {overall_avg*100:.1f}%

Per-Attack Type:
"""
    for attack_type in attack_types:
        recalls = [r["attacks"].get(attack_type, 0) for r in all_results]
        avg = np.mean(recalls)
        report += f"  {attack_type:15s}: {avg*100:.1f}%\n"

    report += f"""
COMPARISON WITH RAW FEATURES:
                    Raw Features    PINN Residuals
  Overall Recall:      22.7%          {overall_avg*100:.1f}%
  Bias Recall:          3.1%          {bias_recall*100:.1f}%
  FPR:                  6.5%          {avg_fpr*100:.1f}%

"""

    if overall_avg > 0.227:
        report += "CONCLUSION: PINN-residual approach IMPROVES detection.\n"
    else:
        report += "CONCLUSION: PINN-residual approach needs more work.\n"

    report_path = OUTPUT_DIR / "PINN_RESIDUAL_RESULTS.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nResults saved to: {report_path}")

    return all_results


if __name__ == "__main__":
    run_pinn_residual_evaluation()
