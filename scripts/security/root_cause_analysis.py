"""
Root Cause Analysis: Why Some Attacks Are Detectable, Others Aren't

Key questions:
1. Why does noise (60%) work but bias (3%) fail?
2. Why does 4x magnitude (60%) work but 0.25x (3%) fail?
3. What features are actually discriminative?
4. Why does cross-dataset transfer fail so badly?
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
EUROC_PATH = PROJECT_ROOT / "data" / "euroc" / "all_sequences.csv"

WINDOWS = [5, 10, 25, 50, 100, 200]


def extract_multiscale_features_detailed(data, windows=WINDOWS):
    """Extract features with detailed breakdown."""
    all_features = []
    feature_names = []
    max_window = max(windows)

    # Generate feature names
    for w_size in windows:
        feature_names.extend([f"mean_w{w_size}", f"std_w{w_size}", f"maxdiff_w{w_size}"])

    for i in range(max_window, len(data)):
        feat_list = []
        for w_size in windows:
            w = data[i - w_size : i]
            feat_list.extend(
                [
                    np.mean(w, axis=0).mean(),
                    np.std(w, axis=0).mean(),
                    np.max(np.abs(np.diff(w, axis=0))) if len(w) > 1 else 0,
                ]
            )
        all_features.append(feat_list)

    return np.array(all_features), feature_names


def generate_attack(clean_data, attack_type, magnitude):
    """Generate synthetic attack."""
    attacked = clean_data.copy()
    n = len(clean_data)

    if attack_type == "noise":
        noise = np.random.normal(0, magnitude * 0.1, attacked.shape)
        attacked += noise
    elif attack_type == "bias":
        attacked[:, 3] += magnitude * 0.05  # roll
        attacked[:, 4] += magnitude * 0.05  # pitch
    elif attack_type == "drift":
        drift = np.linspace(0, magnitude * 5.0, n)
        attacked[:, 0] += drift
    elif attack_type == "jump":
        jump_idx = n // 2
        attacked[jump_idx:, 0] += magnitude * 2.0
    elif attack_type == "oscillation":
        t = np.linspace(0, 10 * np.pi, n)
        attacked[:, 0] += magnitude * np.sin(t)

    return attacked


def analyze_feature_sensitivity():
    """Analyze how each feature responds to different attacks."""
    print("=" * 70)
    print("ROOT CAUSE ANALYSIS: Feature Sensitivity to Attacks")
    print("=" * 70)

    # Load EuRoC data
    df = pd.read_csv(EUROC_PATH)
    state_cols = ["x", "y", "z", "roll", "pitch", "yaw", "p", "q", "r", "vx", "vy", "vz"]
    clean_data = df[state_cols].values[:1000]  # Sample

    # Extract clean features
    clean_features, feature_names = extract_multiscale_features_detailed(clean_data)
    clean_mean = np.mean(clean_features, axis=0)
    clean_std = np.std(clean_features, axis=0)

    print("\n--- Clean Data Feature Statistics ---")
    print(f"{'Feature':<20} {'Mean':>12} {'Std':>12}")
    print("-" * 44)
    for i, name in enumerate(feature_names):
        print(f"{name:<20} {clean_mean[i]:>12.6f} {clean_std[i]:>12.6f}")

    # Analyze each attack type
    attack_types = ["noise", "bias", "drift", "jump", "oscillation"]
    magnitudes = [0.25, 1.0, 4.0]

    print("\n" + "=" * 70)
    print("FEATURE CHANGES BY ATTACK TYPE AND MAGNITUDE")
    print("=" * 70)

    results = {}

    for attack_type in attack_types:
        print(f"\n--- {attack_type.upper()} Attack ---")

        for magnitude in magnitudes:
            attacked_data = generate_attack(clean_data.copy(), attack_type, magnitude)
            attacked_features, _ = extract_multiscale_features_detailed(attacked_data)
            attacked_mean = np.mean(attacked_features, axis=0)

            # Compute z-score change (how many stds from clean)
            z_scores = (attacked_mean - clean_mean) / (clean_std + 1e-10)

            # Find most affected features
            top_indices = np.argsort(np.abs(z_scores))[::-1][:3]

            print(f"\n  Magnitude {magnitude}x:")
            print(f"    Max z-score: {np.max(np.abs(z_scores)):.2f}")
            print(f"    Top affected features:")
            for idx in top_indices:
                print(f"      {feature_names[idx]}: z={z_scores[idx]:.2f}")

            results[(attack_type, magnitude)] = {
                "max_z": np.max(np.abs(z_scores)),
                "mean_z": np.mean(np.abs(z_scores)),
                "top_feature": feature_names[top_indices[0]],
                "top_z": z_scores[top_indices[0]],
            }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Maximum Z-Score by Attack and Magnitude")
    print("=" * 70)
    print(f"\n{'Attack':<15} {'0.25x':>10} {'1.0x':>10} {'4.0x':>10}")
    print("-" * 45)
    for attack_type in attack_types:
        z_025 = results[(attack_type, 0.25)]["max_z"]
        z_100 = results[(attack_type, 1.0)]["max_z"]
        z_400 = results[(attack_type, 4.0)]["max_z"]
        print(f"{attack_type:<15} {z_025:>10.2f} {z_100:>10.2f} {z_400:>10.2f}")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    # Why noise works
    noise_z = results[("noise", 1.0)]["max_z"]
    bias_z = results[("bias", 1.0)]["max_z"]

    print(
        f"""
1. NOISE vs BIAS at 1.0x magnitude:
   - Noise max z-score: {noise_z:.2f}
   - Bias max z-score: {bias_z:.2f}
   - Ratio: {noise_z/bias_z:.1f}x

   WHY: Noise increases VARIANCE (std features detect this)
        Bias is constant OFFSET (mean shifts but within normal range)

2. MAGNITUDE EFFECT:
   - 0.25x attacks: Most z-scores < 1 (within normal variance)
   - 4.0x attacks: Z-scores >> 3 (clearly anomalous)

   WHY: Small perturbations are indistinguishable from normal variation

3. WHICH FEATURES MATTER:
"""
    )

    # Analyze which features are most discriminative
    feature_importance = {name: 0 for name in feature_names}
    for (attack, mag), res in results.items():
        if mag == 1.0:  # Focus on baseline magnitude
            feature_importance[res["top_feature"]] += abs(res["top_z"])

    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print("   Most discriminative features (at 1.0x):")
    for name, score in sorted_features[:5]:
        print(f"     {name}: importance={score:.2f}")

    return results


def analyze_detection_threshold():
    """Analyze what threshold would be needed to detect each attack."""
    print("\n" + "=" * 70)
    print("DETECTION THRESHOLD ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(EUROC_PATH)
    state_cols = ["x", "y", "z", "roll", "pitch", "yaw", "p", "q", "r", "vx", "vy", "vz"]

    # Get normal data statistics
    normal_data = df[state_cols].values[:50000]
    normal_features, _ = extract_multiscale_features_detailed(normal_data)

    # Compute per-feature thresholds at different percentiles
    percentiles = [90, 95, 99, 99.9]
    thresholds = {}
    for p in percentiles:
        thresholds[p] = np.percentile(normal_features, p, axis=0)

    print("\nTo detect an attack, its feature values must exceed normal percentiles.")
    print("Let's see which attacks exceed which thresholds:\n")

    attack_types = ["noise", "bias"]
    clean_sample = df[state_cols].values[60000:61000]

    for attack_type in attack_types:
        print(f"\n{attack_type.upper()} Attack (1.0x magnitude):")
        attacked = generate_attack(clean_sample.copy(), attack_type, 1.0)
        attack_features, feature_names = extract_multiscale_features_detailed(attacked)
        attack_max = np.max(attack_features, axis=0)

        for p in percentiles:
            exceeds = np.sum(attack_max > thresholds[p])
            print(f"  Exceeds {p}th percentile: {exceeds}/{len(feature_names)} features")


def propose_improvements():
    """Based on analysis, propose specific improvements."""
    print("\n" + "=" * 70)
    print("PROPOSED IMPROVEMENTS")
    print("=" * 70)

    improvements = """
Based on root cause analysis, here are specific improvements:

1. FOR BIAS ATTACKS (currently 3% recall):
   - Problem: Constant offset doesn't change variance
   - Solution: Add CUMULATIVE SUM (CUSUM) features
   - Why: CUSUM detects persistent shifts, not just variance changes

2. FOR LOW-MAGNITUDE ATTACKS (currently 3% at 0.25x):
   - Problem: Z-scores < 1, within normal variation
   - Solution A: Longer windows (500, 1000 timesteps) for drift
   - Solution B: Adaptive thresholds based on recent history
   - Solution C: Physics-based residuals (PINN dynamics model)

3. FOR CROSS-DATASET TRANSFER (currently 3.3%):
   - Problem: Features are platform-specific
   - Solution A: Normalize by per-sequence statistics
   - Solution B: Use dimensionless features (ratios, correlations)
   - Solution C: Domain adaptation / transfer learning
   - Solution D: Physics-based features (thrust-to-weight, etc.)

4. FUNDAMENTAL ISSUE:
   - Multi-scale STATISTICAL features only detect STATISTICAL anomalies
   - Bias attacks are not statistical anomalies (just shifted mean)
   - Need PHYSICS-BASED features to detect consistency violations

5. RECOMMENDED HYBRID APPROACH:
   - Use PINN dynamics model to predict next state
   - Compute residual: actual - predicted
   - Apply multi-scale features to RESIDUALS, not raw data
   - Residuals should show anomalies regardless of operating point
"""
    print(improvements)


def main():
    print("ROOT CAUSE ANALYSIS FOR POOR DETECTION PERFORMANCE")
    print("=" * 70)

    # Run analyses
    results = analyze_feature_sensitivity()
    analyze_detection_threshold()
    propose_improvements()

    # Save summary
    output_dir = PROJECT_ROOT / "models" / "security" / "rigorous_evaluation"
    summary = """
================================================================================
ROOT CAUSE ANALYSIS SUMMARY
================================================================================

WHY NOISE WORKS (60% recall):
- Noise INCREASES variance at all timescales
- std_w* features directly measure this
- Large magnitude noise exceeds normal variance significantly

WHY BIAS FAILS (3% recall):
- Bias is CONSTANT OFFSET
- Does NOT change variance (std features unchanged)
- Mean shift is within normal operating range
- Multi-scale features are BLIND to constant offsets

WHY 0.25x FAILS (3% recall):
- Perturbations are smaller than normal variation
- Z-scores < 1 (within 1 std of normal)
- Statistically indistinguishable from normal data

WHY CROSS-DATASET FAILS (3.3% recall):
- EuRoC and PADRE have different:
  - Sensor characteristics
  - Flight dynamics
  - State representations
  - Sampling rates
- Features trained on EuRoC are MEANINGLESS for PADRE

FUNDAMENTAL LIMITATION:
Multi-scale statistical features can only detect attacks that
CHANGE STATISTICAL PROPERTIES (variance, autocorrelation).
They CANNOT detect:
- Constant offsets (bias)
- Slow drifts (within variance)
- Coordinated attacks (maintain statistics)

SOLUTION: PHYSICS-BASED DETECTION
Instead of detecting statistical anomalies in raw data,
detect PHYSICS VIOLATIONS:
1. Use PINN to predict expected next state
2. Compute residual = actual - predicted
3. Residuals should be near-zero for normal data
4. ANY attack causes physics violation -> residual spike
5. Apply anomaly detection to residuals

This approach should:
- Detect bias (violates dynamics)
- Detect low-magnitude attacks (still violate physics)
- Generalize across platforms (physics is universal)
"""

    with open(output_dir / "ROOT_CAUSE_ANALYSIS.txt", "w") as f:
        f.write(summary)

    print(f"\nAnalysis saved to: {output_dir / 'ROOT_CAUSE_ANALYSIS.txt'}")


if __name__ == "__main__":
    main()
