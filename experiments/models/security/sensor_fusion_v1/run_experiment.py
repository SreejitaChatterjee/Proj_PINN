"""
Run the Sensor Fusion Attack Detector experiment.

This script:
1. Loads EuRoC data with proper sequence-based splitting
2. Trains the detector on normal data only
3. Evaluates on all 30 attack types
4. Compares with PINN baseline
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import json
from datetime import datetime

from detector import AttackDetector, DetectorConfig, train_detector
from scripts.security.generate_synthetic_attacks import SyntheticAttackGenerator


def load_euroc_data(data_path: Path) -> pd.DataFrame:
    """Load and normalize EuRoC data."""
    df = pd.read_csv(data_path / "all_sequences.csv")

    # Normalize columns
    for old, new in [("roll", "phi"), ("pitch", "theta"), ("yaw", "psi")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "thrust" not in df.columns:
        df["thrust"] = df["az"] + 9.81 if "az" in df.columns else 9.81
    for col in ["torque_x", "torque_y", "torque_z"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


def split_by_sequence(df: pd.DataFrame, seed: int = 42):
    """Split data by flight sequence (prevents trajectory leakage)."""
    sequences = df["sequence"].unique()
    np.random.seed(seed)
    np.random.shuffle(sequences)

    n = len(sequences)
    train_seqs = list(sequences[:int(n*0.6)])
    val_seqs = list(sequences[int(n*0.6):int(n*0.8)])
    test_seqs = list(sequences[int(n*0.8):])

    train_df = df[df["sequence"].isin(train_seqs)].reset_index(drop=True)
    val_df = df[df["sequence"].isin(val_seqs)].reset_index(drop=True)
    test_df = df[df["sequence"].isin(test_seqs)].reset_index(drop=True)

    print(f"  Train sequences: {train_seqs}")
    print(f"  Val sequences:   {val_seqs}")
    print(f"  Test sequences:  {test_seqs}")

    return train_df, val_df, test_df


def prepare_tensor(df: pd.DataFrame) -> torch.Tensor:
    """Convert DataFrame to tensor."""
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]
    data = df[state_cols + control_cols].values
    return torch.FloatTensor(data)


def evaluate_detector(
    model: AttackDetector,
    test_df: pd.DataFrame,
    device: str = 'cpu'
) -> dict:
    """Evaluate detector on all attack types."""

    # Generate attacks on test data
    generator = SyntheticAttackGenerator(test_df, seed=42, randomize=False)
    attacks = generator.generate_all_attacks(handle_nan=True)

    results = {}
    model.eval()

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n{'Attack Type':<30} {'Recall':>8} {'Precision':>10} {'F1':>8} {'FPR':>8}")
    print("-" * 70)

    for attack_name, attack_data in attacks.items():
        # Prepare data
        data_tensor = prepare_tensor(attack_data)
        labels = attack_data["label"].values

        # Create sequences
        seq_len = 100
        all_preds = []
        all_labels = []

        for i in range(0, len(data_tensor) - seq_len, seq_len // 2):
            seq = data_tensor[i:i+seq_len].unsqueeze(0).to(device)
            seq_labels = labels[i+1:i+seq_len]  # +1 because physics layer reduces by 1

            with torch.no_grad():
                out = model(seq)
                preds = out['predictions'].squeeze().cpu().numpy()

            # Align lengths
            min_len = min(len(preds), len(seq_labels))
            all_preds.extend(preds[:min_len].tolist())
            all_labels.extend(seq_labels[:min_len].tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        tn = np.sum((all_preds == 0) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        results[attack_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }

        if attack_name == "clean":
            print(f"{attack_name:<30} {'---':>8} {'---':>10} {'---':>8} {fpr*100:>7.1f}%")
        else:
            print(f"{attack_name:<30} {recall*100:>7.1f}% {precision*100:>9.1f}% {f1*100:>7.1f}% {fpr*100:>7.1f}%")

    # Aggregate
    attack_results = {k: v for k, v in results.items() if k != "clean" and results[k]["tp"] + results[k]["fn"] > 0}

    total_tp = sum(r["tp"] for r in attack_results.values())
    total_fp = sum(r["fp"] for r in attack_results.values())
    total_tn = sum(r["tn"] for r in attack_results.values())
    total_fn = sum(r["fn"] for r in attack_results.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print("-" * 70)
    print(f"{'OVERALL':<30} {overall_recall*100:>7.1f}% {overall_precision*100:>9.1f}% {overall_f1*100:>7.1f}%")

    results["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "clean_fpr": results["clean"]["fpr"]
    }

    return results


def main():
    print("=" * 70)
    print("SENSOR FUSION ATTACK DETECTOR")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\n[1/5] Loading EuRoC data...")
    data_path = project_root / "data" / "euroc"
    df = load_euroc_data(data_path)
    print(f"  Loaded {len(df):,} samples")

    # Split by sequence
    print("\n[2/5] Splitting by sequence...")
    train_df, val_df, test_df = split_by_sequence(df, seed=42)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Prepare tensors
    train_tensor = prepare_tensor(train_df)
    val_tensor = prepare_tensor(val_df)

    # Create model
    print("\n[3/5] Creating model...")
    config = DetectorConfig(
        dt=0.005,
        seq_len=100,
        hidden_dim=64,
        dropout=0.1
    )
    model = AttackDetector(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Train
    print("\n[4/5] Training on normal data...")
    history = train_detector(
        model=model,
        train_data=train_tensor,
        val_data=val_tensor,
        epochs=50,
        batch_size=32,
        lr=1e-3,
        device=device,
        verbose=True
    )

    # Calibrate thresholds
    print("\n  Calibrating thresholds...")
    val_seqs = []
    for i in range(0, len(val_tensor) - 100, 50):
        val_seqs.append(val_tensor[i:i+100])
    val_seqs = torch.stack(val_seqs).to(device)

    thresholds = model.calibrate(val_seqs, percentile=99.0)
    print(f"  Physics threshold: {thresholds['physics_threshold']:.4f}")
    print(f"  Learned threshold: {thresholds['learned_threshold']:.4f}")

    # Evaluate
    print("\n[5/5] Evaluating on attacks...")
    model = model.to(device)
    results = evaluate_detector(model, test_df, device=device)

    # Save results
    output_dir = project_root / "sensor_fusion_detector" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), output_dir / "model.pth")

    # Save results
    results_serializable = {}
    for k, v in results.items():
        results_serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()}

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nOverall Results:")
    print(f"  Recall:    {results['overall']['recall']*100:.1f}%")
    print(f"  Precision: {results['overall']['precision']*100:.1f}%")
    print(f"  F1 Score:  {results['overall']['f1']*100:.1f}%")
    print(f"  Clean FPR: {results['overall']['clean_fpr']*100:.1f}%")

    print(f"\nResults saved to: {output_dir}")

    # Compare with PINN baseline
    print("\n" + "-" * 70)
    print("COMPARISON WITH PINN BASELINE")
    print("-" * 70)
    print(f"  PINN Recall:           18.7%")
    print(f"  Sensor Fusion Recall:  {results['overall']['recall']*100:.1f}%")
    improvement = (results['overall']['recall'] - 0.187) / 0.187 * 100
    print(f"  Improvement:           {improvement:+.1f}%")


if __name__ == "__main__":
    main()
