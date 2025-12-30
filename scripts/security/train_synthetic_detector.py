"""
Train PINN-based anomaly detector on synthetic attack dataset.

This trains on normal EuRoC data and evaluates on synthetic attacks.
The detector learns normal dynamics and flags deviations as anomalies.

Usage:
    python scripts/security/train_synthetic_detector.py
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

try:
    from pinn_dynamics import QuadrotorPINN
    from pinn_dynamics.training import Trainer
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pinn_dynamics import QuadrotorPINN
    from pinn_dynamics.training import Trainer


def load_euroc_data(data_path: Path) -> pd.DataFrame:
    """Load EuRoC processed data."""
    csv_files = list(data_path.glob("*.csv"))

    # Try common names
    for name in ["all_sequences.csv", "euroc_processed.csv", "all_sequences_full.csv"]:
        path = data_path / name
        if path.exists():
            df = pd.read_csv(path)
            print(f"Loaded {len(df):,} samples from {name}")
            return df

    if csv_files:
        df = pd.read_csv(csv_files[0])
        print(f"Loaded {len(df):,} samples from {csv_files[0].name}")
        return df

    raise FileNotFoundError(f"No CSV files found in {data_path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard format."""
    # Map roll/pitch/yaw to phi/theta/psi
    column_map = {
        "roll": "phi",
        "pitch": "theta",
        "yaw": "psi"
    }

    for old, new in column_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Add control columns if missing
    if "thrust" not in df.columns:
        if "az" in df.columns:
            df["thrust"] = df["az"] + 9.81
        else:
            df["thrust"] = 9.81

    for col in ["torque_x", "torque_y", "torque_z"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


def prepare_sequences(df: pd.DataFrame, seq_len: int = 1):
    """
    Prepare state transition data for PINN training.

    Returns:
        X: [N, 16] - current state (12) + control (4)
        y: [N, 12] - next state
    """
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    states = df[state_cols].values
    controls = df[control_cols].values

    # Current state + control -> next state
    X = np.concatenate([states[:-1], controls[:-1]], axis=1)
    y = states[1:]

    return X, y


def split_by_sequence(df: pd.DataFrame, test_ratio: float = 0.2, val_ratio: float = 0.2, seed: int = 42):
    """
    Split data by flight sequence to prevent temporal leakage.

    This ensures that:
    - Train, val, and test use DIFFERENT flight sequences
    - No information leaks from test sequences to training
    - Evaluation is on truly unseen trajectory patterns

    Returns:
        train_df, val_df, test_df: DataFrames for each split
        train_seqs, val_seqs, test_seqs: List of sequences in each split
    """
    if "sequence" not in df.columns:
        print("WARNING: No 'sequence' column found. Falling back to random split.")
        return None

    sequences = df["sequence"].unique()
    np.random.seed(seed)
    np.random.shuffle(sequences)

    n_seqs = len(sequences)
    n_test = max(1, int(n_seqs * test_ratio))
    n_val = max(1, int(n_seqs * val_ratio))
    n_train = n_seqs - n_test - n_val

    train_seqs = list(sequences[:n_train])
    val_seqs = list(sequences[n_train:n_train + n_val])
    test_seqs = list(sequences[n_train + n_val:])

    train_df = df[df["sequence"].isin(train_seqs)]
    val_df = df[df["sequence"].isin(val_seqs)]
    test_df = df[df["sequence"].isin(test_seqs)]

    print(f"  Sequence-based split:")
    print(f"    Train sequences ({len(train_seqs)}): {train_seqs}")
    print(f"    Val sequences   ({len(val_seqs)}): {val_seqs}")
    print(f"    Test sequences  ({len(test_seqs)}): {test_seqs}")

    return train_df, val_df, test_df, train_seqs, val_seqs, test_seqs


def train_pinn_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "auto",
):
    """Train PINN for dynamics prediction."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on {device}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Model
    model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)

    # Trainer
    trainer = Trainer(model=model, lr=lr, device=device)

    # Train with data loss only (w=0 shown to be best)
    weights = {"physics": 0, "temporal": 0, "stability": 0, "reg": 0, "energy": 0}

    print(f"\nTraining for {epochs} epochs...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        weights=weights,
        verbose=True,
    )

    return model, history


def compute_detection_threshold(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler_y: StandardScaler,
    percentile: float = 99.0,
    device: str = "cpu",
):
    """
    Compute detection threshold from validation errors.

    Uses percentile of prediction errors on clean data.
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        X_t = torch.FloatTensor(X_val).to(device)
        y_pred = model(X_t).cpu().numpy()

    # Inverse transform to original scale
    y_val_orig = scaler_y.inverse_transform(y_val)
    y_pred_orig = scaler_y.inverse_transform(y_pred)

    # Compute per-sample L2 error
    errors = np.linalg.norm(y_val_orig - y_pred_orig, axis=1)

    threshold = np.percentile(errors, percentile)
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    print(f"\nCalibration on clean data:")
    print(f"  Mean error: {mean_error:.4f}")
    print(f"  Std error:  {std_error:.4f}")
    print(f"  {percentile}th percentile threshold: {threshold:.4f}")

    return threshold, mean_error, std_error


def main():
    parser = argparse.ArgumentParser(description="Train synthetic attack detector")
    parser.add_argument("--data", type=str, default="data/euroc",
                        help="Path to EuRoC data")
    parser.add_argument("--output", type=str, default="models/security",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PINN SYNTHETIC ATTACK DETECTOR TRAINING")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading EuRoC data...")
    df = load_euroc_data(data_path)
    df = normalize_columns(df)

    # =========================================================================
    # CRITICAL: Split BY SEQUENCE to prevent temporal/trajectory leakage
    # =========================================================================
    print("\n[2/5] Splitting data by flight sequence...")
    split_result = split_by_sequence(df, test_ratio=0.2, val_ratio=args.val_split, seed=args.seed)

    if split_result is not None:
        train_df, val_df, test_df, train_seqs, val_seqs, test_seqs = split_result

        # Prepare sequences for each split separately
        X_train, y_train = prepare_sequences(train_df)
        X_val, y_val = prepare_sequences(val_df)
        X_test, y_test = prepare_sequences(test_df)

        # Save sequence split info for evaluation
        sequence_split_info = {
            "train_sequences": train_seqs,
            "val_sequences": val_seqs,
            "test_sequences": test_seqs,
        }
    else:
        # Fallback to random split if no sequence column
        print("  Falling back to random split...")
        X, y = prepare_sequences(df)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=args.val_split / (1 - 0.2),
            random_state=args.seed
        )
        sequence_split_info = None

    print(f"\n  Total samples after split:")
    print(f"    Train: {len(X_train):,}")
    print(f"    Val:   {len(X_val):,}")
    print(f"    Test:  {len(X_test):,} (held out for final evaluation)")

    # Scale data - FIT ONLY ON TRAINING DATA
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit scalers on training data only
    scaler_X.fit(X_train)
    scaler_y.fit(y_train)

    # Transform all splits using training-fitted scalers
    X_train_scaled = scaler_X.transform(X_train)
    y_train_scaled = scaler_y.transform(y_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)


    # Train model
    print("\n[3/5] Training PINN detector...")
    model, history = train_pinn_detector(
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Compute detection threshold on VALIDATION set only
    print("\n[4/5] Computing detection threshold on validation set...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold, mean_err, std_err = compute_detection_threshold(
        model, X_val_scaled, y_val_scaled, scaler_y,
        percentile=99.0, device=device
    )

    # Evaluate on held-out TEST set (final unbiased estimate)
    print("\n[4.5/5] Evaluating on held-out test set...")
    test_threshold, test_mean_err, test_std_err = compute_detection_threshold(
        model, X_test_scaled, y_test_scaled, scaler_y,
        percentile=99.0, device=device
    )
    print(f"  Test mean error: {test_mean_err:.4f}")
    print(f"  Test std error:  {test_std_err:.4f}")

    # Save model and artifacts
    print("\n[5/5] Saving model and artifacts...")

    # Model
    model_path = output_path / "pinn_synthetic_detector.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  Model: {model_path}")

    # Scalers
    scaler_path = output_path / "scalers_synthetic.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)
    print(f"  Scalers: {scaler_path}")

    # Config and results
    config = {
        "state_dim": 12,
        "control_dim": 4,
        "hidden_size": 256,
        "num_layers": 5,
        "dropout": 0.1,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "final_train_loss": float(history["train"][-1]) if history["train"] else None,
        "final_val_loss": float(history["val"][-1]) if history["val"] else None,
        "detection_threshold": float(threshold),
        "calibration_mean_error": float(mean_err),
        "calibration_std_error": float(std_err),
        # Held-out test set metrics (unbiased estimate)
        "test_mean_error": float(test_mean_err),
        "test_std_error": float(test_std_err),
        # Sequence-based splitting info (for proper evaluation)
        "sequence_split": sequence_split_info,
        # Methodology note
        "methodology": "Sequence-based split; scaler fit on train only; threshold tuned on validation; test held out",
    }

    config_path = output_path / "synthetic_detector_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  Final train loss: {config['final_train_loss']:.6f}")
    print(f"  Final val loss:   {config['final_val_loss']:.6f}")
    print(f"  Detection threshold: {threshold:.4f}")
    print(f"\nNext step:")
    print(f"  python scripts/security/evaluate_synthetic_detector.py")


if __name__ == "__main__":
    main()
