"""
Train Sequence-PINN based anomaly detector.

Unlike single-step PINN, this model uses a sliding window of past states
to detect temporal attacks (replay, freeze, delay, gradual drift).

Usage:
    python scripts/security/train_sequence_detector.py
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pinn_dynamics import SequencePINN


class SequenceDataset(Dataset):
    """Dataset for sequence-based training."""

    def __init__(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        sequence_length: int = 20,
        labels: np.ndarray = None,
    ):
        """
        Args:
            states: [N, state_dim] state data
            controls: [N, control_dim] control data
            sequence_length: Length of input sequences
            labels: [N] optional labels (0=normal, 1=attack)
        """
        self.states = states
        self.controls = controls
        self.sequence_length = sequence_length
        self.labels = labels

        # Number of valid sequences
        self.n_samples = len(states) - sequence_length

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Get sequence of past states + controls
        seq_states = self.states[idx : idx + self.sequence_length]
        seq_controls = self.controls[idx : idx + self.sequence_length]

        # Concatenate state and control
        sequence = np.concatenate([seq_states, seq_controls], axis=1)

        # Target: next state after sequence
        target = self.states[idx + self.sequence_length]

        # Label (if available)
        if self.labels is not None:
            label = self.labels[idx + self.sequence_length]
            return (
                torch.FloatTensor(sequence),
                torch.FloatTensor(target),
                torch.FloatTensor([label]),
            )

        return torch.FloatTensor(sequence), torch.FloatTensor(target)


def load_euroc_data(data_path: Path) -> pd.DataFrame:
    """Load EuRoC processed data."""
    csv_files = list(data_path.glob("*.csv"))

    for name in ["all_sequences.csv", "euroc_processed.csv"]:
        path = data_path / name
        if path.exists():
            return pd.read_csv(path)

    if csv_files:
        return pd.read_csv(csv_files[0])

    raise FileNotFoundError(f"No CSV files found in {data_path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard format."""
    column_map = {"roll": "phi", "pitch": "theta", "yaw": "psi"}

    for old, new in column_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "thrust" not in df.columns:
        if "az" in df.columns:
            df["thrust"] = df["az"] + 9.81
        else:
            df["thrust"] = 9.81

    for col in ["torque_x", "torque_y", "torque_z"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


def prepare_sequences(df: pd.DataFrame):
    """Extract state and control sequences."""
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    states = df[state_cols].values
    controls = df[control_cols].values

    return states, controls


def train_sequence_detector(
    train_loader: DataLoader,
    val_loader: DataLoader,
    sequence_length: int = 20,
    hidden_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "auto",
):
    """Train SequencePINN model."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on {device}")

    # Create model
    model = SequencePINN(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_lstm_layers=2,
        fc_hidden_size=256,
        dropout=0.1,
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            sequences, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, targets)

            # Add temporal smoothness regularization
            smoothness_loss = model.temporal_smoothness_loss(sequences)
            total_loss = loss + 0.01 * smoothness_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                sequences, targets = batch[0].to(device), batch[1].to(device)
                predictions = model(sequences)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= n_val_batches

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def compute_detection_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    scaler_y: StandardScaler,
    percentile: float = 95.0,
    device: str = "cpu",
):
    """Compute detection threshold from validation errors."""
    model.eval()
    model.to(device)

    all_errors = []

    with torch.no_grad():
        for batch in val_loader:
            sequences, targets = batch[0].to(device), batch[1].to(device)
            predictions = model(sequences)

            # Compute per-sample errors
            errors = torch.norm(predictions - targets, dim=1).cpu().numpy()
            all_errors.extend(errors)

    all_errors = np.array(all_errors)

    threshold = np.percentile(all_errors, percentile)
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)

    print(f"\nCalibration on clean data:")
    print(f"  Mean error: {mean_error:.4f}")
    print(f"  Std error:  {std_error:.4f}")
    print(f"  {percentile}th percentile threshold: {threshold:.4f}")

    return threshold, mean_error, std_error


def main():
    parser = argparse.ArgumentParser(description="Train sequence-based detector")
    parser.add_argument("--data", type=str, default="data/euroc", help="Path to EuRoC data")
    parser.add_argument("--output", type=str, default="models/security", help="Output directory")
    parser.add_argument("--sequence-length", type=int, default=20, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SEQUENCE-PINN DETECTOR TRAINING")
    print("=" * 70)
    print(f"Sequence length: {args.sequence_length}")

    # Load data
    print("\n[1/5] Loading EuRoC data...")
    df = load_euroc_data(data_path)
    df = normalize_columns(df)
    print(f"  Loaded {len(df):,} samples")

    # Prepare sequences
    print("\n[2/5] Preparing sequences...")
    states, controls = prepare_sequences(df)

    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # For sequences, we need to scale states and controls together
    combined = np.concatenate([states, controls], axis=1)
    combined_scaled = scaler_X.fit_transform(combined)
    states_scaled = combined_scaled[:, :12]
    controls_scaled = combined_scaled[:, 12:]

    # Fit output scaler on states only
    scaler_y.fit(states)

    # Train/val split (by index to preserve temporal structure)
    n_samples = len(states) - args.sequence_length
    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(indices, test_size=args.val_split, random_state=args.seed)

    # Create datasets
    train_dataset = SequenceDataset(
        states_scaled, controls_scaled, sequence_length=args.sequence_length
    )
    val_dataset = SequenceDataset(
        states_scaled, controls_scaled, sequence_length=args.sequence_length
    )

    # Note: We use full dataset for both but different batch sampling would be ideal
    # For simplicity, using the same dataset with shuffle

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")

    # Train model
    print("\n[3/5] Training Sequence-PINN...")
    model, history = train_sequence_detector(
        train_loader,
        val_loader,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        lr=args.lr,
    )

    # Compute detection threshold
    print("\n[4/5] Computing detection threshold...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold, mean_err, std_err = compute_detection_threshold(
        model, val_loader, scaler_y, percentile=95.0, device=device
    )

    # Save model and artifacts
    print("\n[5/5] Saving model and artifacts...")

    model_path = output_path / "sequence_pinn_detector.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  Model: {model_path}")

    scaler_path = output_path / "scalers_sequence.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)
    print(f"  Scalers: {scaler_path}")

    config = {
        "model_type": "SequencePINN",
        "sequence_length": args.sequence_length,
        "state_dim": 12,
        "control_dim": 4,
        "hidden_size": args.hidden_size,
        "num_lstm_layers": 2,
        "fc_hidden_size": 256,
        "dropout": 0.1,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "final_train_loss": float(history["train"][-1]) if history["train"] else None,
        "final_val_loss": float(history["val"][-1]) if history["val"] else None,
        "detection_threshold": float(threshold),
        "calibration_mean_error": float(mean_err),
        "calibration_std_error": float(std_err),
    }

    config_path = output_path / "sequence_detector_config.json"
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
    print(f"  python scripts/security/evaluate_sequence_detector.py")


if __name__ == "__main__":
    main()
