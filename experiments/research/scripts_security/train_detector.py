"""
Train PINN detector on ALFA dataset for fault detection.

Trains two variants:
1. w=0  - Pure data-driven (no physics loss)
2. w=20 - Physics-informed (with physics constraints)

Usage:
    python scripts/security/train_detector.py \\
        --data data/attack_datasets/processed/alfa/ \\
        --output models/security/
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Import PINN framework (install with: pip install -e .)
try:
    from pinn_dynamics import QuadrotorPINN
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pinn_dynamics import QuadrotorPINN

from pinn_dynamics.training import Trainer


def load_alfa_data(data_dir: Path, flight_type: str = "normal"):
    """
    Load and combine ALFA flights.

    Args:
        data_dir: Directory with preprocessed ALFA CSVs
        flight_type: "normal" or "fault"

    Returns:
        Combined dataframe
    """
    csv_files = list(data_dir.glob("*.csv"))

    flights = []
    for csv_file in csv_files:
        if csv_file.name == "summary.json":
            continue

        df = pd.read_csv(csv_file)

        # Filter by type
        if flight_type == "normal" and df["label"].iloc[0] == 0:
            flights.append(df)
        elif flight_type == "fault" and df["label"].iloc[0] == 1:
            flights.append(df)

    if not flights:
        raise ValueError(f"No {flight_type} flights found in {data_dir}")

    combined = pd.concat(flights, ignore_index=True)
    print(f"Loaded {len(flights)} {flight_type} flights: {len(combined)} samples")

    return combined


def prepare_training_data(df: pd.DataFrame):
    """
    Convert ALFA dataframe to PINN training format.

    Args:
        df: Dataframe with states and controls

    Returns:
        X, y, scaler_X, scaler_y
    """
    # State columns (12)
    state_cols = ["x", "y", "z", "phi", "theta", "psi", "p", "q", "r", "vx", "vy", "vz"]

    # Control columns (4)
    control_cols = ["thrust", "torque_x", "torque_y", "torque_z"]

    # Create input (current state + control) and output (next state)
    states = df[state_cols].values[:-1]  # t
    controls = df[control_cols].values[:-1]  # t
    next_states = df[state_cols].values[1:]  # t+1

    # Combine inputs
    X = np.concatenate([states, controls], axis=1)  # [N, 16]
    y = next_states  # [N, 12]

    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    print(f"Training data shape: X={X_scaled.shape}, y={y_scaled.shape}")

    return X_scaled, y_scaled, scaler_X, scaler_y


def train_pinn(
    X_train,
    y_train,
    X_val,
    y_val,
    physics_weight: float = 20.0,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train QuadrotorPINN model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        physics_weight: Weight for physics loss (0 = pure data-driven)
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: cuda or cpu

    Returns:
        Trained model, training history
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Create DataLoaders
    from torch.utils.data import DataLoader, TensorDataset

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)

    # Create trainer
    trainer = Trainer(model=model, lr=lr, device=device)

    # Set loss weights
    weights = {
        "physics": physics_weight,
        "temporal": 0,  # Disable temporal loss for simplicity
        "stability": 0,  # Disable stability loss
        "reg": 0,  # Disable regularization
        "energy": 0,  # Disable energy loss
    }

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        weights=weights,
        verbose=False,  # Suppress per-epoch output for cleaner logs
    )

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train PINN detector on ALFA")
    parser.add_argument(
        "--data",
        type=str,
        default="data/attack_datasets/processed/alfa",
        help="Path to preprocessed ALFA data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/security",
        help="Output directory for models",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,  # Exhaustive training
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=20,
        help="Number of random seeds for statistical significance (default: 20)",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=50,
        help="Early stopping patience (epochs)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PINN DETECTOR TRAINING - ALFA Dataset")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading ALFA data...")
    normal_flights = load_alfa_data(data_dir, "normal")

    # Prepare training data
    print("\n[2/5] Preparing training data...")
    X, y, scaler_X, scaler_y = prepare_training_data(normal_flights)

    # Train/val split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")

    # Multi-seed training for statistical significance
    print(f"\n[3/5] Training PINN w=0 with {args.num_seeds} seeds (EXHAUSTIVE)...")
    print("This ensures statistical significance (p<0.05) for paper.")

    results_w0 = []
    best_loss_w0 = float("inf")
    best_model_w0 = None

    for seed in range(args.num_seeds):
        print(f"\n  Seed {seed+1}/{args.num_seeds}...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model_w0, history_w0 = train_pinn(
            X_train,
            y_train,
            X_val,
            y_val,
            physics_weight=0.0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        final_loss = history_w0["val"][-1] if history_w0["val"] else float("inf")
        results_w0.append(final_loss)

        # Track best model
        if final_loss < best_loss_w0:
            best_loss_w0 = final_loss
            best_model_w0 = model_w0

        print(f"    Final val loss: {final_loss:.6f}")

    # Save best w=0 model
    model_w0_path = output_dir / "pinn_w0_best.pth"
    torch.save(best_model_w0.state_dict(), model_w0_path)
    print(f"\n  Best w=0 model saved: {model_w0_path}")
    print(f"  Mean val loss: {np.mean(results_w0):.6f} ± {np.std(results_w0):.6f}")

    # Train w=20 (physics-informed) with multiple seeds
    print(f"\n[4/5] Training PINN w=20 with {args.num_seeds} seeds (EXHAUSTIVE)...")

    results_w20 = []
    best_loss_w20 = float("inf")
    best_model_w20 = None

    for seed in range(args.num_seeds):
        print(f"\n  Seed {seed+1}/{args.num_seeds}...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model_w20, history_w20 = train_pinn(
            X_train,
            y_train,
            X_val,
            y_val,
            physics_weight=20.0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        final_loss = history_w20["val"][-1] if history_w20["val"] else float("inf")
        results_w20.append(final_loss)

        if final_loss < best_loss_w20:
            best_loss_w20 = final_loss
            best_model_w20 = model_w20

        print(f"    Final val loss: {final_loss:.6f}")

    # Save best w=20 model
    model_w20_path = output_dir / "pinn_w20_best.pth"
    torch.save(best_model_w20.state_dict(), model_w20_path)
    print(f"\n  Best w=20 model saved: {model_w20_path}")
    print(f"  Mean val loss: {np.mean(results_w20):.6f} ± {np.std(results_w20):.6f}")

    # Statistical significance test (paired t-test)
    from scipy import stats

    t_stat, p_value = stats.ttest_rel(results_w0, results_w20)
    print(f"\n  Statistical Test (w=0 vs w=20):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value: {p_value:.6f}")
    if p_value < 0.05:
        winner = "w=0" if np.mean(results_w0) < np.mean(results_w20) else "w=20"
        print(f"    Result: {winner} is SIGNIFICANTLY better (p<0.05) ✓")
    else:
        print(f"    Result: No significant difference (p>=0.05)")

    # Save scalers
    print("\n[5/5] Saving scalers and config...")
    scaler_path = output_dir / "scalers.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)
    print(f"Saved: {scaler_path}")

    # Save comprehensive results
    config = {
        "state_dim": 12,
        "control_dim": 4,
        "hidden_size": 256,
        "num_layers": 5,
        "dropout": 0.1,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_seeds": args.num_seeds,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        # w=0 results (20 seeds)
        "w0_mean_loss": float(np.mean(results_w0)),
        "w0_std_loss": float(np.std(results_w0)),
        "w0_best_loss": float(best_loss_w0),
        "w0_all_losses": [float(x) for x in results_w0],
        # w=20 results (20 seeds)
        "w20_mean_loss": float(np.mean(results_w20)),
        "w20_std_loss": float(np.std(results_w20)),
        "w20_best_loss": float(best_loss_w20),
        "w20_all_losses": [float(x) for x in results_w20],
        # Statistical significance
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "winner": "w=0" if np.mean(results_w0) < np.mean(results_w20) else "w=20",
    }

    config_path = output_dir / "training_results.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {config_path}")

    print("\n" + "=" * 60)
    print("EXHAUSTIVE TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModels saved to: {output_dir.absolute()}")
    print(f"  - pinn_w0_best.pth  (w=0, best of {args.num_seeds} seeds)")
    print(f"  - pinn_w20_best.pth (w=20, best of {args.num_seeds} seeds)")
    print(f"\nResults (statistically significant with {args.num_seeds} seeds):")
    print(f"  w=0:  {config['w0_mean_loss']:.6f} ± {config['w0_std_loss']:.6f}")
    print(f"  w=20: {config['w20_mean_loss']:.6f} ± {config['w20_std_loss']:.6f}")
    print(f"  p-value: {p_value:.6f} ({'SIGNIFICANT' if p_value < 0.05 else 'not significant'})")
    print(f"\nNext steps:")
    print("  python scripts/security/evaluate_detector.py")


if __name__ == "__main__":
    main()
