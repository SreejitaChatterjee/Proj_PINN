"""
Train QuadrotorPINN on PADRE dataset for motor fault detection.

This script:
1. Loads PADRE sensor data and converts to PINN-compatible format
2. Trains a QuadrotorPINN model on the converted data
3. Evaluates fault detection via physics residuals

Best Architecture (from CLAUDE.md research):
- hidden_size: 256
- num_layers: 5
- dropout: 0.1
- physics_weight: Tunable (0-20, research suggests lower may be better)

Usage:
    python scripts/train_padre_pinn.py --epochs 100 --physics_weight 5.0
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinn_dynamics.data.padre import PADREtoPINNConverter
from pinn_dynamics.systems.quadrotor import QuadrotorPINN


def parse_args():
    parser = argparse.ArgumentParser(description="Train PINN on PADRE dataset")

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/PADRE_dataset/Parrot_Bebop_2/Normalized_data",
        help="Path to PADRE Normalized_data folder",
    )
    parser.add_argument(
        "--precomputed",
        type=str,
        default="data/PADRE_PINN_converted/padre_pinn_data.npz",
        help="Path to precomputed .npz file (faster loading)",
    )
    parser.add_argument(
        "--window_size", type=int, default=128, help="Window size for training samples"
    )
    parser.add_argument("--stride", type=int, default=64, help="Stride between windows")

    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden layer width")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of hidden layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for regularization"
    )

    # Loss weights
    parser.add_argument("--physics_weight", type=float, default=5.0, help="Weight for physics loss")
    parser.add_argument(
        "--temporal_weight", type=float, default=2.0, help="Weight for temporal smoothness loss"
    )
    parser.add_argument(
        "--stability_weight", type=float, default=1.0, help="Weight for stability loss"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/padre_pinn",
        help="Output directory for model and results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def denormalize_padre_data(data, accel_range_g=16, gyro_range_dps=1000):
    """
    Denormalize PADRE data from [-1, 1] to physical units.

    PADRE data is normalized:
    - Accelerometer: ±accel_range_g (e.g., ±16g)
    - Gyroscope: ±gyro_range_dps (e.g., ±1000 deg/s)

    Returns data in SI units:
    - Accelerometer: m/s²
    - Gyroscope: rad/s
    """
    g = 9.81
    data_out = data.copy()

    # PADRE columns: [A_aX, A_aY, A_aZ, A_gX, A_gY, A_gZ, B_..., C_..., D_...]
    for motor_idx in range(4):
        base = motor_idx * 6
        # Accelerometer (columns 0, 1, 2 for each motor)
        data_out[:, base : base + 3] *= accel_range_g * g  # Convert to m/s²
        # Gyroscope (columns 3, 4, 5 for each motor)
        data_out[:, base + 3 : base + 6] *= gyro_range_dps * (np.pi / 180)  # Convert to rad/s

    return data_out


def load_precomputed_data(npz_path):
    """Load precomputed PADRE-PINN data from .npz file."""
    print(f"Loading precomputed data from {npz_path}")
    data = np.load(npz_path)
    X = data["X"]
    Y = data["Y"]
    labels = data["labels"]
    print(f"Loaded {len(X)} samples")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  Normal: {(labels == 0).sum()}")
    print(f"  Faulty: {(labels == 1).sum()}")
    return X, Y, labels


def load_padre_data(
    data_dir, converter, window_size, stride, max_files=None, max_samples_per_file=10000
):
    """Load and convert PADRE dataset."""
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))

    if max_files:
        csv_files = csv_files[:max_files]

    print(f"Loading {len(csv_files)} files from {data_path}")
    print(f"Max samples per file: {max_samples_per_file}")

    all_X = []
    all_Y = []
    all_labels = []
    file_info = []

    for csv_file in csv_files:
        try:
            # Parse fault label from filename
            import re

            match = re.search(r"normalized_(\d{4})\.csv$", csv_file.name)
            if not match:
                continue

            codes = match.group(1)
            motor_faults = {
                "A": int(codes[0]),
                "B": int(codes[1]),
                "C": int(codes[2]),
                "D": int(codes[3]),
            }

            # Binary label: 0 = normal, 1 = any fault
            is_faulty = 1 if any(f > 0 for f in motor_faults.values()) else 0

            # Load and denormalize data (subsample for efficiency)
            df = pd.read_csv(csv_file)
            padre_data = df.values[:max_samples_per_file].astype(np.float32)
            padre_data = denormalize_padre_data(padre_data)  # Convert to physical units

            # Convert to PINN format
            X, Y = converter.convert_windowed(padre_data, window_size, stride)

            # Store
            for i in range(len(X)):
                all_X.append(X[i])
                all_Y.append(Y[i])
                all_labels.append(is_faulty)

            file_info.append(
                {
                    "file": csv_file.name,
                    "motor_faults": motor_faults,
                    "is_faulty": is_faulty,
                    "n_windows": len(X),
                }
            )

            print(f"  {csv_file.name}: {len(X)} windows, fault={is_faulty}")

        except Exception as e:
            print(f"  Warning: Could not process {csv_file.name}: {e}")

    X = np.array(all_X)
    Y = np.array(all_Y)
    labels = np.array(all_labels)

    print(f"\nTotal: {len(X)} windows")
    print(f"  Normal: {(labels == 0).sum()}")
    print(f"  Faulty: {(labels == 1).sum()}")

    return X, Y, labels, file_info


def create_data_loaders(
    X, Y, labels, batch_size, val_split=0.15, test_split=0.15, seed=42, preflattened=False
):
    """Create train/val/test data loaders."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if preflattened:
        # Data is already flattened from precomputation
        X_flat = X
        Y_flat = Y
        labels_flat = labels
    else:
        # Flatten windows for single-step prediction
        # X: (n_windows, seq_len, 16) -> (n_samples, 16)
        # Y: (n_windows, seq_len, 12) -> (n_samples, 12)
        n_windows, seq_len, input_dim = X.shape

        X_flat = X.reshape(-1, input_dim)
        Y_flat = Y.reshape(-1, Y.shape[-1])

        # Repeat labels for each timestep in window
        labels_flat = np.repeat(labels, seq_len)

    print(f"Total samples: {X_flat.shape[0]}")

    # Convert to tensors
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_flat, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_flat, dtype=torch.long)

    # Create dataset
    dataset = TensorDataset(X_tensor, Y_tensor, labels_tensor)

    # Split
    n_total = len(dataset)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val - n_test

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
    )

    # Create loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    return train_loader, val_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device, weights):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    data_loss_sum = 0
    physics_loss_sum = 0
    n_batches = 0

    for X, Y, _ in loader:
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()

        # Forward
        output = model(X)

        # Data loss
        data_loss = criterion(output, Y)

        # Physics loss
        physics_loss = torch.tensor(0.0, device=device)
        if weights.get("physics", 0) > 0:
            physics_loss = model.physics_loss(X, output)

        # Temporal smoothness
        temporal_loss = torch.tensor(0.0, device=device)
        if weights.get("temporal", 0) > 0:
            temporal_loss = model.temporal_smoothness_loss(X, output)

        # Stability
        stability_loss = torch.tensor(0.0, device=device)
        if weights.get("stability", 0) > 0:
            stability_loss = model.stability_loss(X, output)

        # Total loss
        loss = (
            data_loss
            + weights.get("physics", 0) * physics_loss
            + weights.get("temporal", 0) * temporal_loss
            + weights.get("stability", 0) * stability_loss
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        data_loss_sum += data_loss.item()
        physics_loss_sum += physics_loss.item()
        n_batches += 1

    return {
        "total": total_loss / n_batches,
        "data": data_loss_sum / n_batches,
        "physics": physics_loss_sum / n_batches,
    }


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    all_residuals = []
    all_labels = []

    with torch.no_grad():
        for X, Y, labels in loader:
            X, Y = X.to(device), Y.to(device)

            output = model(X)
            loss = criterion(output, Y)

            # Compute prediction residuals (for fault detection)
            residuals = (output - Y).pow(2).sum(dim=1).sqrt()
            all_residuals.extend(residuals.cpu().numpy())
            all_labels.extend(labels.numpy())

            total_loss += loss.item()
            n_batches += 1

    all_residuals = np.array(all_residuals)
    all_labels = np.array(all_labels)

    # Compute fault detection metrics
    normal_residuals = all_residuals[all_labels == 0]
    faulty_residuals = all_residuals[all_labels == 1]

    metrics = {
        "loss": total_loss / n_batches,
        "normal_residual_mean": normal_residuals.mean() if len(normal_residuals) > 0 else 0,
        "faulty_residual_mean": faulty_residuals.mean() if len(faulty_residuals) > 0 else 0,
    }

    # Separation ratio (higher = better fault detection)
    if len(normal_residuals) > 0 and len(faulty_residuals) > 0:
        metrics["separation_ratio"] = faulty_residuals.mean() / (normal_residuals.mean() + 1e-8)
    else:
        metrics["separation_ratio"] = 1.0

    return metrics


def main():
    args = parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create converter with appropriate parameters for Bebop 2
    # Bebop 2 specs: mass ~0.5kg, sample rate 500Hz
    converter = PADREtoPINNConverter(
        dt=0.002,  # 500 Hz
        mass=0.5,  # kg
        Jxx=0.005,
        Jyy=0.005,
        Jzz=0.009,
        complementary_alpha=0.98,
        drag_coeff=0.01,
    )

    # Load data
    print("\n" + "=" * 60)
    print("Loading PADRE dataset...")
    print("=" * 60)

    precomputed_path = Path(args.precomputed)
    if precomputed_path.exists():
        # Use precomputed data (much faster)
        X, Y, labels = load_precomputed_data(precomputed_path)
        preflattened = True
    else:
        print(f"Precomputed file not found at {precomputed_path}")
        print("Converting from raw data (slower)...")
        X, Y, labels, file_info = load_padre_data(
            args.data_dir, converter, args.window_size, args.stride
        )
        preflattened = False

    # Create data loaders
    print("\n" + "=" * 60)
    print("Creating data loaders...")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_data_loaders(
        X, Y, labels, args.batch_size, seed=args.seed, preflattened=preflattened
    )

    # Create model
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)

    model = QuadrotorPINN(
        hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: hidden_size={args.hidden_size}, num_layers={args.num_layers}")
    print(f"Parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.MSELoss()

    # Loss weights
    weights = {
        "physics": args.physics_weight,
        "temporal": args.temporal_weight,
        "stability": args.stability_weight,
    }

    print(f"\nLoss weights: {weights}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    history = {"train_loss": [], "val_loss": [], "separation_ratio": [], "lr": []}

    best_val_loss = float("inf")
    best_separation = 0

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, weights)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history["train_loss"].append(train_metrics["total"])
        history["val_loss"].append(val_metrics["loss"])
        history["separation_ratio"].append(val_metrics["separation_ratio"])
        history["lr"].append(current_lr)

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": vars(args),
                },
                output_dir / "best_model.pth",
            )

        if val_metrics["separation_ratio"] > best_separation:
            best_separation = val_metrics["separation_ratio"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "separation_ratio": best_separation,
                    "config": vars(args),
                },
                output_dir / "best_separation_model.pth",
            )

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Train: {train_metrics['total']:.4f} (data: {train_metrics['data']:.4f}, phys: {train_metrics['physics']:.4f}) | "
                f"Val: {val_metrics['loss']:.4f} | "
                f"Sep: {val_metrics['separation_ratio']:.2f} | "
                f"LR: {current_lr:.2e}"
            )

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final evaluation on test set...")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(output_dir / "best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = validate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Normal residual mean: {test_metrics['normal_residual_mean']:.4f}")
    print(f"  Faulty residual mean: {test_metrics['faulty_residual_mean']:.4f}")
    print(f"  Separation ratio: {test_metrics['separation_ratio']:.2f}")

    # Save results
    results = {
        "config": vars(args),
        "history": history,
        "test_metrics": test_metrics,
        "best_val_loss": best_val_loss,
        "best_separation": best_separation,
        "n_params": n_params,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nModel saved to: {output_dir / 'best_model.pth'}")
    print(f"Results saved to: {output_dir / 'results.json'}")

    return results


if __name__ == "__main__":
    main()
