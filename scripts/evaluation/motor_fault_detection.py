#!/usr/bin/env python3
"""
Motor Fault Detection using PADRE Dataset
==========================================

Dataset: AeroLab UAV Measurement Data (Poznan University of Technology)
Source: https://github.com/AeroLabPUT/UAV_measurement_data
Paper: https://link.springer.com/article/10.1007/s10846-024-02101-7

Fault Types:
    0 = Normal (no fault)
    1 = Chipped edge
    2 = Bent tip

Architecture: 1D CNN with attention for time-series classification
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================


class PADREDataset(Dataset):
    """
    PADRE UAV Fault Detection Dataset

    Each CSV file contains sensor data from 4 propellers (A, B, C, D).
    Labels are encoded in filename: e.g., Bebop2_16g_1kdps_normalized_0122.csv
    means A=0 (normal), B=1 (chipped), C=2 (bent), D=2 (bent)
    """

    COLUMNS = [
        "A_aX",
        "A_aY",
        "A_aZ",
        "A_gX",
        "A_gY",
        "A_gZ",
        "B_aX",
        "B_aY",
        "B_aZ",
        "B_gX",
        "B_gY",
        "B_gZ",
        "C_aX",
        "C_aY",
        "C_aZ",
        "C_gX",
        "C_gY",
        "C_gZ",
        "D_aX",
        "D_aY",
        "D_aZ",
        "D_gX",
        "D_gY",
        "D_gZ",
    ]

    FAULT_NAMES = {0: "Normal", 1: "Chipped", 2: "Bent"}

    def __init__(
        self,
        data_dir: str,
        window_size: int = 256,
        stride: int = 128,
        task: str = "binary",  # 'binary', 'multiclass', 'per_motor'
        transform=None,
        max_samples_per_file: int = None,
    ):
        """
        Args:
            data_dir: Path to Normalized_data folder
            window_size: Number of timesteps per sample (256 @ 500Hz = 0.512s)
            stride: Sliding window stride
            task: Classification task type
                - 'binary': Faulty vs Normal (any motor)
                - 'multiclass': Normal, Chipped, Bent (overall)
                - 'per_motor': 4-way classification per motor (12 classes)
            transform: Optional data augmentation
            max_samples_per_file: Limit samples per file (for faster debugging)
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.task = task
        self.transform = transform
        self.max_samples_per_file = max_samples_per_file

        self.samples = []  # List of (data_array, label)
        self.file_labels = []  # For reference

        self._load_data()

    def _parse_filename(self, filename: str) -> Dict:
        """Extract fault labels from filename."""
        # Pattern: UAVname_range_range_normalized_ABCD.csv
        match = re.search(r"normalized_(\d{4})\.csv$", filename)
        if not match:
            raise ValueError(f"Cannot parse filename: {filename}")

        codes = match.group(1)
        return {
            "A": int(codes[0]),
            "B": int(codes[1]),
            "C": int(codes[2]),
            "D": int(codes[3]),
        }

    def _get_label(self, motor_faults: Dict) -> int:
        """Convert motor faults to classification label."""
        faults = list(motor_faults.values())

        if self.task == "binary":
            # 0 = all normal, 1 = any fault
            return 0 if all(f == 0 for f in faults) else 1

        elif self.task == "multiclass":
            # Dominant fault type (0=normal, 1=chipped, 2=bent)
            if all(f == 0 for f in faults):
                return 0
            # Return the highest severity fault present
            return max(faults)

        elif self.task == "per_motor":
            # Encode as single label: A*27 + B*9 + C*3 + D
            return (
                motor_faults["A"] * 27
                + motor_faults["B"] * 9
                + motor_faults["C"] * 3
                + motor_faults["D"]
            )

        raise ValueError(f"Unknown task: {self.task}")

    def _load_data(self):
        """Load all CSV files and create windows."""
        csv_files = sorted(self.data_dir.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        print(f"Loading {len(csv_files)} files from {self.data_dir}")

        for csv_file in csv_files:
            # Parse label from filename
            motor_faults = self._parse_filename(csv_file.name)
            label = self._get_label(motor_faults)

            # Load data
            df = pd.read_csv(csv_file)
            data = df.values.astype(np.float32)

            # Create sliding windows
            n_samples = (len(data) - self.window_size) // self.stride + 1
            if self.max_samples_per_file:
                n_samples = min(n_samples, self.max_samples_per_file)

            for i in range(n_samples):
                start = i * self.stride
                end = start + self.window_size
                window = data[start:end]
                self.samples.append((window, label))

            self.file_labels.append(
                {
                    "file": csv_file.name,
                    "motor_faults": motor_faults,
                    "label": label,
                    "n_windows": n_samples,
                }
            )

        print(f"Created {len(self.samples)} windows")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class distribution."""
        labels = [s[1] for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        print("Class distribution:")
        for u, c in zip(unique, counts):
            pct = 100 * c / len(labels)
            print(f"  Class {u}: {c} samples ({pct:.1f}%)")

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse class frequency weights."""
        labels = [s[1] for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(unique)
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> List[float]:
        """Get per-sample weights for WeightedRandomSampler."""
        labels = np.array([s[1] for s in self.samples])
        unique, counts = np.unique(labels, return_counts=True)
        class_weights = 1.0 / counts
        return [class_weights[label] for label in labels]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data, label = self.samples[idx]

        # Transpose to (channels, time) for 1D CNN
        data = data.T  # (24, window_size)

        if self.transform:
            data = self.transform(data)

        return torch.FloatTensor(data), label


class DataAugmentation:
    """Data augmentation for time-series sensor data."""

    def __init__(
        self,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        time_shift_max: int = 10,
        channel_dropout_prob: float = 0.1,
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_shift_max = time_shift_max
        self.channel_dropout_prob = channel_dropout_prob

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        # Gaussian noise
        if self.noise_std > 0:
            data = data + np.random.randn(*data.shape).astype(np.float32) * self.noise_std

        # Random scaling
        if self.scale_range != (1.0, 1.0):
            scale = np.random.uniform(*self.scale_range)
            data = data * scale

        # Time shift (circular)
        if self.time_shift_max > 0:
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
            data = np.roll(data, shift, axis=1)

        # Channel dropout
        if self.channel_dropout_prob > 0:
            mask = np.random.random(data.shape[0]) > self.channel_dropout_prob
            data = data * mask[:, np.newaxis]

        return data


# =============================================================================
# Neural Network Model
# =============================================================================


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding),
            nn.BatchNorm1d(out_channels),
        )

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1, stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.conv(x) + self.residual(x))


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attn = self.attention(x).unsqueeze(-1)
        return x * attn


class MotorFaultCNN(nn.Module):
    """
    1D CNN for motor fault detection.

    Architecture:
        - Input: (batch, 24 sensors, window_size)
        - 4 ConvBlocks with increasing channels and downsampling
        - Channel attention after each block
        - Global average pooling
        - Classifier head
    """

    def __init__(
        self,
        n_input_channels: int = 24,
        n_classes: int = 2,
        base_channels: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_classes = n_classes

        # Feature extraction
        self.features = nn.Sequential(
            # Block 1: 24 -> 64
            ConvBlock(n_input_channels, base_channels, kernel_size=15),
            ChannelAttention(base_channels),
            nn.MaxPool1d(2),
            # Block 2: 64 -> 128
            ConvBlock(base_channels, base_channels * 2, kernel_size=11),
            ChannelAttention(base_channels * 2),
            nn.MaxPool1d(2),
            # Block 3: 128 -> 256
            ConvBlock(base_channels * 2, base_channels * 4, kernel_size=7),
            ChannelAttention(base_channels * 4),
            nn.MaxPool1d(2),
            # Block 4: 256 -> 512
            ConvBlock(base_channels * 4, base_channels * 8, kernel_size=5),
            ChannelAttention(base_channels * 8),
            nn.AdaptiveAvgPool1d(1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(base_channels * 4, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


# =============================================================================
# Training
# =============================================================================


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
) -> Dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    if scheduler:
        scheduler.step()

    return {"loss": total_loss / len(loader), "accuracy": 100.0 * correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Dict:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * (all_preds == all_labels).mean()

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy,
        "predictions": all_preds,
        "labels": all_labels,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    class_weights: torch.Tensor = None,
    save_path: str = None,
) -> Dict:
    """Full training loop."""

    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Scheduler: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss with class weights
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining for {epochs} epochs on {device}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Log
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.1f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.1f}%"
        )

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            if save_path:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": best_val_acc,
                        "n_classes": model.n_classes,
                    },
                    save_path,
                )
                print(f"  -> Saved best model (acc: {best_val_acc:.1f}%)")

    return history


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Motor Fault Detection Training")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/PADRE_dataset/Parrot_Bebop_2/Normalized_data",
        help="Path to normalized data folder",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="binary",
        choices=["binary", "multiclass", "per_motor"],
        help="Classification task",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=256,
        help="Window size in samples (256 @ 500Hz = 0.5s)",
    )
    parser.add_argument("--stride", type=int, default=128, help="Sliding window stride")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="models/fault_detection")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n" + "=" * 60)
    print("Loading PADRE Dataset")
    print("=" * 60)

    full_dataset = PADREDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        task=args.task,
        transform=None,
    )

    # Train/val/test split
    indices = list(range(len(full_dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)

    # Create augmented training dataset
    train_transform = DataAugmentation(
        noise_std=0.01,
        scale_range=(0.95, 1.05),
        time_shift_max=10,
        channel_dropout_prob=0.1,
    )

    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

    print(
        f"\nSplit sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    # Handle class imbalance with weighted sampler
    class_weights = full_dataset.get_class_weights()
    sample_weights = [full_dataset.get_sample_weights()[i] for i in train_idx]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    n_classes = {"binary": 2, "multiclass": 3, "per_motor": 81}[args.task]
    model = MotorFaultCNN(n_input_channels=24, n_classes=n_classes, dropout=0.3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    # Train
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    save_path = save_dir / f"fault_detector_{args.task}.pth"
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        class_weights=class_weights,
        save_path=str(save_path),
    )

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Accuracy: {test_metrics['accuracy']:.2f}%")
    print("\nClassification Report:")

    if args.task == "binary":
        target_names = ["Normal", "Faulty"]
    elif args.task == "multiclass":
        target_names = ["Normal", "Chipped", "Bent"]
    else:
        target_names = None

    print(
        classification_report(
            test_metrics["labels"],
            test_metrics["predictions"],
            target_names=target_names,
        )
    )

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_metrics["labels"], test_metrics["predictions"])
    print(cm)

    # Save results
    results = {
        "task": args.task,
        "test_accuracy": test_metrics["accuracy"],
        "n_classes": n_classes,
        "window_size": args.window_size,
        "stride": args.stride,
        "epochs": args.epochs,
        "best_epoch": checkpoint["epoch"],
        "confusion_matrix": cm.tolist(),
        "history": history,
    }

    results_path = save_dir / f"results_{args.task}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
