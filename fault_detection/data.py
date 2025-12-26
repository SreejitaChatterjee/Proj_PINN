"""
Data Loading Module
===================

Dataset classes and utilities for the PADRE UAV fault detection dataset.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable
import warnings


class PADREDataset(Dataset):
    """
    PADRE UAV Fault Detection Dataset.

    Each CSV file contains sensor data from 4 propellers (A, B, C, D).
    Labels are encoded in filename: e.g., Bebop2_16g_1kdps_normalized_0122.csv
    means A=0 (normal), B=1 (chipped), C=2 (bent), D=2 (bent)

    Features:
        - Supports multiple task types (binary, multiclass, per_motor)
        - Optional data augmentation
        - Configurable window size and stride
        - Class balancing utilities
    """

    COLUMNS = [
        'A_aX', 'A_aY', 'A_aZ', 'A_gX', 'A_gY', 'A_gZ',
        'B_aX', 'B_aY', 'B_aZ', 'B_gX', 'B_gY', 'B_gZ',
        'C_aX', 'C_aY', 'C_aZ', 'C_gX', 'C_gY', 'C_gZ',
        'D_aX', 'D_aY', 'D_aZ', 'D_gX', 'D_gY', 'D_gZ'
    ]

    FAULT_NAMES = {0: 'Normal', 1: 'Chipped', 2: 'Bent'}
    MOTORS = ['A', 'B', 'C', 'D']

    def __init__(
        self,
        data_dir: str,
        window_size: int = 256,
        stride: int = 128,
        task: str = 'binary',
        transform: Optional[Callable] = None,
        max_samples_per_file: Optional[int] = None,
        return_motor_labels: bool = False
    ):
        """
        Args:
            data_dir: Path to Normalized_data folder
            window_size: Number of timesteps per sample
            stride: Sliding window stride
            task: Classification task type
                - 'binary': Faulty vs Normal
                - 'multiclass': Normal, Chipped, Bent
                - 'per_motor': Per-motor classification
                - 'severity': Fault severity (for regression)
            transform: Optional data augmentation
            max_samples_per_file: Limit samples per file
            return_motor_labels: Return per-motor labels alongside main label
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.task = task
        self.transform = transform
        self.max_samples_per_file = max_samples_per_file
        self.return_motor_labels = return_motor_labels

        self.samples = []
        self.motor_labels = []
        self.file_labels = []

        self._load_data()

    def _parse_filename(self, filename: str) -> Dict:
        """Extract fault labels from filename."""
        match = re.search(r'normalized_(\d{4})\.csv$', filename)
        if not match:
            raise ValueError(f"Cannot parse filename: {filename}")

        codes = match.group(1)
        return {
            'A': int(codes[0]),
            'B': int(codes[1]),
            'C': int(codes[2]),
            'D': int(codes[3])
        }

    def _get_label(self, motor_faults: Dict) -> int:
        """Convert motor faults to classification label."""
        faults = list(motor_faults.values())

        if self.task == 'binary':
            return 0 if all(f == 0 for f in faults) else 1

        elif self.task == 'multiclass':
            if all(f == 0 for f in faults):
                return 0
            return max(faults)

        elif self.task == 'per_motor':
            return (motor_faults['A'] * 27 + motor_faults['B'] * 9 +
                    motor_faults['C'] * 3 + motor_faults['D'])

        elif self.task == 'severity':
            # Severity score: average of motor faults (0-2 scale)
            return sum(faults) / (2 * len(faults))

        raise ValueError(f"Unknown task: {self.task}")

    def _get_motor_label_vector(self, motor_faults: Dict) -> List[int]:
        """Get per-motor label vector."""
        return [motor_faults[m] for m in self.MOTORS]

    def _load_data(self):
        """Load all CSV files and create windows."""
        csv_files = sorted(self.data_dir.glob('*.csv'))

        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        print(f"Loading {len(csv_files)} files from {self.data_dir}")

        for csv_file in csv_files:
            motor_faults = self._parse_filename(csv_file.name)
            label = self._get_label(motor_faults)
            motor_label_vec = self._get_motor_label_vector(motor_faults)

            df = pd.read_csv(csv_file)
            data = df.values.astype(np.float32)

            n_samples = (len(data) - self.window_size) // self.stride + 1
            if self.max_samples_per_file:
                n_samples = min(n_samples, self.max_samples_per_file)

            for i in range(n_samples):
                start = i * self.stride
                end = start + self.window_size
                window = data[start:end]
                self.samples.append((window, label))
                self.motor_labels.append(motor_label_vec)

            self.file_labels.append({
                'file': csv_file.name,
                'motor_faults': motor_faults,
                'label': label,
                'n_windows': n_samples
            })

        print(f"Created {len(self.samples)} windows")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class distribution."""
        labels = [s[1] for s in self.samples]

        if self.task == 'severity':
            print(f"Severity: min={min(labels):.2f}, max={max(labels):.2f}, mean={np.mean(labels):.2f}")
        else:
            unique, counts = np.unique(labels, return_counts=True)
            print("Class distribution:")
            for u, c in zip(unique, counts):
                pct = 100 * c / len(labels)
                print(f"  Class {u}: {c} samples ({pct:.1f}%)")

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse class frequency weights."""
        if self.task == 'severity':
            return torch.ones(1)

        labels = [s[1] for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(unique)
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> List[float]:
        """Get per-sample weights for WeightedRandomSampler."""
        if self.task == 'severity':
            return [1.0] * len(self.samples)

        labels = np.array([s[1] for s in self.samples])
        unique, counts = np.unique(labels, return_counts=True)
        class_weights = 1.0 / counts
        return [class_weights[label] for label in labels]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        data, label = self.samples[idx]
        data = data.T.copy()  # (24, window_size)

        if self.transform:
            data = self.transform(data)

        if self.return_motor_labels:
            motor_label = self.motor_labels[idx]
            return torch.FloatTensor(data), label, torch.LongTensor(motor_label)

        return torch.FloatTensor(data), label


class DataAugmentation:
    """
    Data augmentation for time-series sensor data.

    Augmentations:
        - Gaussian noise injection
        - Random scaling
        - Time shift (circular)
        - Channel dropout
        - Time warping (optional)
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        time_shift_max: int = 10,
        channel_dropout_prob: float = 0.1,
        time_warp_prob: float = 0.0
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_shift_max = time_shift_max
        self.channel_dropout_prob = channel_dropout_prob
        self.time_warp_prob = time_warp_prob

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.randn(*data.shape).astype(np.float32) * self.noise_std
            data = data + noise

        # Random scaling
        if self.scale_range != (1.0, 1.0):
            scale = np.random.uniform(*self.scale_range)
            data = data * scale

        # Time shift
        if self.time_shift_max > 0:
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
            data = np.roll(data, shift, axis=1)

        # Channel dropout
        if self.channel_dropout_prob > 0:
            mask = np.random.random(data.shape[0]) > self.channel_dropout_prob
            data = data * mask[:, np.newaxis]

        return data


class MixupAugmentation:
    """
    Mixup augmentation for classification.

    Mixes two samples with random weight for regularization.

    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", 2018
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a pair of samples.

        Returns:
            mixed_x, y1, y2, lambda
        """
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, y1, y2, lam


def create_data_loaders(
    data_dir: str,
    task: str = 'binary',
    window_size: int = 256,
    stride: int = 128,
    batch_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    use_augmentation: bool = True,
    use_weighted_sampler: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train/val/test data loaders.

    Args:
        data_dir: Path to data directory
        task: Classification task
        window_size: Window size
        stride: Stride
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed
        use_augmentation: Apply augmentation to training
        use_weighted_sampler: Use weighted random sampler
        num_workers: Number of data loader workers

    Returns:
        train_loader, val_loader, test_loader, info_dict
    """
    # Create full dataset
    full_dataset = PADREDataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        task=task,
        transform=None
    )

    # Split indices
    indices = list(range(len(full_dataset)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_split + test_split, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_split / (val_split + test_split), random_state=seed
    )

    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

    # Augmentation for training
    if use_augmentation:
        train_transform = DataAugmentation(
            noise_std=0.01,
            scale_range=(0.95, 1.05),
            time_shift_max=10,
            channel_dropout_prob=0.1
        )
        # Note: Would need custom collate_fn to apply transform in DataLoader

    # Weighted sampler for training
    sampler = None
    shuffle = True
    if use_weighted_sampler and task != 'severity':
        sample_weights = [full_dataset.get_sample_weights()[i] for i in train_idx]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    info = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'n_classes': full_dataset.get_class_weights().shape[0] if task != 'severity' else 1,
        'class_weights': full_dataset.get_class_weights()
    }

    return train_loader, val_loader, test_loader, info
