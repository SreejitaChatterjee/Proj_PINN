"""
Generate Synthetic Normal Flight Data for PADRE Dataset.

The PADRE dataset has only 2 normal flights (1 per drone).
This script generates synthetic normal data using:
1. Gaussian noise injection
2. Time warping (stretch/compress)
3. Amplitude scaling
4. Segment mixing between drones
5. Jittering and smoothing

This helps the model learn generalizable "normal" patterns.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import warnings

# Suppress specific interpolation warnings from scipy
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Default number of columns in PADRE data
DEFAULT_N_COLUMNS = 24


class SyntheticNormalGenerator:
    """Generate synthetic normal flight data from real normal flights."""

    def __init__(self, seed: int = 42, n_columns: Optional[int] = None):
        """
        Initialize the synthetic normal generator.

        Args:
            seed: Random seed for reproducibility
            n_columns: Number of columns to use from input data. If None, uses all columns.
        """
        self.rng = np.random.default_rng(seed)
        self.real_normals: List[Dict[str, Any]] = []
        self.n_columns = n_columns

    def load_real_normals(self, file_paths: List[Path]) -> None:
        """
        Load real normal flight data.

        Args:
            file_paths: List of paths to CSV files containing normal flight data

        Raises:
            FileNotFoundError: If a file does not exist
            ValueError: If a file has fewer columns than expected
        """
        for path in file_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"File not found: {path}")

            df = pd.read_csv(path)

            # Determine number of columns to use
            if self.n_columns is not None:
                if df.shape[1] < self.n_columns:
                    raise ValueError(
                        f"File {path} has {df.shape[1]} columns, "
                        f"but {self.n_columns} were requested"
                    )
                data = df.values.astype(np.float32)[:, :self.n_columns]
            else:
                data = df.values.astype(np.float32)

            self.real_normals.append({
                'data': data,
                'path': path,
                'mean': data.mean(axis=0),
                'std': data.std(axis=0)
            })
        print(f"Loaded {len(self.real_normals)} real normal flights")

    def add_noise(self, data: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """Add Gaussian noise scaled to each channel's std."""
        noise = self.rng.standard_normal(data.shape) * data.std(axis=0) * noise_level
        return data + noise

    def time_warp(
        self, data: np.ndarray, factor_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        """Stretch or compress time axis."""
        factor = self.rng.uniform(*factor_range)
        n_samples = data.shape[0]
        new_n = max(1, int(n_samples * factor))  # Ensure at least 1 sample

        x_old = np.linspace(0, 1, n_samples)
        x_new = np.linspace(0, 1, new_n)

        warped = np.zeros((new_n, data.shape[1]))
        for col in range(data.shape[1]):
            f = interp1d(x_old, data[:, col], kind='linear', fill_value='extrapolate')
            warped[:, col] = f(x_new)

        return warped

    def amplitude_scale(
        self, data: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """Scale amplitude per channel."""
        scales = self.rng.uniform(scale_range[0], scale_range[1], size=data.shape[1])
        return data * scales

    def jitter(self, data: np.ndarray, sigma: float = 0.01) -> np.ndarray:
        """Add jitter (small random walk)."""
        jitter_vals = np.cumsum(
            self.rng.standard_normal(data.shape) * sigma, axis=0
        )
        # Remove drift
        jitter_vals -= np.linspace(0, 1, data.shape[0])[:, None] * jitter_vals[-1]
        return data + jitter_vals

    def smooth(self, data: np.ndarray, sigma: float = 2) -> np.ndarray:
        """Apply Gaussian smoothing."""
        smoothed = np.zeros_like(data)
        for col in range(data.shape[1]):
            smoothed[:, col] = gaussian_filter1d(data[:, col], sigma)
        return smoothed

    def mix_segments(
        self, data1: np.ndarray, data2: np.ndarray, n_segments: int = 5
    ) -> np.ndarray:
        """Mix segments from two normal flights."""
        n = min(len(data1), len(data2))

        # Ensure we have enough samples for the requested segments
        if n < n_segments:
            n_segments = max(1, n)

        seg_len = n // n_segments
        if seg_len == 0:
            seg_len = 1
            n_segments = n

        mixed = np.zeros((n, data1.shape[1]))
        for i in range(n_segments):
            start = i * seg_len
            end = start + seg_len if i < n_segments - 1 else n

            # Randomly choose source
            if self.rng.random() > 0.5:
                mixed[start:end] = data1[start:end]
            else:
                mixed[start:end] = data2[start:end]

        return mixed

    def channel_dropout(
        self, data: np.ndarray, dropout_prob: float = 0.1
    ) -> np.ndarray:
        """Randomly zero out some channels (simulates sensor issues)."""
        mask = self.rng.random(data.shape[1]) > dropout_prob
        return data * mask

    def generate_synthetic(
        self, n_synthetic: int = 10, augmentations_per_sample: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic normal flights.

        Args:
            n_synthetic: Number of synthetic flights to generate
            augmentations_per_sample: Number of augmentation combinations per base sample

        Returns:
            List of dictionaries containing synthetic flight data and metadata

        Raises:
            ValueError: If no real normal data has been loaded
        """
        if len(self.real_normals) == 0:
            raise ValueError("No real normal data loaded. Call load_real_normals first.")

        synthetic_flights: List[Dict[str, Any]] = []

        for i in range(n_synthetic):
            # Pick a random base normal
            base_idx = self.rng.integers(len(self.real_normals))
            base_data = self.real_normals[base_idx]['data'].copy()

            # Define augmentations with captured parameters
            # Use partial functions to avoid lambda closure issues
            noise_level = float(self.rng.uniform(0.02, 0.1))
            jitter_sigma = float(self.rng.uniform(0.005, 0.02))
            smooth_sigma = float(self.rng.uniform(1, 4))

            augmentations = [
                ('noise', lambda d, nl=noise_level: self.add_noise(d, nl)),
                ('time_warp', lambda d: self.time_warp(d, (0.85, 1.15))),
                ('amplitude', lambda d: self.amplitude_scale(d, (0.7, 1.3))),
                ('jitter', lambda d, js=jitter_sigma: self.jitter(d, js)),
                ('smooth', lambda d, ss=smooth_sigma: self.smooth(d, ss)),
            ]

            # Apply 2-4 random augmentations
            n_aug = self.rng.integers(2, min(5, len(augmentations) + 1))
            selected = self.rng.choice(len(augmentations), size=n_aug, replace=False)

            synth = base_data.copy()
            applied: List[str] = []
            for idx in selected:
                name, func = augmentations[idx]
                synth = func(synth)
                applied.append(name)

            # Optionally mix with another normal
            if len(self.real_normals) > 1 and self.rng.random() > 0.5:
                other_idx = (base_idx + 1) % len(self.real_normals)
                other_data = self.real_normals[other_idx]['data']
                n_seg = int(self.rng.integers(3, 8))
                synth = self.mix_segments(synth, other_data, n_seg)
                applied.append('mix')

            synthetic_flights.append({
                'data': synth.astype(np.float32),
                'base': base_idx,
                'augmentations': applied
            })

        print(f"Generated {len(synthetic_flights)} synthetic normal flights")
        return synthetic_flights


def extract_features(window: np.ndarray) -> List[float]:
    """
    Extract features from a window of sensor data.

    Args:
        window: 2D array of shape (window_size, n_channels)

    Returns:
        List of extracted features
    """
    feat: List[float] = []
    for col in range(window.shape[1]):
        ch = window[:, col]
        feat.extend([float(ch.mean()), float(ch.std()), float(ch.max() - ch.min())])
        fft = np.abs(np.fft.rfft(ch))
        feat.extend([
            float(fft[1:10].sum()),
            float(fft[10:50].sum()) if len(fft) > 50 else float(fft[10:].sum()),
            float(fft[50:].sum()) if len(fft) > 50 else 0.0,
            float(np.argmax(fft[1:]) + 1) if len(fft) > 1 else 0.0  # +1 to account for slice
        ])
    return feat


def main() -> Tuple[np.ndarray, np.ndarray]:
    """Main function to generate synthetic normal flight data."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic normal flight data from real normal flights"
    )
    parser.add_argument(
        "--bebop-path",
        type=str,
        default="data/PADRE_dataset/Parrot_Bebop_2/Normalized_data/Bebop2_16g_1kdps_normalized_0000.csv",
        help="Path to Bebop normal flight CSV"
    )
    parser.add_argument(
        "--solo-path",
        type=str,
        default="data/PADRE_dataset/3DR_Solo/Normalized_data/extracted/Solo_ACCEL_16g_GYRO_2kdps_BAR_16bit_normalized_0000.csv",
        help="Path to Solo normal flight CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/PADRE_dataset/synthetic_normals",
        help="Output directory for synthetic data"
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=20,
        help="Number of synthetic flights to generate"
    )
    parser.add_argument(
        "--n-columns",
        type=int,
        default=DEFAULT_N_COLUMNS,
        help=f"Number of columns to use from input data (default: {DEFAULT_N_COLUMNS})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="Window size for feature extraction"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride for feature extraction"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SYNTHETIC NORMAL FLIGHT GENERATOR")
    print("=" * 70)

    # Paths to real normal files
    bebop_normal = Path(args.bebop_path)
    solo_normal = Path(args.solo_path)

    # Collect existing files
    input_files = []
    for path in [bebop_normal, solo_normal]:
        if path.exists():
            input_files.append(path)
        else:
            print(f"Warning: File not found, skipping: {path}")

    if not input_files:
        raise FileNotFoundError(
            "No input files found. Please provide valid paths to normal flight data."
        )

    # Initialize generator
    generator = SyntheticNormalGenerator(seed=args.seed, n_columns=args.n_columns)
    generator.load_real_normals(input_files)

    # Generate synthetic normals
    synthetic = generator.generate_synthetic(n_synthetic=args.n_synthetic)

    # Extract windows and features
    window_size = args.window_size
    stride = args.stride

    print(f"\nExtracting features (window={window_size}, stride={stride})...")

    X_synth: List[List[float]] = []
    y_synth: List[int] = []
    sources: List[str] = []

    for i, synth in enumerate(synthetic):
        data = synth['data']
        n_windows = (len(data) - window_size) // stride + 1

        for j in range(n_windows):
            window = data[j * stride: j * stride + window_size]
            if window.shape[0] == window_size:
                X_synth.append(extract_features(window))
                y_synth.append(0)  # Normal
                sources.append(f"synth_{i}")

        print(f"  Synthetic {i}: {n_windows} windows, augmentations: {synth['augmentations']}")

    X_synth_arr = np.array(X_synth)
    y_synth_arr = np.array(y_synth)

    print(f"\nTotal synthetic normal samples: {len(X_synth_arr)}")

    # Save synthetic data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "synthetic_normal_features.npz",
        X=X_synth_arr,
        y=y_synth_arr,
        sources=sources
    )
    print(f"Saved to: {output_dir / 'synthetic_normal_features.npz'}")

    # Also save as individual CSV files for inspection
    for i, synth in enumerate(synthetic[:5]):  # Save first 5 as CSV
        df = pd.DataFrame(synth['data'])
        df.to_csv(output_dir / f"synthetic_normal_{i:02d}.csv", index=False)
    print(f"Saved 5 sample CSVs to: {output_dir}")

    return X_synth_arr, y_synth_arr


if __name__ == "__main__":
    main()
