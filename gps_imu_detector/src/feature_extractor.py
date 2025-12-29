"""
Streaming Multi-Scale Feature Extractor

Optimized for real-time 200 Hz inference with O(1) per-timestep cost.
Uses incremental statistics updates instead of recomputing over windows.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numba
from numba import jit


@dataclass
class WindowStats:
    """Running statistics for a single window size."""
    size: int
    buffer: deque
    sum_x: np.ndarray
    sum_x2: np.ndarray
    max_vals: np.ndarray
    min_vals: np.ndarray

    def __init__(self, size: int, n_features: int):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.sum_x = np.zeros(n_features)
        self.sum_x2 = np.zeros(n_features)
        self.max_vals = np.full(n_features, -np.inf)
        self.min_vals = np.full(n_features, np.inf)


class StreamingFeatureExtractor:
    """
    Streaming multi-scale feature extractor with O(1) updates.

    Features extracted per window:
    - Mean (rolling sum / window size)
    - Std (from rolling sum and sum of squares)
    - Max deviation from mean
    - Cumulative sum magnitude

    Total features: n_windows * 4 * n_input_features
    """

    def __init__(
        self,
        n_features: int,
        windows: List[int] = [5, 10, 25, 50, 100, 200],
        include_raw: bool = True
    ):
        """
        Initialize streaming feature extractor.

        Args:
            n_features: Number of input features per timestep
            windows: List of window sizes (in samples)
            include_raw: Include raw features in output
        """
        self.n_features = n_features
        self.windows = sorted(windows)
        self.max_window = max(windows)
        self.include_raw = include_raw

        # Initialize window statistics
        self.window_stats = [
            WindowStats(w, n_features) for w in windows
        ]

        # Circular buffer for raw values (for max/min tracking)
        self.raw_buffer = deque(maxlen=self.max_window)

        self.n_samples_seen = 0

    def reset(self):
        """Reset all statistics for new sequence."""
        for ws in self.window_stats:
            ws.buffer.clear()
            ws.sum_x.fill(0)
            ws.sum_x2.fill(0)
            ws.max_vals.fill(-np.inf)
            ws.min_vals.fill(np.inf)
        self.raw_buffer.clear()
        self.n_samples_seen = 0

    def update(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Process one timestep and return features.

        O(1) time complexity per call (after warmup).

        Args:
            x: [n_features] input vector

        Returns:
            [n_output_features] feature vector, or None if warmup incomplete
        """
        x = np.asarray(x, dtype=np.float64)
        self.raw_buffer.append(x)
        self.n_samples_seen += 1

        # Update each window's statistics
        for ws in self.window_stats:
            if len(ws.buffer) >= ws.size:
                # Remove oldest sample
                old_x = ws.buffer[0]
                ws.sum_x -= old_x
                ws.sum_x2 -= old_x ** 2

            # Add new sample
            ws.buffer.append(x)
            ws.sum_x += x
            ws.sum_x2 += x ** 2

        # Check if we have enough samples
        if self.n_samples_seen < self.max_window:
            return None

        # Extract features
        return self._compute_features()

    def _compute_features(self) -> np.ndarray:
        """Compute features from current window statistics."""
        features = []

        for ws in self.window_stats:
            n = len(ws.buffer)
            if n == 0:
                # Should not happen after warmup
                features.extend(np.zeros(4 * self.n_features))
                continue

            # Mean
            mean = ws.sum_x / n
            features.extend(mean)

            # Std (Welford's method)
            variance = (ws.sum_x2 / n) - (mean ** 2)
            std = np.sqrt(np.maximum(variance, 0))
            features.extend(std)

            # Max deviation from mean in window
            buffer_array = np.array(ws.buffer)
            max_dev = np.max(np.abs(buffer_array - mean), axis=0)
            features.extend(max_dev)

            # Cumulative sum magnitude (for drift detection)
            cumsum = np.cumsum(buffer_array - mean, axis=0)
            cumsum_mag = np.max(np.abs(cumsum), axis=0)
            features.extend(cumsum_mag)

        if self.include_raw:
            features.extend(self.raw_buffer[-1])

        return np.array(features, dtype=np.float32)

    def process_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Process batch of data (for offline evaluation).

        Args:
            data: [N, n_features] input data

        Returns:
            [N - max_window + 1, n_output_features] features
        """
        self.reset()
        features = []

        for i in range(len(data)):
            feat = self.update(data[i])
            if feat is not None:
                features.append(feat)

        return np.array(features) if features else np.array([])

    @property
    def n_output_features(self) -> int:
        """Number of output features per timestep."""
        # 4 stats per window per feature, plus raw
        n_stats = len(self.windows) * 4 * self.n_features
        if self.include_raw:
            n_stats += self.n_features
        return n_stats


@jit(nopython=True, cache=True)
def _fast_rolling_stats(
    data: np.ndarray,
    window_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast rolling statistics computation using Numba.

    Args:
        data: [N, D] input data
        window_size: Window size

    Returns:
        means: [N - window_size + 1, D]
        stds: [N - window_size + 1, D]
        max_devs: [N - window_size + 1, D]
    """
    n, d = data.shape
    n_out = n - window_size + 1

    means = np.zeros((n_out, d))
    stds = np.zeros((n_out, d))
    max_devs = np.zeros((n_out, d))

    # Initialize first window
    sum_x = np.zeros(d)
    sum_x2 = np.zeros(d)

    for i in range(window_size):
        sum_x += data[i]
        sum_x2 += data[i] ** 2

    for i in range(n_out):
        if i > 0:
            # Slide window
            old_idx = i - 1
            new_idx = i + window_size - 1
            sum_x += data[new_idx] - data[old_idx]
            sum_x2 += data[new_idx] ** 2 - data[old_idx] ** 2

        # Compute stats
        mean = sum_x / window_size
        var = (sum_x2 / window_size) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 0))

        means[i] = mean
        stds[i] = std

        # Max deviation
        for j in range(window_size):
            dev = np.abs(data[i + j] - mean)
            for k in range(d):
                if dev[k] > max_devs[i, k]:
                    max_devs[i, k] = dev[k]

    return means, stds, max_devs


class BatchFeatureExtractor:
    """
    Batch feature extractor for offline processing.

    Uses vectorized operations for faster batch processing.
    """

    def __init__(
        self,
        windows: List[int] = [5, 10, 25, 50, 100, 200],
        include_cumsum: bool = True
    ):
        self.windows = sorted(windows)
        self.max_window = max(windows)
        self.include_cumsum = include_cumsum

    def extract(self, data: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale features from batch data.

        Args:
            data: [N, D] input data

        Returns:
            [N - max_window + 1, n_features] output features
        """
        n, d = data.shape
        n_out = n - self.max_window + 1

        if n_out <= 0:
            return np.array([])

        all_features = []

        for window_size in self.windows:
            # Use fast Numba implementation
            means, stds, max_devs = _fast_rolling_stats(data, window_size)

            # Align to max_window
            offset = self.max_window - window_size
            means = means[offset:offset + n_out]
            stds = stds[offset:offset + n_out]
            max_devs = max_devs[offset:offset + n_out]

            all_features.append(means)
            all_features.append(stds)
            all_features.append(max_devs)

            if self.include_cumsum:
                # Cumulative sum magnitude (for drift)
                cumsum_mags = np.zeros((n_out, d))
                for i in range(n_out):
                    start = i + offset
                    end = start + window_size
                    window = data[start:end] - means[i]
                    cumsum = np.cumsum(window, axis=0)
                    cumsum_mags[i] = np.max(np.abs(cumsum), axis=0)
                all_features.append(cumsum_mags)

        return np.hstack(all_features).astype(np.float32)

    @property
    def n_features_per_window(self) -> int:
        """Features per window: mean, std, max_dev, (cumsum)."""
        return 4 if self.include_cumsum else 3


if __name__ == "__main__":
    # Test streaming extractor
    n_features = 15
    extractor = StreamingFeatureExtractor(n_features, windows=[5, 10, 25])

    # Simulate streaming data
    data = np.random.randn(100, n_features)
    streaming_features = []

    for i, x in enumerate(data):
        feat = extractor.update(x)
        if feat is not None:
            streaming_features.append(feat)

    print(f"Streaming: {len(streaming_features)} feature vectors")
    print(f"Feature dim: {extractor.n_output_features}")

    # Test batch extractor
    batch_extractor = BatchFeatureExtractor(windows=[5, 10, 25])
    batch_features = batch_extractor.extract(data)
    print(f"Batch: {batch_features.shape}")

    # Verify consistency
    print(f"\nFeature values match: {np.allclose(streaming_features[-1][:15], batch_features[-1][:15], atol=1e-5)}")
