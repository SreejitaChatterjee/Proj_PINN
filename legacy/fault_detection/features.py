"""
Feature Extraction Module
=========================

Advanced feature extraction for motor fault detection:
- FFT (Frequency domain)
- Wavelet (DWT - multi-resolution)
- Statistical (time-domain statistics)
- Cross-motor (correlation between motors)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Union
from scipy import signal
from scipy.stats import kurtosis, skew
import warnings


class FFTFeatureExtractor(nn.Module):
    """
    Fast Fourier Transform feature extraction.

    Extracts frequency-domain features that capture motor vibration signatures.
    Motor faults create distinct frequency patterns due to imbalance/damage.
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        n_freq_bins: int = 64,
        sample_rate: float = 500.0,
        log_scale: bool = True
    ):
        """
        Args:
            n_fft: FFT window size
            hop_length: Hop between windows (for STFT)
            n_freq_bins: Number of frequency bins to keep
            sample_rate: Sampling rate in Hz
            log_scale: Apply log scaling to magnitude
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq_bins = n_freq_bins
        self.sample_rate = sample_rate
        self.log_scale = log_scale

        # Precompute frequency bins
        self.register_buffer(
            'freq_bins',
            torch.linspace(0, sample_rate / 2, n_freq_bins)
        )

        # Hann window for STFT
        self.register_buffer(
            'window',
            torch.hann_window(n_fft)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract FFT features.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            FFT magnitude features (batch, channels, n_freq_bins)
        """
        batch, channels, time = x.shape

        # Compute FFT for each channel
        # Pad to n_fft if needed
        if time < self.n_fft:
            x = torch.nn.functional.pad(x, (0, self.n_fft - time))

        # Apply window
        x_windowed = x * self.window[:x.shape[-1]]

        # FFT
        fft = torch.fft.rfft(x_windowed, n=self.n_fft, dim=-1)
        magnitude = torch.abs(fft)

        # Keep only first n_freq_bins
        magnitude = magnitude[..., :self.n_freq_bins]

        # Log scale
        if self.log_scale:
            magnitude = torch.log1p(magnitude)

        return magnitude

    def extract_spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract additional spectral features beyond raw FFT.

        Returns:
            - Spectral centroid
            - Spectral bandwidth
            - Spectral rolloff
            - Peak frequency
            - Spectral entropy
        """
        magnitude = self.forward(x)
        batch, channels, freq = magnitude.shape

        # Normalize magnitude
        mag_sum = magnitude.sum(dim=-1, keepdim=True) + 1e-8
        mag_norm = magnitude / mag_sum

        # Frequency axis
        freqs = torch.linspace(0, self.sample_rate / 2, freq, device=x.device)

        # Spectral centroid: weighted mean of frequencies
        centroid = (mag_norm * freqs).sum(dim=-1)

        # Spectral bandwidth: weighted std of frequencies
        bandwidth = torch.sqrt(
            (mag_norm * (freqs - centroid.unsqueeze(-1))**2).sum(dim=-1)
        )

        # Spectral rolloff: frequency below which 85% of energy is contained
        cumsum = torch.cumsum(mag_norm, dim=-1)
        rolloff_idx = (cumsum < 0.85).sum(dim=-1)
        rolloff = freqs[rolloff_idx.clamp(max=freq-1)]

        # Peak frequency
        peak_idx = magnitude.argmax(dim=-1)
        peak_freq = freqs[peak_idx]

        # Spectral entropy
        entropy = -(mag_norm * torch.log(mag_norm + 1e-8)).sum(dim=-1)

        # Stack features: (batch, channels, 5)
        features = torch.stack([
            centroid, bandwidth, rolloff, peak_freq, entropy
        ], dim=-1)

        return features


class WaveletFeatureExtractor(nn.Module):
    """
    Discrete Wavelet Transform (DWT) feature extraction.

    Multi-resolution analysis captures both time and frequency information.
    Excellent for detecting transient fault signatures.
    """

    def __init__(
        self,
        wavelet: str = 'db4',
        levels: int = 4,
        mode: str = 'symmetric'
    ):
        """
        Args:
            wavelet: Wavelet type (db4, haar, sym4, etc.)
            levels: Number of decomposition levels
            mode: Signal extension mode
        """
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.mode = mode

        # Create wavelet filters
        self._create_filters(wavelet)

    def _create_filters(self, wavelet: str):
        """Create decomposition and reconstruction filters."""
        try:
            import pywt
            w = pywt.Wavelet(wavelet)
            # Decomposition filters
            lo_d = torch.FloatTensor(w.dec_lo)
            hi_d = torch.FloatTensor(w.dec_hi)
        except ImportError:
            # Fallback: Haar wavelet
            warnings.warn("PyWavelets not installed, using Haar wavelet")
            lo_d = torch.FloatTensor([0.7071, 0.7071])
            hi_d = torch.FloatTensor([0.7071, -0.7071])

        self.register_buffer('lo_d', lo_d)
        self.register_buffer('hi_d', hi_d)
        self.filter_len = len(lo_d)

    def _dwt_1d(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single level DWT."""
        # Pad signal
        pad_len = self.filter_len - 1
        x_padded = torch.nn.functional.pad(x, (pad_len, pad_len), mode='reflect')

        # Convolve with filters
        # Reshape for conv1d: (batch*channels, 1, time)
        batch, channels, time = x.shape
        x_flat = x_padded.view(batch * channels, 1, -1)

        lo_filter = self.lo_d.flip(0).view(1, 1, -1)
        hi_filter = self.hi_d.flip(0).view(1, 1, -1)

        # Approximation and detail coefficients
        cA = torch.nn.functional.conv1d(x_flat, lo_filter)[:, :, ::2]
        cD = torch.nn.functional.conv1d(x_flat, hi_filter)[:, :, ::2]

        # Reshape back
        cA = cA.view(batch, channels, -1)
        cD = cD.view(batch, channels, -1)

        return cA, cD

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Multi-level DWT decomposition.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            List of coefficient tensors [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        """
        coeffs = []
        approx = x

        for level in range(self.levels):
            approx, detail = self._dwt_1d(approx)
            coeffs.append(detail)

        coeffs.append(approx)
        coeffs.reverse()  # [cA, cD_1, cD_2, ...]

        return coeffs

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract statistical features from wavelet coefficients.

        Returns features per level: energy, mean, std, max
        """
        coeffs = self.forward(x)
        batch, channels, _ = x.shape

        features = []
        for coeff in coeffs:
            # Energy
            energy = (coeff ** 2).mean(dim=-1)
            # Mean absolute value
            mean_abs = coeff.abs().mean(dim=-1)
            # Standard deviation
            std = coeff.std(dim=-1)
            # Max absolute value
            max_abs = coeff.abs().max(dim=-1)[0]

            level_features = torch.stack([energy, mean_abs, std, max_abs], dim=-1)
            features.append(level_features)

        # Concatenate all levels: (batch, channels, levels * 4)
        features = torch.cat(features, dim=-1)
        return features


class StatisticalFeatureExtractor(nn.Module):
    """
    Time-domain statistical feature extraction.

    Classical features used in vibration analysis and fault detection.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract statistical features.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Statistical features (batch, channels, n_features)
        """
        # Basic statistics
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)
        var = x.var(dim=-1)

        # RMS (Root Mean Square) - proportional to vibration energy
        rms = torch.sqrt((x ** 2).mean(dim=-1))

        # Peak values
        peak = x.max(dim=-1)[0]
        peak_neg = x.min(dim=-1)[0]
        peak_to_peak = peak - peak_neg

        # Crest factor = peak / RMS (detects impulsive signals)
        crest_factor = peak.abs() / (rms + self.eps)

        # Shape factor = RMS / mean_abs
        mean_abs = x.abs().mean(dim=-1)
        shape_factor = rms / (mean_abs + self.eps)

        # Impulse factor = peak / mean_abs
        impulse_factor = peak.abs() / (mean_abs + self.eps)

        # Clearance factor = peak / (mean of sqrt(abs))
        sqrt_mean = torch.sqrt(x.abs()).mean(dim=-1)
        clearance_factor = peak.abs() / (sqrt_mean ** 2 + self.eps)

        # Skewness (asymmetry)
        centered = x - mean.unsqueeze(-1)
        skewness = (centered ** 3).mean(dim=-1) / (std ** 3 + self.eps)

        # Kurtosis (peakedness) - high for impulsive faults
        kurtosis_val = (centered ** 4).mean(dim=-1) / (std ** 4 + self.eps) - 3

        # Zero crossing rate
        signs = torch.sign(x)
        sign_changes = (signs[:, :, 1:] != signs[:, :, :-1]).float()
        zcr = sign_changes.mean(dim=-1)

        # Mean crossing rate
        centered_signs = torch.sign(centered)
        centered_changes = (centered_signs[:, :, 1:] != centered_signs[:, :, :-1]).float()
        mcr = centered_changes.mean(dim=-1)

        # Margin factor
        margin_factor = peak.abs() / (sqrt_mean + self.eps)

        # Energy
        energy = (x ** 2).sum(dim=-1)

        # Stack all features
        features = torch.stack([
            mean, std, var, rms, peak, peak_neg, peak_to_peak,
            crest_factor, shape_factor, impulse_factor, clearance_factor,
            skewness, kurtosis_val, zcr, mcr, margin_factor, energy
        ], dim=-1)

        return features


class CrossMotorFeatureExtractor(nn.Module):
    """
    Cross-motor correlation feature extraction.

    Compares signals between motors to detect asymmetry caused by faults.
    A healthy drone has similar vibration patterns across all motors.
    """

    MOTOR_CHANNELS = {
        'A': [0, 1, 2, 3, 4, 5],    # A_aX, A_aY, A_aZ, A_gX, A_gY, A_gZ
        'B': [6, 7, 8, 9, 10, 11],
        'C': [12, 13, 14, 15, 16, 17],
        'D': [18, 19, 20, 21, 22, 23]
    }

    def __init__(self):
        super().__init__()
        self.motors = ['A', 'B', 'C', 'D']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract cross-motor features.

        Args:
            x: Input tensor (batch, 24, time)

        Returns:
            Cross-motor features (batch, n_features)
        """
        batch = x.shape[0]
        features = []

        # Extract per-motor signals
        motor_signals = {}
        for motor in self.motors:
            channels = self.MOTOR_CHANNELS[motor]
            motor_signals[motor] = x[:, channels, :]  # (batch, 6, time)

        # Compute per-motor energy
        motor_energy = {}
        for motor in self.motors:
            energy = (motor_signals[motor] ** 2).mean(dim=(1, 2))  # (batch,)
            motor_energy[motor] = energy

        # Energy imbalance between motors
        energies = torch.stack([motor_energy[m] for m in self.motors], dim=1)  # (batch, 4)
        energy_std = energies.std(dim=1)  # Imbalance measure
        energy_max_ratio = energies.max(dim=1)[0] / (energies.min(dim=1)[0] + 1e-8)

        features.extend([energy_std, energy_max_ratio])

        # Pairwise correlation between motors
        for i, m1 in enumerate(self.motors):
            for m2 in self.motors[i+1:]:
                # Flatten motor signals
                sig1 = motor_signals[m1].view(batch, -1)  # (batch, 6*time)
                sig2 = motor_signals[m2].view(batch, -1)

                # Correlation coefficient
                sig1_centered = sig1 - sig1.mean(dim=1, keepdim=True)
                sig2_centered = sig2 - sig2.mean(dim=1, keepdim=True)

                corr = (sig1_centered * sig2_centered).sum(dim=1) / (
                    torch.sqrt((sig1_centered**2).sum(dim=1) * (sig2_centered**2).sum(dim=1)) + 1e-8
                )
                features.append(corr)

        # Opposite motor pairs (A-C, B-D should be similar in healthy drone)
        opposite_pairs = [('A', 'C'), ('B', 'D')]
        for m1, m2 in opposite_pairs:
            diff = (motor_signals[m1] - motor_signals[m2]).abs().mean(dim=(1, 2))
            features.append(diff)

        # Adjacent motor differences (should detect single motor faults)
        adjacent_pairs = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
        for m1, m2 in adjacent_pairs:
            diff = (motor_signals[m1] - motor_signals[m2]).abs().mean(dim=(1, 2))
            features.append(diff)

        # Stack all features: (batch, n_features)
        features = torch.stack(features, dim=1)
        return features


class CombinedFeatureExtractor(nn.Module):
    """
    Combined feature extraction using all methods.

    Produces a comprehensive feature vector for fault detection.
    """

    def __init__(
        self,
        use_fft: bool = True,
        use_wavelet: bool = True,
        use_stats: bool = True,
        use_cross_motor: bool = True,
        n_freq_bins: int = 32,
        wavelet_levels: int = 3
    ):
        super().__init__()

        self.use_fft = use_fft
        self.use_wavelet = use_wavelet
        self.use_stats = use_stats
        self.use_cross_motor = use_cross_motor

        if use_fft:
            self.fft = FFTFeatureExtractor(n_freq_bins=n_freq_bins)
        if use_wavelet:
            self.wavelet = WaveletFeatureExtractor(levels=wavelet_levels)
        if use_stats:
            self.stats = StatisticalFeatureExtractor()
        if use_cross_motor:
            self.cross_motor = CrossMotorFeatureExtractor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract combined features.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Combined feature tensor
        """
        batch, channels, time = x.shape
        features = []

        if self.use_fft:
            fft_feats = self.fft.forward(x)  # (batch, channels, freq)
            fft_spectral = self.fft.extract_spectral_features(x)  # (batch, channels, 5)
            features.append(fft_feats.view(batch, -1))
            features.append(fft_spectral.view(batch, -1))

        if self.use_wavelet:
            wavelet_feats = self.wavelet.extract_features(x)  # (batch, channels, levels*4)
            features.append(wavelet_feats.view(batch, -1))

        if self.use_stats:
            stat_feats = self.stats(x)  # (batch, channels, 17)
            features.append(stat_feats.view(batch, -1))

        if self.use_cross_motor:
            cross_feats = self.cross_motor(x)  # (batch, n_cross_features)
            features.append(cross_feats)

        # Concatenate all features
        combined = torch.cat(features, dim=1)
        return combined

    def get_feature_dim(self, n_channels: int = 24, time_steps: int = 256) -> int:
        """Calculate output feature dimension."""
        # Create dummy input to get dimension
        dummy = torch.zeros(1, n_channels, time_steps)
        with torch.no_grad():
            out = self.forward(dummy)
        return out.shape[1]
