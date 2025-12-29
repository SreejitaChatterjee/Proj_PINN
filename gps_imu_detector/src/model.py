"""
Compact ML Detector Model

Architecture: 1D CNN + GRU
Optimized for CPU inference with small channel counts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class CNNGRUDetector(nn.Module):
    """
    1D CNN + GRU detector for temporal anomaly detection.

    Architecture:
    - 1D CNN for local pattern extraction
    - GRU for temporal modeling
    - Dense layers for classification

    Optimized for CPU:
    - Small channel counts (32, 64)
    - Single GRU layer with hidden_size 64
    - Total params < 100K
    """

    def __init__(
        self,
        input_dim: int,
        cnn_channels: Tuple[int, ...] = (32, 64),
        cnn_kernel_size: int = 3,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 1,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.gru_hidden_size = gru_hidden_size

        # 1D CNN layers
        cnn_layers = []
        in_channels = input_dim

        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, cnn_kernel_size, padding=cnn_kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # GRU
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size // 2, output_dim)
        )

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, input_dim] or [batch, input_dim] for single step
            hidden: [num_layers, batch, hidden_size] GRU hidden state

        Returns:
            output: [batch, seq_len, 1] or [batch, 1] logits
            hidden: Updated hidden state
        """
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]

        batch_size, seq_len, _ = x.shape

        # CNN expects [batch, channels, length]
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        x = self.cnn(x)
        x = x.transpose(1, 2)  # [batch, seq_len, cnn_out]

        # GRU
        if hidden is None:
            hidden = torch.zeros(
                self.gru.num_layers, batch_size, self.gru_hidden_size,
                device=x.device, dtype=x.dtype
            )

        x, hidden = self.gru(x, hidden)

        # Output
        output = self.fc(x)

        if single_step:
            output = output.squeeze(1)

        return output, hidden

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get anomaly probability (sigmoid of logits)."""
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)


class StreamingDetector:
    """
    Streaming wrapper for real-time inference.

    Maintains GRU hidden state across timesteps.
    """

    def __init__(self, model: CNNGRUDetector, device: str = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.hidden = None

    def reset(self):
        """Reset hidden state for new sequence."""
        self.hidden = None

    def predict_step(self, features: np.ndarray) -> float:
        """
        Predict anomaly score for single timestep.

        Args:
            features: [input_dim] feature vector

        Returns:
            Anomaly score (0-1)
        """
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32, device=self.device)
            x = x.unsqueeze(0)  # [1, input_dim]

            logits, self.hidden = self.model(x, self.hidden)
            prob = torch.sigmoid(logits).item()

        return prob

    def predict_sequence(self, features: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores for entire sequence.

        Args:
            features: [N, input_dim] feature sequence

        Returns:
            [N] anomaly scores
        """
        self.reset()

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32, device=self.device)
            x = x.unsqueeze(0)  # [1, N, input_dim]

            logits, _ = self.model(x, None)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        return probs.squeeze()


class TemporalDetector(nn.Module):
    """
    Temporal anomaly detector using attention over sliding window.

    Lighter alternative to GRU for some use cases.
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int = 50,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.window_size = window_size

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            [batch, seq_len, 1] anomaly logits
        """
        # Project input
        x = self.input_proj(x)

        # Self-attention
        x, _ = self.attention(x, x, x)

        # Output
        return self.fc(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    input_dim = 100  # Multi-scale features
    batch_size = 32
    seq_len = 100

    model = CNNGRUDetector(input_dim=input_dim)
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    output, hidden = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {hidden.shape}")

    # Test streaming
    detector = StreamingDetector(model)

    features = np.random.randn(seq_len, input_dim).astype(np.float32)

    # Single-step inference
    import time
    start = time.time()
    for i in range(seq_len):
        score = detector.predict_step(features[i])
    elapsed = (time.time() - start) * 1000
    print(f"\nStreaming inference: {elapsed/seq_len:.2f} ms per step")

    # Batch inference
    detector.reset()
    start = time.time()
    scores = detector.predict_sequence(features)
    elapsed = (time.time() - start) * 1000
    print(f"Batch inference: {elapsed:.2f} ms total, {elapsed/seq_len:.2f} ms per step")
