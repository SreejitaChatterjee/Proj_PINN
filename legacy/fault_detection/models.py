"""
Neural Network Models for Fault Detection
==========================================

Advanced architectures:
- MotorFaultCNN: 1D CNN with attention
- TransformerDetector: Transformer encoder for time series
- MultiScaleCNN: Parallel convolutions with different kernel sizes
- TCNDetector: Temporal Convolutional Network with dilated convolutions
- EnsembleDetector: Combines multiple models for robust predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import math


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding),
            nn.BatchNorm1d(out_channels),
        )

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1, stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.conv(x) + self.residual(x))


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attention(x).unsqueeze(-1)
        return x * attn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


# =============================================================================
# Model 1: Enhanced CNN (baseline)
# =============================================================================

class MotorFaultCNN(nn.Module):
    """
    1D CNN for motor fault detection with channel attention.
    """

    def __init__(
        self,
        n_input_channels: int = 24,
        n_classes: int = 2,
        base_channels: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        self.n_classes = n_classes
        self.model_name = 'cnn'

        self.features = nn.Sequential(
            ConvBlock(n_input_channels, base_channels, kernel_size=15),
            ChannelAttention(base_channels),
            nn.MaxPool1d(2),

            ConvBlock(base_channels, base_channels * 2, kernel_size=11),
            ChannelAttention(base_channels * 2),
            nn.MaxPool1d(2),

            ConvBlock(base_channels * 2, base_channels * 4, kernel_size=7),
            ChannelAttention(base_channels * 4),
            nn.MaxPool1d(2),

            ConvBlock(base_channels * 4, base_channels * 8, kernel_size=5),
            ChannelAttention(base_channels * 8),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(base_channels * 4, n_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        return self.features(x).flatten(1)


# =============================================================================
# Model 2: Transformer Encoder
# =============================================================================

class TransformerDetector(nn.Module):
    """
    Transformer encoder for time series fault detection.

    Uses self-attention to capture long-range dependencies in sensor signals.
    """

    def __init__(
        self,
        n_input_channels: int = 24,
        n_classes: int = 2,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 512
    ):
        super().__init__()
        self.n_classes = n_classes
        self.model_name = 'transformer'
        self.d_model = d_model

        # Input embedding: project channels to d_model
        self.input_proj = nn.Linear(n_input_channels, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Class logits (batch, n_classes)
        """
        batch = x.shape[0]

        # Transpose to (batch, time, channels)
        x = x.transpose(1, 2)

        # Project to d_model
        x = self.input_proj(x)  # (batch, time, d_model)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)

        # Use CLS token for classification
        cls_output = x[:, 0]

        return self.classifier(cls_output)

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention weights for visualization."""
        batch = x.shape[0]
        x = x.transpose(1, 2)
        x = self.input_proj(x)

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)

        attention_weights = []
        for layer in self.transformer.layers:
            # Get attention weights from self-attention
            attn_output, attn_weights = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=True
            )
            attention_weights.append(attn_weights)
            x = layer(x)

        return attention_weights


# =============================================================================
# Model 3: Multi-Scale CNN
# =============================================================================

class MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN with parallel convolutions.

    Different kernel sizes capture features at different temporal scales:
    - Small kernels: high-frequency transients
    - Large kernels: low-frequency trends
    """

    def __init__(
        self,
        n_input_channels: int = 24,
        n_classes: int = 2,
        base_channels: int = 32,
        kernel_sizes: List[int] = [3, 7, 15, 31],
        dropout: float = 0.3
    ):
        super().__init__()
        self.n_classes = n_classes
        self.model_name = 'multiscale_cnn'

        # Parallel branches with different kernel sizes
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(n_input_channels, base_channels, ks, padding=ks//2),
                nn.BatchNorm1d(base_channels),
                nn.GELU(),
                nn.Conv1d(base_channels, base_channels * 2, ks, padding=ks//2),
                nn.BatchNorm1d(base_channels * 2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(16)
            )
            self.branches.append(branch)

        # Merge features from all branches
        merged_channels = base_channels * 2 * len(kernel_sizes)

        self.merge = nn.Sequential(
            nn.Conv1d(merged_channels, base_channels * 4, 3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.GELU(),
            ChannelAttention(base_channels * 4),
            nn.Conv1d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm1d(base_channels * 8),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(base_channels * 4, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process through all branches
        branch_outputs = [branch(x) for branch in self.branches]

        # Concatenate along channel dimension
        merged = torch.cat(branch_outputs, dim=1)

        # Merge and classify
        features = self.merge(merged)
        return self.classifier(features)


# =============================================================================
# Model 4: Temporal Convolutional Network (TCN)
# =============================================================================

class TCNBlock(nn.Module):
    """TCN residual block with dilated causal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv
        out = self.conv1(x)
        out = out[:, :, :-self.padding]  # Causal: remove future
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout1(out)

        # Second conv
        out = self.conv2(out)
        out = out[:, :, :-self.padding]
        out = self.bn2(out)
        out = F.gelu(out)
        out = self.dropout2(out)

        # Residual
        res = self.residual(x)
        # Match temporal dimension
        if res.shape[-1] > out.shape[-1]:
            res = res[:, :, :out.shape[-1]]

        return F.gelu(out + res)


class TCNDetector(nn.Module):
    """
    Temporal Convolutional Network for fault detection.

    Uses dilated causal convolutions for large receptive field
    while maintaining temporal causality.
    """

    def __init__(
        self,
        n_input_channels: int = 24,
        n_classes: int = 2,
        n_channels: List[int] = [64, 128, 256, 512],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_classes = n_classes
        self.model_name = 'tcn'

        layers = []
        in_ch = n_input_channels

        for i, out_ch in enumerate(n_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_channels[-1], n_channels[-1] // 2),
            nn.GELU(),
            nn.Linear(n_channels[-1] // 2, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tcn(x)
        return self.classifier(features)


# =============================================================================
# Model 5: Ensemble
# =============================================================================

class EnsembleDetector(nn.Module):
    """
    Ensemble model combining multiple architectures.

    Aggregates predictions from CNN, Transformer, Multi-scale CNN, and TCN
    for more robust fault detection.
    """

    def __init__(
        self,
        n_input_channels: int = 24,
        n_classes: int = 2,
        models: Optional[List[str]] = None,
        aggregation: str = 'weighted_avg',  # 'avg', 'weighted_avg', 'learned'
        dropout: float = 0.3
    ):
        super().__init__()
        self.n_classes = n_classes
        self.model_name = 'ensemble'
        self.aggregation = aggregation

        if models is None:
            models = ['cnn', 'transformer', 'multiscale', 'tcn']

        self.model_names = models
        self.models = nn.ModuleDict()

        # Create sub-models
        if 'cnn' in models:
            self.models['cnn'] = MotorFaultCNN(n_input_channels, n_classes, dropout=dropout)
        if 'transformer' in models:
            self.models['transformer'] = TransformerDetector(n_input_channels, n_classes, dropout=dropout)
        if 'multiscale' in models:
            self.models['multiscale'] = MultiScaleCNN(n_input_channels, n_classes, dropout=dropout)
        if 'tcn' in models:
            self.models['tcn'] = TCNDetector(n_input_channels, n_classes, dropout=dropout)

        n_models = len(self.models)

        # Learnable weights for weighted averaging
        if aggregation == 'weighted_avg':
            self.model_weights = nn.Parameter(torch.ones(n_models) / n_models)
        elif aggregation == 'learned':
            # Learned fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(n_classes * n_models, n_classes * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(n_classes * 2, n_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Aggregated class logits (batch, n_classes)
        """
        # Get predictions from all models
        predictions = []
        for name, model in self.models.items():
            pred = model(x)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=1)  # (batch, n_models, n_classes)

        # Aggregate
        if self.aggregation == 'avg':
            return predictions.mean(dim=1)

        elif self.aggregation == 'weighted_avg':
            weights = F.softmax(self.model_weights, dim=0)
            return (predictions * weights.view(1, -1, 1)).sum(dim=1)

        elif self.aggregation == 'learned':
            # Flatten and fuse
            flat = predictions.view(predictions.shape[0], -1)
            return self.fusion(flat)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def get_individual_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predictions from each sub-model."""
        return {name: model(x) for name, model in self.models.items()}

    def get_model_weights(self) -> Dict[str, float]:
        """Get normalized model weights (for weighted_avg aggregation)."""
        if self.aggregation != 'weighted_avg':
            return {name: 1.0 / len(self.models) for name in self.models}

        weights = F.softmax(self.model_weights, dim=0)
        return {name: w.item() for name, w in zip(self.models.keys(), weights)}


# =============================================================================
# Feature-based Classifier (for extracted features)
# =============================================================================

class FeatureClassifier(nn.Module):
    """
    MLP classifier for pre-extracted features.

    Use with CombinedFeatureExtractor for feature-based classification.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3
    ):
        super().__init__()
        self.n_classes = n_classes
        self.model_name = 'feature_mlp'

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, n_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
