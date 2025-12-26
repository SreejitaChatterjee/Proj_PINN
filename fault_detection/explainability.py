"""
Explainability Module
=====================

Interpretability tools for fault detection:
- GradCAM-1D: Which time regions trigger detection
- Sensor Importance: Which sensors matter most
- Attention Visualization: What the model focuses on
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# GradCAM for 1D Time Series
# =============================================================================

class GradCAM1D:
    """
    Gradient-weighted Class Activation Mapping for 1D signals.

    Visualizes which temporal regions are most important for predictions.
    Adapted from the original GradCAM for images.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", 2017
    """

    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """
        Args:
            model: The neural network model
            target_layer: Name of the layer to compute CAM for.
                         If None, uses the last convolutional layer.
        """
        self.model = model
        self.model.eval()

        # Find target layer
        if target_layer is None:
            target_layer = self._find_last_conv_layer()

        self.target_layer = self._get_layer(target_layer)
        self.target_layer_name = target_layer

        # Storage for activations and gradients
        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

    def _find_last_conv_layer(self) -> str:
        """Find the last convolutional layer in the model."""
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d):
                last_conv = name
        if last_conv is None:
            raise ValueError("No Conv1d layer found in model")
        return last_conv

    def _get_layer(self, name: str) -> nn.Module:
        """Get layer by name."""
        module = self.model
        for part in name.split('.'):
            module = getattr(module, part)
        return module

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """Remove registered hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        relu: bool = True
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute GradCAM heatmap.

        Args:
            x: Input tensor (batch, channels, time)
            target_class: Class to explain (None = predicted class)
            relu: Apply ReLU to heatmap

        Returns:
            heatmap: (time,) importance scores
            predicted_class: Model's prediction
            confidence: Prediction confidence
        """
        self.model.eval()

        # Forward pass
        output = self.model(x)
        probs = F.softmax(output, dim=-1)
        predicted_class = probs.argmax(dim=-1).item()
        confidence = probs.max(dim=-1)[0].item()

        if target_class is None:
            target_class = predicted_class

        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute weights: global average pooling of gradients
        weights = self.gradients.mean(dim=-1, keepdim=True)  # (1, C, 1)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1)  # (1, T)

        if relu:
            cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze(0).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Upsample to input size
        cam_upsampled = np.interp(
            np.linspace(0, 1, x.shape[-1]),
            np.linspace(0, 1, len(cam)),
            cam
        )

        return cam_upsampled, predicted_class, confidence

    def visualize(
        self,
        x: torch.Tensor,
        channel_idx: int = 0,
        target_class: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 4),
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize GradCAM overlay on signal.

        Args:
            x: Input tensor
            channel_idx: Which channel to display
            target_class: Class to explain
            figsize: Figure size
            title: Plot title

        Returns:
            matplotlib Figure
        """
        heatmap, pred_class, conf = self(x, target_class)
        signal = x[0, channel_idx].cpu().numpy()

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Signal with heatmap overlay
        time_axis = np.arange(len(signal))

        axes[0].plot(signal, 'b-', linewidth=0.5, alpha=0.7)
        axes[0].fill_between(
            time_axis,
            signal.min(),
            signal.max(),
            where=heatmap > 0.5,
            alpha=0.3,
            color='red',
            label='Important regions'
        )
        axes[0].set_ylabel('Signal')
        axes[0].legend()

        # Heatmap
        axes[1].imshow(
            heatmap.reshape(1, -1),
            aspect='auto',
            cmap='hot',
            extent=[0, len(signal), 0, 1]
        )
        axes[1].set_ylabel('Importance')
        axes[1].set_xlabel('Time step')

        if title is None:
            title = f'GradCAM: Class {pred_class} (conf: {conf:.1%})'
        fig.suptitle(title)
        plt.tight_layout()

        return fig


# =============================================================================
# Sensor Importance Analysis
# =============================================================================

class SensorImportance:
    """
    Analyze which sensors (channels) are most important for predictions.

    Methods:
    - Gradient-based: Sensitivity of output to input channels
    - Occlusion: Drop each channel and measure accuracy change
    - Permutation: Shuffle each channel and measure accuracy change
    """

    SENSOR_NAMES = [
        'A_aX', 'A_aY', 'A_aZ', 'A_gX', 'A_gY', 'A_gZ',
        'B_aX', 'B_aY', 'B_aZ', 'B_gX', 'B_gY', 'B_gZ',
        'C_aX', 'C_aY', 'C_aZ', 'C_gX', 'C_gY', 'C_gZ',
        'D_aX', 'D_aY', 'D_aZ', 'D_gX', 'D_gY', 'D_gZ'
    ]

    MOTOR_GROUPS = {
        'Motor A': [0, 1, 2, 3, 4, 5],
        'Motor B': [6, 7, 8, 9, 10, 11],
        'Motor C': [12, 13, 14, 15, 16, 17],
        'Motor D': [18, 19, 20, 21, 22, 23]
    }

    SENSOR_TYPE_GROUPS = {
        'Accel X': [0, 6, 12, 18],
        'Accel Y': [1, 7, 13, 19],
        'Accel Z': [2, 8, 14, 20],
        'Gyro X': [3, 9, 15, 21],
        'Gyro Y': [4, 10, 16, 22],
        'Gyro Z': [5, 11, 17, 23]
    }

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def gradient_importance(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute gradient-based importance for each channel.

        Args:
            x: Input tensor (batch, channels, time)
            target_class: Target class (None = predicted)

        Returns:
            importance: (channels,) importance scores
        """
        x = x.clone().requires_grad_(True)

        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=-1)

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1)
        output.backward(gradient=one_hot)

        # Importance = mean absolute gradient per channel
        gradients = x.grad.abs()
        importance = gradients.mean(dim=(0, 2)).cpu().numpy()

        # Normalize
        importance = importance / (importance.sum() + 1e-8)

        return importance

    def occlusion_importance(
        self,
        x: torch.Tensor,
        baseline: str = 'zero'  # 'zero', 'mean', 'noise'
    ) -> np.ndarray:
        """
        Compute importance by occluding each channel.

        Args:
            x: Input tensor
            baseline: Replacement value for occluded channel

        Returns:
            importance: (channels,) importance scores
        """
        n_channels = x.shape[1]
        importance = np.zeros(n_channels)

        # Get original prediction
        with torch.no_grad():
            original_output = self.model(x)
            original_prob = F.softmax(original_output, dim=-1)
            original_class = original_prob.argmax(dim=-1)
            original_confidence = original_prob.max(dim=-1)[0]

        # Occlude each channel
        for ch in range(n_channels):
            x_occluded = x.clone()

            if baseline == 'zero':
                x_occluded[:, ch, :] = 0
            elif baseline == 'mean':
                x_occluded[:, ch, :] = x[:, ch, :].mean()
            elif baseline == 'noise':
                x_occluded[:, ch, :] = torch.randn_like(x[:, ch, :])

            with torch.no_grad():
                occluded_output = self.model(x_occluded)
                occluded_prob = F.softmax(occluded_output, dim=-1)
                occluded_confidence = occluded_prob.gather(
                    1, original_class.unsqueeze(1)
                ).squeeze()

            # Importance = drop in confidence for original class
            importance[ch] = (original_confidence - occluded_confidence).item()

        # Normalize (positive = important)
        importance = np.maximum(importance, 0)
        importance = importance / (importance.sum() + 1e-8)

        return importance

    def permutation_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        n_permutations: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Compute permutation importance on a dataset.

        Args:
            dataloader: Data loader
            device: Device to use
            n_permutations: Number of permutations per channel

        Returns:
            Dictionary with importance scores and statistics
        """
        n_channels = 24
        importance_scores = np.zeros((n_channels, n_permutations))

        # Get baseline accuracy
        baseline_correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                output = self.model(data)
                preds = output.argmax(dim=-1)
                baseline_correct += (preds == labels).sum().item()
                total += len(labels)

        baseline_accuracy = baseline_correct / total

        # Permute each channel
        for ch in range(n_channels):
            for perm in range(n_permutations):
                perm_correct = 0

                with torch.no_grad():
                    for data, labels in dataloader:
                        data, labels = data.to(device), labels.to(device)

                        # Permute channel across batch
                        data_perm = data.clone()
                        perm_idx = torch.randperm(data.shape[0])
                        data_perm[:, ch, :] = data[perm_idx, ch, :]

                        output = self.model(data_perm)
                        preds = output.argmax(dim=-1)
                        perm_correct += (preds == labels).sum().item()

                perm_accuracy = perm_correct / total
                importance_scores[ch, perm] = baseline_accuracy - perm_accuracy

        return {
            'importance': importance_scores.mean(axis=1),
            'std': importance_scores.std(axis=1),
            'baseline_accuracy': baseline_accuracy
        }

    def get_grouped_importance(
        self,
        importance: np.ndarray,
        grouping: str = 'motor'  # 'motor' or 'sensor_type'
    ) -> Dict[str, float]:
        """
        Aggregate importance by motor or sensor type.

        Args:
            importance: Per-channel importance scores
            grouping: How to group channels

        Returns:
            Dictionary of group -> importance
        """
        groups = self.MOTOR_GROUPS if grouping == 'motor' else self.SENSOR_TYPE_GROUPS

        grouped = {}
        for name, channels in groups.items():
            grouped[name] = importance[channels].sum()

        # Normalize
        total = sum(grouped.values())
        grouped = {k: v / total for k, v in grouped.items()}

        return grouped

    def visualize(
        self,
        importance: np.ndarray,
        figsize: Tuple[int, int] = (12, 6),
        title: str = 'Sensor Importance'
    ) -> plt.Figure:
        """
        Visualize sensor importance.

        Args:
            importance: Per-channel importance scores
            figsize: Figure size
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Per-sensor bar chart
        colors = []
        for i in range(24):
            if i < 6:
                colors.append('tab:blue')
            elif i < 12:
                colors.append('tab:orange')
            elif i < 18:
                colors.append('tab:green')
            else:
                colors.append('tab:red')

        axes[0].bar(range(24), importance, color=colors)
        axes[0].set_xticks(range(24))
        axes[0].set_xticklabels(self.SENSOR_NAMES, rotation=45, ha='right')
        axes[0].set_xlabel('Sensor')
        axes[0].set_ylabel('Importance')
        axes[0].set_title('Per-Sensor Importance')

        # Add motor legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='tab:blue', label='Motor A'),
            Patch(facecolor='tab:orange', label='Motor B'),
            Patch(facecolor='tab:green', label='Motor C'),
            Patch(facecolor='tab:red', label='Motor D'),
        ]
        axes[0].legend(handles=legend_elements, loc='upper right')

        # Grouped importance heatmap
        motor_imp = self.get_grouped_importance(importance, 'motor')
        type_imp = self.get_grouped_importance(importance, 'sensor_type')

        # Create 4x6 heatmap (motors x sensor types)
        heatmap = np.zeros((4, 6))
        for i, motor in enumerate(['Motor A', 'Motor B', 'Motor C', 'Motor D']):
            for j, stype in enumerate(['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z']):
                ch_idx = i * 6 + j
                heatmap[i, j] = importance[ch_idx]

        im = axes[1].imshow(heatmap, cmap='YlOrRd', aspect='auto')
        axes[1].set_xticks(range(6))
        axes[1].set_xticklabels(['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ'])
        axes[1].set_yticks(range(4))
        axes[1].set_yticklabels(['A', 'B', 'C', 'D'])
        axes[1].set_xlabel('Sensor Type')
        axes[1].set_ylabel('Motor')
        axes[1].set_title('Importance Heatmap')
        plt.colorbar(im, ax=axes[1])

        fig.suptitle(title)
        plt.tight_layout()

        return fig


# =============================================================================
# Attention Visualization
# =============================================================================

class AttentionVisualizer:
    """
    Visualize attention weights from Transformer models.

    Shows what parts of the input the model attends to.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Model with get_attention_weights method
                   (e.g., TransformerDetector)
        """
        self.model = model
        self.model.eval()

        if not hasattr(model, 'get_attention_weights'):
            raise ValueError("Model must have get_attention_weights method")

    def get_attention(self, x: torch.Tensor) -> List[np.ndarray]:
        """
        Extract attention weights for input.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            List of attention weight matrices, one per layer
        """
        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(x)

        return [aw.cpu().numpy() for aw in attention_weights]

    def visualize_attention(
        self,
        x: torch.Tensor,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Visualize attention weights.

        Args:
            x: Input tensor
            layer_idx: Which layer to visualize (-1 = last)
            head_idx: Which attention head (None = average)
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        attention_weights = self.get_attention(x)
        attn = attention_weights[layer_idx][0]  # First batch item

        if head_idx is not None and len(attn.shape) > 2:
            attn = attn[head_idx]

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Attention Weights (Layer {layer_idx})')
        plt.colorbar(im, ax=ax)

        return fig

    def get_cls_attention(self, x: torch.Tensor) -> np.ndarray:
        """
        Get attention from CLS token to all other positions.

        Useful for understanding what the model focuses on for classification.

        Args:
            x: Input tensor

        Returns:
            attention: (time,) attention scores
        """
        attention_weights = self.get_attention(x)

        # Average attention to CLS token across all layers
        cls_attention = []
        for layer_attn in attention_weights:
            # layer_attn shape: (batch, seq_len, seq_len)
            # CLS token is at position 0
            attn_to_cls = layer_attn[0, 0, 1:]  # Attention from CLS to other tokens
            cls_attention.append(attn_to_cls)

        # Average across layers
        cls_attention = np.mean(cls_attention, axis=0)

        return cls_attention

    def visualize_cls_attention(
        self,
        x: torch.Tensor,
        channel_idx: int = 0,
        figsize: Tuple[int, int] = (12, 4)
    ) -> plt.Figure:
        """
        Visualize CLS attention overlaid on signal.

        Args:
            x: Input tensor
            channel_idx: Which channel to display
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        cls_attention = self.get_cls_attention(x)
        signal = x[0, channel_idx].cpu().numpy()

        # Upsample attention to signal length
        attn_upsampled = np.interp(
            np.linspace(0, 1, len(signal)),
            np.linspace(0, 1, len(cls_attention)),
            cls_attention
        )

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Signal with attention overlay
        time_axis = np.arange(len(signal))

        axes[0].plot(signal, 'b-', linewidth=0.5)
        axes[0].fill_between(
            time_axis,
            signal.min(),
            signal.max(),
            alpha=attn_upsampled * 0.5,
            color='red'
        )
        axes[0].set_ylabel('Signal')
        axes[0].set_title('Signal with Attention Overlay')

        # Attention weights
        axes[1].bar(range(len(cls_attention)), cls_attention, color='red', alpha=0.7)
        axes[1].set_ylabel('Attention')
        axes[1].set_xlabel('Token Position')
        axes[1].set_title('CLS Token Attention Weights')

        plt.tight_layout()

        return fig
