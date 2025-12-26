"""
Deployment Module
=================

Production-ready deployment utilities:
- ONNX export for cross-platform deployment
- Streaming inference for real-time detection
- Model quantization for edge devices
- Latency optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import quantize_dynamic, prepare, convert
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable
from pathlib import Path
from collections import deque
import time
import warnings


# =============================================================================
# ONNX Export
# =============================================================================

def export_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 24, 256),
    opset_version: int = 14,
    dynamic_axes: Optional[Dict] = None,
    simplify: bool = True,
    verify: bool = True
) -> str:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        save_path: Path to save ONNX model
        input_shape: Input tensor shape (batch, channels, time)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable batch/sequence length
        simplify: Simplify the ONNX graph
        verify: Verify exported model matches PyTorch output

    Returns:
        Path to saved ONNX model
    """
    model.eval()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Default dynamic axes
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'time_steps'},
            'output': {0: 'batch_size'}
        }

    # Export
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )

    print(f"Exported ONNX model to {save_path}")

    # Simplify if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            onnx_model = onnx.load(str(save_path))
            simplified_model, check = onnx_simplify(onnx_model)

            if check:
                onnx.save(simplified_model, str(save_path))
                print("ONNX model simplified successfully")
            else:
                print("Warning: ONNX simplification check failed")
        except ImportError:
            print("Note: Install onnx-simplifier for graph optimization")

    # Verify if requested
    if verify:
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(save_path))
            onnx_output = session.run(None, {'input': dummy_input.numpy()})[0]

            with torch.no_grad():
                pytorch_output = model(dummy_input).numpy()

            max_diff = np.abs(onnx_output - pytorch_output).max()
            print(f"ONNX verification: max difference = {max_diff:.6f}")

            if max_diff > 1e-4:
                warnings.warn(f"ONNX output differs from PyTorch by {max_diff}")
        except ImportError:
            print("Note: Install onnxruntime for verification")

    return str(save_path)


class ONNXInference:
    """
    ONNX Runtime inference wrapper.

    Provides consistent interface for ONNX model inference.
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None
    ):
        """
        Args:
            model_path: Path to ONNX model
            providers: Execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Install onnxruntime: pip install onnxruntime")

        if providers is None:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Run inference."""
        return self.session.run([self.output_name], {self.input_name: x})[0]

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and probabilities."""
        logits = self(x)
        probs = self._softmax(logits)
        predictions = np.argmax(probs, axis=-1)
        return predictions, probs

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# =============================================================================
# Streaming Inference
# =============================================================================

class StreamingInference:
    """
    Real-time streaming inference for continuous fault monitoring.

    Maintains a sliding window buffer and triggers predictions
    at specified intervals.
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: int = 256,
        stride: int = 64,
        device: str = 'cpu',
        preprocessing: Optional[Callable] = None
    ):
        """
        Args:
            model: Fault detection model
            window_size: Number of samples per window
            stride: Samples between predictions
            device: Device for inference
            preprocessing: Optional preprocessing function
        """
        self.model = model
        self.model.eval()
        self.window_size = window_size
        self.stride = stride
        self.device = torch.device(device)
        self.preprocessing = preprocessing

        # Circular buffer for sensor data
        self.buffer = deque(maxlen=window_size)
        self.sample_count = 0
        self.last_prediction_sample = 0

        # Move model to device
        self.model = self.model.to(self.device)

        # Performance tracking
        self.inference_times = deque(maxlen=100)

    def reset(self):
        """Reset buffer and counters."""
        self.buffer.clear()
        self.sample_count = 0
        self.last_prediction_sample = 0
        self.inference_times.clear()

    def add_sample(self, sample: np.ndarray) -> Optional[Dict]:
        """
        Add a single sample to the buffer.

        Args:
            sample: Single time step of sensor data (24 channels)

        Returns:
            Prediction dict if stride reached, else None
        """
        self.buffer.append(sample)
        self.sample_count += 1

        # Check if we should make a prediction
        if (len(self.buffer) >= self.window_size and
            self.sample_count - self.last_prediction_sample >= self.stride):

            self.last_prediction_sample = self.sample_count
            return self._predict()

        return None

    def add_batch(self, samples: np.ndarray) -> List[Dict]:
        """
        Add multiple samples at once.

        Args:
            samples: Array of shape (n_samples, 24)

        Returns:
            List of predictions
        """
        predictions = []
        for sample in samples:
            pred = self.add_sample(sample)
            if pred is not None:
                predictions.append(pred)
        return predictions

    def _predict(self) -> Dict:
        """Make prediction on current buffer."""
        start_time = time.perf_counter()

        # Convert buffer to tensor
        window = np.array(list(self.buffer))  # (window_size, 24)

        if self.preprocessing:
            window = self.preprocessing(window)

        # Shape: (1, 24, window_size)
        x = torch.FloatTensor(window.T).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            prediction = probs.argmax(dim=-1).item()
            confidence = probs.max(dim=-1)[0].item()

        inference_time = time.perf_counter() - start_time
        self.inference_times.append(inference_time)

        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probs.cpu().numpy()[0],
            'sample_idx': self.sample_count,
            'inference_time_ms': inference_time * 1000,
            'timestamp': time.time()
        }

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {'avg_inference_ms': 0, 'max_inference_ms': 0}

        times = list(self.inference_times)
        return {
            'avg_inference_ms': np.mean(times) * 1000,
            'max_inference_ms': np.max(times) * 1000,
            'min_inference_ms': np.min(times) * 1000,
            'samples_processed': self.sample_count,
            'predictions_made': len(self.inference_times)
        }


class StreamingBuffer:
    """
    Efficient circular buffer for streaming data.

    Uses numpy for memory efficiency with large buffers.
    """

    def __init__(self, max_size: int, n_channels: int = 24):
        self.max_size = max_size
        self.n_channels = n_channels
        self.buffer = np.zeros((max_size, n_channels), dtype=np.float32)
        self.write_idx = 0
        self.count = 0

    def append(self, data: np.ndarray):
        """Add data to buffer."""
        n_samples = len(data) if data.ndim > 1 else 1

        if data.ndim == 1:
            data = data.reshape(1, -1)

        for sample in data:
            self.buffer[self.write_idx] = sample
            self.write_idx = (self.write_idx + 1) % self.max_size
            self.count = min(self.count + 1, self.max_size)

    def get_window(self, window_size: int) -> Optional[np.ndarray]:
        """Get the last window_size samples."""
        if self.count < window_size:
            return None

        if self.write_idx >= window_size:
            return self.buffer[self.write_idx - window_size:self.write_idx].copy()
        else:
            # Wrap around
            first_part = self.buffer[self.max_size - (window_size - self.write_idx):]
            second_part = self.buffer[:self.write_idx]
            return np.vstack([first_part, second_part])

    def clear(self):
        """Clear buffer."""
        self.buffer.fill(0)
        self.write_idx = 0
        self.count = 0


# =============================================================================
# Model Quantization
# =============================================================================

class QuantizedModel:
    """
    Quantized model wrapper for efficient inference.

    Supports:
    - Dynamic quantization (INT8 weights)
    - Static quantization (INT8 activations + weights)
    - Quantization-aware training (QAT)
    """

    def __init__(
        self,
        model: nn.Module,
        quantization_type: str = 'dynamic'  # 'dynamic', 'static'
    ):
        self.original_model = model
        self.quantization_type = quantization_type
        self.quantized_model = None

    def quantize_dynamic(self) -> nn.Module:
        """
        Apply dynamic quantization.

        Quantizes weights to INT8 at load time.
        Best for models with Linear/LSTM layers.
        """
        self.quantized_model = quantize_dynamic(
            self.original_model,
            {nn.Linear, nn.Conv1d},
            dtype=torch.qint8
        )
        return self.quantized_model

    def quantize_static(
        self,
        calibration_data: torch.Tensor,
        backend: str = 'fbgemm'
    ) -> nn.Module:
        """
        Apply static quantization.

        Quantizes both weights and activations using calibration data.

        Args:
            calibration_data: Representative data for calibration
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        """
        torch.backends.quantized.engine = backend

        # Prepare model for quantization
        model = self.original_model
        model.eval()

        # Fuse modules if possible (Conv + BN + ReLU)
        model = self._fuse_modules(model)

        # Insert observers
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        prepared_model = prepare(model)

        # Calibrate with representative data
        with torch.no_grad():
            prepared_model(calibration_data)

        # Convert to quantized model
        self.quantized_model = convert(prepared_model)
        return self.quantized_model

    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse Conv-BN-ReLU sequences for better quantization."""
        # This is a simplified version; production code would need
        # model-specific fusion patterns
        return model

    def get_size_reduction(self) -> Dict[str, float]:
        """Compare model sizes."""
        import io

        # Original size
        orig_buffer = io.BytesIO()
        torch.save(self.original_model.state_dict(), orig_buffer)
        orig_size = orig_buffer.tell()

        if self.quantized_model is None:
            return {'original_mb': orig_size / 1e6}

        # Quantized size
        quant_buffer = io.BytesIO()
        torch.save(self.quantized_model.state_dict(), quant_buffer)
        quant_size = quant_buffer.tell()

        return {
            'original_mb': orig_size / 1e6,
            'quantized_mb': quant_size / 1e6,
            'reduction_pct': (1 - quant_size / orig_size) * 100
        }

    def benchmark(
        self,
        input_shape: Tuple[int, ...] = (1, 24, 256),
        n_runs: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark original vs quantized model.

        Returns inference time comparison.
        """
        dummy_input = torch.randn(*input_shape)

        results = {}

        # Benchmark original
        self.original_model.eval()
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.original_model(dummy_input)

        start = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = self.original_model(dummy_input)
        orig_time = (time.perf_counter() - start) / n_runs

        results['original_ms'] = orig_time * 1000

        # Benchmark quantized
        if self.quantized_model is not None:
            for _ in range(warmup):
                with torch.no_grad():
                    _ = self.quantized_model(dummy_input)

            start = time.perf_counter()
            for _ in range(n_runs):
                with torch.no_grad():
                    _ = self.quantized_model(dummy_input)
            quant_time = (time.perf_counter() - start) / n_runs

            results['quantized_ms'] = quant_time * 1000
            results['speedup'] = orig_time / quant_time

        return results


# =============================================================================
# TorchScript Export
# =============================================================================

def export_torchscript(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 24, 256),
    method: str = 'trace'  # 'trace' or 'script'
) -> str:
    """
    Export model to TorchScript format.

    Args:
        model: PyTorch model
        save_path: Path to save model
        input_shape: Input tensor shape
        method: 'trace' for tracing, 'script' for scripting

    Returns:
        Path to saved model
    """
    model.eval()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*input_shape)

    if method == 'trace':
        with torch.no_grad():
            scripted = torch.jit.trace(model, dummy_input)
    else:
        scripted = torch.jit.script(model)

    # Optimize for inference
    scripted = torch.jit.optimize_for_inference(scripted)

    scripted.save(str(save_path))
    print(f"Exported TorchScript model to {save_path}")

    return str(save_path)


# =============================================================================
# Latency Profiler
# =============================================================================

class LatencyProfiler:
    """Profile model latency by layer."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_times = {}
        self.hooks = []

    def _register_hooks(self):
        """Register forward hooks on all modules."""
        for name, module in self.model.named_modules():
            hook = module.register_forward_hook(self._create_hook(name))
            self.hooks.append(hook)

    def _create_hook(self, name: str):
        def hook(module, input, output):
            if name not in self.layer_times:
                self.layer_times[name] = []
        return hook

    def profile(
        self,
        input_shape: Tuple[int, ...] = (1, 24, 256),
        n_runs: int = 50,
        warmup: int = 10
    ) -> Dict[str, Dict]:
        """
        Profile model latency.

        Returns per-layer timing information.
        """
        dummy_input = torch.randn(*input_shape)
        self.model.eval()

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(dummy_input)

        # Profile with torch.profiler
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=True
            ) as prof:
                for _ in range(n_runs):
                    with torch.no_grad():
                        _ = self.model(dummy_input)

            # Aggregate results
            key_averages = prof.key_averages()

            results = {}
            for item in key_averages:
                if item.cpu_time_total > 0:
                    results[item.key] = {
                        'cpu_time_ms': item.cpu_time_total / n_runs / 1000,
                        'count': item.count / n_runs,
                        'input_shapes': str(item.input_shapes) if hasattr(item, 'input_shapes') else 'N/A'
                    }

            return results

        except Exception as e:
            print(f"Profiling failed: {e}")
            return {}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
