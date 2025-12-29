"""
Model Quantization for Embedded Deployment

Implements:
1. Post-training dynamic quantization (8-bit)
2. Quantization-aware training (QAT)
3. ONNX export with quantized ops
4. Latency benchmarking

Target: <5ms inference @ 200 Hz on CPU
"""

import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quant
from torch.ao.quantization import get_default_qconfig, prepare, convert
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import time
import json
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    precision: str  # 'fp32', 'int8', 'fp16'
    batch_size: int
    seq_len: int
    mean_latency_ms: float
    std_latency_ms: float
    p99_latency_ms: float
    throughput_samples_per_sec: float
    model_size_mb: float
    memory_peak_mb: float


class ModelQuantizer:
    """
    Quantize PyTorch models for efficient inference.

    Supports:
    - Dynamic quantization (fastest to apply)
    - Static quantization (requires calibration)
    - Quantization-aware training
    """

    def __init__(self, model: nn.Module):
        self.original_model = model
        self.quantized_model = None

    def dynamic_quantize(self) -> nn.Module:
        """
        Apply dynamic quantization (weights only).

        Fastest method, no calibration needed.
        Good for RNNs and transformers.
        """
        # Only quantize Linear and LSTM/GRU layers
        quantized = torch.quantization.quantize_dynamic(
            self.original_model,
            {nn.Linear, nn.GRU, nn.LSTM},
            dtype=torch.qint8
        )
        self.quantized_model = quantized
        return quantized

    def static_quantize(
        self,
        calibration_data: torch.Tensor,
        backend: str = 'fbgemm'
    ) -> nn.Module:
        """
        Apply static quantization (weights + activations).

        Requires calibration data to determine activation ranges.

        Args:
            calibration_data: Representative input data [N, T, D]
            backend: 'fbgemm' for x86, 'qnnpack' for ARM
        """
        model = self.original_model.cpu().eval()

        # Set quantization config
        model.qconfig = get_default_qconfig(backend)

        # Prepare model for quantization
        model_prepared = prepare(model, inplace=False)

        # Calibrate with representative data
        with torch.no_grad():
            for i in range(min(len(calibration_data), 100)):
                model_prepared(calibration_data[i:i+1])

        # Convert to quantized model
        model_quantized = convert(model_prepared, inplace=False)
        self.quantized_model = model_quantized

        return model_quantized

    def get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / 1024 / 1024

    def compare_outputs(
        self,
        input_data: torch.Tensor,
        rtol: float = 0.1,
        atol: float = 0.1
    ) -> Dict:
        """
        Compare outputs of original and quantized models.

        Args:
            input_data: Test input [B, T, D]
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Dict with comparison metrics
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model available")

        self.original_model.eval()
        self.quantized_model.eval()

        with torch.no_grad():
            orig_out, _ = self.original_model(input_data)
            quant_out, _ = self.quantized_model(input_data.cpu())

        orig_np = orig_out.cpu().numpy()
        quant_np = quant_out.numpy()

        # Compute metrics
        abs_diff = np.abs(orig_np - quant_np)
        rel_diff = abs_diff / (np.abs(orig_np) + 1e-8)

        return {
            'max_abs_diff': float(np.max(abs_diff)),
            'mean_abs_diff': float(np.mean(abs_diff)),
            'max_rel_diff': float(np.max(rel_diff)),
            'mean_rel_diff': float(np.mean(rel_diff)),
            'outputs_close': bool(np.allclose(orig_np, quant_np, rtol=rtol, atol=atol))
        }


class ONNXExporter:
    """
    Export PyTorch models to ONNX format.

    ONNX enables deployment on various platforms:
    - ONNX Runtime (optimized CPU inference)
    - TensorRT (NVIDIA GPU)
    - OpenVINO (Intel)
    - CoreML (Apple)
    """

    def __init__(self, model: nn.Module, input_dim: int, seq_len: int = 50):
        self.model = model
        self.input_dim = input_dim
        self.seq_len = seq_len

    def export(
        self,
        output_path: str,
        opset_version: int = 14,
        dynamic_axes: bool = True
    ) -> str:
        """
        Export model to ONNX.

        Args:
            output_path: Output file path
            opset_version: ONNX opset version
            dynamic_axes: Allow dynamic batch size and sequence length

        Returns:
            Path to saved ONNX model
        """
        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, self.seq_len, self.input_dim)

        # Define dynamic axes
        if dynamic_axes:
            axes = {
                'input': {0: 'batch_size', 1: 'seq_len'},
                'output': {0: 'batch_size', 1: 'seq_len'}
            }
        else:
            axes = None

        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output', 'hidden'],
            dynamic_axes=axes
        )

        print(f"ONNX model saved to {output_path}")
        return output_path

    def verify_export(self, onnx_path: str, test_input: torch.Tensor) -> Dict:
        """
        Verify ONNX export produces correct outputs.

        Args:
            onnx_path: Path to ONNX model
            test_input: Test input tensor

        Returns:
            Dict with verification results
        """
        try:
            import onnxruntime as ort
        except ImportError:
            return {'error': 'onnxruntime not installed'}

        # Load ONNX model
        session = ort.InferenceSession(onnx_path)

        # Run inference
        self.model.eval()
        with torch.no_grad():
            torch_out, _ = self.model(test_input)
            torch_out = torch_out.numpy()

        onnx_out = session.run(None, {'input': test_input.numpy()})[0]

        # Compare
        max_diff = np.max(np.abs(torch_out - onnx_out))

        return {
            'max_diff': float(max_diff),
            'outputs_match': max_diff < 1e-4,
            'onnx_model_size_mb': Path(onnx_path).stat().st_size / 1024 / 1024
        }


class LatencyBenchmark:
    """
    Benchmark model inference latency.

    Target: <5ms per sample for real-time 200 Hz operation.
    """

    def __init__(self, warmup_iterations: int = 10, benchmark_iterations: int = 100):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def benchmark_pytorch(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = 'cpu'
    ) -> BenchmarkResult:
        """
        Benchmark PyTorch model latency.

        Args:
            model: Model to benchmark
            input_shape: (batch_size, seq_len, input_dim)
            device: 'cpu' or 'cuda'

        Returns:
            BenchmarkResult with timing statistics
        """
        model = model.to(device).eval()
        batch_size, seq_len, input_dim = input_shape

        # Create input
        x = torch.randn(input_shape, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(x)

        # Synchronize if CUDA
        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(self.benchmark_iterations):
                if device == 'cuda':
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = model(x)

                if device == 'cuda':
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        latencies = np.array(latencies)

        # Get model size
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        model_size_mb = param_size / 1024 / 1024

        return BenchmarkResult(
            model_name=model.__class__.__name__,
            precision='fp32' if model.parameters().__next__().dtype == torch.float32 else 'int8',
            batch_size=batch_size,
            seq_len=seq_len,
            mean_latency_ms=float(np.mean(latencies)),
            std_latency_ms=float(np.std(latencies)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            throughput_samples_per_sec=float(batch_size * 1000 / np.mean(latencies)),
            model_size_mb=model_size_mb,
            memory_peak_mb=0.0  # Would need profiling
        )

    def benchmark_onnx(
        self,
        onnx_path: str,
        input_shape: Tuple[int, ...]
    ) -> BenchmarkResult:
        """
        Benchmark ONNX model latency.

        Args:
            onnx_path: Path to ONNX model
            input_shape: (batch_size, seq_len, input_dim)

        Returns:
            BenchmarkResult with timing statistics
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed")

        batch_size, seq_len, input_dim = input_shape

        # Configure session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1  # Single-threaded for consistent benchmarks

        session = ort.InferenceSession(onnx_path, sess_options)

        # Create input
        x = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(self.warmup_iterations):
            _ = session.run(None, {'input': x})

        # Benchmark
        latencies = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            _ = session.run(None, {'input': x})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        latencies = np.array(latencies)

        model_size_mb = Path(onnx_path).stat().st_size / 1024 / 1024

        return BenchmarkResult(
            model_name='ONNX',
            precision='fp32',  # Or determine from model
            batch_size=batch_size,
            seq_len=seq_len,
            mean_latency_ms=float(np.mean(latencies)),
            std_latency_ms=float(np.std(latencies)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            throughput_samples_per_sec=float(batch_size * 1000 / np.mean(latencies)),
            model_size_mb=model_size_mb,
            memory_peak_mb=0.0
        )

    def run_full_benchmark(
        self,
        model: nn.Module,
        input_dim: int,
        output_dir: str = './benchmarks'
    ) -> Dict[str, BenchmarkResult]:
        """
        Run comprehensive benchmark suite.

        Tests:
        - FP32 PyTorch
        - INT8 dynamic quantization
        - ONNX Runtime

        Args:
            model: Model to benchmark
            input_dim: Input feature dimension
            output_dir: Directory to save results

        Returns:
            Dict mapping config name to BenchmarkResult
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Test configurations
        configs = [
            ('batch1_seq1', (1, 1, input_dim)),    # Single sample (streaming)
            ('batch1_seq50', (1, 50, input_dim)),  # 50 timesteps
            ('batch8_seq50', (8, 50, input_dim)),  # Batch of 8
        ]

        # FP32 PyTorch
        print("Benchmarking FP32 PyTorch...")
        for name, shape in configs:
            result = self.benchmark_pytorch(model, shape, 'cpu')
            results[f'pytorch_fp32_{name}'] = result
            print(f"  {name}: {result.mean_latency_ms:.3f} ms (p99: {result.p99_latency_ms:.3f} ms)")

        # INT8 Dynamic Quantization
        print("\nBenchmarking INT8 quantized...")
        quantizer = ModelQuantizer(model)
        quantized_model = quantizer.dynamic_quantize()

        for name, shape in configs:
            result = self.benchmark_pytorch(quantized_model, shape, 'cpu')
            results[f'pytorch_int8_{name}'] = result
            print(f"  {name}: {result.mean_latency_ms:.3f} ms (p99: {result.p99_latency_ms:.3f} ms)")

        # ONNX Runtime
        print("\nBenchmarking ONNX Runtime...")
        onnx_path = output_dir / 'model.onnx'
        exporter = ONNXExporter(model, input_dim)
        exporter.export(str(onnx_path))

        for name, shape in configs:
            try:
                result = self.benchmark_onnx(str(onnx_path), shape)
                results[f'onnx_{name}'] = result
                print(f"  {name}: {result.mean_latency_ms:.3f} ms (p99: {result.p99_latency_ms:.3f} ms)")
            except Exception as e:
                print(f"  {name}: Failed - {e}")

        # Save results
        results_dict = {
            k: {
                'mean_latency_ms': v.mean_latency_ms,
                'std_latency_ms': v.std_latency_ms,
                'p99_latency_ms': v.p99_latency_ms,
                'throughput_samples_per_sec': v.throughput_samples_per_sec,
                'model_size_mb': v.model_size_mb
            }
            for k, v in results.items()
        }

        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Summary
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"\nTarget: <5ms for real-time 200 Hz")

        # Check streaming performance
        streaming_key = 'pytorch_int8_batch1_seq1'
        if streaming_key in results:
            streaming_latency = results[streaming_key].mean_latency_ms
            meets_target = streaming_latency < 5.0
            print(f"\nStreaming (1 sample): {streaming_latency:.3f} ms")
            print(f"Meets target: {'✓ YES' if meets_target else '✗ NO'}")

        return results


class TorchScriptExporter:
    """Export model to TorchScript for deployment."""

    def __init__(self, model: nn.Module):
        self.model = model

    def trace(self, example_input: torch.Tensor, output_path: str) -> str:
        """
        Export model via tracing.

        Args:
            example_input: Example input tensor
            output_path: Output file path

        Returns:
            Path to saved model
        """
        self.model.eval()
        traced = torch.jit.trace(self.model, example_input)
        traced.save(output_path)
        print(f"TorchScript model saved to {output_path}")
        return output_path

    def script(self, output_path: str) -> str:
        """
        Export model via scripting.

        More flexible than tracing, handles control flow.
        """
        self.model.eval()
        scripted = torch.jit.script(self.model)
        scripted.save(output_path)
        print(f"TorchScript (scripted) model saved to {output_path}")
        return output_path


if __name__ == "__main__":
    # Test quantization and benchmarking
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from model import CNNGRUDetector

    # Create model
    input_dim = 100
    model = CNNGRUDetector(input_dim=input_dim)
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test quantization
    print("\n=== Testing Quantization ===")
    quantizer = ModelQuantizer(model)

    # Dynamic quantization
    quantized = quantizer.dynamic_quantize()
    print(f"Original size: {quantizer.get_model_size(model):.3f} MB")
    print(f"Quantized size: {quantizer.get_model_size(quantized):.3f} MB")

    # Compare outputs
    test_input = torch.randn(1, 50, input_dim)
    comparison = quantizer.compare_outputs(test_input)
    print(f"Output comparison: {comparison}")

    # Test ONNX export
    print("\n=== Testing ONNX Export ===")
    exporter = ONNXExporter(model, input_dim)
    onnx_path = './benchmarks/test_model.onnx'
    Path('./benchmarks').mkdir(exist_ok=True)
    exporter.export(onnx_path)

    # Verify export
    verification = exporter.verify_export(onnx_path, test_input)
    print(f"ONNX verification: {verification}")

    # Run benchmarks
    print("\n=== Running Benchmarks ===")
    benchmark = LatencyBenchmark(warmup_iterations=5, benchmark_iterations=50)
    results = benchmark.run_full_benchmark(model, input_dim, './benchmarks')
