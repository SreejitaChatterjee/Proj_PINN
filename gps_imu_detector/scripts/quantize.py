#!/usr/bin/env python3
"""
Quantization and ONNX Export Script

Usage:
    python scripts/quantize.py --model models/baseline.pth --out models/baseline.onnx
    python scripts/quantize.py --model models/baseline.pth --out models/baseline.onnx --int8
    python scripts/quantize.py --model models/baseline.pth --benchmark

Exit codes:
    0: Success
    1: Error
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from typing import Optional, Dict, Any
import json
import time


def load_model(model_path: str, input_dim: int = 90) -> torch.nn.Module:
    """Load PyTorch model from checkpoint."""
    from model import CNNGRUDetector

    model = CNNGRUDetector(input_dim=input_dim)

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}, using random weights")

    model.eval()
    return model


def export_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_dim: int = 90,
    seq_len: int = 100,
    opset_version: int = 13
) -> Dict[str, Any]:
    """Export model to ONNX format."""
    from quantization import ONNXExporter

    exporter = ONNXExporter(model, input_dim)

    # Export
    exporter.export(output_path, opset_version=opset_version)

    # Verify
    test_input = torch.randn(1, seq_len, input_dim)
    verification = exporter.verify_export(output_path, test_input)

    return verification


def quantize_model(
    model: torch.nn.Module,
    method: str = 'dynamic'
) -> torch.nn.Module:
    """Quantize model to INT8."""
    from quantization import ModelQuantizer

    # Get input dim from model
    input_dim = model.conv1.in_channels if hasattr(model, 'conv1') else 90

    quantizer = ModelQuantizer(model)

    if method == 'dynamic':
        quantized = quantizer.dynamic_quantize()
    else:
        # For static quantization, we'd need calibration data
        print("Static quantization requires calibration data, using dynamic")
        quantized = quantizer.dynamic_quantize()

    # Report size reduction
    original_size = quantizer.get_model_size(model)
    quantized_size = quantizer.get_model_size(quantized)

    print(f"Original size: {original_size:.3f} MB")
    print(f"Quantized size: {quantized_size:.3f} MB")
    print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")

    return quantized


def benchmark_model(
    model: torch.nn.Module,
    input_dim: int = 90,
    seq_len: int = 100,
    n_warmup: int = 50,
    n_iterations: int = 500
) -> Dict[str, float]:
    """Benchmark model latency."""
    model.eval()

    # Warmup
    test_input = torch.randn(1, seq_len, input_dim)
    hidden = None

    for _ in range(n_warmup):
        with torch.no_grad():
            _, hidden = model(test_input, hidden)

    # Benchmark
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _, hidden = model(test_input, hidden)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)

    results = {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'n_iterations': n_iterations
    }

    return results


def benchmark_onnx(
    onnx_path: str,
    input_dim: int = 90,
    seq_len: int = 100,
    n_warmup: int = 50,
    n_iterations: int = 500
) -> Dict[str, float]:
    """Benchmark ONNX model latency."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping ONNX benchmark")
        return {}

    # Create session with single thread
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(onnx_path, sess_options)

    # Get input name
    input_name = session.get_inputs()[0].name

    # Warmup
    test_input = np.random.randn(1, seq_len, input_dim).astype(np.float32)

    for _ in range(n_warmup):
        session.run(None, {input_name: test_input})

    # Benchmark
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        session.run(None, {input_name: test_input})
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)

    results = {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'n_iterations': n_iterations,
        'threads': 1
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Quantize and export model to ONNX'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to PyTorch model checkpoint'
    )
    parser.add_argument(
        '--out', type=str, default=None,
        help='Output path for ONNX model'
    )
    parser.add_argument(
        '--int8', action='store_true',
        help='Apply INT8 quantization before export'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Run latency benchmark'
    )
    parser.add_argument(
        '--input-dim', type=int, default=90,
        help='Input feature dimension (default: 90)'
    )
    parser.add_argument(
        '--seq-len', type=int, default=100,
        help='Sequence length for benchmark (default: 100)'
    )
    parser.add_argument(
        '--iterations', type=int, default=500,
        help='Number of benchmark iterations (default: 500)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("QUANTIZATION AND EXPORT SCRIPT")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {args.model}...")
    model = load_model(args.model, args.input_dim)

    # Quantize if requested
    if args.int8:
        print("\nApplying INT8 quantization...")
        model = quantize_model(model, method='dynamic')

    # Export to ONNX if output path provided
    if args.out:
        print(f"\nExporting to ONNX: {args.out}...")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        verification = export_onnx(model, args.out, args.input_dim, args.seq_len)
        print(f"Verification: {verification}")

    # Benchmark if requested
    if args.benchmark:
        print("\n" + "=" * 60)
        print("LATENCY BENCHMARK")
        print("=" * 60)

        print("\nBenchmarking PyTorch model...")
        pytorch_results = benchmark_model(
            model, args.input_dim, args.seq_len,
            n_iterations=args.iterations
        )

        print(f"\nPyTorch Results:")
        print(f"  Mean: {pytorch_results['mean_ms']:.2f} ms")
        print(f"  P50:  {pytorch_results['p50_ms']:.2f} ms")
        print(f"  P95:  {pytorch_results['p95_ms']:.2f} ms")
        print(f"  P99:  {pytorch_results['p99_ms']:.2f} ms")

        # Benchmark ONNX if available
        if args.out and Path(args.out).exists():
            print("\nBenchmarking ONNX model (single thread)...")
            onnx_results = benchmark_onnx(
                args.out, args.input_dim, args.seq_len,
                n_iterations=args.iterations
            )

            if onnx_results:
                print(f"\nONNX Results:")
                print(f"  Mean: {onnx_results['mean_ms']:.2f} ms")
                print(f"  P50:  {onnx_results['p50_ms']:.2f} ms")
                print(f"  P95:  {onnx_results['p95_ms']:.2f} ms")
                print(f"  P99:  {onnx_results['p99_ms']:.2f} ms")

                # Save results
                results_path = Path(args.out).with_suffix('.benchmark.json')
                with open(results_path, 'w') as f:
                    json.dump({
                        'pytorch': pytorch_results,
                        'onnx': onnx_results
                    }, f, indent=2)
                print(f"\nBenchmark results saved to {results_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
