"""
Measure computational costs for deployment feasibility analysis.

Measures:
1. Inference time (mean, std, min, max)
2. Memory footprint (model size, RAM usage)
3. FLOPs estimation
4. Throughput (samples per second)
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Import PINN framework (install with: pip install -e .)
try:
    from pinn_dynamics import Predictor, QuadrotorPINN
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pinn_dynamics import QuadrotorPINN, Predictor

from pinn_dynamics.security.anomaly_detector import AnomalyDetector

print("=" * 60)
print("COMPUTATIONAL COST ANALYSIS")
print("=" * 60)

# Paths
MODELS_DIR = Path("models/security")
DATA_DIR = Path("data/ALFA_processed")
RESULTS_DIR = Path("research/security/computational_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. LOAD MODEL AND DATA
# ============================================================================
print("\n[1/5] Loading model and data...")

# Load best model
model_path = MODELS_DIR / "pinn_w0_best.pth"
if not model_path.exists():
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

# Create model architecture (matching trained model)
model = QuadrotorPINN(hidden_size=256, num_layers=5, dropout=0.1)

# Load weights
model.load_state_dict(torch.load(model_path, weights_only=False))
model.eval()

print(f"  Model loaded: {model_path.name}")
print(f"  Architecture: 5 layers x 256 hidden units")

# Load scalers
scalers_path = MODELS_DIR / "scalers.pkl"
with open(scalers_path, "rb") as f:
    scalers = pickle.load(f)

scaler_X = scalers["scaler_X"]
scaler_y = scalers["scaler_y"]

# Create predictor
predictor = Predictor(model, scaler_X, scaler_y, device="cpu")

# Create detector
detector = AnomalyDetector(
    predictor=predictor, threshold=0.1707, use_physics=False, n_mc_samples=50
)

print("  Detector initialized (MC samples: 50)")

# Generate random test data (1000 samples)
# This is just for performance measurement, not actual flight data
np.random.seed(42)
n_samples = 1000
states = np.random.randn(n_samples, 12).astype(np.float32)
controls = np.random.randn(n_samples, 4).astype(np.float32)

print(f"  Generated {len(states)} test samples for timing measurement")

# ============================================================================
# 2. MODEL SIZE AND MEMORY FOOTPRINT
# ============================================================================
print("\n[2/5] Analyzing model size and memory...")

# Model file size
model_size_bytes = model_path.stat().st_size
model_size_mb = model_size_bytes / (1024 * 1024)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Estimate memory (float32 = 4 bytes)
params_memory_mb = (total_params * 4) / (1024 * 1024)

print(f"  Model file size: {model_size_mb:.2f} MB")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Parameters memory: {params_memory_mb:.2f} MB")

# ============================================================================
# 3. INFERENCE TIME MEASUREMENT
# ============================================================================
print("\n[3/5] Measuring inference time...")

# Warm-up (compile JIT, load cache)
print("  Warming up (100 iterations)...")
for i in range(100):
    _ = predictor.predict(states[i], controls[i])

# Measure single-sample inference time
print("  Measuring single-sample inference (1000 iterations)...")
times_single = []
for i in range(1000):
    state = states[i % len(states)]
    control = controls[i % len(controls)]

    start = time.perf_counter()
    _ = predictor.predict(state, control)
    end = time.perf_counter()

    times_single.append((end - start) * 1000)  # Convert to ms

mean_time = np.mean(times_single)
std_time = np.std(times_single)
min_time = np.min(times_single)
max_time = np.max(times_single)
p95_time = np.percentile(times_single, 95)
p99_time = np.percentile(times_single, 99)

print(f"  Single-sample inference time:")
print(f"    Mean: {mean_time:.4f} ms")
print(f"    Std:  {std_time:.4f} ms")
print(f"    Min:  {min_time:.4f} ms")
print(f"    Max:  {max_time:.4f} ms")
print(f"    P95:  {p95_time:.4f} ms")
print(f"    P99:  {p99_time:.4f} ms")

# Throughput
throughput = 1000 / mean_time  # samples per second
print(f"  Throughput: {throughput:.1f} samples/sec")

# Check real-time capability (100 Hz = 10ms per sample)
realtime_freq = 100  # Hz
realtime_budget = 1000 / realtime_freq  # ms
realtime_capable = mean_time < realtime_budget
print(
    f"  Real-time capable at {realtime_freq} Hz: {realtime_capable} ({mean_time:.2f} ms < {realtime_budget:.2f} ms budget)"
)

# ============================================================================
# 4. DETECTION TIME WITH MC DROPOUT
# ============================================================================
print("\n[4/5] Measuring detection time with MC dropout (50 samples)...")

times_detection = []
for i in range(100):  # Fewer iterations due to MC overhead
    state = states[i % len(states)]
    control = controls[i % len(controls)]
    next_state = states[(i + 1) % len(states)]

    start = time.perf_counter()
    result = detector.detect(state, control, next_state)
    end = time.perf_counter()

    times_detection.append((end - start) * 1000)

mean_det_time = np.mean(times_detection)
std_det_time = np.std(times_detection)

print(f"  Detection time (with uncertainty):")
print(f"    Mean: {mean_det_time:.4f} ms")
print(f"    Std:  {std_det_time:.4f} ms")
print(f"  MC overhead: {mean_det_time / mean_time:.1f}x slower")

detection_throughput = 1000 / mean_det_time
print(f"  Detection throughput: {detection_throughput:.1f} samples/sec")

# ============================================================================
# 5. ESTIMATE FLOPs
# ============================================================================
print("\n[5/5] Estimating FLOPs...")


def count_flops(model, input_dim):
    """Estimate FLOPs for feed-forward pass."""
    total_flops = 0

    # Input layer
    prev_dim = input_dim

    # Hidden layers (assuming linear + activation)
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            # Linear: (prev_dim * out_dim) multiply-adds + out_dim bias adds
            out_dim = layer.out_features
            total_flops += prev_dim * out_dim * 2  # MAC = 2 ops
            total_flops += out_dim  # Bias addition
            prev_dim = out_dim

    return total_flops


input_dim = 16  # 12 states + 4 controls
flops = count_flops(model, input_dim)
gflops = flops / 1e9

print(f"  Estimated FLOPs per forward pass: {flops:,}")
print(f"  Estimated GFLOPs: {gflops:.6f}")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n[6/6] Saving results...")

results = {
    "model_size": {
        "file_size_mb": round(model_size_mb, 3),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameters_memory_mb": round(params_memory_mb, 3),
    },
    "inference_time_ms": {
        "mean": round(mean_time, 4),
        "std": round(std_time, 4),
        "min": round(min_time, 4),
        "max": round(max_time, 4),
        "p95": round(p95_time, 4),
        "p99": round(p99_time, 4),
    },
    "detection_time_ms": {
        "mean": round(mean_det_time, 4),
        "std": round(std_det_time, 4),
        "mc_samples": 50,
        "overhead_factor": round(mean_det_time / mean_time, 2),
    },
    "throughput": {
        "inference_samples_per_sec": round(throughput, 1),
        "detection_samples_per_sec": round(detection_throughput, 1),
    },
    "real_time_capability": {
        "target_frequency_hz": realtime_freq,
        "time_budget_ms": realtime_budget,
        "mean_inference_time_ms": round(mean_time, 4),
        "capable": str(realtime_capable),
        "headroom_factor": round(realtime_budget / mean_time, 2),
    },
    "computational_complexity": {
        "flops_per_forward_pass": flops,
        "gflops": round(gflops, 6),
    },
}

output_file = RESULTS_DIR / "computational_costs.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"  Saved: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("COMPUTATIONAL COST ANALYSIS COMPLETE")
print("=" * 60)
print("\nKEY FINDINGS:")
print(f"  Model Size: {model_size_mb:.2f} MB ({total_params:,} parameters)")
print(f"  Inference Time: {mean_time:.4f} ± {std_time:.4f} ms")
print(f"  Detection Time: {mean_det_time:.2f} ms (with MC dropout)")
print(f"  Real-time Capable: {realtime_capable} at {realtime_freq} Hz")
print(f"  Throughput: {throughput:.0f} inference/sec, {detection_throughput:.0f} detection/sec")
print(f"  Computational Cost: {gflops:.6f} GFLOPs per inference")
print("\nDEPLOYMENT FEASIBILITY:")
if realtime_capable:
    print(f"  ✓ Suitable for embedded UAV autopilots (100 Hz control loop)")
    print(f"  ✓ {round(realtime_budget / mean_time, 1)}x time budget remaining for other tasks")
else:
    print(
        f"  X Exceeds 100 Hz time budget by {round((mean_time - realtime_budget) / realtime_budget * 100, 1)}%"
    )
    print(f"  Recommendation: Reduce MC samples or use GPU acceleration")

print(f"\nResults saved to: {output_file}")
print("=" * 60)
