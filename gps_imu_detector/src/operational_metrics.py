"""
Operational Metrics for Deployment

Computes deployment-relevant metrics:
1. Latency CDF (Cumulative Distribution Function)
2. False alarms per hour
3. CPU utilization
4. Memory footprint
5. Detection delay distribution

These metrics are critical for real-world deployment decisions.

Usage:
    profiler = OperationalProfiler(model, sample_rate_hz=200)
    metrics = profiler.run_full_profile(test_data)
    print(metrics.summary())
"""

import numpy as np
import torch
import time
import psutil
import os
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class LatencyStats:
    """Latency statistics."""
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float
    samples: int
    cdf_values: np.ndarray = field(default_factory=lambda: np.array([]))
    cdf_percentiles: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class FalseAlarmStats:
    """False alarm statistics."""
    total_false_alarms: int
    total_normal_samples: int
    false_alarm_rate: float
    false_alarms_per_hour: float
    mean_time_between_false_alarms_sec: float
    sample_rate_hz: float


@dataclass
class DetectionDelayStats:
    """Detection delay statistics."""
    mean_delay_samples: float
    std_delay_samples: float
    mean_delay_ms: float
    p50_delay_ms: float
    p95_delay_ms: float
    delays_per_attack: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResourceStats:
    """Resource utilization statistics."""
    cpu_percent_mean: float
    cpu_percent_max: float
    memory_mb_mean: float
    memory_mb_max: float
    model_size_mb: float


@dataclass
class OperationalMetrics:
    """Complete operational metrics."""
    latency: LatencyStats
    false_alarms: FalseAlarmStats
    detection_delay: DetectionDelayStats
    resources: ResourceStats
    meets_latency_target: bool
    target_latency_ms: float

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("OPERATIONAL METRICS SUMMARY")
        lines.append("=" * 60)

        lines.append("\n[LATENCY]")
        lines.append(f"  Mean:   {self.latency.mean_ms:.2f} ms")
        lines.append(f"  P95:    {self.latency.p95_ms:.2f} ms")
        lines.append(f"  P99:    {self.latency.p99_ms:.2f} ms")
        lines.append(f"  Target: {self.target_latency_ms:.2f} ms")
        lines.append(f"  Status: {'PASS' if self.meets_latency_target else 'FAIL'}")

        lines.append("\n[FALSE ALARMS]")
        lines.append(f"  Rate:     {self.false_alarms.false_alarm_rate:.4f}")
        lines.append(f"  Per hour: {self.false_alarms.false_alarms_per_hour:.1f}")
        lines.append(f"  MTBFA:    {self.false_alarms.mean_time_between_false_alarms_sec:.1f} sec")

        lines.append("\n[DETECTION DELAY]")
        lines.append(f"  Mean: {self.detection_delay.mean_delay_ms:.1f} ms")
        lines.append(f"  P95:  {self.detection_delay.p95_delay_ms:.1f} ms")

        lines.append("\n[RESOURCES]")
        lines.append(f"  CPU (mean):  {self.resources.cpu_percent_mean:.1f}%")
        lines.append(f"  Memory:      {self.resources.memory_mb_mean:.1f} MB")
        lines.append(f"  Model size:  {self.resources.model_size_mb:.2f} MB")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export to JSON."""
        return json.dumps({
            'latency': {
                'mean_ms': self.latency.mean_ms,
                'p95_ms': self.latency.p95_ms,
                'p99_ms': self.latency.p99_ms,
                'max_ms': self.latency.max_ms
            },
            'false_alarms': {
                'rate': self.false_alarms.false_alarm_rate,
                'per_hour': self.false_alarms.false_alarms_per_hour,
                'mtbfa_sec': self.false_alarms.mean_time_between_false_alarms_sec
            },
            'detection_delay': {
                'mean_ms': self.detection_delay.mean_delay_ms,
                'p95_ms': self.detection_delay.p95_delay_ms
            },
            'resources': {
                'cpu_percent': self.resources.cpu_percent_mean,
                'memory_mb': self.resources.memory_mb_mean,
                'model_size_mb': self.resources.model_size_mb
            },
            'meets_latency_target': self.meets_latency_target
        }, indent=2)


class OperationalProfiler:
    """Profile model for operational deployment metrics."""

    def __init__(
        self,
        model: torch.nn.Module,
        sample_rate_hz: float = 200.0,
        target_latency_ms: float = 5.0
    ):
        """
        Initialize profiler.

        Args:
            model: PyTorch model to profile
            sample_rate_hz: Sensor sample rate
            target_latency_ms: Target latency constraint
        """
        self.model = model
        self.sample_rate_hz = sample_rate_hz
        self.target_latency_ms = target_latency_ms
        self.device = next(model.parameters()).device if list(model.parameters()) else 'cpu'

    def profile_latency(
        self,
        input_shape: Tuple[int, ...],
        n_warmup: int = 50,
        n_iterations: int = 500
    ) -> LatencyStats:
        """
        Profile inference latency with CDF.

        Args:
            input_shape: Shape of input tensor
            n_warmup: Warmup iterations (excluded from timing)
            n_iterations: Benchmark iterations

        Returns:
            LatencyStats with full CDF
        """
        self.model.eval()

        # Create random input
        x = torch.randn(*input_shape, device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(x)

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(n_iterations):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = self.model(x)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        latencies = np.array(latencies)

        # Compute CDF
        percentiles = np.arange(0, 101, 1)
        cdf_values = np.percentile(latencies, percentiles)

        return LatencyStats(
            mean_ms=float(np.mean(latencies)),
            std_ms=float(np.std(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p95_ms=float(np.percentile(latencies, 95)),
            p99_ms=float(np.percentile(latencies, 99)),
            max_ms=float(np.max(latencies)),
            min_ms=float(np.min(latencies)),
            samples=n_iterations,
            cdf_values=cdf_values,
            cdf_percentiles=percentiles
        )

    def compute_false_alarms(
        self,
        normal_predictions: np.ndarray,
        threshold: float
    ) -> FalseAlarmStats:
        """
        Compute false alarm statistics.

        Args:
            normal_predictions: Model predictions on normal data
            threshold: Detection threshold

        Returns:
            FalseAlarmStats
        """
        false_alarms = (normal_predictions > threshold).astype(int)
        total_fa = false_alarms.sum()
        total_samples = len(normal_predictions)
        fa_rate = total_fa / total_samples

        # Compute false alarms per hour
        samples_per_hour = self.sample_rate_hz * 3600
        fa_per_hour = fa_rate * samples_per_hour

        # Mean time between false alarms
        if total_fa > 0:
            mtbfa_samples = total_samples / total_fa
            mtbfa_sec = mtbfa_samples / self.sample_rate_hz
        else:
            mtbfa_sec = float('inf')

        return FalseAlarmStats(
            total_false_alarms=int(total_fa),
            total_normal_samples=total_samples,
            false_alarm_rate=float(fa_rate),
            false_alarms_per_hour=float(fa_per_hour),
            mean_time_between_false_alarms_sec=float(mtbfa_sec),
            sample_rate_hz=self.sample_rate_hz
        )

    def compute_detection_delay(
        self,
        attack_predictions: Dict[str, np.ndarray],
        attack_labels: Dict[str, np.ndarray],
        threshold: float
    ) -> DetectionDelayStats:
        """
        Compute detection delay statistics.

        Detection delay = time from attack start to first detection.

        Args:
            attack_predictions: Dict of attack_type -> predictions
            attack_labels: Dict of attack_type -> binary labels
            threshold: Detection threshold

        Returns:
            DetectionDelayStats
        """
        all_delays = []
        delays_per_attack = {}

        for attack_type, preds in attack_predictions.items():
            labels = attack_labels[attack_type]
            detections = preds > threshold

            # Find attack start indices
            attack_starts = np.where(np.diff(labels.astype(int)) == 1)[0] + 1
            if len(attack_starts) == 0 and labels[0] == 1:
                attack_starts = np.array([0])

            attack_delays = []
            for start in attack_starts:
                # Find first detection after attack starts
                detection_after_start = np.where(detections[start:])[0]
                if len(detection_after_start) > 0:
                    delay = detection_after_start[0]
                else:
                    # Never detected - use remaining length as delay
                    delay = len(labels) - start
                attack_delays.append(delay)
                all_delays.append(delay)

            if attack_delays:
                delays_per_attack[attack_type] = float(np.mean(attack_delays))

        all_delays = np.array(all_delays) if all_delays else np.array([0])

        # Convert to ms
        sample_period_ms = 1000.0 / self.sample_rate_hz

        return DetectionDelayStats(
            mean_delay_samples=float(np.mean(all_delays)),
            std_delay_samples=float(np.std(all_delays)),
            mean_delay_ms=float(np.mean(all_delays) * sample_period_ms),
            p50_delay_ms=float(np.percentile(all_delays, 50) * sample_period_ms),
            p95_delay_ms=float(np.percentile(all_delays, 95) * sample_period_ms),
            delays_per_attack=delays_per_attack
        )

    def profile_resources(
        self,
        input_shape: Tuple[int, ...],
        duration_sec: float = 5.0
    ) -> ResourceStats:
        """
        Profile CPU and memory usage.

        Args:
            input_shape: Input shape for inference
            duration_sec: Duration to run profiling

        Returns:
            ResourceStats
        """
        self.model.eval()

        # Get model size
        model_size_mb = self._get_model_size_mb()

        # Profile during inference
        cpu_samples = []
        memory_samples = []

        x = torch.randn(*input_shape, device=self.device)
        process = psutil.Process(os.getpid())

        start_time = time.time()
        with torch.no_grad():
            while time.time() - start_time < duration_sec:
                _ = self.model(x)
                cpu_samples.append(process.cpu_percent())
                memory_samples.append(process.memory_info().rss / 1024 / 1024)

        return ResourceStats(
            cpu_percent_mean=float(np.mean(cpu_samples)) if cpu_samples else 0.0,
            cpu_percent_max=float(np.max(cpu_samples)) if cpu_samples else 0.0,
            memory_mb_mean=float(np.mean(memory_samples)) if memory_samples else 0.0,
            memory_mb_max=float(np.max(memory_samples)) if memory_samples else 0.0,
            model_size_mb=model_size_mb
        )

    def _get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / 1024 / 1024

    def run_full_profile(
        self,
        input_shape: Tuple[int, ...],
        normal_predictions: Optional[np.ndarray] = None,
        attack_predictions: Optional[Dict[str, np.ndarray]] = None,
        attack_labels: Optional[Dict[str, np.ndarray]] = None,
        threshold: float = 0.5
    ) -> OperationalMetrics:
        """
        Run complete operational profiling.

        Args:
            input_shape: Input tensor shape
            normal_predictions: Predictions on normal data (for false alarms)
            attack_predictions: Dict of attack predictions (for delay)
            attack_labels: Dict of attack labels
            threshold: Detection threshold

        Returns:
            Complete OperationalMetrics
        """
        print("Profiling latency...")
        latency = self.profile_latency(input_shape)

        print("Computing false alarms...")
        if normal_predictions is not None:
            false_alarms = self.compute_false_alarms(normal_predictions, threshold)
        else:
            # Generate synthetic for demo
            normal_preds = np.random.rand(10000) * 0.5
            false_alarms = self.compute_false_alarms(normal_preds, threshold)

        print("Computing detection delay...")
        if attack_predictions is not None and attack_labels is not None:
            detection_delay = self.compute_detection_delay(
                attack_predictions, attack_labels, threshold
            )
        else:
            # Dummy values
            detection_delay = DetectionDelayStats(
                mean_delay_samples=5.0,
                std_delay_samples=2.0,
                mean_delay_ms=25.0,
                p50_delay_ms=20.0,
                p95_delay_ms=50.0,
                delays_per_attack={}
            )

        print("Profiling resources...")
        resources = self.profile_resources(input_shape, duration_sec=3.0)

        meets_target = latency.p99_ms <= self.target_latency_ms

        return OperationalMetrics(
            latency=latency,
            false_alarms=false_alarms,
            detection_delay=detection_delay,
            resources=resources,
            meets_latency_target=meets_target,
            target_latency_ms=self.target_latency_ms
        )


def plot_latency_cdf(latency_stats: LatencyStats, save_path: Optional[str] = None):
    """Plot latency CDF."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(latency_stats.cdf_values, latency_stats.cdf_percentiles,
                'b-', linewidth=2, label='Latency CDF')

        # Add target line
        ax.axvline(x=5.0, color='r', linestyle='--', label='Target (5ms)')

        # Add percentile markers
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=99, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Percentile')
        ax.set_title('Inference Latency CDF')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved CDF plot to {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("matplotlib not available for plotting")


if __name__ == "__main__":
    # Demo with simple model
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self, input_dim=100):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return torch.sigmoid(self.fc(x))

    model = SimpleModel(input_dim=100)
    profiler = OperationalProfiler(model, sample_rate_hz=200, target_latency_ms=5.0)

    # Run full profile
    metrics = profiler.run_full_profile(input_shape=(1, 100))

    print(metrics.summary())
    print("\nJSON output:")
    print(metrics.to_json())

    # Plot CDF
    plot_latency_cdf(metrics.latency, save_path=None)
