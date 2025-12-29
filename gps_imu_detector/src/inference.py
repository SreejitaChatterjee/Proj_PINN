"""
Real-Time Inference Pipeline

Optimized for 200 Hz streaming inference:
- O(1) feature extraction
- Batched/streaming model inference
- Efficient state management

Target latency: <5ms per sample
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import time
import json


@dataclass
class InferenceResult:
    """Container for inference result."""
    anomaly_score: float
    is_anomaly: bool
    physics_score: float
    ekf_score: float
    ml_score: float
    latency_ms: float
    timestamp: float


class StreamingInferencePipeline:
    """
    Real-time streaming inference pipeline.

    Processes samples one at a time with O(1) time complexity per step.
    Maintains all necessary state for temporal features.
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = 'cpu',
        use_quantized: bool = True
    ):
        """
        Initialize streaming pipeline.

        Args:
            model_path: Path to model checkpoint
            config_path: Path to config file (optional)
            device: 'cpu' or 'cuda'
            use_quantized: Use INT8 quantized model
        """
        self.device = device
        self.use_quantized = use_quantized

        # Load config
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()

        # Initialize components
        self._init_feature_extractor()
        self._init_model(model_path)
        self._init_physics_checker()
        self._init_ekf()

        # State
        self.hidden_state = None
        self.sample_count = 0

    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'features': {
                'windows': [5, 10, 25],
                'n_raw_features': 15
            },
            'physics': {
                'dt': 0.005,
                'use_pinn': False
            },
            'ekf': {
                'dt': 0.005,
                'window_size': 50
            },
            'threshold': 0.5
        }

    def _init_feature_extractor(self):
        """Initialize streaming feature extractor."""
        from feature_extractor import StreamingFeatureExtractor

        self.feature_extractor = StreamingFeatureExtractor(
            n_features=self.config['features']['n_raw_features'],
            windows=self.config['features']['windows']
        )
        self.n_output_features = self.feature_extractor.n_output_features

    def _init_model(self, model_path: str):
        """Initialize and optionally quantize model."""
        from model import CNNGRUDetector

        # Load model
        self.model = CNNGRUDetector(input_dim=self.n_output_features)

        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model path {model_path} not found, using random weights")

        self.model.eval()

        # Quantize if requested
        if self.use_quantized:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.GRU},
                dtype=torch.qint8
            )
            print("Applied INT8 dynamic quantization")

    def _init_physics_checker(self):
        """Initialize physics residual checker."""
        from physics_residuals import AnalyticPhysicsChecker

        self.physics_checker = AnalyticPhysicsChecker(
            dt=self.config['physics']['dt']
        )

        # Rolling buffers for physics residuals
        self.physics_buffer = {
            'position': [],
            'velocity': [],
            'acceleration': [],
            'attitude': [],
            'angular_rates': []
        }
        self.physics_buffer_size = 10

    def _init_ekf(self):
        """Initialize EKF."""
        from ekf import SimpleEKF

        self.ekf = SimpleEKF(dt=self.config['ekf']['dt'])

    def reset(self):
        """Reset pipeline state."""
        self.feature_extractor.reset()
        self.hidden_state = None
        self.sample_count = 0

        # Reset physics buffers
        for key in self.physics_buffer:
            self.physics_buffer[key] = []

        # Reset EKF
        self._init_ekf()

    def process_sample(
        self,
        raw_data: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Optional[InferenceResult]:
        """
        Process single sample.

        Args:
            raw_data: [15] raw sensor data
                [pos(3), vel(3), att(3), gyro(3), accel(3)]
            timestamp: Optional timestamp

        Returns:
            InferenceResult or None if still warming up
        """
        start_time = time.perf_counter()

        if timestamp is None:
            timestamp = time.time()

        # 1. Extract features (O(1))
        features = self.feature_extractor.update(raw_data)

        if features is None:
            # Still warming up
            self.sample_count += 1
            return None

        # 2. Compute physics residuals
        physics_score = self._compute_physics_score(raw_data)

        # 3. Compute EKF NIS
        ekf_score = self._compute_ekf_score(raw_data)

        # 4. ML model inference
        ml_score = self._compute_ml_score(features)

        # 5. Combine scores
        anomaly_score = self._combine_scores(physics_score, ekf_score, ml_score)
        is_anomaly = anomaly_score > self.config['threshold']

        self.sample_count += 1
        latency_ms = (time.perf_counter() - start_time) * 1000

        return InferenceResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            physics_score=physics_score,
            ekf_score=ekf_score,
            ml_score=ml_score,
            latency_ms=latency_ms,
            timestamp=timestamp
        )

    def _compute_physics_score(self, raw_data: np.ndarray) -> float:
        """Compute physics residual score."""
        # Parse raw data
        pos = raw_data[0:3]
        vel = raw_data[3:6]
        att = raw_data[6:9]
        gyro = raw_data[9:12]
        accel = raw_data[12:15]

        # Update buffers
        self.physics_buffer['position'].append(pos)
        self.physics_buffer['velocity'].append(vel)
        self.physics_buffer['acceleration'].append(accel)
        self.physics_buffer['attitude'].append(att)
        self.physics_buffer['angular_rates'].append(gyro)

        # Trim buffers
        for key in self.physics_buffer:
            if len(self.physics_buffer[key]) > self.physics_buffer_size:
                self.physics_buffer[key].pop(0)

        # Need at least 3 samples for residuals
        if len(self.physics_buffer['position']) < 3:
            return 0.0

        # Compute residuals on buffer
        residuals = self.physics_checker.compute_residuals(
            np.array(self.physics_buffer['position']),
            np.array(self.physics_buffer['velocity']),
            np.array(self.physics_buffer['acceleration']),
            np.array(self.physics_buffer['attitude']),
            np.array(self.physics_buffer['angular_rates'])
        )

        # Return last PVA residual normalized
        pva_mag = np.linalg.norm(residuals.pva_residual[-1])
        return float(min(pva_mag / 10.0, 1.0))  # Normalize to 0-1

    def _compute_ekf_score(self, raw_data: np.ndarray) -> float:
        """Compute EKF NIS score."""
        pos = raw_data[0:3]
        vel = raw_data[3:6]
        gyro = raw_data[9:12]
        accel = raw_data[12:15]

        # Predict
        self.ekf.predict(gyro, accel)

        # Update
        nis, is_consistent = self.ekf.update_gps(pos, vel)

        # Normalize NIS (chi-squared with 6 DOF, 95% threshold ~12.6)
        nis_normalized = min(nis / 12.6, 1.0)

        return float(nis_normalized)

    def _compute_ml_score(self, features: np.ndarray) -> float:
        """Compute ML detector score."""
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            output, self.hidden_state = self.model(x, self.hidden_state)
            score = torch.sigmoid(output).item()

        return float(score)

    def _combine_scores(
        self,
        physics_score: float,
        ekf_score: float,
        ml_score: float
    ) -> float:
        """Combine component scores."""
        # Simple weighted average (could be replaced with learned weights)
        weights = {
            'physics': 0.2,
            'ekf': 0.3,
            'ml': 0.5
        }

        combined = (
            weights['physics'] * physics_score +
            weights['ekf'] * ekf_score +
            weights['ml'] * ml_score
        )

        return float(combined)


class BatchInferencePipeline:
    """
    Batch inference pipeline for offline analysis.

    More efficient for processing recorded data.
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.device = device

        # Load config
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {
                'features': {'windows': [5, 10, 25]},
                'threshold': 0.5
            }

        self._init_model(model_path)

    def _init_model(self, model_path: str):
        """Initialize model."""
        from model import CNNGRUDetector
        from feature_extractor import BatchFeatureExtractor

        # Feature extractor
        self.feature_extractor = BatchFeatureExtractor(
            windows=self.config['features']['windows']
        )

        # Determine input dim from extractor
        test_data = np.random.randn(100, 15)
        test_features = self.feature_extractor.extract(test_data)
        input_dim = test_features.shape[1]

        # Load model
        self.model = CNNGRUDetector(input_dim=input_dim)
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device).eval()

    def process_batch(
        self,
        data: np.ndarray,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process batch of data.

        Args:
            data: [N, D] raw sensor data
            batch_size: Batch size for model inference

        Returns:
            scores: [N] anomaly scores
            is_anomaly: [N] boolean flags
        """
        # Extract features
        features = self.feature_extractor.extract(data)
        n_samples = len(features)

        # Process in batches
        scores = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = features[i:i+batch_size]
                x = torch.tensor(batch, dtype=torch.float32, device=self.device)
                x = x.unsqueeze(0)  # Add batch dimension

                output, _ = self.model(x)
                batch_scores = torch.sigmoid(output).squeeze().cpu().numpy()
                scores.extend(batch_scores.flatten())

        scores = np.array(scores)
        is_anomaly = scores > self.config['threshold']

        # Pad to match original data length
        pad_size = len(data) - len(scores)
        if pad_size > 0:
            scores = np.concatenate([np.zeros(pad_size), scores])
            is_anomaly = np.concatenate([np.zeros(pad_size, dtype=bool), is_anomaly])

        return scores, is_anomaly


class ONNXInferencePipeline:
    """
    ONNX Runtime inference pipeline for deployment.
    """

    def __init__(
        self,
        onnx_path: str,
        feature_windows: List[int] = [5, 10, 25]
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed")

        self.feature_windows = feature_windows

        # Configure session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1

        self.session = ort.InferenceSession(onnx_path, sess_options)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference.

        Args:
            features: [N, D] or [B, T, D] feature array

        Returns:
            scores: Anomaly scores
        """
        if features.ndim == 2:
            features = features[np.newaxis, ...]

        features = features.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: features})

        scores = 1 / (1 + np.exp(-outputs[0]))  # Sigmoid
        return scores.squeeze()


def create_deployment_package(
    model_path: str,
    output_dir: str,
    input_dim: int = 100
) -> Dict[str, str]:
    """
    Create deployment package with all artifacts.

    Args:
        model_path: Path to trained model
        output_dir: Output directory
        input_dim: Model input dimension

    Returns:
        Dict mapping artifact name to path
    """
    from model import CNNGRUDetector
    from quantization import ModelQuantizer, ONNXExporter, TorchScriptExporter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # Load model
    model = CNNGRUDetector(input_dim=input_dim)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()

    # 1. Original PyTorch model
    torch_path = output_dir / 'model_fp32.pth'
    torch.save(model.state_dict(), torch_path)
    artifacts['pytorch_fp32'] = str(torch_path)

    # 2. Quantized model
    quantizer = ModelQuantizer(model)
    quantized = quantizer.dynamic_quantize()
    quant_path = output_dir / 'model_int8.pth'
    torch.save(quantized.state_dict(), quant_path)
    artifacts['pytorch_int8'] = str(quant_path)

    # 3. ONNX export
    onnx_exporter = ONNXExporter(model, input_dim)
    onnx_path = output_dir / 'model.onnx'
    onnx_exporter.export(str(onnx_path))
    artifacts['onnx'] = str(onnx_path)

    # 4. TorchScript
    ts_exporter = TorchScriptExporter(model)
    example_input = torch.randn(1, 50, input_dim)
    ts_path = output_dir / 'model_traced.pt'
    ts_exporter.trace(example_input, str(ts_path))
    artifacts['torchscript'] = str(ts_path)

    # 5. Configuration
    config = {
        'input_dim': input_dim,
        'feature_windows': [5, 10, 25],
        'threshold': 0.5,
        'dt': 0.005
    }
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    artifacts['config'] = str(config_path)

    print(f"\nDeployment package created at {output_dir}")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")

    return artifacts


if __name__ == "__main__":
    # Test inference pipelines
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Create test model
    from model import CNNGRUDetector

    input_dim = 100
    model = CNNGRUDetector(input_dim=input_dim)
    model_path = './test_model.pth'
    torch.save(model.state_dict(), model_path)

    # Test streaming pipeline
    print("=== Testing Streaming Pipeline ===")
    pipeline = StreamingInferencePipeline(model_path, use_quantized=True)

    # Simulate streaming data
    latencies = []
    for i in range(200):
        raw_data = np.random.randn(15).astype(np.float32)
        result = pipeline.process_sample(raw_data)

        if result:
            latencies.append(result.latency_ms)
            if i % 50 == 0:
                print(f"Sample {i}: score={result.anomaly_score:.3f}, "
                      f"latency={result.latency_ms:.3f}ms")

    if latencies:
        print(f"\nLatency stats:")
        print(f"  Mean: {np.mean(latencies):.3f} ms")
        print(f"  P99: {np.percentile(latencies, 99):.3f} ms")
        print(f"  Max: {np.max(latencies):.3f} ms")

    # Test batch pipeline
    print("\n=== Testing Batch Pipeline ===")
    batch_pipeline = BatchInferencePipeline(model_path)

    batch_data = np.random.randn(1000, 15).astype(np.float32)
    start = time.perf_counter()
    scores, is_anomaly = batch_pipeline.process_batch(batch_data)
    elapsed = time.perf_counter() - start

    print(f"Processed {len(batch_data)} samples in {elapsed*1000:.1f}ms")
    print(f"Throughput: {len(batch_data)/elapsed:.0f} samples/sec")

    # Cleanup
    Path(model_path).unlink()
