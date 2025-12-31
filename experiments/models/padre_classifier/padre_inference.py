"""
PADRE Fault Detection - Inference Module.

Usage:
    from padre_inference import PADREDetector

    detector = PADREDetector('models/padre_classifier')
    is_faulty, confidence = detector.predict(sensor_window)
"""

import pickle
import numpy as np
from pathlib import Path


class PADREDetector:
    """Production fault detector for PADRE sensor data."""

    def __init__(self, model_dir: str, threshold: float = 0.5):
        model_dir = Path(model_dir)

        with open(model_dir / 'rf_binary_final.pkl', 'rb') as f:
            self.clf = pickle.load(f)
        with open(model_dir / 'scaler_final.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        self.threshold = threshold
        self.window_size = 256

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """Extract features from sensor window."""
        features = []
        for col in range(min(window.shape[1], 24)):
            ch = window[:, col]
            features.extend([ch.mean(), ch.std(), ch.max() - ch.min()])
            fft = np.abs(np.fft.rfft(ch))
            features.extend([fft[1:10].sum(), fft[10:50].sum(), fft[50:].sum(),
                           np.argmax(fft[1:]) if len(fft) > 1 else 0])
        return np.array(features)

    def predict(self, window: np.ndarray) -> tuple:
        """
        Predict if sensor window indicates motor fault.

        Args:
            window: Shape (256, 24) sensor data

        Returns:
            (is_faulty: bool, confidence: float)
        """
        if window.shape[0] < self.window_size:
            raise ValueError(f"Window must have at least {self.window_size} samples")

        features = self.extract_features(window[:self.window_size])
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        proba = self.clf.predict_proba(features_scaled)[0, 1]
        is_faulty = proba >= self.threshold

        return bool(is_faulty), float(proba)

    def predict_stream(self, data: np.ndarray, stride: int = 128):
        """
        Process streaming sensor data.

        Args:
            data: Shape (n_samples, 24) sensor data
            stride: Samples between predictions

        Yields:
            (window_idx, is_faulty, confidence)
        """
        for i in range((len(data) - self.window_size) // stride + 1):
            window = data[i * stride: i * stride + self.window_size]
            is_faulty, conf = self.predict(window)
            yield i, is_faulty, conf


if __name__ == "__main__":
    # Test
    detector = PADREDetector('models/padre_classifier')

    # Simulate normal data
    test_window = np.random.randn(256, 24).astype(np.float32)
    is_faulty, conf = detector.predict(test_window)
    print(f"Test prediction: faulty={is_faulty}, confidence={conf:.3f}")
