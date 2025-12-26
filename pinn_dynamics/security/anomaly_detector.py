"""
Physics-Informed Anomaly Detector for Cyber-Physical Systems.

Uses trained PINN models to detect sensor spoofing attacks by identifying
violations of physical laws and prediction errors weighted by uncertainty.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from ..inference.predictor import Predictor


@dataclass
class AnomalyScore:
    """Container for anomaly detection results."""

    total_score: float  # Combined anomaly score
    prediction_error: float  # ||measured - predicted||
    physics_violation: float  # Physics loss on measured state
    uncertainty: float  # Epistemic uncertainty (std)
    is_anomaly: bool  # Binary detection (score > threshold)
    components: Dict[str, float]  # Detailed breakdown


class AnomalyDetector:
    """
    Physics-informed anomaly detector for UAV sensor attacks.

    Detection strategy:
        1. Predict next state using PINN
        2. Compare measured vs predicted (prediction error)
        3. Compute physics violation on measured state
        4. Weight by epistemic uncertainty (MC dropout)
        5. Combine into anomaly score
        6. Threshold for binary detection

    Args:
        predictor: Trained Predictor instance
        threshold: Anomaly score threshold (tune on validation set)
        lambda_physics: Weight for physics violation term (default: 1.0)
        n_mc_samples: Number of MC dropout samples for uncertainty (default: 50)

    Example:
        # Train PINN on clean data
        model = QuadrotorPINN()
        # ... train model ...

        predictor = Predictor(model, scaler_X, scaler_y)
        detector = AnomalyDetector(predictor, threshold=3.0)

        # Online detection
        for state, control, next_state in data_stream:
            result = detector.detect(state, control, next_state)
            if result.is_anomaly:
                print(f"ATTACK DETECTED! Score: {result.total_score:.2f}")
    """

    def __init__(
        self,
        predictor: Predictor,
        threshold: float = 3.0,
        lambda_physics: float = 0.0,  # Default to 0 (prediction error only)
        n_mc_samples: int = 50,
        use_physics: bool = False,  # New: disable physics by default
    ):
        self.predictor = predictor
        self.threshold = threshold
        self.lambda_physics = lambda_physics
        self.n_mc_samples = n_mc_samples
        self.use_physics = use_physics

        # Statistics for normalization (fit on clean validation data)
        self.error_mean = 0.0
        self.error_std = 1.0
        self.physics_mean = 0.0
        self.physics_std = 1.0

    def calibrate(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
    ):
        """
        Calibrate detector on clean validation data.

        Computes mean/std of prediction errors and physics violations
        for normalization. This ensures anomaly scores are standardized.

        Args:
            states: [N, state_dim] clean states
            controls: [N, control_dim] controls
            next_states: [N, state_dim] clean next states
        """
        print("Calibrating anomaly detector on clean data...")

        errors = []
        physics_scores = []

        for i in range(len(states)):
            # Prediction error
            predicted = self.predictor.predict(states[i], controls[i])
            error = np.linalg.norm(next_states[i] - predicted)
            errors.append(error)

            # Physics violation (only if enabled)
            if self.use_physics:
                physics_score = self._compute_physics_violation(
                    states[i], controls[i], next_states[i]
                )
                physics_scores.append(physics_score)

        # Compute statistics
        self.error_mean = np.mean(errors)
        self.error_std = np.std(errors) + 1e-6  # avoid division by zero

        if self.use_physics:
            self.physics_mean = np.mean(physics_scores)
            self.physics_std = np.std(physics_scores) + 1e-6
            print(f"  Error: mean={self.error_mean:.4f}, std={self.error_std:.4f}")
            print(f"  Physics: mean={self.physics_mean:.4f}, std={self.physics_std:.4f}")
        else:
            print(f"  Prediction error: mean={self.error_mean:.4f}, std={self.error_std:.4f}")
            print(f"  [Note] Physics term disabled (use_physics=False)")

        print("  [OK] Calibration complete")

    def detect(
        self,
        state: np.ndarray,
        control: np.ndarray,
        measured_next_state: np.ndarray,
    ) -> AnomalyScore:
        """
        Detect anomaly in a single timestep.

        Args:
            state: [state_dim] current state
            control: [control_dim] control input
            measured_next_state: [state_dim] measured next state (potentially attacked)

        Returns:
            AnomalyScore with detection result
        """
        # 1. Prediction with uncertainty
        result = self.predictor.rollout_with_uncertainty(
            state, control.reshape(1, -1), n_samples=self.n_mc_samples, steps=1
        )
        predicted_mean = result.mean[0]
        predicted_std = result.std[0]

        # 2. Prediction error
        prediction_error = np.linalg.norm(measured_next_state - predicted_mean)

        # 3. Physics violation on measured state (optional)
        physics_violation = 0.0
        if self.use_physics:
            physics_violation = self._compute_physics_violation(
                state, control, measured_next_state
            )

        # 4. Epistemic uncertainty (average std across state dimensions)
        uncertainty = predicted_std.mean()

        # 5. Normalize prediction error
        norm_error = (prediction_error - self.error_mean) / self.error_std

        if self.use_physics:
            # Full anomaly score with physics
            norm_physics = (physics_violation - self.physics_mean) / self.physics_std
            uncertainty_weight = 1.0 / (1.0 + uncertainty)
            anomaly_score = (norm_error + self.lambda_physics * norm_physics) * uncertainty_weight
        else:
            # Simplified: normalized prediction error only
            anomaly_score = norm_error
            norm_physics = 0.0
            uncertainty_weight = 1.0

        # 6. Binary detection
        is_anomaly = anomaly_score > self.threshold

        return AnomalyScore(
            total_score=anomaly_score,
            prediction_error=prediction_error,
            physics_violation=physics_violation,
            uncertainty=uncertainty,
            is_anomaly=is_anomaly,
            components={
                "norm_error": norm_error,
                "norm_physics": norm_physics,
                "uncertainty_weight": uncertainty_weight,
            },
        )

    def _compute_physics_violation(
        self,
        state: np.ndarray,
        control: np.ndarray,
        next_state: np.ndarray,
    ) -> float:
        """
        Compute physics violation score for a state transition.

        Args:
            state: Current state
            control: Control input
            next_state: Next state (measured)

        Returns:
            Physics violation score (higher = more violation)
        """
        # Convert to torch tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.predictor.device)
        control_tensor = torch.FloatTensor(control).unsqueeze(0).to(self.predictor.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.predictor.device)

        # Concatenate input
        input_tensor = torch.cat([state_tensor, control_tensor], dim=-1)

        # Scale if scaler exists
        if self.predictor.scaler_X is not None:
            input_np = input_tensor.cpu().numpy()
            input_np = self.predictor.scaler_X.transform(input_np)
            input_tensor = torch.FloatTensor(input_np).to(self.predictor.device)

        # Scale next state
        next_state_scaled = next_state_tensor
        if self.predictor.scaler_y is not None:
            next_state_np = next_state_tensor.cpu().numpy()
            next_state_np = self.predictor.scaler_y.transform(next_state_np)
            next_state_scaled = torch.FloatTensor(next_state_np).to(self.predictor.device)

        # Compute physics loss
        with torch.no_grad():
            physics_loss = self.predictor.model.physics_loss(
                input_tensor, next_state_scaled
            )

        return physics_loss.item()

    def detect_batch(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
    ) -> list[AnomalyScore]:
        """
        Detect anomalies in a batch of transitions.

        Args:
            states: [N, state_dim]
            controls: [N, control_dim]
            next_states: [N, state_dim]

        Returns:
            List of AnomalyScore for each transition
        """
        results = []
        for i in range(len(states)):
            result = self.detect(states[i], controls[i], next_states[i])
            results.append(result)
        return results

    def evaluate(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
        true_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate detector performance on labeled test set.

        Args:
            states: [N, state_dim]
            controls: [N, control_dim]
            next_states: [N, state_dim]
            true_labels: [N] binary labels (0=normal, 1=attack)

        Returns:
            Dictionary with metrics (accuracy, precision, recall, F1, FPR, TPR)
        """
        results = self.detect_batch(states, controls, next_states)
        predicted_labels = np.array([r.is_anomaly for r in results], dtype=int)

        # Compute metrics
        TP = np.sum((predicted_labels == 1) & (true_labels == 1))
        TN = np.sum((predicted_labels == 0) & (true_labels == 0))
        FP = np.sum((predicted_labels == 1) & (true_labels == 0))
        FN = np.sum((predicted_labels == 0) & (true_labels == 1))

        accuracy = (TP + TN) / len(true_labels) if len(true_labels) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        tpr = recall  # TPR = recall

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "tpr": tpr,
            "TP": int(TP),
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN),
        }

    def tune_threshold(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
        true_labels: np.ndarray,
        metric: str = "f1",
    ) -> float:
        """
        Tune detection threshold on validation set.

        Args:
            states: [N, state_dim] validation states
            controls: [N, control_dim]
            next_states: [N, state_dim]
            true_labels: [N] binary labels
            metric: Metric to optimize ('f1', 'accuracy', 'balanced')

        Returns:
            Optimal threshold value
        """
        print(f"Tuning threshold to maximize {metric}...")

        # Compute anomaly scores
        results = self.detect_batch(states, controls, next_states)
        scores = np.array([r.total_score for r in results])

        # Try different thresholds
        thresholds = np.percentile(scores, np.linspace(10, 90, 50))
        best_threshold = self.threshold
        best_metric_value = 0

        for thresh in thresholds:
            # Temporarily set threshold
            old_thresh = self.threshold
            self.threshold = thresh

            # Evaluate
            metrics = self.evaluate(states, controls, next_states, true_labels)

            # Select metric
            if metric == "f1":
                metric_value = metrics["f1"]
            elif metric == "accuracy":
                metric_value = metrics["accuracy"]
            elif metric == "balanced":
                metric_value = (metrics["tpr"] + (1 - metrics["fpr"])) / 2
            else:
                raise ValueError(f"Unknown metric: {metric}")

            # Update best
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = thresh

            # Restore threshold
            self.threshold = old_thresh

        # Set optimal threshold
        self.threshold = best_threshold
        print(f"  Optimal threshold: {best_threshold:.4f}")
        print(f"  {metric.upper()}: {best_metric_value:.4f}")

        return best_threshold
