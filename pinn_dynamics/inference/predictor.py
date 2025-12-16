"""
High-level prediction interface for PINN models.

Provides the Predictor class for easy inference with optional
uncertainty quantification using Monte Carlo dropout.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Container for prediction results with optional uncertainty."""

    trajectory: np.ndarray  # [n_steps, state_dim] predicted states
    mean: Optional[np.ndarray] = None  # [n_steps, state_dim] mean prediction
    std: Optional[np.ndarray] = None  # [n_steps, state_dim] standard deviation
    samples: Optional[np.ndarray] = None  # [n_samples, n_steps, state_dim] MC samples


class Predictor:
    """
    High-level prediction interface for PINN models.

    Provides:
        - Single-step and multi-step (rollout) prediction
        - Optional feature scaling/unscaling
        - Uncertainty quantification via Monte Carlo dropout

    Args:
        model: A trained DynamicsPINN model
        scaler_X: Optional sklearn scaler for inputs
        scaler_y: Optional sklearn scaler for outputs
        device: Device for inference

    Example:
        from pinn_dynamics import QuadrotorPINN, Predictor

        model = QuadrotorPINN()
        model.load_state_dict(torch.load('model.pth'))

        predictor = Predictor(model, scaler_X, scaler_y)

        # Single step prediction
        next_state = predictor.predict(current_state, control)

        # Multi-step rollout
        trajectory = predictor.rollout(initial_state, control_sequence, steps=100)

        # With uncertainty
        result = predictor.rollout_with_uncertainty(
            initial_state, control_sequence,
            n_samples=50
        )
        print(f"Mean: {result.mean}, Std: {result.std}")
    """

    def __init__(
        self,
        model,
        scaler_X=None,
        scaler_y=None,
        device: str = "cpu",
    ):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        state: Union[np.ndarray, torch.Tensor],
        control: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Single-step prediction.

        Args:
            state: [state_dim] or [batch, state_dim] current state
            control: [control_dim] or [batch, control_dim] control input

        Returns:
            [state_dim] or [batch, state_dim] predicted next state
        """
        # Convert to numpy if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(control, torch.Tensor):
            control = control.cpu().numpy()

        # Handle single sample
        squeeze = False
        if state.ndim == 1:
            state = state.reshape(1, -1)
            control = control.reshape(1, -1)
            squeeze = True

        # Concatenate inputs
        X = np.concatenate([state, control], axis=1)

        # Scale if scaler provided
        if self.scaler_X is not None:
            X = self.scaler_X.transform(X)

        # Predict
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()

        # Unscale if scaler provided
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred)

        if squeeze:
            y_pred = y_pred.squeeze(0)

        return y_pred

    def rollout(
        self,
        initial_state: Union[np.ndarray, torch.Tensor],
        controls: Union[np.ndarray, torch.Tensor],
        steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Multi-step autoregressive rollout.

        Args:
            initial_state: [state_dim] initial state
            controls: [n_steps, control_dim] control sequence
            steps: Number of steps (default: len(controls))

        Returns:
            [n_steps, state_dim] predicted trajectory
        """
        # Convert to numpy
        if isinstance(initial_state, torch.Tensor):
            initial_state = initial_state.cpu().numpy()
        if isinstance(controls, torch.Tensor):
            controls = controls.cpu().numpy()

        # Determine steps
        if steps is None:
            steps = len(controls)

        # Rollout
        predictions = []
        state = initial_state.copy()

        for i in range(steps):
            control = controls[i] if i < len(controls) else controls[-1]
            state = self.predict(state, control)
            predictions.append(state)

        return np.array(predictions)

    def rollout_with_uncertainty(
        self,
        initial_state: Union[np.ndarray, torch.Tensor],
        controls: Union[np.ndarray, torch.Tensor],
        n_samples: int = 50,
        steps: Optional[int] = None,
    ) -> PredictionResult:
        """
        Multi-step rollout with uncertainty quantification using MC dropout.

        Args:
            initial_state: [state_dim] initial state
            controls: [n_steps, control_dim] control sequence
            n_samples: Number of MC samples for uncertainty
            steps: Number of steps

        Returns:
            PredictionResult with mean, std, and samples
        """
        # Convert to numpy
        if isinstance(initial_state, torch.Tensor):
            initial_state = initial_state.cpu().numpy()
        if isinstance(controls, torch.Tensor):
            controls = controls.cpu().numpy()

        if steps is None:
            steps = len(controls)

        # Enable dropout for MC sampling
        self.model.train()

        samples = []
        for _ in range(n_samples):
            trajectory = self._rollout_single(initial_state, controls, steps)
            samples.append(trajectory)

        # Disable dropout
        self.model.eval()

        samples = np.array(samples)  # [n_samples, n_steps, state_dim]
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)

        # Deterministic rollout for main trajectory
        trajectory = self.rollout(initial_state, controls, steps)

        return PredictionResult(
            trajectory=trajectory,
            mean=mean,
            std=std,
            samples=samples,
        )

    def _rollout_single(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        steps: int,
    ) -> np.ndarray:
        """Single rollout (internal, doesn't toggle eval mode)."""
        predictions = []
        state = initial_state.copy()

        for i in range(steps):
            control = controls[i] if i < len(controls) else controls[-1]

            # Build input
            X = np.concatenate([state, control]).reshape(1, -1)
            if self.scaler_X is not None:
                X = self.scaler_X.transform(X)

            # Predict
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                y_pred = self.model(X_tensor).cpu().numpy()

            if self.scaler_y is not None:
                y_pred = self.scaler_y.inverse_transform(y_pred)

            state = y_pred.squeeze(0)
            predictions.append(state)

        return np.array(predictions)

    def compute_rollout_error(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute rollout error against ground truth.

        Args:
            initial_state: [state_dim] initial state
            controls: [n_steps, control_dim] control sequence
            ground_truth: [n_steps, state_dim] true trajectory

        Returns:
            (mean_error, per_step_errors)
        """
        predictions = self.rollout(initial_state, controls, len(ground_truth))
        errors = np.abs(predictions - ground_truth).mean(axis=1)
        return errors.mean(), errors
