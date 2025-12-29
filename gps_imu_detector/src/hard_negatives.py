"""
Hard Negative Mining for Detector Hardening

Generates stealth attacks that evade the current detector,
then retrains to improve worst-case recall.

Attack types:
1. AR(1) slow drift (high autocorrelation)
2. Coordinated GPS+IMU co-bias
3. Intermittent on/off attacks
4. Ramp attacks below threshold
5. Adversarial perturbations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import torch
from scipy.optimize import minimize


@dataclass
class StealthAttack:
    """Container for stealth attack specification."""
    name: str
    data: np.ndarray
    labels: np.ndarray
    parameters: Dict
    evasion_score: float  # How well it evades detection


class HardNegativeGenerator:
    """
    Generate hard negative examples (stealth attacks).

    Iteratively finds attacks that evade current detector.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.attack_history = []

    def generate_ar1_drift(
        self,
        data: np.ndarray,
        ar_coef: float = 0.995,
        magnitude: float = 0.1,
        sensor_cols: Tuple[int, int] = (0, 3)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate AR(1) slow drift attack.

        Very slow drift that accumulates over time but has low
        instantaneous rate of change.

        Args:
            data: [N, D] input data
            ar_coef: AR(1) coefficient (closer to 1 = slower drift)
            magnitude: Final drift magnitude (relative to std)
            sensor_cols: (start, end) columns to attack
        """
        n = len(data)
        start_col, end_col = sensor_cols
        n_cols = end_col - start_col

        # Generate AR(1) process
        drift = np.zeros((n, n_cols))
        innovation_std = magnitude * np.std(data[:, start_col:end_col], axis=0) * (1 - ar_coef**2)**0.5

        for i in range(1, n):
            drift[i] = ar_coef * drift[i-1] + self.rng.randn(n_cols) * innovation_std

        # Normalize to target magnitude
        drift = drift / (np.std(drift, axis=0) + 1e-8) * magnitude * np.std(data[:, start_col:end_col], axis=0)

        attacked = data.copy()
        attacked[:, start_col:end_col] += drift

        labels = np.ones(n)

        return attacked, labels

    def generate_coordinated_attack(
        self,
        data: np.ndarray,
        magnitude: float = 0.5,
        consistency_factor: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate coordinated GPS+IMU attack.

        Attacks multiple sensors consistently to maintain cross-modal
        consistency and evade sensor fusion detection.

        Args:
            data: [N, D] input data (assumes [pos(3), vel(3), att(3), rate(3), acc(3)])
            magnitude: Attack magnitude
            consistency_factor: How well to maintain consistency (1.0 = perfect)
        """
        n = len(data)
        attacked = data.copy()

        # Compute baseline statistics
        pos_std = np.std(data[:, 0:3], axis=0)
        vel_std = np.std(data[:, 3:6], axis=0)
        acc_std = np.std(data[:, 12:15], axis=0)

        # Generate coordinated bias
        pos_bias = magnitude * pos_std * self.rng.randn(3)

        # Velocity bias should be consistent with position bias rate of change
        # For a constant position bias, velocity bias should be zero
        # But attacker adds small consistent velocity bias
        vel_bias = magnitude * vel_std * self.rng.randn(3) * (1 - consistency_factor)

        # Acceleration bias should be consistent with velocity bias
        acc_bias = magnitude * acc_std * self.rng.randn(3) * (1 - consistency_factor)

        # Apply coordinated attack
        attacked[:, 0:3] += pos_bias
        attacked[:, 3:6] += vel_bias
        attacked[:, 12:15] += acc_bias

        labels = np.ones(n)

        return attacked, labels

    def generate_intermittent_attack(
        self,
        data: np.ndarray,
        on_probability: float = 0.1,
        on_duration_mean: int = 10,
        magnitude: float = 1.0,
        sensor_cols: Tuple[int, int] = (0, 3)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate intermittent on/off attack.

        Short bursts of attack interspersed with normal operation.
        Hard to detect because statistical properties remain mostly normal.

        Args:
            data: [N, D] input data
            on_probability: Probability of attack starting
            on_duration_mean: Mean duration of attack burst
            magnitude: Attack magnitude during active period
            sensor_cols: Columns to attack
        """
        n = len(data)
        start_col, end_col = sensor_cols
        n_cols = end_col - start_col

        attacked = data.copy()
        labels = np.zeros(n)

        baseline_std = np.std(data[:, start_col:end_col], axis=0)

        # Generate attack schedule
        i = 0
        while i < n:
            if self.rng.rand() < on_probability:
                # Start attack burst
                duration = max(1, int(self.rng.exponential(on_duration_mean)))
                end_idx = min(i + duration, n)

                # Apply attack
                offset = magnitude * baseline_std * self.rng.randn(n_cols)
                attacked[i:end_idx, start_col:end_col] += offset
                labels[i:end_idx] = 1

                i = end_idx
            else:
                i += 1

        return attacked, labels

    def generate_below_threshold_ramp(
        self,
        data: np.ndarray,
        threshold_estimate: float = 0.5,
        safety_margin: float = 0.8,
        sensor_cols: Tuple[int, int] = (0, 3)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ramp attack that stays below detection threshold.

        Args:
            data: [N, D] input data
            threshold_estimate: Estimated detector threshold
            safety_margin: Stay this fraction below threshold
            sensor_cols: Columns to attack
        """
        n = len(data)
        start_col, end_col = sensor_cols
        n_cols = end_col - start_col

        attacked = data.copy()
        baseline_std = np.std(data[:, start_col:end_col], axis=0)

        # Compute max safe magnitude
        max_magnitude = threshold_estimate * safety_margin

        # Generate slow linear ramp
        ramp = np.linspace(0, max_magnitude, n).reshape(-1, 1)
        attacked[:, start_col:end_col] += ramp * baseline_std

        labels = np.ones(n)

        return attacked, labels

    def find_evasive_attacks(
        self,
        data: np.ndarray,
        detector_fn: Callable[[np.ndarray], np.ndarray],
        n_attempts: int = 20,
        target_evasion: float = 0.9
    ) -> List[StealthAttack]:
        """
        Find attacks that evade the given detector.

        Args:
            data: [N, D] clean input data
            detector_fn: Function that returns anomaly scores [N]
            n_attempts: Number of attack variants to try
            target_evasion: Target evasion rate (fraction undetected)

        Returns:
            List of successful evasive attacks
        """
        evasive_attacks = []

        attack_generators = [
            ('ar1_drift_slow', lambda: self.generate_ar1_drift(data, ar_coef=0.999, magnitude=0.2)),
            ('ar1_drift_medium', lambda: self.generate_ar1_drift(data, ar_coef=0.995, magnitude=0.3)),
            ('coordinated_high', lambda: self.generate_coordinated_attack(data, magnitude=0.5, consistency_factor=0.95)),
            ('coordinated_med', lambda: self.generate_coordinated_attack(data, magnitude=0.3, consistency_factor=0.9)),
            ('intermittent_sparse', lambda: self.generate_intermittent_attack(data, on_probability=0.05, magnitude=1.0)),
            ('intermittent_dense', lambda: self.generate_intermittent_attack(data, on_probability=0.15, magnitude=0.5)),
            ('below_threshold', lambda: self.generate_below_threshold_ramp(data, threshold_estimate=0.5)),
        ]

        for name, gen_fn in attack_generators:
            for attempt in range(n_attempts // len(attack_generators) + 1):
                attacked, labels = gen_fn()

                # Get detector scores
                scores = detector_fn(attacked)

                # Compute evasion rate (fraction with score < 0.5)
                attack_mask = labels == 1
                if np.sum(attack_mask) > 0:
                    evasion_rate = np.mean(scores[attack_mask] < 0.5)

                    if evasion_rate >= target_evasion:
                        evasive_attacks.append(StealthAttack(
                            name=f"{name}_v{attempt}",
                            data=attacked,
                            labels=labels,
                            parameters={'evasion_target': target_evasion},
                            evasion_score=evasion_rate
                        ))

        # Sort by evasion score (higher = harder to detect)
        evasive_attacks.sort(key=lambda x: x.evasion_score, reverse=True)

        return evasive_attacks


class AdversarialAttackGenerator:
    """
    Generate adversarial perturbations using gradient-based optimization.

    Finds minimal perturbations that reduce detector score.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def generate_adversarial(
        self,
        model: torch.nn.Module,
        data: np.ndarray,
        epsilon: float = 0.1,
        n_steps: int = 20,
        step_size: float = 0.01
    ) -> Tuple[np.ndarray, float]:
        """
        Generate adversarial perturbation using PGD.

        Args:
            model: PyTorch detector model
            data: [N, D] input data
            epsilon: Maximum perturbation magnitude
            n_steps: Number of PGD steps
            step_size: Step size for gradient descent

        Returns:
            perturbed_data, perturbation_magnitude
        """
        model.eval()

        # Convert to tensor
        x = torch.tensor(data, dtype=torch.float32, device=self.device, requires_grad=True)
        x_orig = x.clone().detach()

        for step in range(n_steps):
            x.requires_grad_(True)

            # Forward pass
            if x.dim() == 2:
                x_input = x.unsqueeze(0)
            else:
                x_input = x

            output, _ = model(x_input)
            score = torch.sigmoid(output).mean()

            # We want to MINIMIZE score (make attack undetectable)
            loss = score

            # Backward
            model.zero_grad()
            loss.backward()

            # PGD step
            with torch.no_grad():
                grad = x.grad.sign()
                x = x - step_size * grad

                # Project back to epsilon ball
                delta = x - x_orig
                delta = torch.clamp(delta, -epsilon, epsilon)
                x = x_orig + delta

        perturbed = x.detach().cpu().numpy()
        perturbation_mag = np.mean(np.abs(perturbed - data))

        return perturbed, perturbation_mag

    def evaluate_adversarial_robustness(
        self,
        model: torch.nn.Module,
        data: np.ndarray,
        labels: np.ndarray,
        epsilon_values: List[float] = [0.01, 0.05, 0.1, 0.2]
    ) -> Dict[str, float]:
        """
        Evaluate model robustness to adversarial attacks.

        Args:
            model: Detector model
            data: [N, D] attacked data
            labels: [N] ground truth labels
            epsilon_values: List of epsilon values to test

        Returns:
            Dict with recall at each epsilon
        """
        results = {}

        for epsilon in epsilon_values:
            perturbed, _ = self.generate_adversarial(model, data, epsilon=epsilon)

            # Get scores on perturbed data
            model.eval()
            with torch.no_grad():
                x = torch.tensor(perturbed, dtype=torch.float32, device=self.device)
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                output, _ = model(x)
                scores = torch.sigmoid(output).squeeze().cpu().numpy()

            # Compute recall on attacked samples
            attack_mask = labels == 1
            if np.sum(attack_mask) > 0:
                recall = np.mean(scores[attack_mask] > 0.5)
                results[f'recall@eps={epsilon}'] = recall

        return results


class DomainRandomizer:
    """
    Domain randomization for improved generalization.

    Randomizes sensor noise, sampling jitter, and motion regimes.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def randomize_noise(
        self,
        data: np.ndarray,
        noise_scale_range: Tuple[float, float] = (0.5, 2.0)
    ) -> np.ndarray:
        """
        Randomize sensor noise level.

        Args:
            data: [N, D] input data
            noise_scale_range: (min, max) scale for noise
        """
        scale = self.rng.uniform(*noise_scale_range)
        baseline_std = np.std(data, axis=0)
        noise = self.rng.randn(*data.shape) * baseline_std * scale * 0.1
        return data + noise

    def randomize_sampling_jitter(
        self,
        data: np.ndarray,
        jitter_std: float = 0.001
    ) -> np.ndarray:
        """
        Simulate sampling time jitter via interpolation.

        Args:
            data: [N, D] input data
            jitter_std: Standard deviation of timing jitter
        """
        n = len(data)
        # Jittered sample times
        jitter = self.rng.randn(n) * jitter_std
        jittered_times = np.arange(n) + jitter
        jittered_times = np.clip(jittered_times, 0, n-1)

        # Interpolate
        from scipy.interpolate import interp1d
        result = np.zeros_like(data)
        for col in range(data.shape[1]):
            f = interp1d(np.arange(n), data[:, col], kind='linear', fill_value='extrapolate')
            result[:, col] = f(jittered_times)

        return result

    def randomize_motion_regime(
        self,
        data: np.ndarray,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        offset_range: Tuple[float, float] = (-0.5, 0.5)
    ) -> np.ndarray:
        """
        Randomize motion regime (scale and offset).

        Args:
            data: [N, D] input data
            scale_range: Range for scaling motion
            offset_range: Range for offsetting
        """
        scale = self.rng.uniform(*scale_range, size=data.shape[1])
        offset = self.rng.uniform(*offset_range, size=data.shape[1]) * np.std(data, axis=0)

        return data * scale + offset

    def augment_batch(
        self,
        data: np.ndarray,
        augment_prob: float = 0.5
    ) -> np.ndarray:
        """
        Apply random augmentations with given probability.

        Args:
            data: [N, D] input data
            augment_prob: Probability of each augmentation
        """
        result = data.copy()

        if self.rng.rand() < augment_prob:
            result = self.randomize_noise(result)

        if self.rng.rand() < augment_prob:
            result = self.randomize_sampling_jitter(result)

        if self.rng.rand() < augment_prob:
            result = self.randomize_motion_regime(result)

        return result


if __name__ == "__main__":
    # Test hard negative generation
    n = 1000
    d = 15

    data = np.random.randn(n, d)

    generator = HardNegativeGenerator(seed=42)

    # Test AR1 drift
    attacked, labels = generator.generate_ar1_drift(data, ar_coef=0.995, magnitude=0.5)
    print(f"AR1 drift: {np.sum(labels)} attacked samples")

    # Test coordinated
    attacked, labels = generator.generate_coordinated_attack(data, magnitude=0.5)
    print(f"Coordinated: {np.sum(labels)} attacked samples")

    # Test intermittent
    attacked, labels = generator.generate_intermittent_attack(data, on_probability=0.1)
    print(f"Intermittent: {np.sum(labels)} attacked samples")

    # Test domain randomization
    randomizer = DomainRandomizer(seed=42)
    augmented = randomizer.augment_batch(data)
    print(f"Augmented data shape: {augmented.shape}")
