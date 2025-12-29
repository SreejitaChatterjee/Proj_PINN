"""
Physics Residual Computation

Analytic physics constraints:
1. Bounded jerk (derivative of acceleration)
2. Position-velocity-acceleration consistency
3. Energy conservation bounds
4. Attitude-rate consistency

Optional: Lightweight PINN for learned residuals.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class PhysicsResiduals:
    """Container for physics residual values."""
    jerk_residual: np.ndarray          # [N, 3] - bounded jerk violation
    pva_residual: np.ndarray           # [N, 3] - pos-vel-acc consistency
    energy_residual: np.ndarray        # [N, 1] - energy conservation
    attitude_rate_residual: np.ndarray # [N, 3] - attitude-rate consistency
    total_residual: np.ndarray         # [N, 1] - weighted sum


class AnalyticPhysicsChecker:
    """
    Analytic physics constraint checker.

    No learning required - uses known physical laws.
    """

    def __init__(
        self,
        dt: float = 0.005,
        max_jerk: float = 100.0,       # m/s^3
        max_accel: float = 50.0,        # m/s^2
        energy_tolerance: float = 0.1,  # relative
        gravity: float = 9.81
    ):
        self.dt = dt
        self.max_jerk = max_jerk
        self.max_accel = max_accel
        self.energy_tolerance = energy_tolerance
        self.g = gravity

    def compute_residuals(
        self,
        position: np.ndarray,    # [N, 3]
        velocity: np.ndarray,    # [N, 3]
        acceleration: np.ndarray, # [N, 3]
        attitude: np.ndarray,     # [N, 3] - roll, pitch, yaw
        angular_rates: np.ndarray # [N, 3] - p, q, r
    ) -> PhysicsResiduals:
        """
        Compute all physics residuals.

        Args:
            position: [N, 3] position (x, y, z)
            velocity: [N, 3] velocity (vx, vy, vz)
            acceleration: [N, 3] acceleration (ax, ay, az)
            attitude: [N, 3] attitude (roll, pitch, yaw)
            angular_rates: [N, 3] angular rates (p, q, r)

        Returns:
            PhysicsResiduals with all residual types
        """
        n = len(position)

        # 1. Jerk residual (rate of change of acceleration)
        jerk_residual = self._compute_jerk_residual(acceleration)

        # 2. Position-Velocity-Acceleration consistency
        pva_residual = self._compute_pva_residual(position, velocity, acceleration)

        # 3. Energy conservation residual
        energy_residual = self._compute_energy_residual(position, velocity, acceleration)

        # 4. Attitude-rate consistency
        att_rate_residual = self._compute_attitude_rate_residual(attitude, angular_rates)

        # Combine into total residual
        total_residual = (
            np.linalg.norm(jerk_residual, axis=1, keepdims=True) / self.max_jerk +
            np.linalg.norm(pva_residual, axis=1, keepdims=True) * 10 +
            np.abs(energy_residual) +
            np.linalg.norm(att_rate_residual, axis=1, keepdims=True)
        )

        return PhysicsResiduals(
            jerk_residual=jerk_residual,
            pva_residual=pva_residual,
            energy_residual=energy_residual,
            attitude_rate_residual=att_rate_residual,
            total_residual=total_residual
        )

    def _compute_jerk_residual(self, acceleration: np.ndarray) -> np.ndarray:
        """Compute jerk and check against bounds."""
        n = len(acceleration)
        jerk = np.zeros_like(acceleration)

        if n > 1:
            jerk[1:] = np.diff(acceleration, axis=0) / self.dt

        # Residual: how much jerk exceeds bound
        jerk_magnitude = np.linalg.norm(jerk, axis=1, keepdims=True)
        excess = np.maximum(0, jerk_magnitude - self.max_jerk)
        residual = jerk * (excess / (jerk_magnitude + 1e-8))

        return residual

    def _compute_pva_residual(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray
    ) -> np.ndarray:
        """
        Check position-velocity-acceleration consistency.

        pos_dot should equal velocity
        vel_dot should equal acceleration
        """
        n = len(position)
        residual = np.zeros((n, 3))

        if n > 1:
            # Velocity should match position derivative
            pos_dot = np.diff(position, axis=0) / self.dt
            vel_error = velocity[:-1] - pos_dot
            residual[:-1] += vel_error

            # Acceleration should match velocity derivative
            vel_dot = np.diff(velocity, axis=0) / self.dt
            acc_error = acceleration[:-1] - vel_dot
            residual[:-1] += acc_error

        return residual

    def _compute_energy_residual(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray
    ) -> np.ndarray:
        """
        Check energy conservation (approximately).

        For a free-falling body with drag:
        dE/dt = F_drag * v (energy dissipation)
        Large jumps in energy indicate anomaly.
        """
        n = len(position)
        residual = np.zeros((n, 1))

        # Kinetic energy (assuming unit mass)
        ke = 0.5 * np.sum(velocity ** 2, axis=1)

        # Potential energy
        pe = self.g * position[:, 2]  # z is up

        # Total energy
        total_energy = ke + pe

        if n > 1:
            # Energy change rate
            de_dt = np.diff(total_energy) / self.dt

            # Power from acceleration (F*v = m*a*v)
            power = np.sum(acceleration[:-1] * velocity[:-1], axis=1)

            # Residual: unexplained energy change
            residual[:-1, 0] = de_dt - power

        return residual

    def _compute_attitude_rate_residual(
        self,
        attitude: np.ndarray,
        angular_rates: np.ndarray
    ) -> np.ndarray:
        """
        Check attitude-rate consistency.

        For small angles: d(attitude)/dt ≈ angular_rates
        """
        n = len(attitude)
        residual = np.zeros((n, 3))

        if n > 1:
            # Attitude derivative
            att_dot = np.diff(attitude, axis=0) / self.dt

            # Should match angular rates (simplified, for small angles)
            residual[:-1] = att_dot - angular_rates[:-1]

        return residual


class LightweightPINN(nn.Module):
    """
    Lightweight PINN for learned physics residuals.

    Optimized for CPU: 2-3 layers, small hidden size.
    """

    def __init__(
        self,
        input_dim: int = 15,  # pos(3) + vel(3) + att(3) + rate(3) + acc(3)
        hidden_size: int = 64,
        output_dim: int = 15,
        num_layers: int = 2
    ):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
        layers.append(nn.Linear(hidden_size, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def compute_residual(
        self,
        current_state: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute residual: actual_next - predicted_next."""
        predicted_next = self.forward(current_state)
        return next_state - predicted_next


class HybridPhysicsChecker:
    """
    Hybrid physics checker combining analytic and learned residuals.
    """

    def __init__(
        self,
        dt: float = 0.005,
        use_pinn: bool = False,
        pinn_checkpoint: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.dt = dt
        self.device = device

        # Analytic checker
        self.analytic = AnalyticPhysicsChecker(dt=dt)

        # Optional PINN
        self.use_pinn = use_pinn
        self.pinn = None
        if use_pinn:
            self.pinn = LightweightPINN()
            if pinn_checkpoint:
                self.pinn.load_state_dict(torch.load(pinn_checkpoint, map_location=device))
            self.pinn.to(device)
            self.pinn.eval()

    def compute_all_residuals(
        self,
        data: np.ndarray  # [N, 15] - all state
    ) -> Dict[str, np.ndarray]:
        """
        Compute all residuals (analytic + PINN).

        Args:
            data: [N, 15] state data

        Returns:
            Dict with residual arrays
        """
        position = data[:, 0:3]
        velocity = data[:, 3:6]
        attitude = data[:, 6:9]
        angular_rates = data[:, 9:12]
        acceleration = data[:, 12:15]

        # Analytic residuals
        analytic_res = self.analytic.compute_residuals(
            position, velocity, acceleration, attitude, angular_rates
        )

        result = {
            'jerk': analytic_res.jerk_residual,
            'pva': analytic_res.pva_residual,
            'energy': analytic_res.energy_residual,
            'attitude_rate': analytic_res.attitude_rate_residual,
            'analytic_total': analytic_res.total_residual,
        }

        # PINN residuals
        if self.use_pinn and self.pinn is not None:
            pinn_res = self._compute_pinn_residuals(data)
            result['pinn'] = pinn_res
            result['pinn_norm'] = np.linalg.norm(pinn_res, axis=1, keepdims=True)

        return result

    def _compute_pinn_residuals(self, data: np.ndarray) -> np.ndarray:
        """Compute PINN prediction residuals."""
        n = len(data)
        residuals = np.zeros((n, 15))

        if n < 2:
            return residuals

        with torch.no_grad():
            current = torch.tensor(data[:-1], dtype=torch.float32, device=self.device)
            next_state = torch.tensor(data[1:, :15], dtype=torch.float32, device=self.device)

            predicted = self.pinn(current)
            res = (next_state - predicted).cpu().numpy()
            residuals[:-1] = res

        return residuals


if __name__ == "__main__":
    # Test physics checker
    n = 1000
    dt = 0.005

    # Generate test data
    t = np.arange(n) * dt
    position = np.column_stack([np.sin(t), np.cos(t), t * 0.1])
    velocity = np.column_stack([np.cos(t), -np.sin(t), np.ones(n) * 0.1])
    acceleration = np.column_stack([-np.sin(t), -np.cos(t), np.zeros(n)])
    attitude = np.column_stack([0.1 * np.sin(2*t), 0.1 * np.cos(2*t), np.zeros(n)])
    angular_rates = np.column_stack([0.2 * np.cos(2*t), -0.2 * np.sin(2*t), np.zeros(n)])

    # Test analytic checker
    checker = AnalyticPhysicsChecker(dt=dt)
    residuals = checker.compute_residuals(position, velocity, acceleration, attitude, angular_rates)

    print("Analytic Physics Residuals:")
    print(f"  Jerk: mean={np.mean(np.abs(residuals.jerk_residual)):.4f}")
    print(f"  PVA: mean={np.mean(np.abs(residuals.pva_residual)):.4f}")
    print(f"  Energy: mean={np.mean(np.abs(residuals.energy_residual)):.4f}")
    print(f"  Att-Rate: mean={np.mean(np.abs(residuals.attitude_rate_residual)):.4f}")
    print(f"  Total: mean={np.mean(residuals.total_residual):.4f}")

    # Test with attack
    print("\n--- With Bias Attack ---")
    attacked_position = position.copy()
    attacked_position[:, 0] += 0.5  # Bias on x

    residuals_attack = checker.compute_residuals(
        attacked_position, velocity, acceleration, attitude, angular_rates
    )
    print(f"  PVA: mean={np.mean(np.abs(residuals_attack.pva_residual)):.4f}")
    print(f"  Total: mean={np.mean(residuals_attack.total_residual):.4f}")
