"""
Stable Physics-Informed Neural Network for Quadrotor Dynamics

Architecture design for autoregressive stability:
1. Unified network with shared trunk (maintains dynamic coupling)
2. Small task-specific heads (translation + rotation)
3. Residual connections for better gradient flow
4. Swish activation for smooth training
5. Optional low-frequency Fourier features (safe extrapolation)

Based on lessons learned from optimization failures:
- Simple architectures with long training > complex architectures with short training
- Coupled dynamics require unified networks, not hard modularity
- Scheduled sampling and curriculum rollout are essential for stability
"""

import torch
import torch.nn as nn
import numpy as np
import math


class LowFrequencyFourierFeatures(nn.Module):
    """
    Optional: Single low-frequency Fourier term for periodic states
    Avoids extrapolation catastrophe of high-frequency harmonics
    """
    def __init__(self, input_dim, num_frequencies=1, periodic_indices=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies

        # Indices of periodic states: phi, theta, psi, p, q, r
        if periodic_indices is None:
            periodic_indices = [1, 2, 3, 4, 5, 6]  # angles and rates
        self.periodic_indices = periodic_indices

        # Output: original + sin/cos for each frequency and periodic variable
        self.output_dim = input_dim + 2 * num_frequencies * len(periodic_indices)

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] - normalized inputs
        Returns:
            encoded: [batch, output_dim] - with Fourier features
        """
        batch_size = x.shape[0]
        encoded = [x]

        # Add low-frequency Fourier for periodic states only
        for idx in self.periodic_indices:
            x_periodic = x[:, idx:idx+1]  # [batch, 1]
            for k in range(1, self.num_frequencies + 1):
                encoded.append(torch.sin(k * np.pi * x_periodic))
                encoded.append(torch.cos(k * np.pi * x_periodic))

        return torch.cat(encoded, dim=1)


class ResidualBlock(nn.Module):
    """Residual block with Swish activation"""
    def __init__(self, hidden_size, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return x + self.net(x)


class StablePINN(nn.Module):
    """
    Stable PINN with unified architecture

    Design principles:
    - Shared trunk maintains dynamic coupling (z, vz, phi, theta, psi, p, q, r)
    - Small task heads reduce parameters while maintaining expressiveness
    - Residual connections improve gradient flow
    - Optional low-frequency Fourier (safe extrapolation)
    """
    def __init__(self,
                 input_dim=12,  # [z, phi, theta, psi, p, q, r, vz, thrust, tx, ty, tz]
                 output_dim=8,  # [z, phi, theta, psi, p, q, r, vz]
                 hidden_size=128,
                 num_residual_blocks=2,
                 dropout=0.1,
                 use_fourier=False,
                 num_fourier_freq=1,
                 physics_params=None):
        super().__init__()

        # Physics parameters (learnable)
        if physics_params is None:
            physics_params = {
                'Jxx': 6.86e-5,
                'Jyy': 9.20e-5,
                'Jzz': 1.366e-4,
                'kt': 0.01,
                'kq': 7.8263e-4,
                'm': 0.068
            }

        self.Jxx = nn.Parameter(torch.tensor(physics_params['Jxx'], dtype=torch.float32))
        self.Jyy = nn.Parameter(torch.tensor(physics_params['Jyy'], dtype=torch.float32))
        self.Jzz = nn.Parameter(torch.tensor(physics_params['Jzz'], dtype=torch.float32))
        self.kt = nn.Parameter(torch.tensor(physics_params['kt'], dtype=torch.float32))
        self.kq = nn.Parameter(torch.tensor(physics_params['kq'], dtype=torch.float32))
        self.m = nn.Parameter(torch.tensor(physics_params['m'], dtype=torch.float32))

        # Fourier features (optional)
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = LowFrequencyFourierFeatures(
                input_dim,
                num_frequencies=num_fourier_freq,
                periodic_indices=[1, 2, 3, 4, 5, 6]  # angles and rates
            )
            trunk_input_dim = self.fourier.output_dim
        else:
            trunk_input_dim = input_dim

        # Shared trunk (maintains coupling between all states)
        self.trunk_input = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_size),
            nn.SiLU()
        )

        # Residual blocks for deep feature extraction
        residual_blocks = []
        for _ in range(num_residual_blocks):
            residual_blocks.append(ResidualBlock(hidden_size, dropout))
        self.trunk_residual = nn.Sequential(*residual_blocks)

        # Task-specific heads (small, just final projection)
        head_size = hidden_size // 2

        # Translational head: vz (vertical velocity derivative)
        self.head_translation = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.SiLU(),
            nn.Linear(head_size, 1)  # vz prediction
        )

        # Rotational head: p, q, r (angular acceleration)
        self.head_rotation = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.SiLU(),
            nn.Linear(head_size, 3)  # p, q, r predictions
        )

        # Kinematic head: z, phi, theta, psi (integrated from velocities)
        self.head_kinematics = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.SiLU(),
            nn.Linear(head_size, 4)  # z, phi, theta, psi
        )

        self.output_dim = output_dim
        self.input_dim = input_dim

    def forward(self, x, add_noise=False, noise_level=0.01):
        """
        Forward pass with optional process noise for training robustness

        Args:
            x: [batch, input_dim] - state and control inputs
            add_noise: bool - add process noise during training
            noise_level: float - std dev of Gaussian noise

        Returns:
            next_state: [batch, output_dim] - predicted next state
        """
        # Optional: Add process noise during training
        if add_noise and self.training:
            x = x + noise_level * torch.randn_like(x)

        # Fourier encoding (optional)
        if self.use_fourier:
            h = self.fourier(x)
        else:
            h = x

        # Shared trunk (maintains coupling)
        h = self.trunk_input(h)
        h = self.trunk_residual(h)

        # Task-specific heads
        kinematics = self.head_kinematics(h)  # [z, phi, theta, psi]
        vz_pred = self.head_translation(h)    # [vz]
        rates = self.head_rotation(h)         # [p, q, r]

        # Combine: [z, phi, theta, psi, p, q, r, vz]
        return torch.cat([
            kinematics[:, 0:1],  # z
            kinematics[:, 1:2],  # phi
            kinematics[:, 2:3],  # theta
            kinematics[:, 3:4],  # psi
            rates[:, 0:1],       # p
            rates[:, 1:2],       # q
            rates[:, 2:3],       # r
            vz_pred              # vz
        ], dim=1)

    def physics_loss(self, x, y_pred, y_true, dt=0.001):
        """
        Physics-informed loss based on quadrotor dynamics

        Args:
            x: [batch, 12] - current state + controls
            y_pred: [batch, 8] - predicted next state
            y_true: [batch, 8] - true next state
            dt: float - time step

        Returns:
            physics_loss: scalar tensor
        """
        # Extract current state
        z = x[:, 0:1]
        phi = x[:, 1:2]
        theta = x[:, 2:3]
        psi = x[:, 3:4]
        p = x[:, 4:5]
        q = x[:, 5:6]
        r = x[:, 6:7]
        vz = x[:, 7:8]
        thrust = x[:, 8:9]
        tx = x[:, 9:10]
        ty = x[:, 10:11]
        tz = x[:, 11:12]

        # Extract predicted next state
        z_next_pred = y_pred[:, 0:1]
        phi_next = y_pred[:, 1:2]
        theta_next = y_pred[:, 2:3]
        psi_next = y_pred[:, 3:4]
        p_next = y_pred[:, 4:5]
        q_next = y_pred[:, 5:6]
        r_next = y_pred[:, 6:7]
        vz_next_pred = y_pred[:, 7:8]

        # Physics equations
        g = 9.81

        # Vertical dynamics: az = -g + T/(m*cos(phi)*cos(theta))
        cos_phi = torch.cos(phi)
        cos_theta = torch.cos(theta)
        az_physics = -g + thrust / (self.m * cos_phi * cos_theta + 1e-6)
        vz_physics = vz + az_physics * dt
        z_physics = z + vz * dt

        # Rotational dynamics
        # p_dot = (Jyy - Jzz)/Jxx * q * r + tx/Jxx
        # q_dot = (Jzz - Jxx)/Jyy * p * r + ty/Jyy
        # r_dot = (Jxx - Jyy)/Jzz * p * q + tz/Jzz
        p_dot = ((self.Jyy - self.Jzz) / self.Jxx) * q * r + tx / self.Jxx
        q_dot = ((self.Jzz - self.Jxx) / self.Jyy) * p * r + ty / self.Jyy
        r_dot = ((self.Jxx - self.Jyy) / self.Jzz) * p * q + tz / self.Jzz

        p_physics = p + p_dot * dt
        q_physics = q + q_dot * dt
        r_physics = r + r_dot * dt

        # Angular kinematics (simplified Euler angle rates)
        phi_physics = phi + p * dt
        theta_physics = theta + q * dt
        psi_physics = psi + r * dt

        # Physics loss: MSE between physics-based prediction and NN prediction
        loss_z = torch.mean((z_next_pred - z_physics) ** 2)
        loss_vz = torch.mean((vz_next_pred - vz_physics) ** 2)
        loss_phi = torch.mean((phi_next - phi_physics) ** 2)
        loss_theta = torch.mean((theta_next - theta_physics) ** 2)
        loss_psi = torch.mean((psi_next - psi_physics) ** 2)
        loss_p = torch.mean((p_next - p_physics) ** 2)
        loss_q = torch.mean((q_next - q_physics) ** 2)
        loss_r = torch.mean((r_next - r_physics) ** 2)

        return loss_z + loss_vz + loss_phi + loss_theta + loss_psi + loss_p + loss_q + loss_r

    def multistep_rollout_loss(self, x_initial, u_sequence, y_true_sequence, num_steps, add_noise=False):
        """
        Multi-step autoregressive rollout loss with curriculum learning

        Args:
            x_initial: [batch, 8] - initial state
            u_sequence: [batch, num_steps, 4] - control sequence
            y_true_sequence: [batch, num_steps, 8] - ground truth trajectory
            num_steps: int - rollout horizon
            add_noise: bool - add noise for robustness

        Returns:
            rollout_loss: scalar tensor
        """
        batch_size = x_initial.shape[0]
        x_current = x_initial

        total_loss = 0.0
        for step in range(num_steps):
            # Get control for this step
            u_step = u_sequence[:, step, :]  # [batch, 4]

            # Concatenate state + control
            x_input = torch.cat([x_current, u_step], dim=1)  # [batch, 12]

            # Predict next state
            x_next_pred = self.forward(x_input, add_noise=add_noise)

            # Compare to ground truth
            x_next_true = y_true_sequence[:, step, :]
            step_loss = torch.mean((x_next_pred - x_next_true) ** 2)
            total_loss += step_loss

            # Update current state for next step
            x_current = x_next_pred

        return total_loss / num_steps

    def get_jacobian_loss(self, x_batch):
        """
        Jacobian regularization: penalize large sensitivities
        Encourages contractive dynamics for stability

        Args:
            x_batch: [batch, input_dim]

        Returns:
            jac_loss: scalar - penalizes ||J||_F > 1
        """
        # Compute Jacobian for a subset of batch (expensive)
        x_sample = x_batch[:min(8, x_batch.shape[0])]  # Use small subset
        x_sample.requires_grad = True

        y_sample = self.forward(x_sample)

        # Approximate Frobenius norm using random projections
        batch_size = x_sample.shape[0]
        jac_norms = []

        for i in range(batch_size):
            # Compute gradient of each output w.r.t input
            grads = torch.autograd.grad(
                outputs=y_sample[i].sum(),
                inputs=x_sample,
                create_graph=True,
                retain_graph=True
            )[0]

            # Frobenius norm for this sample
            jac_norm = torch.norm(grads[i])
            jac_norms.append(jac_norm)

        jac_norms = torch.stack(jac_norms)

        # Penalize norms > 1 (contractive mapping)
        jac_loss = torch.mean(torch.clamp(jac_norms - 1.0, min=0.0) ** 2)

        return jac_loss

    def get_parameters_dict(self):
        """Return physics parameters as dict"""
        return {
            'Jxx': self.Jxx.item(),
            'Jyy': self.Jyy.item(),
            'Jzz': self.Jzz.item(),
            'kt': self.kt.item(),
            'kq': self.kq.item(),
            'm': self.m.item()
        }


# Parameter count
if __name__ == "__main__":
    model = StablePINN(
        hidden_size=128,
        num_residual_blocks=2,
        use_fourier=False
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stable PINN parameters: {total_params:,}")

    # Test forward pass
    x_test = torch.randn(32, 12)
    y_pred = model(x_test)
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {y_pred.shape}")

    # Test Jacobian loss
    jac_loss = model.get_jacobian_loss(x_test)
    print(f"Jacobian loss: {jac_loss.item():.6f}")

    print("\nPhysics parameters:")
    for name, value in model.get_parameters_dict().items():
        print(f"  {name}: {value:.6e}")
