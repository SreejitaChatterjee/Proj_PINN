"""Quick test of optimized model forward pass"""
import torch
import numpy as np
from pinn_model_optimized import QuadrotorPINNOptimized

print("Testing optimized model...")

# Create model
model = QuadrotorPINNOptimized(hidden_size=128, dropout=0.1, num_fourier_freq=3)
print(f"Model created successfully")

# Count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")

# Test forward pass
batch_size = 4
test_input = torch.randn(batch_size, 12)  # [z, phi, theta, psi, p, q, r, vz, thrust, tx, ty, tz]

print(f"\nTesting forward pass with batch_size={batch_size}...")
output = model(test_input)
print(f"Output shape: {output.shape}")
print(f"Expected: ({batch_size}, 8)")

# Test physics loss
print(f"\nTesting physics loss...")
phys_loss = model.physics_loss(test_input, output)
print(f"Physics loss: {phys_loss.item():.4f}")

# Test energy loss
print(f"\nTesting energy loss...")
energy_loss = model.energy_loss(test_input, output)
print(f"Energy loss: {energy_loss.item():.4f}")

# Test multistep rollout
print(f"\nTesting 3-step rollout...")
rollout_loss = model.multistep_rollout_loss(test_input, num_steps=3)
print(f"Rollout loss: {rollout_loss.item():.4f}")

print("\nAll tests passed!")
