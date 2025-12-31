#!/usr/bin/env python3
"""
Industry-Grade Training for ALL Dynamics Models.

Applies best practices:
1. Gradient clipping (1.0)
2. Adaptive curriculum on dynamics
3. Inverse-sigmoid scheduled sampling (delayed)
4. Weight decay (no dropout)
5. Normalized energy constraint (soft, gated)
6. Rollout loss during training
7. Extended validation horizon (extrapolation)

Usage:
    python scripts/train_industry_all_models.py --model quadrotor
    python scripts/train_industry_all_models.py --model all
    python scripts/train_industry_all_models.py --model all --seeds 5
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinn_dynamics.systems import QuadrotorPINN, PendulumPINN, CartPolePINN
from pinn_dynamics.training.trainer_v2 import IndustryTrainer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Model Configurations
# ============================================================================

MODEL_CONFIGS = {
    "quadrotor": {
        "class": QuadrotorPINN,
        "state_dim": 12,
        "control_dim": 4,
        "hidden_size": 256,
        "num_layers": 5,
        "data_source": "euroc",  # or "synthetic"
        "epochs": 200,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "weights": {"physics": 0.0, "energy": 1e-3},  # No physics loss (hurts)
        "rollout_horizon": 10,
        "val_horizon": 20,
    },
    "pendulum": {
        "class": PendulumPINN,
        "state_dim": 2,
        "control_dim": 1,
        "hidden_size": 64,
        "num_layers": 3,
        "data_source": "synthetic",
        "epochs": 100,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "weights": {"physics": 0.1, "energy": 1e-3},  # Small physics helps for simple systems
        "rollout_horizon": 20,
        "val_horizon": 40,
    },
    "cartpole": {
        "class": CartPolePINN,
        "state_dim": 4,
        "control_dim": 1,
        "hidden_size": 128,
        "num_layers": 4,
        "data_source": "synthetic",
        "epochs": 150,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "weights": {"physics": 0.05, "energy": 1e-3},
        "rollout_horizon": 15,
        "val_horizon": 30,
    },
}


# ============================================================================
# Data Generation
# ============================================================================

def generate_quadrotor_data(n_samples: int = 10000, seed: int = 42):
    """Generate synthetic quadrotor data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # State: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    # Control: [f1, f2, f3, f4] (motor forces)

    dt = 0.01
    state_dim = 12
    control_dim = 4

    states = []
    next_states = []
    controls = []

    for _ in range(n_samples):
        # Random state
        pos = torch.randn(3) * 2
        vel = torch.randn(3) * 1
        angles = torch.randn(3) * 0.3
        ang_vel = torch.randn(3) * 0.5

        state = torch.cat([pos, vel, angles, ang_vel])

        # Random control (hover + perturbation)
        ctrl = torch.ones(4) * 2.5 + torch.randn(4) * 0.5

        # Simplified dynamics (for training data)
        g = 9.81
        m = 1.0

        # Next velocity (simplified)
        total_thrust = ctrl.sum()
        acc = torch.zeros(3)
        acc[2] = total_thrust / m - g

        # Next position
        next_pos = pos + vel * dt + 0.5 * acc * dt ** 2
        next_vel = vel + acc * dt

        # Next angles (simplified)
        next_angles = angles + ang_vel * dt
        next_ang_vel = ang_vel * 0.99  # Damping

        next_state = torch.cat([next_pos, next_vel, next_angles, next_ang_vel])

        states.append(state)
        next_states.append(next_state)
        controls.append(ctrl)

    X = torch.stack([torch.cat([s, c]) for s, c in zip(states, controls)])
    Y = torch.stack(next_states)

    return X, Y


def generate_pendulum_data(n_samples: int = 5000, seed: int = 42):
    """Generate synthetic pendulum data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dt = 0.02
    g = 9.81
    L = 1.0
    m = 1.0
    b = 0.1  # Damping

    states = []
    next_states = []
    controls = []

    for _ in range(n_samples):
        # State: [theta, omega]
        theta = torch.randn(1) * 0.5
        omega = torch.randn(1) * 1.0
        state = torch.cat([theta, omega])

        # Control: torque
        ctrl = torch.randn(1) * 0.5

        # Dynamics
        alpha = (-g / L * torch.sin(theta) - b / (m * L ** 2) * omega + ctrl / (m * L ** 2))
        next_omega = omega + alpha * dt
        next_theta = theta + omega * dt

        next_state = torch.cat([next_theta, next_omega])

        states.append(state)
        next_states.append(next_state)
        controls.append(ctrl)

    X = torch.stack([torch.cat([s, c]) for s, c in zip(states, controls)])
    Y = torch.stack(next_states)

    return X, Y


def generate_cartpole_data(n_samples: int = 5000, seed: int = 42):
    """Generate synthetic cartpole data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dt = 0.02
    g = 9.81
    mc = 1.0  # Cart mass
    mp = 0.1  # Pole mass
    L = 0.5   # Pole length

    states = []
    next_states = []
    controls = []

    for _ in range(n_samples):
        # State: [x, x_dot, theta, theta_dot]
        x = torch.randn(1) * 0.5
        x_dot = torch.randn(1) * 0.5
        theta = torch.randn(1) * 0.2
        theta_dot = torch.randn(1) * 0.5
        state = torch.cat([x, x_dot, theta, theta_dot])

        # Control: force on cart
        ctrl = torch.randn(1) * 5.0

        # Simplified dynamics
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        temp = (ctrl + mp * L * theta_dot ** 2 * sin_t) / (mc + mp)
        theta_acc = (g * sin_t - cos_t * temp) / (L * (4/3 - mp * cos_t ** 2 / (mc + mp)))
        x_acc = temp - mp * L * theta_acc * cos_t / (mc + mp)

        next_x = x + x_dot * dt
        next_x_dot = x_dot + x_acc * dt
        next_theta = theta + theta_dot * dt
        next_theta_dot = theta_dot + theta_acc * dt

        next_state = torch.cat([next_x, next_x_dot, next_theta, next_theta_dot])

        states.append(state)
        next_states.append(next_state)
        controls.append(ctrl)

    X = torch.stack([torch.cat([s, c]) for s, c in zip(states, controls)])
    Y = torch.stack(next_states)

    return X, Y


def load_euroc_data():
    """Load real EuRoC MAV data if available."""
    data_path = Path("data/euroc_processed")
    if not data_path.exists():
        logger.warning("EuRoC data not found, using synthetic")
        return generate_quadrotor_data()

    # Try to load preprocessed data
    try:
        X = torch.load(data_path / "X_train.pt")
        Y = torch.load(data_path / "Y_train.pt")
        return X, Y
    except:
        logger.warning("Could not load EuRoC data, using synthetic")
        return generate_quadrotor_data()


DATA_GENERATORS = {
    "quadrotor": lambda seed: generate_quadrotor_data(10000, seed),
    "pendulum": lambda seed: generate_pendulum_data(5000, seed),
    "cartpole": lambda seed: generate_cartpole_data(5000, seed),
}


# ============================================================================
# Training
# ============================================================================

def train_single_model(
    model_name: str,
    seed: int = 42,
    device: str = "cpu",
    output_dir: Path = Path("models/industry"),
) -> dict:
    """Train a single model with industry settings."""

    config = MODEL_CONFIGS[model_name]
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name.upper()} (seed={seed})")
    logger.info(f"{'='*60}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate data
    X, Y = DATA_GENERATORS[model_name](seed)

    # Split
    n = len(X)
    train_idx = int(0.8 * n)
    X_train, X_val = X[:train_idx], X[train_idx:]
    Y_train, Y_val = Y[:train_idx], Y[train_idx:]

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=config["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=config["batch_size"]
    )

    # Create model
    ModelClass = config["class"]
    model = ModelClass(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
    )

    # Create industry trainer
    trainer = IndustryTrainer(
        model=model,
        device=device,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        grad_clip=1.0,
    )

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        weights=config["weights"],
        rollout_horizon=config["rollout_horizon"],
        val_horizon=config["val_horizon"],
        verbose=True,
    )

    # Final evaluation
    final_val, final_rollout = trainer.validate_rollout(val_loader, config["val_horizon"])

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name}_seed{seed}.pt"
    trainer.save(str(model_path))

    results = {
        "model": model_name,
        "seed": seed,
        "final_val_loss": final_val,
        "final_rollout_mae": final_rollout,
        "best_rollout_mae": min(history["rollout_mae"]),
        "epochs": config["epochs"],
        "config": {k: str(v) if isinstance(v, type) else v for k, v in config.items()},
    }

    logger.info(f"\nResults for {model_name} (seed={seed}):")
    logger.info(f"  Final Val Loss: {final_val:.6f}")
    logger.info(f"  Final Rollout MAE: {final_rollout:.4f}m")
    logger.info(f"  Best Rollout MAE: {min(history['rollout_mae']):.4f}m")
    logger.info(f"  Model saved: {model_path}")

    return results


def train_all_models(
    seeds: int = 5,
    device: str = "cpu",
    output_dir: Path = Path("models/industry"),
):
    """Train all models with multiple seeds."""

    all_results = {}

    for model_name in MODEL_CONFIGS.keys():
        model_results = []

        for seed in range(seeds):
            result = train_single_model(
                model_name=model_name,
                seed=seed,
                device=device,
                output_dir=output_dir,
            )
            model_results.append(result)

        # Aggregate
        rollout_maes = [r["final_rollout_mae"] for r in model_results]
        all_results[model_name] = {
            "per_seed": model_results,
            "mean_rollout_mae": np.mean(rollout_maes),
            "std_rollout_mae": np.std(rollout_maes),
            "best_rollout_mae": np.min(rollout_maes),
        }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INDUSTRY TRAINING SUMMARY")
    logger.info("=" * 60)

    for model_name, results in all_results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Rollout MAE: {results['mean_rollout_mae']:.4f} ± {results['std_rollout_mae']:.4f}m")
        logger.info(f"  Best: {results['best_rollout_mae']:.4f}m")

    # Save summary
    summary_path = output_dir / "industry_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSummary saved: {summary_path}")

    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Industry-grade training for all models")
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        help="Model to train")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--output", type=str, default="models/industry",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)

    logger.info("=" * 60)
    logger.info("INDUSTRY-GRADE DYNAMICS TRAINING")
    logger.info("=" * 60)
    logger.info(f"Improvements applied:")
    logger.info(f"  1. Gradient clipping (1.0)")
    logger.info(f"  2. Adaptive curriculum on dynamics")
    logger.info(f"  3. Inverse-sigmoid scheduled sampling")
    logger.info(f"  4. Weight decay {1e-4} (no dropout)")
    logger.info(f"  5. Normalized energy constraint (gated)")
    logger.info(f"  6. Rollout loss during training")
    logger.info(f"  7. Extended validation horizon")
    logger.info("=" * 60)

    if args.model == "all":
        results = train_all_models(
            seeds=args.seeds,
            device=args.device,
            output_dir=output_dir,
        )
    else:
        results = {}
        model_results = []
        for seed in range(args.seeds):
            result = train_single_model(
                model_name=args.model,
                seed=seed,
                device=args.device,
                output_dir=output_dir,
            )
            model_results.append(result)

        rollout_maes = [r["final_rollout_mae"] for r in model_results]
        results[args.model] = {
            "per_seed": model_results,
            "mean_rollout_mae": np.mean(rollout_maes),
            "std_rollout_mae": np.std(rollout_maes),
        }

    logger.info("\nTraining complete!")
    return results


if __name__ == "__main__":
    main()
