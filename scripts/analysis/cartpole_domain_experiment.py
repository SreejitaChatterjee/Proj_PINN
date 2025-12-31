"""
CartPole Domain Experiment for ICML Paper

This script validates our findings on a second domain (CartPole) to demonstrate
generalizability beyond quadrotor dynamics.

Key hypotheses to test:
1. Physics loss does not improve rollout stability in high-data regime
2. Jacobian spectral radius correlates with rollout error
3. Jacobian regularization improves rollout performance

CartPole: 4-state system (x, x_dot, theta, theta_dot), 1 control (force)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from pinn_dynamics.systems.cartpole import CartPolePINN

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "cartpole_domain"
MODELS_DIR = PROJECT_ROOT / "models" / "cartpole_domain"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training config
BATCH_SIZE = 256
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 30
PHYSICS_WEIGHT = 10.0  # Tuned for cartpole scale
JACOBIAN_WEIGHT = 0.1
JACOBIAN_THRESHOLD = 2.0  # sqrt(4) for 4-state system

# Experiment config
N_SEEDS = 20  # Same rigor as quadrotor experiments
SEEDS = [42, 123, 456, 789, 999] + list(range(1, 16))  # Same seeds as quadrotor
N_TRAJECTORIES = 50
TRAJECTORY_LENGTH = 500
DT = 0.02  # 50 Hz


# ============================================================================
# DATA GENERATION
# ============================================================================


def cartpole_dynamics(state, force, dt=0.02):
    """True cartpole dynamics for data generation."""
    x, x_dot, theta, theta_dot = state

    # Physical parameters
    g = 9.81
    mc = 1.0  # cart mass
    mp = 0.1  # pole mass
    L = 0.5  # pole half-length

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    total_mass = mc + mp

    # Equations of motion
    temp = (force + mp * L * theta_dot**2 * sin_t) / total_mass
    theta_ddot = (g * sin_t - cos_t * temp) / (L * (4 / 3 - mp * cos_t**2 / total_mass))
    x_ddot = temp - mp * L * theta_ddot * cos_t / total_mass

    # Euler integration
    x_new = x + x_dot * dt
    x_dot_new = x_dot + x_ddot * dt
    theta_new = theta + theta_dot * dt
    theta_dot_new = theta_dot + theta_ddot * dt

    return np.array([x_new, x_dot_new, theta_new, theta_dot_new])


def generate_cartpole_data(n_trajectories=50, traj_length=500, dt=0.02, seed=42):
    """Generate diverse cartpole trajectories."""
    np.random.seed(seed)

    all_states = []
    all_controls = []
    all_next_states = []

    for traj_id in range(n_trajectories):
        # Random initial conditions
        x0 = np.random.uniform(-1, 1)
        x_dot0 = np.random.uniform(-0.5, 0.5)
        theta0 = np.random.uniform(-0.3, 0.3)  # Near upright
        theta_dot0 = np.random.uniform(-0.5, 0.5)

        state = np.array([x0, x_dot0, theta0, theta_dot0])

        for t in range(traj_length):
            # Control policy: random + stabilizing feedback
            force = np.random.uniform(-5, 5)
            # Add some stabilizing feedback
            force += -2.0 * state[2] - 0.5 * state[3]  # PD on angle
            force = np.clip(force, -10, 10)

            next_state = cartpole_dynamics(state, force, dt)

            # Check for instability (pole fell over)
            if abs(next_state[2]) > np.pi / 2:
                break

            all_states.append(state.copy())
            all_controls.append([force])
            all_next_states.append(next_state.copy())

            state = next_state

    return (
        np.array(all_states, dtype=np.float32),
        np.array(all_controls, dtype=np.float32),
        np.array(all_next_states, dtype=np.float32),
    )


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================


class CartPoleNN(nn.Module):
    """Pure neural network baseline for CartPole."""

    def __init__(self, hidden_size=64, num_layers=3):
        super().__init__()
        layers = [nn.Linear(5, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CartPolePINNWrapper(nn.Module):
    """Wrapper for CartPolePINN with physics loss."""

    def __init__(self, hidden_size=64, num_layers=3):
        super().__init__()
        self.pinn = CartPolePINN(hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x):
        return self.pinn(x)

    def physics_loss(self, inputs, outputs, dt=0.02):
        return self.pinn.physics_loss(inputs, outputs, dt)


class CartPoleJacobian(nn.Module):
    """CartPole NN with Jacobian regularization."""

    def __init__(self, hidden_size=64, num_layers=3):
        super().__init__()
        layers = [nn.Linear(5, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def jacobian_loss(self, x, threshold=2.0):
        """Compute Jacobian regularization loss."""
        x = x.requires_grad_(True)
        y = self.forward(x)

        batch_size = x.shape[0]
        state_dim = 4  # Only state, not control

        # Compute Jacobian w.r.t. state (first 4 dims)
        jacobian = torch.zeros(batch_size, 4, state_dim, device=x.device)
        for i in range(4):
            grad = torch.autograd.grad(y[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
            jacobian[:, i, :] = grad[:, :state_dim]

        # Frobenius norm
        frob_norm = torch.sqrt((jacobian**2).sum(dim=(1, 2)))

        # ReLU penalty above threshold
        loss = torch.relu(frob_norm - threshold).mean()
        return loss


# ============================================================================
# TRAINING
# ============================================================================


def train_model(
    model,
    train_loader,
    val_loader,
    model_type="baseline",
    seed=42,
    physics_weight=10.0,
    jacobian_weight=0.1,
):
    """Train a CartPole model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()

            y_pred = model(X)
            data_loss = nn.MSELoss()(y_pred, y)

            loss = data_loss

            if model_type == "physics" and hasattr(model, "physics_loss"):
                phys_loss = model.physics_loss(X, y_pred)
                loss = data_loss + physics_weight * phys_loss
            elif model_type == "jacobian" and hasattr(model, "jacobian_loss"):
                jac_loss = model.jacobian_loss(X)
                loss = data_loss + jacobian_weight * jac_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            train_loss += loss.item() * X.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation (supervised loss only for fair comparison)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                y_pred = model(X)
                val_loss += nn.MSELoss()(y_pred, y).item() * X.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_loss


# ============================================================================
# EVALUATION
# ============================================================================


def compute_rollout_error(model, states, controls, n_steps=50):
    """Compute autoregressive rollout error."""
    model.eval()

    with torch.no_grad():
        current_state = torch.FloatTensor(states[0]).unsqueeze(0)
        errors = []

        for t in range(min(n_steps, len(controls))):
            control = torch.FloatTensor(controls[t]).unsqueeze(0)
            inp = torch.cat([current_state, control], dim=1)

            next_state = model(inp)

            # Position error (x and theta)
            true_state = torch.FloatTensor(states[t + 1])
            error = torch.abs(next_state[0, [0, 2]] - true_state[[0, 2]]).sum().item()
            errors.append(error)

            current_state = next_state

        return np.mean(errors)


def compute_jacobian_spectral_radius(model, test_points, n_samples=100):
    """Estimate Jacobian spectral radius."""
    model.eval()
    spectral_radii = []

    indices = np.random.choice(len(test_points), min(n_samples, len(test_points)), replace=False)

    for idx in indices:
        x = torch.FloatTensor(test_points[idx]).unsqueeze(0).requires_grad_(True)
        y = model(x)

        # Compute Jacobian
        jacobian = torch.zeros(4, 4)
        for i in range(4):
            grad = torch.autograd.grad(y[0, i], x, retain_graph=True)[0]
            jacobian[i, :] = grad[0, :4]

        # Spectral radius
        eigenvalues = torch.linalg.eigvals(jacobian)
        spectral_radius = torch.max(torch.abs(eigenvalues)).item()
        spectral_radii.append(spectral_radius)

    return np.mean(spectral_radii), np.std(spectral_radii)


def evaluate_on_trajectories(model, test_states, test_controls, n_rollouts=10, rollout_length=50):
    """Evaluate rollout performance on multiple trajectories."""
    errors = []

    # Sample starting points
    start_indices = np.linspace(0, len(test_states) - rollout_length - 1, n_rollouts).astype(int)

    for start_idx in start_indices:
        states = test_states[start_idx : start_idx + rollout_length + 1]
        controls = test_controls[start_idx : start_idx + rollout_length]

        error = compute_rollout_error(model, states, controls, rollout_length)
        errors.append(error)

    return np.mean(errors), np.std(errors)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_experiment():
    print("=" * 80)
    print("CARTPOLE DOMAIN EXPERIMENT")
    print("Validating inductive bias mismatch on second domain")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Generate data
    print("\n[1/4] Generating CartPole data...")
    states, controls, next_states = generate_cartpole_data(
        n_trajectories=N_TRAJECTORIES, traj_length=TRAJECTORY_LENGTH, seed=0
    )
    print(f"  Generated {len(states):,} samples")

    # Train/val/test split
    n_total = len(states)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    X = np.concatenate([states, controls], axis=1)
    y = next_states

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"  Train: {n_train:,}, Val: {n_val:,}, Test: {n_total - n_train - n_val:,}")

    # Run experiments
    results = {
        "baseline": {"rollout": [], "spectral_radius": []},
        "physics": {"rollout": [], "spectral_radius": []},
        "jacobian": {"rollout": [], "spectral_radius": []},
    }

    print(f"\n[2/4] Training models ({N_SEEDS} seeds per condition)...")

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")

        # Baseline
        model_base = CartPoleNN()
        model_base, _ = train_model(
            model_base, train_loader, val_loader, model_type="baseline", seed=seed
        )
        rollout_mean, _ = evaluate_on_trajectories(model_base, states, controls)
        rho_mean, _ = compute_jacobian_spectral_radius(model_base, X_test)
        results["baseline"]["rollout"].append(rollout_mean)
        results["baseline"]["spectral_radius"].append(rho_mean)
        print(f"    Baseline: rollout={rollout_mean:.3f}, rho={rho_mean:.3f}")

        # Physics
        model_phys = CartPolePINNWrapper()
        model_phys, _ = train_model(
            model_phys, train_loader, val_loader, model_type="physics", seed=seed
        )
        rollout_mean, _ = evaluate_on_trajectories(model_phys, states, controls)
        rho_mean, _ = compute_jacobian_spectral_radius(model_phys, X_test)
        results["physics"]["rollout"].append(rollout_mean)
        results["physics"]["spectral_radius"].append(rho_mean)
        print(f"    Physics:  rollout={rollout_mean:.3f}, rho={rho_mean:.3f}")

        # Jacobian
        model_jac = CartPoleJacobian()
        model_jac, _ = train_model(
            model_jac, train_loader, val_loader, model_type="jacobian", seed=seed
        )
        rollout_mean, _ = evaluate_on_trajectories(model_jac, states, controls)
        rho_mean, _ = compute_jacobian_spectral_radius(model_jac, X_test)
        results["jacobian"]["rollout"].append(rollout_mean)
        results["jacobian"]["spectral_radius"].append(rho_mean)
        print(f"    Jacobian: rollout={rollout_mean:.3f}, rho={rho_mean:.3f}")

    # Statistical analysis
    print("\n[3/4] Statistical Analysis...")
    print("-" * 60)

    # Compute statistics
    stats_results = {}
    for condition in ["baseline", "physics", "jacobian"]:
        rollouts = results[condition]["rollout"]
        rhos = results[condition]["spectral_radius"]
        stats_results[condition] = {
            "rollout_mean": np.mean(rollouts),
            "rollout_std": np.std(rollouts),
            "rho_mean": np.mean(rhos),
            "rho_std": np.std(rhos),
        }

    # Statistical tests
    # Baseline vs Physics
    t_bp, p_bp = stats.ttest_ind(results["baseline"]["rollout"], results["physics"]["rollout"])
    d_bp = (
        np.mean(results["physics"]["rollout"]) - np.mean(results["baseline"]["rollout"])
    ) / np.sqrt(
        (np.std(results["baseline"]["rollout"]) ** 2 + np.std(results["physics"]["rollout"]) ** 2)
        / 2
    )

    # Baseline vs Jacobian
    t_bj, p_bj = stats.ttest_ind(results["baseline"]["rollout"], results["jacobian"]["rollout"])
    d_bj = (
        np.mean(results["baseline"]["rollout"]) - np.mean(results["jacobian"]["rollout"])
    ) / np.sqrt(
        (np.std(results["baseline"]["rollout"]) ** 2 + np.std(results["jacobian"]["rollout"]) ** 2)
        / 2
    )

    # Correlation: rho vs rollout
    all_rhos = (
        results["baseline"]["spectral_radius"]
        + results["physics"]["spectral_radius"]
        + results["jacobian"]["spectral_radius"]
    )
    all_rollouts = (
        results["baseline"]["rollout"]
        + results["physics"]["rollout"]
        + results["jacobian"]["rollout"]
    )
    r_corr, p_corr = stats.pearsonr(all_rhos, all_rollouts)

    print("\nResults Summary:")
    print(f"{'Condition':<12} {'Rollout MAE':<20} {'Spectral ρ':<20}")
    print("-" * 52)
    for cond in ["baseline", "physics", "jacobian"]:
        s = stats_results[cond]
        print(
            f"{cond:<12} {s['rollout_mean']:.3f} ± {s['rollout_std']:.3f}      {s['rho_mean']:.3f} ± {s['rho_std']:.3f}"
        )

    print(f"\nStatistical Tests:")
    print(f"  Baseline vs Physics: t={t_bp:.2f}, p={p_bp:.3f}, d={d_bp:.2f}")
    print(f"  Baseline vs Jacobian: t={t_bj:.2f}, p={p_bj:.3f}, d={d_bj:.2f}")
    print(f"  Correlation (ρ vs rollout): r={r_corr:.2f}, p={p_corr:.3f}")

    # Save results
    print("\n[4/4] Saving results...")

    final_results = {
        "config": {
            "n_seeds": N_SEEDS,
            "n_trajectories": N_TRAJECTORIES,
            "trajectory_length": TRAJECTORY_LENGTH,
            "physics_weight": PHYSICS_WEIGHT,
            "jacobian_weight": JACOBIAN_WEIGHT,
            "jacobian_threshold": JACOBIAN_THRESHOLD,
        },
        "raw_results": {
            k: {kk: [float(v) for v in vv] for kk, vv in v.items()} for k, v in results.items()
        },
        "statistics": {
            k: {kk: float(vv) for kk, vv in v.items()} for k, v in stats_results.items()
        },
        "tests": {
            "baseline_vs_physics": {
                "t": float(t_bp),
                "p": float(p_bp),
                "d": float(d_bp),
            },
            "baseline_vs_jacobian": {
                "t": float(t_bj),
                "p": float(p_bj),
                "d": float(d_bj),
            },
            "rho_rollout_correlation": {"r": float(r_corr), "p": float(p_corr)},
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(RESULTS_DIR / "cartpole_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"  Saved to {RESULTS_DIR / 'cartpole_results.json'}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return final_results


if __name__ == "__main__":
    results = run_experiment()
