"""
Validation script for the counter-intuitive PINN finding.

This implements the quick intuition checklist:
1. Compare supervised train/val loss for PureNN vs PINN
2. Check if PINN underfits supervised loss
3. Analyze physics residual magnitude
4. Compute gradient norms for both loss components
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent))
from pinn_architectures import BaselinePINN, ModularPINN, PhysicsLossMixin
from run_comprehensive_ablations import PureNNBaseline, load_data, load_trajectories

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / "data" / "train_set_diverse.csv"
VAL_DATA = PROJECT_ROOT / "data" / "val_set_diverse.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "comprehensive_ablation"
RESULTS_DIR = PROJECT_ROOT / "results" / "validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_supervised_loss_only(model, data_loader, scaler_y):
    """Compute supervised MSE loss only (no physics)"""
    model.eval()
    total_loss = 0
    n_samples = 0

    with torch.no_grad():
        for X_scaled, y_scaled, X_unscaled, y_unscaled in data_loader:
            y_pred_scaled = model(X_scaled)
            loss = nn.MSELoss(reduction="sum")(y_pred_scaled, y_scaled)
            total_loss += loss.item()
            n_samples += X_scaled.size(0)

    return total_loss / n_samples


def compute_physics_residual(model, data_loader, scaler_y):
    """Compute physics residual magnitude"""
    if not hasattr(model, "physics_loss"):
        return None, None

    model.eval()
    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    residuals = []

    with torch.no_grad():
        for X_scaled, y_scaled, X_unscaled, y_unscaled in data_loader:
            y_pred_scaled = model(X_scaled)
            y_pred_unscaled = y_pred_scaled * y_scale + y_mean

            # Get individual residuals
            batch_residual = model.physics_loss(X_unscaled, y_pred_unscaled)
            residuals.append(batch_residual.item())

    return np.mean(residuals), np.std(residuals)


def compute_gradient_norms(model, X_scaled, y_scaled, X_unscaled, y_unscaled, scaler_y):
    """Compute gradient norms for supervised vs physics loss"""
    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    # Supervised loss gradient
    model.zero_grad()
    y_pred_scaled = model(X_scaled)
    supervised_loss = nn.MSELoss()(y_pred_scaled, y_scaled)
    supervised_loss.backward(retain_graph=True)

    supervised_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            supervised_grad_norm += p.grad.data.norm(2).item() ** 2
    supervised_grad_norm = np.sqrt(supervised_grad_norm)

    # Physics loss gradient (if applicable)
    physics_grad_norm = None
    if hasattr(model, "physics_loss"):
        model.zero_grad()
        y_pred_scaled = model(X_scaled)
        y_pred_unscaled = y_pred_scaled * y_scale + y_mean
        physics_loss = model.physics_loss(X_unscaled, y_pred_unscaled)
        physics_loss.backward()

        physics_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                physics_grad_norm += p.grad.data.norm(2).item() ** 2
        physics_grad_norm = np.sqrt(physics_grad_norm)

    return supervised_grad_norm, physics_grad_norm


def compute_jacobian_spectral_norm(model, x, scaler_X, scaler_y, n_power_iter=10):
    """Compute spectral norm of Jacobian via power iteration"""
    model.eval()
    x_mean = torch.FloatTensor(scaler_X.mean_)
    x_scale = torch.FloatTensor(scaler_X.scale_)
    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    # Normalize input
    x_scaled = (x - x_mean) / x_scale
    x_scaled = x_scaled.detach().requires_grad_(True)

    # Get output dimension
    with torch.no_grad():
        out = model(x_scaled.unsqueeze(0)).squeeze(0)
    out_dim = out.shape[0]

    # Power iteration
    v = torch.randn(x_scaled.shape[0])
    v = v / (v.norm() + 1e-12)

    for _ in range(n_power_iter):
        # Compute Jv via forward-mode AD
        x_scaled = x_scaled.detach().requires_grad_(True)
        out = model(x_scaled.unsqueeze(0)).squeeze(0)

        # Compute J^T J v iteratively
        Jv = torch.zeros(out_dim)
        for i in range(out_dim):
            grad_i = torch.autograd.grad(out[i], x_scaled, retain_graph=True)[0]
            Jv[i] = (grad_i * v).sum()

        # Now compute J^T (Jv)
        JTJv = torch.zeros(x_scaled.shape[0])
        for i in range(out_dim):
            x_scaled_new = x_scaled.detach().requires_grad_(True)
            out_new = model(x_scaled_new.unsqueeze(0)).squeeze(0)
            grad_i = torch.autograd.grad(out_new[i], x_scaled_new, retain_graph=True)[0]
            JTJv += Jv[i] * grad_i

        v = JTJv / (JTJv.norm() + 1e-12)

    # Final sigma estimate
    x_scaled = x_scaled.detach().requires_grad_(True)
    out = model(x_scaled.unsqueeze(0)).squeeze(0)
    Jv = torch.zeros(out_dim)
    for i in range(out_dim):
        grad_i = torch.autograd.grad(out[i], x_scaled, retain_graph=True)[0]
        Jv[i] = (grad_i * v).sum()

    sigma = Jv.norm().item()
    return sigma


def main():
    print("=" * 70)
    print("VALIDATION OF COUNTER-INTUITIVE PINN FINDING")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    X_train, y_train = load_data(TRAIN_DATA)
    X_val, y_val = load_data(VAL_DATA)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_scaled),
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_scaled),
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Load models
    print("\n[2] Loading models...")
    purenn = PureNNBaseline()
    purenn.load_state_dict(torch.load(MODELS_DIR / "PureNN_seed42.pth", map_location="cpu"))

    pinn = BaselinePINN()
    pinn.load_state_dict(torch.load(MODELS_DIR / "PINN_Baseline_seed42.pth", map_location="cpu"))

    modular = ModularPINN()
    modular.load_state_dict(torch.load(MODELS_DIR / "Modular_72K_seed42.pth", map_location="cpu"))

    models = {"PureNN": purenn, "PINN": pinn, "Modular": modular}

    # ========================================================================
    # CHECK 1: Compare supervised loss ONLY
    # ========================================================================
    print("\n" + "=" * 70)
    print("CHECK 1: SUPERVISED LOSS COMPARISON (no physics)")
    print("=" * 70)
    print("\nIf PINN has HIGHER supervised loss, it's underfitting due to physics loss.")
    print("-" * 70)
    print(f"{'Model':<15} {'Train Sup Loss':<18} {'Val Sup Loss':<18}")
    print("-" * 70)

    results = {}
    for name, model in models.items():
        train_sup_loss = compute_supervised_loss_only(model, train_loader, scaler_y)
        val_sup_loss = compute_supervised_loss_only(model, val_loader, scaler_y)
        results[name] = {"train_sup_loss": train_sup_loss, "val_sup_loss": val_sup_loss}
        print(f"{name:<15} {train_sup_loss:<18.6f} {val_sup_loss:<18.6f}")

    # Check if PINN underfits
    ratio = results["PINN"]["val_sup_loss"] / results["PureNN"]["val_sup_loss"]
    print(f"\nPINN/PureNN supervised loss ratio: {ratio:.2f}x")
    if ratio > 1.1:
        print("WARNING: PINN has >10% higher supervised loss - may be underfitting!")
    else:
        print("OK: PINN supervised loss is comparable to PureNN")

    # ========================================================================
    # CHECK 2: Physics residual magnitude
    # ========================================================================
    print("\n" + "=" * 70)
    print("CHECK 2: PHYSICS RESIDUAL MAGNITUDE")
    print("=" * 70)
    print("\nShould decrease during training if physics loss is working.")
    print("-" * 70)

    for name, model in models.items():
        if hasattr(model, "physics_loss"):
            mean_res, std_res = compute_physics_residual(model, val_loader, scaler_y)
            print(f"{name}: Physics residual = {mean_res:.6f} +/- {std_res:.6f}")
            results[name]["physics_residual"] = mean_res

    # ========================================================================
    # CHECK 3: Gradient norms
    # ========================================================================
    print("\n" + "=" * 70)
    print("CHECK 3: GRADIENT NORM COMPARISON")
    print("=" * 70)
    print("\nIf physics grad >> supervised grad, physics loss dominates optimization.")
    print("-" * 70)

    # Get a batch for gradient computation
    X_batch, y_batch, X_unscaled, y_unscaled = next(iter(train_loader))

    print(f"{'Model':<15} {'Sup Grad Norm':<18} {'Phys Grad Norm':<18} {'Ratio':<10}")
    print("-" * 70)

    for name, model in models.items():
        model.train()  # Enable gradients
        sup_norm, phys_norm = compute_gradient_norms(
            model, X_batch, y_batch, X_unscaled, y_unscaled, scaler_y
        )

        if phys_norm is not None:
            ratio = phys_norm / sup_norm if sup_norm > 0 else float("inf")
            print(f"{name:<15} {sup_norm:<18.4f} {phys_norm:<18.4f} {ratio:<10.2f}")
            results[name]["sup_grad_norm"] = sup_norm
            results[name]["phys_grad_norm"] = phys_norm
            results[name]["grad_ratio"] = ratio

            if ratio > 5:
                print(f"  WARNING: Physics gradients dominate by {ratio:.1f}x!")
        else:
            print(f"{name:<15} {sup_norm:<18.4f} {'N/A':<18} {'N/A':<10}")
            results[name]["sup_grad_norm"] = sup_norm

    # ========================================================================
    # CHECK 4: Jacobian spectral norms (sample)
    # ========================================================================
    print("\n" + "=" * 70)
    print("CHECK 4: JACOBIAN SPECTRAL NORM (sample of 50 states)")
    print("=" * 70)
    print("\nIf PINN sigma_max > 1 but PureNN sigma_max <= 1, physics loss causes instability.")
    print("-" * 70)

    n_samples = 50
    sample_indices = np.random.choice(len(X_val), n_samples, replace=False)

    for name, model in models.items():
        sigmas = []
        for idx in sample_indices:
            x = torch.FloatTensor(X_val[idx])
            sigma = compute_jacobian_spectral_norm(model, x, scaler_X, scaler_y, n_power_iter=5)
            sigmas.append(sigma)

        mean_sigma = np.mean(sigmas)
        max_sigma = np.max(sigmas)
        p95_sigma = np.percentile(sigmas, 95)

        print(f"{name}: mean={mean_sigma:.3f}, p95={p95_sigma:.3f}, max={max_sigma:.3f}")
        results[name]["sigma_mean"] = mean_sigma
        results[name]["sigma_max"] = max_sigma
        results[name]["sigma_p95"] = p95_sigma

    # ========================================================================
    # SUMMARY AND DIAGNOSIS
    # ========================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    issues = []

    # Check 1: Supervised loss
    if results["PINN"]["val_sup_loss"] > results["PureNN"]["val_sup_loss"] * 1.1:
        issues.append("PINN underfits supervised loss (>10% higher than PureNN)")

    # Check 2: Gradient dominance
    if "grad_ratio" in results["PINN"] and results["PINN"]["grad_ratio"] > 5:
        issues.append(f"Physics gradients dominate by {results['PINN']['grad_ratio']:.1f}x")

    # Check 3: Spectral norm
    if results["PINN"]["sigma_max"] > 1.0 and results["PureNN"]["sigma_max"] <= 1.0:
        issues.append("PINN sigma_max > 1 while PureNN <= 1 (physics loss increases Lipschitz)")

    if issues:
        print("\nPOTENTIAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        if "underfits" in issues[0].lower():
            print("\n--> The finding may be an OPTIMIZATION ARTIFACT.")
            print("    Try: lower physics weight, normalize physics loss, or longer training.")
        else:
            print("\n--> The finding appears MECHANISTICALLY VALID.")
            print("    Physics loss genuinely increases Lipschitz constant.")
    else:
        print("\nNO OBVIOUS ISSUES FOUND.")
        print("The counter-intuitive finding appears robust.")

    # Save results
    with open(RESULTS_DIR / "validation_results.json", "w") as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR / 'validation_results.json'}")

    return results


if __name__ == "__main__":
    results = main()
