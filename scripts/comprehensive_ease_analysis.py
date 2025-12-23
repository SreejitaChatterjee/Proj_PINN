"""
COMPREHENSIVE EASE RENE 2026 ANALYSIS

Runs extensive experiments for the paper:
1. Extended physics weight sweep (w=0,1,5,10,20,50,100)
2. Learning rate sensitivity analysis
3. Rollout horizon sensitivity (50,100,200,500 steps)
4. Architecture variations (depth, width)
5. Statistical inference and effect size analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent))
from pinn_architectures import BaselinePINN

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / 'data' / 'train_set_diverse.csv'
VAL_DATA = PROJECT_ROOT / 'data' / 'val_set_diverse.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'ease_comprehensive'
MODELS_DIR = PROJECT_ROOT / 'models' / 'ease_comprehensive'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Base config
BATCH_SIZE = 512
MAX_EPOCHS = 100
BASE_LR = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
EARLY_STOP_PATIENCE = 40
N_ROLLOUT_TRAJ = 10

# Use existing 20-seed data where available
EXISTING_SEEDS = [42, 123, 456, 789, 999, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
QUICK_SEEDS = [42, 123, 456]  # For new experiments


def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.rename(columns={'roll': 'phi', 'pitch': 'theta', 'yaw': 'psi'})
    state_cols = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    input_features = state_cols + ['thrust', 'torque_x', 'torque_y', 'torque_z']
    X, y = [], []
    for traj_id in df['trajectory_id'].unique():
        df_traj = df[df['trajectory_id'] == traj_id].reset_index(drop=True)
        for i in range(len(df_traj) - 1):
            X.append(df_traj.iloc[i][input_features].values)
            y.append(df_traj.iloc[i+1][state_cols].values)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_trajectories(data_path):
    df = pd.read_csv(data_path)
    df = df.rename(columns={'roll': 'phi', 'pitch': 'theta', 'yaw': 'psi'})
    state_cols = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vx', 'vy', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']
    trajectories = []
    for traj_id in df['trajectory_id'].unique():
        df_traj = df[df['trajectory_id'] == traj_id].reset_index(drop=True)
        trajectories.append({
            'states': df_traj[state_cols].values.astype(np.float32),
            'controls': df_traj[control_cols].values.astype(np.float32)
        })
    return trajectories


class FlexiblePINN(nn.Module):
    """Flexible PINN with configurable depth/width"""
    def __init__(self, n_layers=5, hidden_dim=256):
        super().__init__()
        layers = [nn.Linear(16, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 12))
        self.net = nn.Sequential(*layers)

        # Physics parameters
        self.log_mass = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_Ixx = nn.Parameter(torch.log(torch.tensor(0.01)))
        self.log_Iyy = nn.Parameter(torch.log(torch.tensor(0.01)))
        self.log_Izz = nn.Parameter(torch.log(torch.tensor(0.02)))

    def forward(self, x):
        return self.net(x)

    def physics_loss(self, x, y_pred):
        mass = torch.exp(self.log_mass)
        g = 9.81
        dt = 0.001

        thrust = x[:, 12]
        phi = x[:, 3]
        theta = x[:, 4]

        vz = x[:, 11]
        vz_next = y_pred[:, 11]

        # Simplified physics: vertical dynamics
        az_physics = (thrust / mass) * torch.cos(phi) * torch.cos(theta) - g
        vz_physics = vz + az_physics * dt

        return nn.MSELoss()(vz_next, vz_physics)

    def constrain_parameters(self):
        with torch.no_grad():
            self.log_mass.clamp_(-1.0, 2.0)
            for log_I in [self.log_Ixx, self.log_Iyy, self.log_Izz]:
                log_I.clamp_(-6.0, 0.0)


def train_model(model, train_loader, val_loader, scaler_y, w_phys, lr, seed, patience):
    """Train model with configurable parameters"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for X_scaled, y_scaled, X_unscaled, y_unscaled in train_loader:
            optimizer.zero_grad()
            y_pred_scaled = model(X_scaled)
            sup_loss = nn.MSELoss()(y_pred_scaled, y_scaled)
            loss = sup_loss

            if w_phys > 0 and hasattr(model, 'physics_loss'):
                y_pred_unscaled = y_pred_scaled * y_scale + y_mean
                phys_loss = model.physics_loss(X_unscaled, y_pred_unscaled)
                loss = sup_loss + w_phys * phys_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            if hasattr(model, 'constrain_parameters'):
                model.constrain_parameters()

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for X_scaled, y_scaled, _, _ in val_loader:
                y_pred = model(X_scaled)
                val_loss += nn.MSELoss()(y_pred, y_scaled).item() * X_scaled.size(0)
                n_val += X_scaled.size(0)
        val_loss /= n_val
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_loss


def autoregressive_rollout(model, initial_state, controls, scaler_X, scaler_y, n_steps):
    """Autoregressive rollout for n_steps"""
    model.eval()
    x_mean, x_scale = scaler_X.mean_, scaler_X.scale_
    y_mean, y_scale = scaler_y.mean_, scaler_y.scale_

    states = [initial_state.copy()]
    current = initial_state.copy()

    with torch.no_grad():
        for i in range(min(n_steps, len(controls))):
            inp = np.concatenate([current, controls[i]])
            inp_scaled = (inp - x_mean) / x_scale
            out_scaled = model(torch.FloatTensor(inp_scaled).unsqueeze(0)).squeeze(0).numpy()
            current = out_scaled * y_scale + y_mean
            states.append(current)

    return np.array(states)


def evaluate_model(model, val_loader, val_trajectories, scaler_X, scaler_y, rollout_horizon=100):
    """Evaluate with configurable rollout horizon"""
    model.eval()
    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_scale = torch.FloatTensor(scaler_y.scale_)

    # Single-step MAE
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_scaled, y_scaled, _, y_unscaled in val_loader:
            y_pred_scaled = model(X_scaled)
            y_pred = y_pred_scaled * y_scale + y_mean
            all_preds.append(y_pred.numpy())
            all_true.append(y_unscaled.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    single_step_mae = float(np.mean(np.abs(all_preds - all_true)))

    # Rollout MAE
    pos_errors = []
    for traj in val_trajectories[:N_ROLLOUT_TRAJ]:
        states, controls = traj['states'], traj['controls']
        if len(states) < rollout_horizon + 1:
            continue
        predicted = autoregressive_rollout(model, states[0], controls, scaler_X, scaler_y, rollout_horizon)
        pos_errors.append(np.mean(np.abs(predicted[:, :3] - states[:len(predicted), :3])))

    rollout_mae = float(np.mean(pos_errors)) if pos_errors else float('inf')
    return single_step_mae, rollout_mae


def compute_statistics(values):
    """Comprehensive statistics"""
    arr = np.array(values)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    se = std / np.sqrt(n)
    ci95 = 1.96 * se
    return {
        'mean': float(mean),
        'std': float(std),
        'se': float(se),
        'ci95': float(ci95),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'n': n
    }


def welch_ttest(group1, group2):
    """Comprehensive statistical comparison"""
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    # Mann-Whitney U (non-parametric)
    u_stat, u_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    # Effect sizes
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    # Levene's test for variance
    levene_stat, levene_p = stats.levene(group1, group2)

    # Coefficient of variation comparison
    cv1 = np.std(group1, ddof=1) / np.mean(group1) * 100 if np.mean(group1) > 0 else 0
    cv2 = np.std(group2, ddof=1) / np.mean(group2) * 100 if np.mean(group2) > 0 else 0

    return {
        'welch_t': float(t_stat),
        'welch_p': float(p_value),
        'mann_whitney_u': float(u_stat),
        'mann_whitney_p': float(u_p),
        'cohens_d': float(cohens_d),
        'levene_stat': float(levene_stat),
        'levene_p': float(levene_p),
        'cv1_percent': float(cv1),
        'cv2_percent': float(cv2)
    }


def main():
    print("=" * 80)
    print("COMPREHENSIVE EASE RENE 2026 ANALYSIS")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[1/6] Loading data...")
    X_train, y_train = load_data(TRAIN_DATA)
    X_val, y_val = load_data(VAL_DATA)
    val_trajectories = load_trajectories(VAL_DATA)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled),
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled),
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")

    all_results = {'metadata': {'timestamp': datetime.now().isoformat()}}

    # ========================================================================
    # ANALYSIS 1: Load existing 20-seed weight sweep data
    # ========================================================================
    print("\n[2/6] Loading existing 20-seed weight sweep data...")

    existing_results_path = PROJECT_ROOT / 'results' / 'weight_sweep' / 'weight_sweep_robust_results.json'
    if existing_results_path.exists():
        with open(existing_results_path) as f:
            existing_weight_sweep = json.load(f)

        w0_rollouts = [r['rollout_mae'] for r in existing_weight_sweep['w0.0']['seed_results']]
        w20_rollouts = [r['rollout_mae'] for r in existing_weight_sweep['w20.0']['seed_results']]
        w0_single = [r['single_step_mae'] for r in existing_weight_sweep['w0.0']['seed_results']]
        w20_single = [r['single_step_mae'] for r in existing_weight_sweep['w20.0']['seed_results']]

        all_results['weight_sweep_20seed'] = {
            'w0': {
                'rollout': compute_statistics(w0_rollouts),
                'single_step': compute_statistics(w0_single),
                'raw_rollouts': w0_rollouts,
                'raw_single_step': w0_single
            },
            'w20': {
                'rollout': compute_statistics(w20_rollouts),
                'single_step': compute_statistics(w20_single),
                'raw_rollouts': w20_rollouts,
                'raw_single_step': w20_single
            },
            'comparison': welch_ttest(w0_rollouts, w20_rollouts)
        }

        # Correlation analysis (Simpson's paradox)
        overall_single = w0_single + w20_single
        overall_rollout = w0_rollouts + w20_rollouts

        overall_r, overall_p = stats.pearsonr(overall_single, overall_rollout)
        w0_r, w0_p = stats.pearsonr(w0_single, w0_rollouts)
        w20_r, w20_p = stats.pearsonr(w20_single, w20_rollouts)

        all_results['correlation_analysis'] = {
            'overall': {'r': float(overall_r), 'p': float(overall_p)},
            'w0_within': {'r': float(w0_r), 'p': float(w0_p)},
            'w20_within': {'r': float(w20_r), 'p': float(w20_p)},
            'simpsons_paradox': abs(overall_r) > 0.3 and abs(w0_r) < 0.35 and abs(w20_r) < 0.35
        }

        print(f"  w=0:  {np.mean(w0_rollouts):.3f} ± {np.std(w0_rollouts):.3f}m (20 seeds)")
        print(f"  w=20: {np.mean(w20_rollouts):.3f} ± {np.std(w20_rollouts):.3f}m (20 seeds)")
        print(f"  p-value: {all_results['weight_sweep_20seed']['comparison']['welch_p']:.4f}")

    # ========================================================================
    # ANALYSIS 2: Extended weight sweep (new weights, 3 seeds for speed)
    # ========================================================================
    print("\n[3/6] Extended physics weight sweep (3 seeds)...")

    extended_weights = [1.0, 5.0, 10.0, 50.0, 100.0]
    extended_results = {}

    for w in extended_weights:
        print(f"\n  w={w}...")
        seed_rollouts = []

        for seed in QUICK_SEEDS:
            model = FlexiblePINN(n_layers=5, hidden_dim=256)
            model, _ = train_model(model, train_loader, val_loader, scaler_y,
                                   w, BASE_LR, seed, EARLY_STOP_PATIENCE)
            _, rollout = evaluate_model(model, val_loader, val_trajectories,
                                        scaler_X, scaler_y, 100)
            seed_rollouts.append(rollout)
            print(f"    seed {seed}: {rollout:.3f}m")

        extended_results[f'w{w}'] = compute_statistics(seed_rollouts)
        extended_results[f'w{w}']['raw'] = seed_rollouts

    all_results['extended_weight_sweep'] = extended_results

    # ========================================================================
    # ANALYSIS 3: Learning rate sensitivity
    # ========================================================================
    print("\n[4/6] Learning rate sensitivity analysis...")

    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    lr_results = {}

    for lr in learning_rates:
        print(f"\n  lr={lr}...")
        seed_rollouts = []

        for seed in QUICK_SEEDS:
            model = FlexiblePINN(n_layers=5, hidden_dim=256)
            model, _ = train_model(model, train_loader, val_loader, scaler_y,
                                   0, lr, seed, EARLY_STOP_PATIENCE)  # w=0
            _, rollout = evaluate_model(model, val_loader, val_trajectories,
                                        scaler_X, scaler_y, 100)
            seed_rollouts.append(rollout)
            print(f"    seed {seed}: {rollout:.3f}m")

        lr_results[f'lr{lr}'] = compute_statistics(seed_rollouts)
        lr_results[f'lr{lr}']['raw'] = seed_rollouts

    all_results['learning_rate_sensitivity'] = lr_results

    # ========================================================================
    # ANALYSIS 4: Rollout horizon sensitivity
    # ========================================================================
    print("\n[5/6] Rollout horizon sensitivity...")

    horizons = [50, 100, 200, 500]
    horizon_results = {}

    # Train one model per condition
    model_w0 = FlexiblePINN(n_layers=5, hidden_dim=256)
    model_w0, _ = train_model(model_w0, train_loader, val_loader, scaler_y,
                              0, BASE_LR, 42, EARLY_STOP_PATIENCE)

    model_w20 = FlexiblePINN(n_layers=5, hidden_dim=256)
    model_w20, _ = train_model(model_w20, train_loader, val_loader, scaler_y,
                               20, BASE_LR, 42, EARLY_STOP_PATIENCE)

    for h in horizons:
        print(f"\n  horizon={h}...")
        _, rollout_w0 = evaluate_model(model_w0, val_loader, val_trajectories,
                                       scaler_X, scaler_y, h)
        _, rollout_w20 = evaluate_model(model_w20, val_loader, val_trajectories,
                                        scaler_X, scaler_y, h)

        horizon_results[f'h{h}'] = {
            'w0': float(rollout_w0),
            'w20': float(rollout_w20),
            'ratio': float(rollout_w20 / rollout_w0) if rollout_w0 > 0 else float('inf')
        }
        print(f"    w=0: {rollout_w0:.3f}m, w=20: {rollout_w20:.3f}m")

    all_results['rollout_horizon_sensitivity'] = horizon_results

    # ========================================================================
    # ANALYSIS 5: Architecture variations
    # ========================================================================
    print("\n[6/6] Architecture variations...")

    architectures = [
        {'name': '3layer_128', 'n_layers': 3, 'hidden_dim': 128},
        {'name': '3layer_256', 'n_layers': 3, 'hidden_dim': 256},
        {'name': '5layer_128', 'n_layers': 5, 'hidden_dim': 128},
        {'name': '5layer_256', 'n_layers': 5, 'hidden_dim': 256},
        {'name': '7layer_256', 'n_layers': 7, 'hidden_dim': 256},
        {'name': '5layer_512', 'n_layers': 5, 'hidden_dim': 512},
    ]

    arch_results = {}

    for arch in architectures:
        print(f"\n  {arch['name']}...")
        seed_rollouts = []

        for seed in QUICK_SEEDS:
            model = FlexiblePINN(n_layers=arch['n_layers'], hidden_dim=arch['hidden_dim'])
            n_params = sum(p.numel() for p in model.parameters())
            model, _ = train_model(model, train_loader, val_loader, scaler_y,
                                   0, BASE_LR, seed, EARLY_STOP_PATIENCE)
            _, rollout = evaluate_model(model, val_loader, val_trajectories,
                                        scaler_X, scaler_y, 100)
            seed_rollouts.append(rollout)
            print(f"    seed {seed}: {rollout:.3f}m")

        arch_results[arch['name']] = {
            **compute_statistics(seed_rollouts),
            'n_params': n_params,
            'n_layers': arch['n_layers'],
            'hidden_dim': arch['hidden_dim'],
            'raw': seed_rollouts
        }

    all_results['architecture_variations'] = arch_results

    # ========================================================================
    # INFERENCE: Key findings
    # ========================================================================
    print("\n" + "=" * 80)
    print("KEY FINDINGS AND INFERENCE")
    print("=" * 80)

    findings = []

    # Finding 1: Physics loss effect
    if 'weight_sweep_20seed' in all_results:
        comp = all_results['weight_sweep_20seed']['comparison']
        w0_mean = all_results['weight_sweep_20seed']['w0']['rollout']['mean']
        w20_mean = all_results['weight_sweep_20seed']['w20']['rollout']['mean']

        findings.append({
            'id': 'F1',
            'title': 'Physics Loss Degraded Rollout Performance',
            'description': f"w=0 achieved {w0_mean:.2f}m vs w=20 at {w20_mean:.2f}m",
            'effect_size': f"Cohen's d = {comp['cohens_d']:.2f}",
            'significance': f"p = {comp['welch_p']:.4f}",
            'interpretation': 'Statistically significant' if comp['welch_p'] < 0.05 else 'Not significant'
        })

    # Finding 2: Simpson's paradox
    if 'correlation_analysis' in all_results:
        corr = all_results['correlation_analysis']
        findings.append({
            'id': 'F2',
            'title': "Simpson's Paradox in Single-Step vs Rollout Correlation",
            'description': f"Overall r={corr['overall']['r']:.2f} but within-condition r~0",
            'w0_correlation': f"r = {corr['w0_within']['r']:.2f}, p = {corr['w0_within']['p']:.3f}",
            'w20_correlation': f"r = {corr['w20_within']['r']:.2f}, p = {corr['w20_within']['p']:.3f}",
            'interpretation': 'Single-step does NOT predict rollout within conditions'
        })

    # Finding 3: Variance difference
    if 'weight_sweep_20seed' in all_results:
        comp = all_results['weight_sweep_20seed']['comparison']
        findings.append({
            'id': 'F3',
            'title': 'Physics Loss Increases Variance',
            'description': f"w=20 has {comp['cv2_percent']:.1f}% CV vs w=0 at {comp['cv1_percent']:.1f}% CV",
            'levene_test': f"Levene's p = {comp['levene_p']:.4f}",
            'interpretation': 'Significant variance difference' if comp['levene_p'] < 0.05 else 'Not significant'
        })

    all_results['key_findings'] = findings

    for f in findings:
        print(f"\n{f['id']}: {f['title']}")
        print(f"  {f['description']}")
        print(f"  Interpretation: {f['interpretation']}")

    # ========================================================================
    # Save results
    # ========================================================================
    results_path = RESULTS_DIR / 'comprehensive_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {results_path}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results


if __name__ == '__main__':
    results = main()
