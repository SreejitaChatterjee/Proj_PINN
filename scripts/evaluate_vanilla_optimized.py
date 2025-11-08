"""Evaluation script for Vanilla Optimized PINN"""
import torch
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from pinn_model_vanilla_optimized import QuadrotorPINNVanillaOptimized

def evaluate_model(model_path, scaler_path, data_path, device='cpu'):
    """Evaluate vanilla optimized PINN model"""

    # Load model
    model = QuadrotorPINNVanillaOptimized(hidden_size=128, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load scalers
    scalers = joblib.load(scaler_path)
    scaler_X, scaler_y = scalers['scaler_X'], scalers['scaler_y']

    # Load data
    df = pd.read_csv(data_path)
    df = df.rename(columns={'roll': 'phi', 'pitch': 'theta', 'yaw': 'psi'})

    print("="*70)
    print("VANILLA OPTIMIZED PINN EVALUATION (NO FOURIER)")
    print("="*70)
    print(f"Model: Residual MLP + Modular Architecture (Stable)")
    print()

    # 1. Teacher-Forced Evaluation
    print("1. TEACHER-FORCED (Single-Step) Evaluation:")
    print("-" * 70)

    features = ['z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vz',
                'thrust', 'torque_x', 'torque_y', 'torque_z']

    X_test, y_test = [], []
    for traj_id in df['trajectory_id'].unique():
        traj = df[df['trajectory_id'] == traj_id].sort_values('timestamp')
        traj_data = traj[features].values
        X_test.append(traj_data[:-1])
        y_test.append(traj_data[1:, :8])

    X_test, y_test = np.vstack(X_test), np.vstack(y_test)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_pred_scaled = model(X_tensor).cpu().numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    state_names = ['z', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'vz']
    print(f"{'State':<10} {'MAE':<12} {'RMSE':<12} {'Max Error':<12}")
    print("-" * 70)

    teacher_forced_metrics = {}
    for i, name in enumerate(state_names):
        mae = np.mean(np.abs(y_pred[:, i] - y_test[:, i]))
        rmse = np.sqrt(np.mean((y_pred[:, i] - y_test[:, i])**2))
        max_err = np.max(np.abs(y_pred[:, i] - y_test[:, i]))
        teacher_forced_metrics[name] = {'mae': mae, 'rmse': rmse, 'max': max_err}
        print(f"{name:<10} {mae:<12.6f} {rmse:<12.6f} {max_err:<12.6f}")

    # 2. Autoregressive Rollout Evaluation
    print("\n2. AUTOREGRESSIVE ROLLOUT (100-step, 0.1s) Evaluation:")
    print("-" * 70)

    traj_0 = df[df['trajectory_id'] == 0].iloc[:101]
    initial_state = traj_0[features].iloc[0].values[:8]
    controls = traj_0[['thrust', 'torque_x', 'torque_y', 'torque_z']].values
    true_states = traj_0[features].values[:, :8]

    # Autoregressive rollout
    predicted_states = [initial_state]
    current_state = initial_state.copy()

    for t in range(100):
        # Prepare input
        model_input = np.concatenate([current_state, controls[t]])
        model_input_scaled = scaler_X.transform(model_input.reshape(1, -1))

        with torch.no_grad():
            input_tensor = torch.FloatTensor(model_input_scaled).to(device)
            pred_scaled = model(input_tensor).cpu().numpy()

        next_state = scaler_y.inverse_transform(pred_scaled)[0]
        predicted_states.append(next_state)
        current_state = next_state

    predicted_states = np.array(predicted_states)

    print(f"{'State':<10} {'MAE':<12} {'RMSE':<12} {'Max Error':<12}")
    print("-" * 70)

    autoregressive_metrics = {}
    for i, name in enumerate(state_names):
        mae = np.mean(np.abs(predicted_states[:, i] - true_states[:, i]))
        rmse = np.sqrt(np.mean((predicted_states[:, i] - true_states[:, i])**2))
        max_err = np.max(np.abs(predicted_states[:, i] - true_states[:, i]))
        autoregressive_metrics[name] = {'mae': mae, 'rmse': rmse, 'max': max_err}
        print(f"{name:<10} {mae:<12.6f} {rmse:<12.6f} {max_err:<12.6f}")

    # 3. Parameter Identification
    print("\n3. PARAMETER IDENTIFICATION:")
    print("-" * 70)
    print(f"{'Parameter':<12} {'Learned':<15} {'True':<15} {'Error %':<10}")
    print("-" * 70)

    for k, v in model.params.items():
        learned = v.item()
        true = model.true_params[k]
        error = abs(learned - true) / true * 100
        print(f"{k:<12} {learned:<15.6e} {true:<15.6e} {error:<10.2f}%")

    # 4. Generate plots
    print("\n4. Generating evaluation plots...")
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    time = np.arange(101) * 0.001

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle('Vanilla Optimized PINN - Autoregressive Rollout (100 steps, 0.1s)',
                 fontsize=14, fontweight='bold')

    for i, (ax, name) in enumerate(zip(axes.flat, state_names)):
        ax.plot(time, true_states[:, i], 'b-', label='True', linewidth=2, alpha=0.7)
        ax.plot(time, predicted_states[:, i], 'r--', label='Predicted', linewidth=1.5)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        mae = autoregressive_metrics[name]['mae']
        rmse = autoregressive_metrics[name]['rmse']
        ax.set_title(f'{name.upper()} - MAE: {mae:.4f}, RMSE: {rmse:.4f}', fontsize=10)

    plt.tight_layout()
    plot_path = results_dir / 'vanilla_optimized_pinn_evaluation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Saved plot to {plot_path}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    return {
        'teacher_forced': teacher_forced_metrics,
        'autoregressive': autoregressive_metrics,
        'predicted_states': predicted_states,
        'true_states': true_states
    }

if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / 'models' / 'quadrotor_pinn_vanilla_optimized.pth'
    scaler_path = Path(__file__).parent.parent / 'models' / 'scalers_vanilla_optimized.pkl'
    data_path = Path(__file__).parent.parent / 'data' / 'quadrotor_training_data.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = evaluate_model(model_path, scaler_path, data_path, device)
