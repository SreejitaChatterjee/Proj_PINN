"""Model evaluation and prediction script"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from pinn_model import QuadrotorPINN
from plot_utils import PlotGenerator

def rollout_predictions(model, initial_state, controls, n_steps):
    """Rollout model predictions for n_steps"""
    model.eval()
    states = [initial_state]

    with torch.no_grad():
        for i in range(n_steps):
            state_input = torch.cat([states[-1], controls[i]])
            next_state = model(state_input.unsqueeze(0)).squeeze(0)[:8]
            states.append(next_state)

    return torch.stack(states)

def evaluate_model(model_path, data_path, output_dir='results'):
    """Evaluate model and generate visualizations"""
    model = QuadrotorPINN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    df = pd.read_csv(data_path)
    states = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']

    # Compute predictions
    predictions = []
    with torch.no_grad():
        for idx in range(len(df) - 1):
            input_data = torch.FloatTensor(df.iloc[idx][['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz',
                                                          'thrust', 'torque_x', 'torque_y', 'torque_z',
                                                          'p_dot', 'q_dot', 'r_dot']].values)
            pred = model(input_data.unsqueeze(0)).squeeze(0)[:8].numpy()
            predictions.append(pred)

    df_pred = pd.DataFrame(predictions, columns=states)
    df_pred['timestamp'] = df['timestamp'].iloc[1:].values

    # Calculate errors
    errors = {}
    for state in states:
        true_vals = df[state].iloc[1:].values
        pred_vals = df_pred[state].values
        errors[state] = {
            'mae': np.mean(np.abs(true_vals - pred_vals)),
            'rmse': np.sqrt(np.mean((true_vals - pred_vals)**2)),
            'mape': np.mean(np.abs((true_vals - pred_vals) / (true_vals + 1e-8))) * 100
        }

    # Generate plots
    plotter = PlotGenerator(output_dir)
    plotter.plot_summary({'true': df.iloc[1:], 'pred': df_pred})
    plotter.plot_state_comparison(df.iloc[1:], df_pred, states)

    print("\nEvaluation Results:")
    print("-" * 60)
    for state, metrics in errors.items():
        print(f"{state:8s}: MAE={metrics['mae']:8.4f}, RMSE={metrics['rmse']:8.4f}, MAPE={metrics['mape']:6.2f}%")

    print(f"\nModel Parameters:")
    print("-" * 60)
    for k, v in model.params.items():
        error = abs(v.item() - model.true_params[k]) / model.true_params[k] * 100
        print(f"{k:4s}: {v.item():.6e} (true: {model.true_params[k]:.6e}, error: {error:5.1f}%)")

    return errors

if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / 'models' / 'quadrotor_pinn.pth'
    data_path = Path(__file__).parent.parent / 'data' / 'quadrotor_training_data.csv'
    evaluate_model(model_path, data_path)
