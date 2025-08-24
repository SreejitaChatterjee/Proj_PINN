import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quadrotor_pinn_model import QuadrotorPINN, QuadrotorDataProcessor
from quadrotor_data_generator import QuadrotorSimulator
import seaborn as sns

class QuadrotorVisualizer:
    """Visualization tools for quadrotor PINN results"""
    
    def __init__(self, model_path, processor):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        self.model = QuadrotorPINN(input_size=12, hidden_size=128, output_size=16, num_layers=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.processor = processor
        
    def predict_trajectory(self, initial_state, duration=5.0, dt=0.001):
        """Predict a full trajectory using the trained model"""
        
        # Input features order
        input_features = ['thrust', 'z', 'torque_x', 'torque_y', 'torque_z', 
                         'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
        
        # Initialize trajectory storage
        trajectory = []
        current_state = initial_state.copy()
        
        times = np.arange(0, duration, dt)
        
        with torch.no_grad():
            for t in times:
                # Prepare input
                input_vector = np.array([current_state[feat] for feat in input_features])
                input_scaled = self.processor.scaler_input.transform(input_vector.reshape(1, -1))
                input_tensor = torch.FloatTensor(input_scaled).to(self.device)
                
                # Predict next state
                output_scaled = self.model(input_tensor).cpu().numpy()
                output = self.processor.inverse_transform_output(output_scaled)[0]
                
                # Store trajectory point
                trajectory_point = current_state.copy()
                trajectory_point['timestamp'] = t
                trajectory.append(trajectory_point)
                
                # Update current state with prediction (first 12 elements)
                output_features = ['thrust', 'z', 'torque_x', 'torque_y', 'torque_z',
                                  'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
                
                for i, feat in enumerate(output_features):
                    current_state[feat] = output[i]
                    
        return pd.DataFrame(trajectory)
    
    def plot_state_comparison(self, true_data, predicted_data, trajectory_id=0):
        """Plot comparison between true and predicted states over time"""
        
        # Filter data for specific trajectory
        true_traj = true_data[true_data['trajectory_id'] == trajectory_id]
        
        # Key states to plot
        states_to_plot = [
            ('z', 'Position Z (m)', 'Height'),
            ('roll', 'Roll (rad)', 'Roll Angle'),
            ('pitch', 'Pitch (rad)', 'Pitch Angle'), 
            ('yaw', 'Yaw (rad)', 'Yaw Angle'),
            ('p', 'Angular Velocity P (rad/s)', 'Roll Rate'),
            ('q', 'Angular Velocity Q (rad/s)', 'Pitch Rate'),
            ('r', 'Angular Velocity R (rad/s)', 'Yaw Rate'),
            ('thrust', 'Thrust (N)', 'Thrust'),
            ('vz', 'Vertical Velocity (m/s)', 'Vertical Velocity')
        ]
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, (state, ylabel, title) in enumerate(states_to_plot):
            ax = axes[i]
            
            # Plot true trajectory
            ax.plot(true_traj['timestamp'], true_traj[state], 
                   'b-', label='Ground Truth', linewidth=2, alpha=0.8)
            
            # Plot predicted trajectory
            if len(predicted_data) > 0:
                ax.plot(predicted_data['timestamp'], predicted_data[state], 
                       'r--', label='PINN Prediction', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(f'state_comparison_trajectory_{trajectory_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_control_inputs(self, true_data, predicted_data, trajectory_id=0):
        """Plot control inputs over time"""
        
        true_traj = true_data[true_data['trajectory_id'] == trajectory_id]
        
        control_inputs = [
            ('thrust', 'Thrust (N)', 'Thrust Command'),
            ('torque_x', 'Torque X (N*m)', 'Roll Torque'),
            ('torque_y', 'Torque Y (N*m)', 'Pitch Torque'),
            ('torque_z', 'Torque Z (N*m)', 'Yaw Torque')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (control, ylabel, title) in enumerate(control_inputs):
            ax = axes[i]
            
            ax.plot(true_traj['timestamp'], true_traj[control], 
                   'b-', label='Ground Truth', linewidth=2, alpha=0.8)
            
            if len(predicted_data) > 0:
                ax.plot(predicted_data['timestamp'], predicted_data[control], 
                       'r--', label='PINN Prediction', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(f'control_inputs_trajectory_{trajectory_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_3d_trajectory(self, true_data, predicted_data=None, trajectory_id=0):
        """Plot 3D trajectory visualization"""
        
        true_traj = true_data[true_data['trajectory_id'] == trajectory_id]
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Add position coordinates if not available (assuming x=y=0 for simplicity)
        if 'x' not in true_traj.columns:
            true_traj = true_traj.copy()
            true_traj['x'] = 0
            true_traj['y'] = 0
            
        # Plot true trajectory
        ax.plot(true_traj['x'], true_traj['y'], -true_traj['z'], 
                'b-', label='Ground Truth', linewidth=3, alpha=0.8)
        ax.scatter(true_traj['x'].iloc[0], true_traj['y'].iloc[0], -true_traj['z'].iloc[0], 
                  color='green', s=100, label='Start')
        ax.scatter(true_traj['x'].iloc[-1], true_traj['y'].iloc[-1], -true_traj['z'].iloc[-1], 
                  color='red', s=100, label='End')
        
        # Plot predicted trajectory if available
        if predicted_data is not None and len(predicted_data) > 0:
            if 'x' not in predicted_data.columns:
                predicted_data = predicted_data.copy()
                predicted_data['x'] = 0
                predicted_data['y'] = 0
                
            ax.plot(predicted_data['x'], predicted_data['y'], -predicted_data['z'], 
                    'r--', label='PINN Prediction', linewidth=3, alpha=0.8)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Height (m)')
        ax.set_title('3D Quadrotor Trajectory')
        ax.legend()
        ax.grid(True)
        
        plt.savefig(f'3d_trajectory_{trajectory_id}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_errors(self, true_data, predicted_data, trajectory_id=0):
        """Plot prediction errors over time"""
        
        if len(predicted_data) == 0:
            print("No predicted data available for error analysis")
            return
            
        true_traj = true_data[true_data['trajectory_id'] == trajectory_id]
        
        # Align time series (interpolate if needed)
        common_times = predicted_data['timestamp'].values
        
        states_to_analyze = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'thrust']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, state in enumerate(states_to_analyze):
            ax = axes[i]
            
            # Interpolate true values to predicted timestamps
            true_interp = np.interp(common_times, true_traj['timestamp'], true_traj[state])
            pred_values = predicted_data[state].values[:len(common_times)]
            
            # Calculate error
            error = pred_values - true_interp[:len(pred_values)]
            
            ax.plot(common_times[:len(error)], error, 'r-', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{state.title()} Error')
            ax.set_title(f'Prediction Error: {state.title()}')
            ax.grid(True, alpha=0.3)
            
            # Add RMSE to title
            rmse = np.sqrt(np.mean(error**2))
            ax.set_title(f'{state.title()} Error (RMSE: {rmse:.6f})')
        
        plt.tight_layout()
        plt.savefig(f'prediction_errors_trajectory_{trajectory_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_physical_parameters_evolution(self):
        """Plot learned physical parameters"""
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Physical parameters
        params = {
            'Mass': self.model.m.item(),
            'Jxx': self.model.Jxx.item(),
            'Jyy': self.model.Jyy.item(), 
            'Jzz': self.model.Jzz.item()
        }
        
        # True values from MATLAB
        true_params = {
            'Mass': 0.068,
            'Jxx': 6.86e-5,
            'Jyy': 9.2e-5,
            'Jzz': 1.366e-4
        }
        
        for i, (param, learned_val) in enumerate(params.items()):
            ax = axes[i]
            
            x = ['True', 'Learned']
            y = [true_params[param], learned_val]
            
            bars = ax.bar(x, y, color=['blue', 'red'], alpha=0.7)
            ax.set_title(f'{param}')
            ax.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, val in zip(bars, y):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.6f}', ha='center', va='bottom')
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('learned_physical_parameters.png', dpi=300, bbox_inches='tight')
        plt.show()

# Main execution for visualization
if __name__ == "__main__":
    print("Loading data and model...")
    
    # Load training data (same reduced dataset as used for training)
    df = pd.read_csv('quadrotor_training_data.csv')
    df = df[df['trajectory_id'] < 3].copy()  # Use same reduced dataset
    print(f"Using reduced dataset with {len(df)} samples for visualization")
    
    # Initialize processor
    processor = QuadrotorDataProcessor()
    X, y = processor.prepare_sequences(df)
    processor.fit_transform(X, y)  # Fit the scalers
    
    # Initialize visualizer
    visualizer = QuadrotorVisualizer('quadrotor_pinn_model.pth', processor)
    
    # Get initial state from trajectory 0
    trajectory_0 = df[df['trajectory_id'] == 0].iloc[0].to_dict()
    
    # Predict trajectory
    print("Generating PINN predictions...")
    predicted_trajectory = visualizer.predict_trajectory(
        trajectory_0, duration=2.0, dt=0.01
    )
    
    print("Creating visualizations...")
    
    # Plot comparisons
    visualizer.plot_state_comparison(df, predicted_trajectory, trajectory_id=0)
    visualizer.plot_control_inputs(df, predicted_trajectory, trajectory_id=0)
    visualizer.plot_3d_trajectory(df, predicted_trajectory, trajectory_id=0)
    visualizer.plot_prediction_errors(df, predicted_trajectory, trajectory_id=0)
    visualizer.plot_physical_parameters_evolution()
    
    print("Visualization plots saved!")
    
    # Summary statistics
    print("\nTrajectory Summary:")
    print(f"Predicted trajectory length: {len(predicted_trajectory)} points")
    print(f"Duration: {predicted_trajectory['timestamp'].max():.2f} seconds")
    print(f"Height range: [{predicted_trajectory['z'].min():.3f}, {predicted_trajectory['z'].max():.3f}] m")
    
    # Save predicted trajectory
    predicted_trajectory.to_csv('predicted_trajectory.csv', index=False)
    print("Predicted trajectory saved to 'predicted_trajectory.csv'")