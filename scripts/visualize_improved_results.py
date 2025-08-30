#!/usr/bin/env python3
"""
Generate updated visualizations with improved PINN model results
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from improved_pinn_model import ImprovedQuadrotorPINN
from quadrotor_pinn_model import QuadrotorDataProcessor
from quadrotor_data_generator import QuadrotorSimulator
import seaborn as sns

class ImprovedVisualizer:
    """Updated visualizer for improved PINN results"""
    
    def __init__(self, improved_model_path, processor):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load improved model
        self.improved_model = ImprovedQuadrotorPINN(input_size=12, hidden_size=128, output_size=16, num_layers=4)
        self.improved_model.load_state_dict(torch.load(improved_model_path, map_location=self.device))
        self.improved_model.eval()
        
        self.processor = processor
    
    def predict_trajectory_improved(self, initial_state, duration=2.0, dt=0.01):
        """Predict trajectory using improved model"""
        input_features = ['thrust', 'z', 'torque_x', 'torque_y', 'torque_z', 
                         'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
        
        trajectory = []
        current_state = initial_state.copy()
        times = np.arange(0, duration, dt)
        
        with torch.no_grad():
            for t in times:
                input_vector = np.array([current_state[feat] for feat in input_features])
                input_scaled = self.processor.scaler_input.transform(input_vector.reshape(1, -1))
                input_tensor = torch.FloatTensor(input_scaled).to(self.device)
                
                output_scaled = self.improved_model(input_tensor).cpu().numpy()
                output = self.processor.inverse_transform_output(output_scaled)[0]
                
                trajectory_point = current_state.copy()
                trajectory_point['timestamp'] = t
                trajectory.append(trajectory_point)
                
                # Update state
                output_features = ['thrust', 'z', 'torque_x', 'torque_y', 'torque_z',
                                  'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
                
                for i, feat in enumerate(output_features):
                    current_state[feat] = output[i]
                    
        return pd.DataFrame(trajectory)
    
    def plot_parameter_comparison(self):
        """Compare true vs improved model parameters"""
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        
        # True values
        true_params = {
            'Mass': 0.068,
            'Jxx': 6.86e-5,
            'Jyy': 9.2e-5,
            'Jzz': 1.366e-4,
            'Gravity': 9.81
        }
        
        # Improved model parameters
        improved_params = {
            'Mass': self.improved_model.m.item(),
            'Jxx': self.improved_model.Jxx.item(),
            'Jyy': self.improved_model.Jyy.item(),
            'Jzz': self.improved_model.Jzz.item(),
            'Gravity': self.improved_model.g.item()
        }
        
        param_names = list(true_params.keys())
        
        for i, param in enumerate(param_names):
            ax = axes[i]
            
            x = ['True', 'Improved\nModel']
            y = [true_params[param], improved_params[param]]
            colors = ['blue', 'green']
            
            bars = ax.bar(x, y, color=colors, alpha=0.7)
            ax.set_title(f'{param}')
            ax.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, val in zip(bars, y):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.6f}', ha='center', va='bottom', fontsize=8)
            
            ax.grid(True, alpha=0.3)
            
            # Calculate and show error percentage
            impr_error = abs(improved_params[param] - true_params[param]) / true_params[param] * 100
            
            ax.text(0.5, 0.95, f'Error: {impr_error:.1f}%', 
                   transform=ax.transAxes, ha='center', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('parameter_comparison_old_vs_new.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return true_params, improved_params
    
    def plot_updated_state_comparison(self, true_data, predicted_data, trajectory_id=0):
        """Updated state comparison with improved predictions"""
        
        true_traj = true_data[true_data['trajectory_id'] == trajectory_id]
        
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
            
            # Plot improved predicted trajectory
            if len(predicted_data) > 0:
                ax.plot(predicted_data['timestamp'], predicted_data[state], 
                       'g--', label='Improved PINN', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title} - Improved Model')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.suptitle('State Comparison - Improved PINN Model', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig('improved_state_comparison_trajectory_0.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def additional_physics_formulas():
    """Display additional physics formulas that could enhance the PINN"""
    
    print("\n" + "="*60)
    print("ADDITIONAL PHYSICS FORMULAS FOR ENHANCED PINN")
    print("="*60)
    
    formulas = [
        {
            'name': 'Complete Translational Dynamics',
            'equations': [
                'x_dot = u*cos(psi)*cos(theta) + v*(cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi)) + w*(sin(psi)*sin(phi) + cos(psi)*sin(theta)*cos(phi))',
                'y_dot = u*sin(psi)*cos(theta) + v*(cos(psi)*cos(phi) + sin(psi)*sin(theta)*sin(phi)) + w*(sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))',
                'z_dot = -u*sin(theta) + v*cos(theta)*sin(phi) + w*cos(theta)*cos(phi)'
            ],
            'benefit': 'Full 3D position dynamics for complete trajectory prediction'
        },
        {
            'name': 'Motor Dynamics',
            'equations': [
                'T = kt * (n1^2 + n2^2 + n3^2 + n4^2)',
                'tau_x = kt * b * (n4^2 - n2^2)',
                'tau_y = kt * b * (n3^2 - n1^2)', 
                'tau_z = kq * (n1^2 + n3^2 - n2^2 - n4^2)'
            ],
            'benefit': 'Connect control inputs to actual motor speeds'
        },
        {
            'name': 'Aerodynamic Effects',
            'equations': [
                'F_drag_x = -0.5 * rho * Cd * A * u * |u|',
                'F_drag_y = -0.5 * rho * Cd * A * v * |v|',
                'F_drag_z = -0.5 * rho * Cd * A * w * |w|'
            ],
            'benefit': 'More realistic aerodynamic modeling at higher speeds'
        },
        {
            'name': 'Gyroscopic Effects',
            'equations': [
                'M_gyro_x = Jr * q * (Omega1 - Omega2 + Omega3 - Omega4)',
                'M_gyro_y = -Jr * p * (Omega1 - Omega2 + Omega3 - Omega4)',
                'M_gyro_z = 0'
            ],
            'benefit': 'Account for propeller gyroscopic moments'
        },
        {
            'name': 'Ground Effect',
            'equations': [
                'T_ground = T * (1 + (R/(4*h))^2)  when h < 2*R',
                'where h = height above ground, R = rotor radius'
            ],
            'benefit': 'Increased thrust efficiency near ground'
        }
    ]
    
    for formula in formulas:
        print(f"\n{formula['name']}:")
        print("-" * 40)
        for eq in formula['equations']:
            print(f"  {eq}")
        print(f"Benefit: {formula['benefit']}")
    
    print(f"\n{'IMPLEMENTATION PRIORITY:'}")
    print("1. Complete Translational Dynamics - HIGH (most impact)")
    print("2. Motor Dynamics - MEDIUM (better control modeling)")  
    print("3. Aerodynamic Effects - LOW (minimal impact at low speeds)")
    print("4. Gyroscopic Effects - LOW (small for small quadrotors)")
    print("5. Ground Effect - LOW (specific flight regime)")

if __name__ == "__main__":
    print("GENERATING UPDATED VISUALIZATIONS")
    print("="*50)
    
    # Load data
    df = pd.read_csv('quadrotor_training_data.csv')
    df = df[df['trajectory_id'] < 3].copy()
    
    # Initialize processor
    processor = QuadrotorDataProcessor()
    X, y = processor.prepare_sequences(df)
    processor.fit_transform(X, y)
    
    # Initialize visualizer with improved model
    visualizer = ImprovedVisualizer('../models/improved_quadrotor_pinn_model.pth', processor)
    
    # Get initial state
    trajectory_0 = df[df['trajectory_id'] == 0].iloc[0].to_dict()
    
    # Generate predictions with improved model
    print("Generating improved PINN predictions...")
    improved_predictions = visualizer.predict_trajectory_improved(trajectory_0, duration=2.0, dt=0.01)
    
    # Create updated visualizations
    print("Creating updated visualizations...")
    
    # Parameter comparison plot
    true_params, impr_params = visualizer.plot_parameter_comparison()
    
    # Updated state comparison
    visualizer.plot_updated_state_comparison(df, improved_predictions, trajectory_id=0)
    
    # Save improved predictions
    improved_predictions.to_csv('improved_predicted_trajectory.csv', index=False)
    
    # Show additional physics formulas
    additional_physics_formulas()
    
    # Summary
    print(f"\n{'IMPROVED MODEL RESULTS:'}")
    print("=" * 40)
    print("Parameter Accuracy:")
    
    for param in true_params:
        impr_error = abs(impr_params[param] - true_params[param]) / true_params[param] * 100
        print(f"{param:<10}: {impr_error:>6.1f}% error")
    
    print(f"\nFiles generated:")
    print("• parameter_comparison_old_vs_new.png")
    print("• improved_state_comparison_trajectory_0.png") 
    print("• improved_predicted_trajectory.csv")