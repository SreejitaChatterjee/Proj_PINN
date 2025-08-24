#!/usr/bin/env python3
"""
Complete pipeline for Quadrotor PINN model training and evaluation
"""

import sys
import os
import subprocess
import time

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        end_time = time.time()
        print(f"[SUCCESS] {description} completed successfully in {end_time-start_time:.2f}s")
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error in {description}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False
        
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        ('torch', 'torch'), 
        ('numpy', 'numpy'), 
        ('pandas', 'pandas'), 
        ('sklearn', 'scikit-learn'),  # sklearn is the import name, scikit-learn is pip name
        ('matplotlib', 'matplotlib'), 
        ('seaborn', 'seaborn'), 
        ('scipy', 'scipy')
    ]
    
    missing_packages = []
    
    for import_name, pip_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Run the complete PINN pipeline"""
    
    print("Quadrotor Physics-Informed Neural Network (PINN) Pipeline")
    print("=========================================================")
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies before running the pipeline.")
        return
    
    # Pipeline steps
    steps = [
        ("quadrotor_data_generator.py", "Generating quadrotor training data"),
        ("quadrotor_pinn_model.py", "Training PINN model"),  
        ("visualize_results.py", "Creating visualization plots")
    ]
    
    # Execute pipeline
    for script, description in steps:
        if not os.path.exists(script):
            print(f"[ERROR] Script not found: {script}")
            return
            
        success = run_script(script, description)
        if not success:
            print(f"Pipeline failed at: {description}")
            return
    
    # Final summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    print("\nGenerated files:")
    files_to_check = [
        ("quadrotor_training_data.csv", "Training dataset"),
        ("quadrotor_pinn_model.pth", "Trained PINN model"),
        ("predicted_trajectory.csv", "PINN predictions"),
        ("training_curves.png", "Training loss curves"),
        ("state_comparison_trajectory_0.png", "State comparison plots"),
        ("control_inputs_trajectory_0.png", "Control input plots"),
        ("3d_trajectory_0.png", "3D trajectory visualization"),
        ("prediction_errors_trajectory_0.png", "Prediction error analysis"),
        ("learned_physical_parameters.png", "Physical parameter comparison")
    ]
    
    for filename, description in files_to_check:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"  [OK] {filename} ({description}) - {size_mb:.2f} MB")
        else:
            print(f"  [MISSING] {filename} (not found)")
    
    print("\nModel Features:")
    print("  • Input: thrust, z, 3 torques, roll, pitch, yaw, p, q, r, vx, vy, vz")
    print("  • Output: next step predictions + mass, inertia parameters")
    print("  • Physics-informed loss function")
    print("  • Time-series prediction capability")
    
    print("\nUsage Examples:")
    print("  • Load model: QuadrotorPINN.load_state_dict(torch.load('quadrotor_pinn_model.pth'))")
    print("  • Predict trajectory: visualizer.predict_trajectory(initial_state)")
    print("  • Analyze results: Check generated PNG files for visualizations")

if __name__ == "__main__":
    main()