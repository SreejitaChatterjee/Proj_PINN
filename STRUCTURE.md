# Project Structure

## PINN Quadrotor Parameter Identification

### 📁 Root Directory
```
Proj_PINN/
├── README.md                           # Main project documentation
├── PROJECT_REPORT.md                   # Detailed technical report
├── STRUCTURE.md                        # This file - project organization
├── .gitignore                         # Git ignore rules
├── quadrotor_data_generator.py        # Main data generation script
├── generate_output_statistics.py     # Final visualization generator
└── run_complete_pipeline.py          # Complete training pipeline
```

### 📁 Core Directories

#### `/scripts/` - Core Implementation
- **`quadrotor_pinn_model.py`** - Main PINN model implementation
- **`enhanced_pinn_model.py`** - Enhanced version with physics constraints
- **`improved_pinn_model.py`** - Improved architecture variant
- **`aggressive_data_generator.py`** - Enhanced flight data generator
- **`simple_aggressive_data.py`** - Simple data generation utility

#### `/models/` - Trained Models
- **`quadrotor_pinn_model.pth`** - Base PINN model
- **`improved_quadrotor_pinn_model.pth`** - Improved model version
- **`enhanced_quadrotor_pinn_model.pth`** - Enhanced model version
- **`best_quick_optimized_model.pth`** - Best performing model

#### `/results/` - Data & Results
- **`quadrotor_training_data.csv`** - Base training dataset (15K samples)
- **`aggressive_quadrotor_training_data.csv`** - Enhanced dataset (97.6K samples)
- **`predicted_trajectory.csv`** - Model predictions
- **`improved_predicted_trajectory.csv`** - Enhanced model predictions

#### `/visualizations/` - Final Output Statistics
- **`01_accuracy_overview.png`** - Individual & category-wise accuracy analysis
- **`02_error_analysis.png`** - RMSE, MAE, bias, and correlation analysis
- **`03_performance_metrics.png`** - R² scores, distribution & ranking
- **`04_detailed_breakdown.png`** - Category-wise detailed analysis

### 🎯 Key Features

#### Neural Network Outputs (12 total)
1. **Control**: thrust, torque_x, torque_y, torque_z
2. **Attitude**: roll, pitch, yaw
3. **Rates**: p_rate, q_rate, r_rate  
4. **Position/Velocity**: z_position, z_velocity

#### Performance Statistics
- **Overall Mean Accuracy**: 83.5%
- **Best Performer**: z_position (92.1%)
- **Most Challenging**: torque_z (71.2%)
- **Mean RMSE**: 16.5%
- **Mean R² Score**: 0.885

### 🚀 Quick Start

1. **Generate Visualizations**:
   ```bash
   python generate_output_statistics.py
   ```

2. **Run Complete Pipeline**:
   ```bash
   python run_complete_pipeline.py
   ```

3. **Generate Training Data**:
   ```bash
   python quadrotor_data_generator.py
   ```

### 📊 Repository Stats
- **Total Files**: ~20 essential files
- **Models**: 4 trained PINN variants
- **Datasets**: 2 (base + enhanced)
- **Visualizations**: 4 comprehensive statistical analyses
- **Documentation**: 3 files (README, PROJECT_REPORT, STRUCTURE)

### 🧹 Cleaned Up
Removed redundant/experimental files:
- Old visualization scripts (5 removed)
- Experimental training variants (7 removed)  
- Old analysis scripts (5 removed)
- Python cache files (__pycache__)

**Clean, organized repository ready for academic presentation and further development.**