# Project Structure

## Complete Files for Submission

### Core Implementation
- `scripts/quadrotor_pinn_model.py` - Foundation PINN implementation
- `scripts/improved_pinn_model.py` - Enhanced PINN with better physics weighting
- `scripts/enhanced_pinn_model.py` - Advanced PINN with direct parameter identification

### Dataset
- `quadrotor_training_data.csv` - Training dataset (50,000 samples, 10 trajectories)

### Summary Visualizations (5 essential plots)
- `01_all_outputs_complete_analysis.png` - Comprehensive trajectory analysis
- `02_key_flight_variables.png` - Flight trajectory analysis
- `03_physical_parameters.png` - Physical parameter analysis
- `04_control_inputs.png` - Control input analysis
- `05_model_summary_statistics.png` - Model performance statistics

### Detailed Analysis - All 16 Individual Output Plots
**State Variables Time-Series (12 plots):**
- `detailed_analysis/01_thrust_time_analysis.png`
- `detailed_analysis/02_z_time_analysis.png`
- `detailed_analysis/03_torque_x_time_analysis.png`
- `detailed_analysis/04_torque_y_time_analysis.png`
- `detailed_analysis/05_torque_z_time_analysis.png`
- `detailed_analysis/06_roll_time_analysis.png`
- `detailed_analysis/07_pitch_time_analysis.png`
- `detailed_analysis/08_yaw_time_analysis.png`
- `detailed_analysis/09_p_time_analysis.png`
- `detailed_analysis/10_q_time_analysis.png`
- `detailed_analysis/11_r_time_analysis.png`
- `detailed_analysis/12_vz_time_analysis.png`

**Physical Parameter Convergence (4 plots):**
- `detailed_analysis/13_mass_convergence_analysis.png`
- `detailed_analysis/14_inertia_xx_convergence_analysis.png`
- `detailed_analysis/15_inertia_yy_convergence_analysis.png`
- `detailed_analysis/16_inertia_zz_convergence_analysis.png`

### Utilities
- `generate_all_16_plots.py` - Script to generate individual analysis plots

### Documentation
- `README.md` - Complete project documentation with tabulated results
- `PROJECT_STRUCTURE.md` - This file structure overview

## Repository Summary
- **Total files**: 26 files
- **Core PINN models**: 3 implementations
- **Individual output plots**: 16 detailed analyses (as requested by professor)
- **Summary visualizations**: 5 comprehensive plots
- **Documentation**: Complete technical specification
- **Repository ready for academic submission with all 16 outputs plotted vs time**