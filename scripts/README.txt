# Scripts Directory

Organized utility scripts for the PINN framework.

## Structure

```
scripts/
├── training/       # Model training scripts
├── evaluation/     # Evaluation and validation scripts
├── generation/     # Data generation scripts
├── analysis/       # Analysis and experiments
├── data/           # Data loading and preprocessing
└── utils/          # Utility and formatting scripts
```

## Subdirectories

### training/
Training scripts for various models:
- `train_euroc.py` - Train on real EuRoC MAV data
- `train.py` - General training script
- `train_padre_*.py` - PADRE fault detection training
- `train_all_architectures.py` - Architecture comparison

### evaluation/
Evaluation and validation:
- `evaluate*.py` - Model evaluation scripts
- `run_*.py` - Batch run scripts
- `show_all_results.py` - Results display
- `validate_finding.py` - Validation checks

### generation/
Data and figure generation:
- `generate_diverse_training_data.py` - Training data
- `generate_*_plots.py` - Plot generation
- `generate_ieee_publication_plots.py` - IEEE figures

### analysis/
Analysis and experiments:
- `physics_weight_sweep*.py` - Physics weight ablation
- `*_experiment.py` - Various experiments
- `pinn_*.py` - PINN architecture analysis
- `robustness_analysis.py` - Robustness studies

### data/
Data loading and preprocessing:
- `load_euroc.py` - EuRoC data loader
- `download_alfa.py` - ALFA dataset download
- `padre_*.py` - PADRE data processing
- `split_*.py` - Train/test splitting

### utils/
Utility scripts:
- `fix_*.py` - Formatting fixes
- `convert_to_ieee.py` - IEEE conversion
- `plot_utils.py` - Plotting utilities
- `export.py` - Model export utilities

## Quick Start

```bash
# Train on real EuRoC data
python training/train_euroc.py

# Generate synthetic data
python generation/generate_diverse_training_data.py

# Run evaluation
python evaluation/evaluate.py
```
