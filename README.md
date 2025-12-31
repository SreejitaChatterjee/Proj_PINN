# PINN Dynamics & GPS-IMU Anomaly Detection

A research repository containing two main contributions:
1. **PINN Dynamics Framework** - Physics-informed neural networks for learning dynamical systems
2. **GPS-IMU Anomaly Detector** - Spoofing detection for UAVs (publication-ready)

## Repository Structure

```
Proj_PINN/
│
├── pinn_dynamics/          # Core PINN framework (Python package)
│   ├── systems/            # System definitions (Quadrotor, Pendulum, etc.)
│   ├── training/           # Training utilities
│   ├── inference/          # Prediction and export (ONNX, TorchScript)
│   └── data/               # Data loaders
│
├── gps_imu_detector/       # GPS-IMU Spoofing Detector (Publication-Ready)
│   ├── src/                # Detection algorithms (ICI, CUSUM, etc.)
│   ├── docs/               # Methodology documentation
│   ├── results/            # Evaluation results
│   └── scripts/            # Evaluation scripts
│
├── papers/                 # Paper submissions (organized by venue)
│   ├── gps_imu_detection/  # DSN - GPS-IMU Detection
│   ├── pinn_stability_envelope/  # ACC/CDC
│   ├── pinn_architecture/  # NeurIPS 2025
│   ├── pinn_modular_design/  # ICRA 2026
│   ├── pinn_curriculum/    # RA-L
│   └── shared/             # Common resources
│
├── experiments/            # All experimental artifacts
│   ├── models/             # Trained checkpoints
│   ├── results/            # Experiment outputs
│   ├── research/           # Research artifacts
│   └── runs/               # Training logs
│
├── data/                   # Datasets
│   ├── euroc/              # EuRoC MAV dataset
│   ├── alfa/               # ALFA dataset
│   └── PADRE_dataset/      # PADRE fault dataset
│
├── docs/                   # Project documentation
├── configs/                # Configuration files
├── scripts/                # Utility scripts
├── examples/               # Usage examples
├── tests/                  # Test suite
└── legacy/                 # Deprecated code
```

## Quick Start

### PINN Dynamics

```bash
pip install -e .
python demo.py --real    # Real EuRoC flight data
```

### GPS-IMU Detector

```bash
cd gps_imu_detector
python run_publication_evaluation.py
```

## Key Results

### GPS-IMU Detector (Publication-Ready)

| Metric | Value |
|--------|-------|
| AUROC | 99.8% |
| FPR | 0.21% |
| Detection Rate | 100% @ 10x magnitude |
| Detectability Floor | ~5-10m offset |

### PINN Dynamics

| Architecture | 100-step MAE | Parameters |
|--------------|--------------|------------|
| Baseline | 5.09m | 205K |
| **Modular** | **1.11m** | **72K** |

## Papers

| Paper | Venue | Status |
|-------|-------|--------|
| GPS-IMU Detection | DSN/Security | Publication-Ready |
| Stability Envelope | ACC/CDC | Needs Revision |
| Architecture Comparison | NeurIPS 2025 | Needs Revision |
| Modular Design | ICRA 2026 | Needs Revision |
| Curriculum Training | RA-L | Needs Revision |

See `papers/README.md` for full index.

## Installation

```bash
# Clone repository
git clone https://github.com/SreejitaChatterjee/Proj_PINN.git
cd Proj_PINN

# Install PINN dynamics package
pip install -e .

# Install GPS-IMU detector dependencies
pip install -r gps_imu_detector/requirements.txt
```

## License

MIT License
