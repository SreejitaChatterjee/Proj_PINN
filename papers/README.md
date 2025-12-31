# Research Papers

This directory contains all paper submissions organized by venue/topic.

## Paper Index

| Paper | Venue | Status | Directory |
|-------|-------|--------|-----------|
| **GPS-IMU Spoofing Detection** | DSN/Security | Publication-Ready | `gps_imu_detection/` |
| **Stability Envelope Framework** | ACC/CDC | Needs Revision | `pinn_stability_envelope/` |
| **Architecture Comparison** | NeurIPS 2025 | Needs Revision | `pinn_architecture/` |
| **Modular Design** | ICRA 2026 | Needs Revision | `pinn_modular_design/` |
| **Curriculum Training** | RA-L | Needs Revision | `pinn_curriculum/` |
| **Parameter Identification** | CDC/L4DC | Draft | `pinn_parameter_id/` |
| **ICML Submission** | ICML 2026 | Draft | `pinn_icml/` |
| **MLSys Submission** | MLSys 2026 | Draft | `pinn_mlsys/` |

## Directory Structure

```
papers/
├── gps_imu_detection/       # GPS-IMU Spoofing Detection (MAIN)
│   ├── paper/               # LaTeX source (dsn_submission.tex)
│   ├── docs/                # Methodology documentation
│   ├── README.md            # Paper overview
│   ├── CLAIMS.md            # Explicit claims with evidence
│   ├── FINAL_RESULTS.md     # Complete results
│   └── RESULTS_TABLE.md     # Results in table format
│
├── pinn_stability_envelope/ # ACC/CDC - Stability Envelope
│   └── ACC_CDC_submission.tex
│
├── pinn_architecture/       # NeurIPS - Architecture Comparison
│   └── NeurIPS_2025_submission.tex
│
├── pinn_modular_design/     # ICRA - Modular Design
│   └── ICRA_2026_submission.tex
│
├── pinn_curriculum/         # RA-L - Curriculum Training
│   └── RAL_submission.tex
│
├── pinn_parameter_id/       # CDC/L4DC - Parameter ID
│   └── new_novelty.tex
│
├── pinn_icml/               # ICML 2026
│   └── ICML_2026.tex
│
├── pinn_mlsys/              # MLSys 2026
│   └── MLSys_2026.tex
│
└── shared/                  # Shared resources
    ├── references.bib       # Common bibliography
    ├── fig_*.pdf            # Shared figures
    └── table_*.tex          # Shared tables
```

## Key Results Summary

### GPS-IMU Detection (Publication-Ready)

| Metric | Value |
|--------|-------|
| AUROC | 99.8% |
| FPR | 0.21% |
| Detection Rate | 100% @ 10x magnitude |
| Detectability Floor | ~5-10m offset |

### PINN Dynamics (Various Stages)

| Paper | Key Result |
|-------|------------|
| Stability Envelope | 4.6x better autoregressive stability |
| Architecture | Modular 1.11m vs Baseline 5.09m |
| Modular Design | 65% fewer parameters (72K vs 205K) |
| Curriculum | 25% improvement with curriculum training |

## Building Papers

Each paper directory contains LaTeX source files. To build:

```bash
cd papers/<paper_directory>
pdflatex <submission>.tex
bibtex <submission>
pdflatex <submission>.tex
pdflatex <submission>.tex
```

For shared references, use:
```latex
\bibliography{../shared/references}
```

## Source Code

- GPS-IMU Detection: `../gps_imu_detector/src/`
- PINN Framework: `../pinn_dynamics/`
