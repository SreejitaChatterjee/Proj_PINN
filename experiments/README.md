# Experiments

All experimental artifacts, trained models, and research results.

## Directory Structure

```
experiments/
├── models/           # Trained model checkpoints
│   ├── ablation_study/
│   ├── architecture_comparison/
│   ├── padre_*/      # PADRE classifier models
│   ├── security/     # Security-related models
│   └── weight_sweep*/
│
├── results/          # Experiment outputs and figures
│   ├── ablation_study/
│   ├── architecture_comparison/
│   ├── ieee_publication_plots/
│   └── ...
│
├── research/         # Research artifacts
│   ├── ablation/     # Ablation studies
│   ├── security/     # Security research (Track C)
│   └── paper/        # LaTeX sources
│
├── reports/          # Generated reports
│
└── runs/             # Training run logs
```

## Key Results

| Experiment | Location | Key Finding |
|------------|----------|-------------|
| Weight Sweep | `results/weight_sweep/` | w=0 outperforms w=20 (p=0.024) |
| Architecture | `results/architecture_comparison/` | Modular 4.6x better |
| Security | `research/security/` | ICI detectability boundary |
