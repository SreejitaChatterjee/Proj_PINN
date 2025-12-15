# Project Direction

## What This Is
A **PINN framework** for learning dynamics from data with physics constraints.
Currently implements quadrotor dynamics as the reference application.

## Key Research Finding
Physics loss doesn't improve (and may hurt) autoregressive rollout stability.
Training regime and architecture matter more than physics constraints.
See `paper_versions/ACC_CDC_submission.tex` for the full analysis.

## Industry Focus
- Clean API for defining custom dynamics systems
- Real data integration (beyond simulation)
- Demo-first development (`python demo.py` just works)

## Current State
- Working quadrotor PINN with trained model
- 100-step rollout demo functional
- Research paper in progress

## Next Steps
1. Integrate real quadrotor dataset (Blackbird/EuRoC)
2. Generalize framework API beyond quadrotor
3. Add more example systems (pendulum, cartpole, etc.)
