# Repository Usage Guide

**This repository supports multiple independent research threads.**

Each track answers a **different question**. They should not be mixed in the same paper or results section.

---

## Track A: Detectability & Self-Healing (Main Contribution)

**Question:** What attacks are fundamentally detectable without labels or extra sensors?

### Scope
- Unsupervised / anomaly detection
- Single-modality (GPS + IMU only)
- Adversarial spoofing (attacker preserves consistency)
- Characterizing WHERE detection is possible, not claiming universal detection

### Key Directories
```
gps_imu_detector/
├── src/inverse_cycle/     # ICI detector
├── src/iasp/              # Self-healing
├── src/ekf/               # EKF-NIS baseline
├── results/               # Track A results ONLY
└── docs/
    ├── DETECTABILITY_FLOOR.md
    └── HYBRID_INTERPRETATION.md
```

### Key Results
| Metric | Value | Interpretation |
|--------|-------|----------------|
| ICI AUROC | 0.972 | Primary detection primitive |
| Worst-case Recall@5%FPR | 66% | **Detectability floor** (not a failure) |
| Self-healing reduction | 77% | Error correction when detected |
| Min detectable offset | 50m | Below this is fundamentally hard |

### Strong Claim
> Residual-based detectors cannot detect consistency-preserving spoofing. ICI breaks this barrier for offsets ≥25m.

---

## Track B: Fault Classification (Engineering)

**Question:** Given labels, how well can faults be classified in practice?

### Scope
- Supervised learning
- Known fault types with labeled training data
- Practical deployment for fault detection
- **NOT** adversarial security

### Key Directories
```
models/
├── padre_classifier/      # Random Forest classifier
├── padre_pinn/            # Physics-informed classifier
├── padre_crossdrone/      # Cross-drone transfer
└── fault_detection/       # CNN classifiers
```

### Key Results
| Model | Accuracy | Note |
|-------|----------|------|
| Random Forest | 99.97% | With labels |
| Cross-drone physics | 100% | Rule-based transfer |
| Binary CNN | 98.01% | Deep learning |

### Strong Claim
> Given labeled fault data, high-accuracy classification is achievable.

### Relationship to Track A
These results answer a **different question**. Supervised classifiers assume:
- Labeled training data exists
- Fault types are known a priori
- No adversarial attacker

Track A assumes none of these. **Do not compare numbers across tracks.**

---

## Track C: Motivation & Baselines (Supporting)

**Question:** Why do naïve physics or residual-based approaches fail, motivating ICI?

### Scope
- Evidence that simpler approaches don't work
- Ablation studies showing what matters
- Historical baseline comparisons
- **Motivation for Track A, not standalone contribution**

### Key Directories
```
models/security/
├── training_results_final.json    # Physics ablation (w=0 wins)
├── rigorous_evaluation/           # LOSO-CV baselines
├── sensor_fusion_*/               # Fusion attempts
└── ensemble_detector/             # Multi-detector attempts

research/security/
├── baselines/
└── residual_detectors/
```

### Key Results
| Finding | Evidence |
|---------|----------|
| Physics loss hurts detection | w=0 beats w=20 (p<0.001) |
| Residuals miss consistent spoofing | 0/5 offsets detected |
| Sensor fusion alone insufficient | High FPR or low recall |

### Usage
- Cite these in **Related Work** or **Motivation** sections
- Never present as standalone contribution
- Supports "why ICI was necessary"

---

## Track D: Latency & Deployment (Cross-Cutting)

**Question:** Is any of this deployable in real time?

### Scope
- Latency benchmarks
- Model size constraints
- Real-time feasibility

### Key Results
| Component | P95 Latency | Target |
|-----------|-------------|--------|
| EKF-NIS | 0.06 ms | < 5ms ✓ |
| ICI (ML) | 0.42 ms | < 5ms ✓ |
| Hybrid | 0.53 ms | < 5ms ✓ |

### Usage
- Include as **one table** in any paper
- Never leads a section
- Validates feasibility, doesn't define contribution

---

## Paper Mapping

### Paper 1: Detectability & Self-Healing (Main)

| Section | Track | Depth |
|---------|-------|-------|
| Introduction | A | Full |
| Related Work | C | 1 page |
| Method | A | Full |
| Results | A | Full |
| Latency | D | 1 table |
| Conclusion | A | Full |

**Track B mention:** One sentence maximum.
> "Complementary supervised fault classifiers (e.g., PADRE) address a different problem setting and are not considered here."

### Paper 2: Fault Classification (Engineering)

| Section | Track | Depth |
|---------|-------|-------|
| Introduction | B | Full |
| Related Work | C | 1 page |
| Method | B | Full |
| Results | B | Full |
| Latency | D | 1 table |
| Future Work | A | 1 paragraph |

**Track A mention:** Future direction only.
> "Extending to adversarial settings without labels remains future work."

---

## Scope Guard (Use in Papers)

Include this paragraph in any Track A paper:

> This work focuses on detectability limits under unlabeled, adversarial spoofing. Supervised fault classification and engineered security pipelines, while effective in constrained settings, address a different problem and are intentionally excluded.

---

## The Litmus Test

Before including anything, ask:

> "If I remove this from this paper, does the main claim break?"

- **Yes** → Keep
- **No** → Move to appendix or another paper

---

## Why This Structure Exists

What looks like "mess" is actually deep exploration collapsed into paths.

Each track:
- **Track A** → defines limits (theory + systems)
- **Track B** → shows engineering strength elsewhere
- **Track C** → explains why Track A was necessary
- **Track D** → proves feasibility everywhere

Nothing is wasted. Nothing competes. Nothing looks accidental.
