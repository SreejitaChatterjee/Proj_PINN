# Repository Structure Management Plan

## Current Issues

### 1. **Duplicate Visualizations** (CRITICAL)
Same PNG files exist in 3 locations:
- `results/detailed/` (9 files)
- `visualizations/detailed/` (9 files)
- `scripts/results/` (9 files)

**Impact:** Wasting Git LFS/storage, confusing structure

---

### 2. **Scattered Documentation**
Documentation spread across:
- Root: `anomaly_report.md`, `physics_deviations_report.md`, `physics_fix_comparison.md`, etc.
- `/docs/`: 7 markdown files
- `/reports/documentation/`: 6 markdown files

**Impact:** Hard to find relevant docs

---

### 3. **Temporary/Build Artifacts**
Should not be in repo:
- `nul` (empty file)
- `training_log.txt`, `training_output.txt`, `training_output_full.txt`
- LaTeX auxiliary files: `.aux`, `.idx`, `.log`, `.out`, `.toc`

**Impact:** Clutters repo history

---

### 4. **Redundant Models**
- `models/enhanced_pinn_realistic.pth` (150KB) - OLD
- `models/quadrotor_pinn.pth` (147KB) - CURRENT

**Impact:** Confusion about which model to use

---

### 5. **Empty/Unused Directories**
- `figures/` (empty)
- `figures/anomaly_validation/` (empty)

---

### 6. **Inconsistent Naming**
- Mix of `results/` vs `visualizations/`
- Root-level docs vs organized docs

---

## Recommended Structure

```
Proj_PINN/
├── README.md                          # Main project readme
├── CHANGELOG.md                       # Keep version history
├── PROJECT_SUMMARY.md                 # High-level summary
│
├── data/                              # Training/test data
│   ├── quadrotor_training_data.csv
│   └── aggressive_test_trajectories.pkl
│
├── scripts/                           # All Python code
│   ├── train.py
│   ├── evaluate.py
│   ├── generate_quadrotor_data.py
│   ├── pinn_model.py
│   ├── plot_utils.py
│   └── README.md
│
├── models/                            # Saved models ONLY
│   ├── quadrotor_pinn.pth            # Current best model
│   └── scalers.pkl
│
├── results/                           # ALL evaluation outputs
│   ├── summary.png
│   └── detailed/
│       ├── 01_z_time_analysis.png
│       ├── 02_phi_time_analysis.png
│       └── ... (8 files total)
│
├── docs/                              # ALL documentation
│   ├── physics/
│   │   ├── physics_deviations_report.md
│   │   └── physics_fix_comparison.md
│   ├── anomalies/
│   │   ├── anomaly_report.md
│   │   └── anomaly_verification.txt
│   ├── progress/
│   │   ├── CONVERGENCE_ANALYSIS_SUMMARY.md
│   │   ├── IMPROVEMENT_SUMMARY.md
│   │   ├── PROGRESS_SUMMARY.md
│   │   └── FINAL_RESULTS_SUMMARY.md
│   └── archive/
│       └── (old/deprecated docs)
│
├── reports/                           # LaTeX report ONLY
│   ├── quadrotor_pinn_report.tex
│   ├── quadrotor_pinn_report.pdf
│   └── figures/                       # Figures for LaTeX
│
└── matlab_reference.m                 # Reference implementation

REMOVE:
├── visualizations/                    # DELETE (duplicate)
├── scripts/results/                   # DELETE (duplicate)
├── figures/ (empty)                   # DELETE
├── nul                                # DELETE
├── training_*.txt                     # DELETE (temp files)
├── models/enhanced_pinn_realistic.pth # DELETE (old model)
└── reports/*.aux, *.idx, *.log, etc.  # DELETE (LaTeX artifacts)
```

---

## Cleanup Actions

### Phase 1: Remove Duplicates and Temp Files
1. ✓ Keep `results/` as the canonical visualization directory
2. ✗ Delete `visualizations/` (duplicate)
3. ✗ Delete `scripts/results/` (duplicate)
4. ✗ Delete temp files: `nul`, `training_*.txt`
5. ✗ Delete LaTeX build artifacts
6. ✗ Delete empty `figures/` directory
7. ✗ Delete old model: `enhanced_pinn_realistic.pth`

### Phase 2: Organize Documentation
1. Create `docs/` subdirectories (physics, anomalies, progress, archive)
2. Move root-level docs to appropriate subdirs
3. Consolidate `/reports/documentation/` into `/docs/`

### Phase 3: Update .gitignore
Add patterns to prevent future clutter:
```
# Temporary files
nul
*.tmp
training_output*.txt

# LaTeX build artifacts
*.aux
*.idx
*.log
*.out
*.toc
*.fdb_latexmk
*.fls
*.synctex.gz

# Python
__pycache__/
*.pyc

# Keep only current results
/visualizations/
/scripts/results/
/models/enhanced_*.pth
```

### Phase 4: Update Documentation
- Update README.md with new structure
- Add CONTRIBUTING.md with file organization guidelines
- Update script paths if needed

---

## Impact Summary

**Storage saved:**
- ~8MB from duplicate PNGs
- ~150KB from old models
- Cleaner git history

**Benefits:**
- Clear separation: code, data, models, results, docs
- Single source of truth for visualizations
- Easy to find documentation
- Professional structure for collaboration/publication

---

## Execution Order

1. **Backup first**: Ensure everything is committed
2. **Remove duplicates**: Delete redundant visualization folders
3. **Remove temp files**: Clean up build artifacts
4. **Organize docs**: Create structure and move files
5. **Update .gitignore**: Prevent future clutter
6. **Test**: Verify scripts still work
7. **Commit**: "Reorganize repository structure for clarity"
8. **Update README**: Document new structure

