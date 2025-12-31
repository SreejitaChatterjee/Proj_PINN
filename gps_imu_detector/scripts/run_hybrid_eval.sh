#!/bin/bash
#
# One-Command Hybrid Detector Evaluation
#
# This script runs the complete evaluation pipeline:
# 1. Leakage check (CI gate)
# 2. EKF detector evaluation
# 3. ML (ICI) detector evaluation
# 4. Hybrid fusion evaluation
# 5. Worst-case table generation
# 6. Bootstrap confidence intervals
#
# Usage:
#     chmod +x run_hybrid_eval.sh
#     ./run_hybrid_eval.sh
#
# Expected runtime: ~5 minutes
#
# Outputs:
#     results/ekf_results.json
#     results/hybrid_results.json
#     results/worst_case_summary.csv
#     results/bootstrap_ci.json
#

set -e  # Exit on first error

echo "============================================================"
echo "GPS-IMU HYBRID DETECTOR EVALUATION"
echo "============================================================"
echo "Started at: $(date)"
echo ""

# Navigate to detector directory
cd "$(dirname "$0")"

# Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi

# Create results directory
mkdir -p results

# ============================================================
# STEP 1: Leakage Check
# ============================================================
echo "[1/6] Running leakage check..."

python ci/leakage_check.py \
    --splits configs/splits.json \
    --check-code \
    --root .

if [ $? -ne 0 ]; then
    echo "ERROR: Leakage check failed. Fix issues before evaluation."
    exit 1
fi

echo "    PASS: No leakage detected"
echo ""

# ============================================================
# STEP 2: EKF Detector
# ============================================================
echo "[2/6] Evaluating EKF-NIS detector..."

python scripts/run_ekf_detector.py \
    --output results/ekf_results.json \
    --n-trajectories 10 \
    --trajectory-length 2000

echo ""

# ============================================================
# STEP 3: Hybrid Evaluation
# ============================================================
echo "[3/6] Evaluating hybrid detector..."

python scripts/run_hybrid_eval.py \
    --output results/hybrid_results.json \
    --n-trajectories 5 \
    --trajectory-length 1000

echo ""

# ============================================================
# STEP 4: Worst-Case Table
# ============================================================
echo "[4/6] Generating worst-case table..."

python scripts/generate_worst_case_table.py \
    --input results/hybrid_results.json \
    --output results/worst_case_summary.csv

echo ""

# ============================================================
# STEP 5: Bootstrap CI
# ============================================================
echo "[5/6] Computing bootstrap confidence intervals..."

python scripts/bootstrap_ci.py \
    --n-bootstrap 500 \
    --output results/bootstrap_ci.json \
    --use-synthetic

echo ""

# ============================================================
# STEP 6: Summary
# ============================================================
echo "[6/6] Generating summary..."

echo "============================================================"
echo "EVALUATION COMPLETE"
echo "============================================================"
echo ""
echo "Output files:"
echo "    results/ekf_results.json"
echo "    results/hybrid_results.json"
echo "    results/worst_case_summary.csv"
echo "    results/bootstrap_ci.json"
echo ""
echo "Completed at: $(date)"
echo "============================================================"

# Print key results
if [ -f results/hybrid_results.json ]; then
    echo ""
    echo "KEY RESULTS:"
    python -c "
import json
with open('results/hybrid_results.json') as f:
    r = json.load(f)
print(f\"  EKF AUROC:    {r['ekf']['auroc']:.4f}\")
print(f\"  ML AUROC:     {r['ml']['auroc']:.4f}\")
print(f\"  Hybrid AUROC: {r['hybrid']['auroc']:.4f}\")
"
fi
