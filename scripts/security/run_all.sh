#!/usr/bin/env bash
#
# Complete Fault Detection Pipeline Automation
# =============================================
#
# Reproduces all experimental results from ACSAC 2025 submission in ~2 hours.
#
# Usage:
#   bash scripts/security/run_all.sh
#
# Or with custom settings:
#   bash scripts/security/run_all.sh --seeds 5 --device cpu
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default parameters
NUM_SEEDS=20
EPOCHS=500
DEVICE="cuda"
PHYSICS_WEIGHT=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --physics_weight)
            PHYSICS_WEIGHT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}UAV FAULT DETECTION - FULL PIPELINE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Seeds: $NUM_SEEDS"
echo "  Epochs: $EPOCHS"
echo "  Device: $DEVICE"
echo "  Physics Weight: $PHYSICS_WEIGHT"
echo ""

# Step 1: Download and preprocess dataset
echo -e "${YELLOW}[1/8] Downloading CMU ALFA dataset...${NC}"
python scripts/security/preprocess_alfa.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dataset ready${NC}"
else
    echo -e "${RED}✗ Dataset download failed${NC}"
    exit 1
fi
echo ""

# Step 2: Train detector
echo -e "${YELLOW}[2/8] Training fault detector ($NUM_SEEDS seeds × $EPOCHS epochs)...${NC}"
echo "Estimated time: ~54 minutes (on GPU)"
python scripts/security/train_detector.py \
    --physics_weight $PHYSICS_WEIGHT \
    --num_seeds $NUM_SEEDS \
    --epochs $EPOCHS \
    --hidden_size 256 \
    --num_layers 5 \
    --dropout 0.1 \
    --lr 0.001 \
    --batch_size 32 \
    --device $DEVICE
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training complete${NC}"
else
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi
echo ""

# Step 3: Evaluate on test flights
echo -e "${YELLOW}[3/8] Evaluating on test flights...${NC}"
python scripts/security/evaluate_detector.py \
    --model_path research/security/models/detector_w${PHYSICS_WEIGHT}_seed0.pth \
    --threshold 0.1707
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Evaluation complete${NC}"
else
    echo -e "${RED}✗ Evaluation failed${NC}"
    exit 1
fi
echo ""

# Step 4: Evaluate baselines
echo -e "${YELLOW}[4/8] Evaluating baseline methods...${NC}"
python scripts/security/evaluate_baselines.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Baselines complete${NC}"
else
    echo -e "${RED}✗ Baseline evaluation failed${NC}"
    exit 1
fi
echo ""

# Step 5: Tune threshold
echo -e "${YELLOW}[5/8] Tuning detection threshold...${NC}"
python scripts/security/tune_threshold.py \
    --model_path research/security/models/detector_w${PHYSICS_WEIGHT}_seed0.pth \
    --min_threshold 0.0 \
    --max_threshold 1.0 \
    --num_thresholds 100
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Threshold tuning complete${NC}"
else
    echo -e "${RED}✗ Threshold tuning failed${NC}"
    exit 1
fi
echo ""

# Step 6: Measure computational cost
echo -e "${YELLOW}[6/8] Measuring computational cost...${NC}"
python scripts/security/measure_computational_cost.py \
    --model_path research/security/models/detector_w${PHYSICS_WEIGHT}_seed0.pth \
    --num_trials 1000
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Computational analysis complete${NC}"
else
    echo -e "${RED}✗ Computational analysis failed${NC}"
    exit 1
fi
echo ""

# Step 7: Generate supplementary figures
echo -e "${YELLOW}[7/8] Generating all figures...${NC}"
python scripts/security/create_supplementary_figures.py
python scripts/security/create_architecture_diagram.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Figures generated${NC}"
else
    echo -e "${RED}✗ Figure generation failed${NC}"
    exit 1
fi
echo ""

# Step 8: Verify results
echo -e "${YELLOW}[8/8] Verifying results...${NC}"

# Check that all expected files exist
MISSING=0

# Check models
if [ ! -f "research/security/models/detector_w${PHYSICS_WEIGHT}_seed0.pth" ]; then
    echo -e "${RED}✗ Missing: detector model${NC}"
    MISSING=1
fi

# Check results
if [ ! -f "research/security/results_optimized/aggregated_results.json" ]; then
    echo -e "${RED}✗ Missing: aggregated results${NC}"
    MISSING=1
fi

# Check baselines
if [ ! -f "research/security/baselines/svm_results.json" ]; then
    echo -e "${RED}✗ Missing: baseline results${NC}"
    MISSING=1
fi

# Check figures
FIG_COUNT=$(ls research/security/figures/*.png 2>/dev/null | wc -l)
if [ $FIG_COUNT -lt 11 ]; then
    echo -e "${YELLOW}⚠ Only $FIG_COUNT/11 figures generated${NC}"
fi

if [ $MISSING -eq 0 ]; then
    echo -e "${GREEN}✓ All files present${NC}"
fi
echo ""

# Display summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PIPELINE COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to:"
echo "  - Models: research/security/models/"
echo "  - Results: research/security/results_optimized/"
echo "  - Baselines: research/security/baselines/"
echo "  - Figures: research/security/figures/"
echo ""
echo "Key metrics (check research/security/results_optimized/aggregated_results.json):"
cat research/security/results_optimized/aggregated_results.json 2>/dev/null | grep -E '"mean_f1"|"mean_fpr"|"mean_precision"' || echo "  (Run evaluation first)"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review results in research/security/"
echo "  2. Compile paper with: see research/security/COMPILE_NOW.md"
echo "  3. Upload to Overleaf: research/security/paper_submission.zip"
echo ""
echo -e "${GREEN}Total runtime: ~2 hours (varies by hardware)${NC}"
