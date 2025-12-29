#!/bin/bash
# CI Leakage Check Script
# Runs correlation test and feature audit for CI pipeline
#
# Usage:
#   ./ci/leakage_check.sh [data_path]
#
# Exit codes:
#   0: PASS - No leakage detected
#   1: FAIL - Leakage or circular sensors detected
#   2: ERROR - Script error

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "CI LEAKAGE CHECK"
echo "=============================================="
echo ""

# Default data path
DATA_PATH="${1:-$PROJECT_ROOT/data}"

FAILED=0

# ---------------------------------------------
# Check 1: Feature Audit (grep for banned patterns)
# ---------------------------------------------
echo -e "${YELLOW}[1/3] Feature Audit - Checking for banned sensor patterns...${NC}"

BANNED_PATTERNS=(
    "baro_alt"
    "barometer"
    "mag_heading"
    "magnetometer"
    "derived_"
    "synthetic_"
    "gt_"
)

for pattern in "${BANNED_PATTERNS[@]}"; do
    # Check in src/ files
    if grep -r --include="*.py" "$pattern" "$PROJECT_ROOT/src/" 2>/dev/null | grep -v "BANNED\|#\|test\|check" | head -1; then
        echo -e "${RED}FAIL: Found banned pattern '$pattern' in source code${NC}"
        FAILED=1
    fi
done

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}PASS: No banned patterns in source code${NC}"
fi

# ---------------------------------------------
# Check 2: Correlation Test (Python script)
# ---------------------------------------------
echo ""
echo -e "${YELLOW}[2/3] Correlation Test - Running circular sensor check...${NC}"

if [ -d "$DATA_PATH" ] || [ -f "$DATA_PATH" ]; then
    python "$PROJECT_ROOT/scripts/ci_circular_check.py" --data "$DATA_PATH" --threshold 0.9
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Circular sensor correlation detected${NC}"
        FAILED=1
    else
        echo -e "${GREEN}PASS: No circular sensors detected${NC}"
    fi
else
    echo -e "${YELLOW}SKIP: No data found at $DATA_PATH${NC}"
fi

# ---------------------------------------------
# Check 3: Run Leakage Tests (pytest)
# ---------------------------------------------
echo ""
echo -e "${YELLOW}[3/3] Leakage Tests - Running pytest...${NC}"

cd "$PROJECT_ROOT"
python -m pytest tests/test_leakage.py -v --tb=short
if [ $? -ne 0 ]; then
    echo -e "${RED}FAIL: Leakage tests failed${NC}"
    FAILED=1
else
    echo -e "${GREEN}PASS: All leakage tests passed${NC}"
fi

# ---------------------------------------------
# Summary
# ---------------------------------------------
echo ""
echo "=============================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}*** CI LEAKAGE CHECK: PASSED ***${NC}"
    exit 0
else
    echo -e "${RED}*** CI LEAKAGE CHECK: FAILED ***${NC}"
    echo "Fix leakage issues before merging."
    exit 1
fi
