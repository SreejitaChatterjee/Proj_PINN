#!/bin/bash
# Verification script for LaTeX report fixes
# Run this after compiling the PDF to ensure all inconsistencies are resolved

echo "========================================="
echo "VERIFYING LATEX REPORT FIXES"
echo "========================================="
echo ""

cd "$(dirname "$0")"

# Counter for issues found
ISSUES=0

echo "[1] Checking for removed content (should have 0 results)..."
echo "----------------------------------------"

# Check for 18×1 references (should only be 16×1 now)
COUNT=$(grep -c "18×1" quadrotor_pinn_report.tex 2>/dev/null || echo "0")
if [ "$COUNT" -ne 0 ]; then
    echo "❌ FAIL: Found $COUNT reference(s) to '18×1' output dimension"
    ISSUES=$((ISSUES + COUNT))
else
    echo "✅ PASS: No '18×1' references found (correct: 16×1)"
fi

# Check for "6 critical physical parameters" or "6 physical parameters"
# Use more precise pattern to avoid false positives
COUNT=$(grep -c "6 \(critical \)\?physical parameter" quadrotor_pinn_report.tex 2>/dev/null || echo "0")
if [ "$COUNT" -ne 0 ]; then
    echo "❌ FAIL: Found $COUNT reference(s) to '6 physical parameters'"
    ISSUES=$((ISSUES + COUNT))
else
    echo "✅ PASS: No '6 physical parameters' references found"
fi

# Check for "remarkable accuracy" (should be removed)
COUNT=$(grep -c "remarkable accuracy" quadrotor_pinn_report.tex 2>/dev/null || echo "0")
if [ "$COUNT" -ne 0 ]; then
    echo "❌ FAIL: Found $COUNT reference(s) to 'remarkable accuracy'"
    ISSUES=$((ISSUES + COUNT))
else
    echo "✅ PASS: No 'remarkable accuracy' language found"
fi

# Check for "perfect convergence"
COUNT=$(grep -c "perfect convergence" quadrotor_pinn_report.tex 2>/dev/null || echo "0")
if [ "$COUNT" -ne 0 ]; then
    echo "❌ FAIL: Found $COUNT reference(s) to 'perfect convergence'"
    ISSUES=$((ISSUES + COUNT))
else
    echo "✅ PASS: No 'perfect convergence' language found"
fi

echo ""
echo "[2] Verifying correct content exists..."
echo "----------------------------------------"

# Check for 16×1 output (should exist)
COUNT=$(grep -c "16×1" quadrotor_pinn_report.tex 2>/dev/null || echo "0")
if [ "$COUNT" -eq 0 ]; then
    echo "❌ FAIL: '16×1' output dimension not found"
    ISSUES=$((ISSUES + 1))
else
    echo "✅ PASS: Found $COUNT reference(s) to correct '16×1' dimension"
fi

# Check for "4 critical physical parameters" or "4 physical parameters"
# Use more precise pattern to avoid false positives from decimals
COUNT=$(grep -c "4 \(critical \)\?physical parameter" quadrotor_pinn_report.tex 2>/dev/null || echo "0")
if [ "$COUNT" -eq 0 ]; then
    echo "❌ FAIL: '4 physical parameters' not found"
    ISSUES=$((ISSUES + 1))
else
    echo "✅ PASS: Found $COUNT reference(s) to '4 physical parameters'"
fi

# Check for 4-7% error range mention
COUNT=$(grep -c "4-7.*error\|4\.4-7\.3.*error" quadrotor_pinn_report.tex 2>/dev/null || echo "0")
if [ "$COUNT" -eq 0 ]; then
    echo "❌ FAIL: Error range '4-7%' or '4.4-7.3%' not found"
    ISSUES=$((ISSUES + 1))
else
    echo "✅ PASS: Found $COUNT reference(s) to realistic error ranges"
fi

# Check for motor coefficient explanation
COUNT=$(grep -c "Note on Motor Coefficients" quadrotor_pinn_report.tex 2>/dev/null || echo "0")
if [ "$COUNT" -eq 0 ]; then
    echo "⚠️  WARNING: Motor coefficient explanation note not found"
else
    echo "✅ PASS: Motor coefficient explanation found"
fi

# Check for thrust modulation explanation
COUNT=$(grep -c "PID.*controller.*modulates.*thrust\|controller.*modulates.*thrust" quadrotor_pinn_report.tex 2>/dev/null || echo "0")
if [ "$COUNT" -eq 0 ]; then
    echo "⚠️  WARNING: Thrust modulation explanation not found"
else
    echo "✅ PASS: Thrust modulation explanation found"
fi

echo ""
echo "[3] Checking specific table values..."
echo "----------------------------------------"

# Check total parameters (should be 53,380 not 53,526)
if grep -q "53,526" quadrotor_pinn_report.tex; then
    echo "❌ FAIL: Old total parameter count (53,526) still exists"
    ISSUES=$((ISSUES + 1))
else
    echo "✅ PASS: Old parameter count (53,526) removed"
fi

if grep -q "53,380" quadrotor_pinn_report.tex; then
    echo "✅ PASS: Correct total parameter count (53,380) found"
else
    echo "❌ FAIL: New total parameter count (53,380) not found"
    ISSUES=$((ISSUES + 1))
fi

# Check output layer parameters (should be 2,176 not 2,322)
if grep -q "2,322" quadrotor_pinn_report.tex; then
    echo "❌ FAIL: Old output layer parameter count (2,322) still exists"
    ISSUES=$((ISSUES + 1))
else
    echo "✅ PASS: Old output parameter count (2,322) removed"
fi

if grep -q "2,176" quadrotor_pinn_report.tex; then
    echo "✅ PASS: Correct output layer count (2,176) found"
else
    echo "❌ FAIL: New output parameter count (2,176) not found"
    ISSUES=$((ISSUES + 1))
fi

echo ""
echo "========================================="
echo "VERIFICATION SUMMARY"
echo "========================================="
if [ "$ISSUES" -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED! Report is consistent."
    echo ""
    echo "Next step: Compile the PDF"
    echo "  cd reports"
    echo "  pdflatex quadrotor_pinn_report.tex"
    echo "  pdflatex quadrotor_pinn_report.tex  # Run twice"
    exit 0
else
    echo "❌ FOUND $ISSUES ISSUE(S)"
    echo ""
    echo "Please review the failed checks above."
    exit 1
fi
