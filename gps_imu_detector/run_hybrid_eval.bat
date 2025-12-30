@echo off
REM One-Command Hybrid Detector Evaluation (Windows)
REM
REM Usage:
REM     run_hybrid_eval.bat
REM
REM Outputs:
REM     results\ekf_results.json
REM     results\hybrid_results.json
REM     results\worst_case_summary.csv
REM     results\bootstrap_ci.json

echo ============================================================
echo GPS-IMU HYBRID DETECTOR EVALUATION
echo ============================================================
echo Started at: %date% %time%
echo.

REM Navigate to detector directory
cd /d "%~dp0"

REM Create results directory
if not exist results mkdir results

REM ============================================================
REM STEP 1: Leakage Check
REM ============================================================
echo [1/6] Running leakage check...

python ci\leakage_check.py --splits configs\splits.json --check-code --root .

if %errorlevel% neq 0 (
    echo ERROR: Leakage check failed. Fix issues before evaluation.
    exit /b 1
)

echo     PASS: No leakage detected
echo.

REM ============================================================
REM STEP 2: EKF Detector
REM ============================================================
echo [2/6] Evaluating EKF-NIS detector...

python scripts\run_ekf_detector.py --output results\ekf_results.json --n-trajectories 10 --trajectory-length 2000

echo.

REM ============================================================
REM STEP 3: Hybrid Evaluation
REM ============================================================
echo [3/6] Evaluating hybrid detector...

python scripts\run_hybrid_eval.py --output results\hybrid_results.json --n-trajectories 5 --trajectory-length 1000

echo.

REM ============================================================
REM STEP 4: Worst-Case Table
REM ============================================================
echo [4/6] Generating worst-case table...

python scripts\generate_worst_case_table.py --input results\hybrid_results.json --output results\worst_case_summary.csv

echo.

REM ============================================================
REM STEP 5: Bootstrap CI
REM ============================================================
echo [5/6] Computing bootstrap confidence intervals...

python scripts\bootstrap_ci.py --n-bootstrap 500 --output results\bootstrap_ci.json --use-synthetic

echo.

REM ============================================================
REM STEP 6: Summary
REM ============================================================
echo [6/6] Generating summary...

echo ============================================================
echo EVALUATION COMPLETE
echo ============================================================
echo.
echo Output files:
echo     results\ekf_results.json
echo     results\hybrid_results.json
echo     results\worst_case_summary.csv
echo     results\bootstrap_ci.json
echo.
echo Completed at: %date% %time%
echo ============================================================
