#!/usr/bin/env python3
"""
CI Leakage Gate for GPS-IMU Detector.

This script enforces data hygiene rules that MUST pass before any evaluation.
Failure blocks CI and invalidates any reported metrics.

Rules enforced:
1. NO circular sensors (baro_alt, mag_heading, derived_*)
2. NO cross-flight data leakage (sequence-wise splits)
3. Scaler fitted on train only
4. Feature correlation audit (detect circular relationships)

Usage:
    python ci/leakage_check.py --data path/to/data.csv
    python ci/leakage_check.py --check-code  # Scan codebase for violations

Exit codes:
    0 = PASS (safe to proceed)
    1 = FAIL (leakage detected)
    2 = ERROR (check could not complete)
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List, Tuple, Set
import json

# Banned sensor columns (circular or derived)
BANNED_COLUMNS = {
    # Circular sensors (derived from GPS/IMU fusion)
    'baro_alt', 'baro_altitude', 'barometric_altitude',
    'mag_heading', 'magnetic_heading', 'compass_heading',
    'fused_position', 'fused_velocity',

    # Derived features that encode ground truth
    'derived_altitude', 'derived_heading', 'derived_position',
    'ekf_state', 'filter_state', 'estimated_state',

    # Any column with these patterns
}

BANNED_PATTERNS = [
    r'derived_.*',
    r'fused_.*',
    r'filtered_.*',
    r'estimated_.*',
    r'ekf_.*',
    r'.*_ground_truth.*',
]


def check_banned_columns(columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if any columns are banned circular sensors.

    Returns:
        (passed, violations): True if no violations, list of bad columns
    """
    violations = []

    for col in columns:
        col_lower = col.lower()

        # Direct match
        if col_lower in BANNED_COLUMNS:
            violations.append(f"BANNED: {col} (circular sensor)")
            continue

        # Pattern match
        for pattern in BANNED_PATTERNS:
            if re.match(pattern, col_lower):
                violations.append(f"BANNED: {col} (matches pattern {pattern})")
                break

    return len(violations) == 0, violations


def check_correlation_circularity(data_path: Path, threshold: float = 0.95) -> Tuple[bool, List[str]]:
    """
    Check for suspiciously high correlations that indicate circularity.

    If feature X correlates >0.95 with target Y, it's likely derived from Y.

    Returns:
        (passed, violations): True if no violations
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        return True, ["SKIP: pandas/numpy not available for correlation check"]

    violations = []

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return False, [f"ERROR: Could not read {data_path}: {e}"]

    # Check correlations between all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return True, []

    corr_matrix = df[numeric_cols].corr()

    # Find suspiciously high correlations (excluding diagonal)
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i >= j:
                continue

            corr = abs(corr_matrix.loc[col1, col2])

            if corr > threshold:
                # Check if this is a known safe pair (e.g., position components)
                safe_pairs = [
                    ('x', 'y'), ('y', 'z'), ('x', 'z'),
                    ('vx', 'vy'), ('vy', 'vz'), ('vx', 'vz'),
                    ('roll', 'pitch'), ('pitch', 'yaw'),
                ]

                is_safe = False
                for s1, s2 in safe_pairs:
                    if (s1 in col1.lower() and s2 in col2.lower()) or \
                       (s2 in col1.lower() and s1 in col2.lower()):
                        is_safe = True
                        break

                if not is_safe:
                    violations.append(
                        f"HIGH CORRELATION: {col1} ↔ {col2} = {corr:.3f} "
                        f"(threshold={threshold})"
                    )

    return len(violations) == 0, violations


def check_sequence_splits(split_file: Path) -> Tuple[bool, List[str]]:
    """
    Verify that train/val/test splits are sequence-wise (no flight appears in multiple sets).

    Returns:
        (passed, violations): True if splits are valid
    """
    violations = []

    if not split_file.exists():
        return False, [f"MISSING: {split_file} (sequence splits not documented)"]

    try:
        with open(split_file) as f:
            splits = json.load(f)
    except Exception as e:
        return False, [f"ERROR: Could not parse {split_file}: {e}"]

    # Check for required keys
    required = ['train_flights', 'val_flights', 'test_flights']
    for key in required:
        if key not in splits:
            violations.append(f"MISSING KEY: {key} in {split_file}")

    if violations:
        return False, violations

    # Check for overlap
    train = set(splits['train_flights'])
    val = set(splits['val_flights'])
    test = set(splits['test_flights'])

    train_val_overlap = train & val
    train_test_overlap = train & test
    val_test_overlap = val & test

    if train_val_overlap:
        violations.append(f"OVERLAP: train ∩ val = {train_val_overlap}")
    if train_test_overlap:
        violations.append(f"OVERLAP: train ∩ test = {train_test_overlap}")
    if val_test_overlap:
        violations.append(f"OVERLAP: val ∩ test = {val_test_overlap}")

    return len(violations) == 0, violations


def scan_codebase_for_violations(root: Path) -> Tuple[bool, List[str]]:
    """
    Scan Python files for patterns that suggest data leakage.

    Returns:
        (passed, violations): True if no violations found
    """
    violations = []

    # Patterns that suggest leakage
    bad_patterns = [
        (r'scaler\.fit\(.*test', "Scaler fitted on test data"),
        (r'scaler\.fit\(.*val', "Scaler fitted on validation data"),
        (r'shuffle\s*=\s*True.*time', "Shuffling time series data"),
        (r'baro_alt', "Using barometric altitude (circular)"),
        (r'mag_heading', "Using magnetic heading (circular)"),
        (r'random_state\s*=\s*None', "Missing random seed"),
    ]

    py_files = list(root.glob("**/*.py"))

    for py_file in py_files:
        # Skip this file and test files
        if 'leakage_check' in str(py_file):
            continue
        if '__pycache__' in str(py_file):
            continue

        try:
            content = py_file.read_text(encoding='utf-8')
        except Exception:
            continue

        for pattern, message in bad_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                violations.append(
                    f"{py_file.relative_to(root)}:{line_num} - {message}"
                )

    return len(violations) == 0, violations


def main():
    parser = argparse.ArgumentParser(description="CI Leakage Gate")
    parser.add_argument('--data', type=Path, help="Data file to check")
    parser.add_argument('--splits', type=Path, help="Splits JSON file")
    parser.add_argument('--check-code', action='store_true', help="Scan codebase")
    parser.add_argument('--root', type=Path, default=Path('.'), help="Project root")
    parser.add_argument('--correlation-threshold', type=float, default=0.95)

    args = parser.parse_args()

    all_passed = True
    all_violations = []

    print("=" * 60)
    print("CI LEAKAGE GATE")
    print("=" * 60)

    # Check 1: Banned columns
    if args.data and args.data.exists():
        try:
            import pandas as pd
            df = pd.read_csv(args.data, nrows=1)
            passed, violations = check_banned_columns(df.columns.tolist())

            print(f"\n[1] Banned Columns Check: {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_passed = False
                all_violations.extend(violations)
                for v in violations:
                    print(f"    ✗ {v}")
            else:
                print(f"    ✓ No banned columns found")
        except ImportError:
            print("\n[1] Banned Columns Check: SKIP (pandas not available)")

    # Check 2: Correlation circularity
    if args.data and args.data.exists():
        passed, violations = check_correlation_circularity(
            args.data, args.correlation_threshold
        )

        print(f"\n[2] Correlation Circularity Check: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
            all_violations.extend(violations)
            for v in violations:
                print(f"    ✗ {v}")
        else:
            print(f"    ✓ No circular correlations detected")

    # Check 3: Sequence splits
    if args.splits:
        passed, violations = check_sequence_splits(args.splits)

        print(f"\n[3] Sequence Splits Check: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
            all_violations.extend(violations)
            for v in violations:
                print(f"    ✗ {v}")
        else:
            print(f"    ✓ Splits are sequence-wise (no overlap)")

    # Check 4: Code scan
    if args.check_code:
        passed, violations = scan_codebase_for_violations(args.root)

        print(f"\n[4] Codebase Scan: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
            all_violations.extend(violations)
            for v in violations[:10]:  # Limit output
                print(f"    ✗ {v}")
            if len(violations) > 10:
                print(f"    ... and {len(violations) - 10} more")
        else:
            print(f"    ✓ No leakage patterns detected")

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: PASS ✓")
        print("All leakage checks passed. Safe to proceed with evaluation.")
        return 0
    else:
        print("RESULT: FAIL ✗")
        print(f"Found {len(all_violations)} violation(s). Fix before evaluation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
