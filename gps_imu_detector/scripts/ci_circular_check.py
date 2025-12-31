#!/usr/bin/env python3
"""
CI Gate: Circular Sensor Detection

This script FAILS (exit code 1) if any circular sensor derivation is detected.
Run this in CI to ensure no data leakage through derived sensors.

Usage:
    python scripts/ci_circular_check.py --data path/to/data.csv

Exit codes:
    0: PASS - No circular sensors detected
    1: FAIL - Circular sensors detected (blocks CI)
    2: ERROR - Script error
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import json


# Correlation threshold - FAIL if exceeded
CORRELATION_THRESHOLD = 0.9

# Sensor pairs to check for circularity
CIRCULAR_PAIRS = [
    # (derived, source, relationship)
    ('vx', 'x', 'velocity from position'),
    ('vy', 'y', 'velocity from position'),
    ('vz', 'z', 'velocity from position'),
    ('ax', 'vx', 'acceleration from velocity'),
    ('ay', 'vy', 'acceleration from velocity'),
    ('az', 'vz', 'acceleration from velocity'),
    ('baro_alt', 'z', 'barometer from ground truth altitude'),
    ('mag_heading', 'yaw', 'magnetometer from ground truth heading'),
]

# Banned columns - these should NEVER appear in validated data
BANNED_COLUMNS = [
    'baro_alt', 'barometer', 'baro',
    'mag_heading', 'magnetometer', 'mag',
    'derived_', 'synthetic_', 'gt_'
]


class CircularSensorChecker:
    """Check for circular sensor derivations in dataset."""

    def __init__(self, threshold: float = CORRELATION_THRESHOLD):
        self.threshold = threshold
        self.violations: List[Dict] = []
        self.warnings: List[str] = []

    def check_dataframe(self, df: pd.DataFrame, dt: float = 0.005) -> bool:
        """
        Check DataFrame for circular sensors.

        Returns:
            True if PASS (no circular sensors), False if FAIL
        """
        self.violations = []
        self.warnings = []

        # Check 1: Banned columns
        for col in df.columns:
            for banned in BANNED_COLUMNS:
                if banned in col.lower():
                    self.violations.append({
                        'type': 'banned_column',
                        'column': col,
                        'reason': f'Column contains banned pattern: {banned}'
                    })

        # Check 2: Derivative correlations
        for derived, source, desc in CIRCULAR_PAIRS:
            if derived in df.columns and source in df.columns:
                corr = self._check_derivative_correlation(
                    df[source].values, df[derived].values, dt
                )
                if corr is not None and abs(corr) > self.threshold:
                    self.violations.append({
                        'type': 'derivative_correlation',
                        'derived': derived,
                        'source': source,
                        'correlation': float(corr),
                        'threshold': self.threshold,
                        'reason': f'{desc}: correlation {corr:.4f} > {self.threshold}'
                    })

        # Check 3: Direct high correlations between sensor groups
        self._check_cross_group_correlations(df)

        return len(self.violations) == 0

    def _check_derivative_correlation(
        self,
        source: np.ndarray,
        derived: np.ndarray,
        dt: float
    ) -> float:
        """Check if derived signal correlates with derivative of source."""
        if len(source) < 10:
            return None

        # Compute numerical derivative
        source_deriv = np.diff(source) / dt
        derived_subset = derived[:-1]

        # Handle NaN/Inf
        mask = np.isfinite(source_deriv) & np.isfinite(derived_subset)
        if mask.sum() < 10:
            return None

        corr = np.corrcoef(source_deriv[mask], derived_subset[mask])[0, 1]
        return corr if np.isfinite(corr) else None

    def _check_cross_group_correlations(self, df: pd.DataFrame):
        """Check for suspiciously high correlations between groups."""
        groups = {
            'position': ['x', 'y', 'z'],
            'velocity': ['vx', 'vy', 'vz'],
            'attitude': ['roll', 'pitch', 'yaw', 'phi', 'theta', 'psi'],
            'rates': ['p', 'q', 'r'],
            'accel': ['ax', 'ay', 'az']
        }

        # Check if any auxiliary sensor perfectly correlates with primary
        for aux_col in df.columns:
            if any(aux_col.startswith(b) for b in ['baro', 'mag', 'derived']):
                for group_name, group_cols in groups.items():
                    for primary_col in group_cols:
                        if primary_col in df.columns:
                            try:
                                corr = df[aux_col].corr(df[primary_col])
                                if abs(corr) > self.threshold:
                                    self.violations.append({
                                        'type': 'auxiliary_correlation',
                                        'auxiliary': aux_col,
                                        'primary': primary_col,
                                        'correlation': float(corr),
                                        'reason': f'Auxiliary {aux_col} correlates with {primary_col}'
                                    })
                            except:
                                pass

    def get_report(self) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append("=" * 60)
        lines.append("CIRCULAR SENSOR CHECK REPORT")
        lines.append("=" * 60)

        if self.violations:
            lines.append(f"\nFAILED: {len(self.violations)} violation(s) detected\n")
            for i, v in enumerate(self.violations, 1):
                lines.append(f"[{i}] {v['type'].upper()}")
                lines.append(f"    Reason: {v['reason']}")
                if 'correlation' in v:
                    lines.append(f"    Correlation: {v['correlation']:.4f}")
                lines.append("")
        else:
            lines.append("\nPASSED: No circular sensors detected")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_json_report(self) -> str:
        """Generate JSON report for CI integration."""
        return json.dumps({
            'passed': len(self.violations) == 0,
            'violations': self.violations,
            'warnings': self.warnings,
            'threshold': self.threshold
        }, indent=2)


def check_file(file_path: str, dt: float = 0.005) -> bool:
    """Check a single file for circular sensors."""
    print(f"\nChecking: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"ERROR: Could not read file: {e}")
        return False

    checker = CircularSensorChecker()
    passed = checker.check_dataframe(df, dt)

    print(checker.get_report())

    return passed


def check_directory(dir_path: str, dt: float = 0.005) -> bool:
    """Check all CSV files in directory."""
    path = Path(dir_path)
    csv_files = list(path.glob("**/*.csv"))

    if not csv_files:
        print(f"WARNING: No CSV files found in {dir_path}")
        return True

    all_passed = True
    for csv_file in csv_files:
        if not check_file(str(csv_file), dt):
            all_passed = False

    return all_passed


def check_source_code(src_dir: str = 'src') -> bool:
    """Check source code for banned sensor patterns."""
    import re

    banned_patterns = [
        r'\bbaro_alt\b',
        r'\bmag_heading\b',
        r'\bderived_\w+',
    ]

    path = Path(src_dir)
    if not path.exists():
        print(f"Source directory not found: {src_dir}")
        return True  # Pass if no src dir

    violations = []
    for py_file in path.glob('**/*.py'):
        content = py_file.read_text(encoding='utf-8', errors='ignore')
        for pattern in banned_patterns:
            matches = re.findall(pattern, content)
            if matches:
                # Skip if in comment or string containing "banned" or "EXCLUDED"
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        if 'banned' in line.lower() or 'EXCLUDED' in line or line.strip().startswith('#'):
                            continue
                        violations.append(f"{py_file}:{i}: {line.strip()[:80]}")

    if violations:
        print("=" * 60)
        print("SOURCE CODE CHECK: FAILED")
        print("=" * 60)
        print(f"\nFound {len(violations)} banned pattern(s):\n")
        for v in violations[:20]:  # Limit output
            print(f"  {v}")
        if len(violations) > 20:
            print(f"  ... and {len(violations) - 20} more")
        return False
    else:
        print("=" * 60)
        print("SOURCE CODE CHECK: PASSED")
        print("=" * 60)
        print("No banned sensor patterns found in source code.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='CI Gate: Check for circular sensor derivations'
    )
    parser.add_argument(
        '--data', type=str, required=False,
        help='Path to data file or directory'
    )
    parser.add_argument(
        '--dt', type=float, default=0.005,
        help='Sampling period in seconds (default: 0.005 = 200Hz)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.9,
        help='Correlation threshold for failure (default: 0.9)'
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Output JSON report'
    )
    parser.add_argument(
        '--source-only', action='store_true',
        help='Only check source code for banned patterns (no data needed)'
    )

    args = parser.parse_args()

    # Source-only mode: just check source code
    if args.source_only:
        passed = check_source_code('src')
        sys.exit(0 if passed else 1)

    # Data mode requires --data argument
    if not args.data:
        parser.error("--data is required unless using --source-only")

    global CORRELATION_THRESHOLD
    CORRELATION_THRESHOLD = args.threshold

    path = Path(args.data)

    if path.is_file():
        passed = check_file(str(path), args.dt)
    elif path.is_dir():
        passed = check_directory(str(path), args.dt)
    else:
        print(f"ERROR: Path does not exist: {args.data}")
        sys.exit(2)

    if passed:
        print("\n*** CI GATE: PASSED ***")
        sys.exit(0)
    else:
        print("\n*** CI GATE: FAILED ***")
        print("Fix circular sensor issues before merging.")
        sys.exit(1)


if __name__ == "__main__":
    main()
