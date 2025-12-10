#!/usr/bin/env python3
"""
Fix figure widths in IEEE LaTeX file - use columnwidth for single-column figures
"""
import re

def fix_figure_widths(filepath):
    """Replace textwidth with columnwidth in single-column figure environments"""

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    in_single_figure = False

    for i, line in enumerate(lines):
        # Detect single-column figure environment
        if r'\begin{figure}[!t]' in line or r'\begin{figure}[!h]' in line:
            in_single_figure = True
        elif r'\begin{figure*}' in line:
            in_single_figure = False
        elif r'\end{figure}' in line:
            in_single_figure = False

        # Replace textwidth with columnwidth in single-column figures
        if in_single_figure and r'width=\textwidth' in line:
            lines[i] = line.replace(r'width=\textwidth', r'width=\columnwidth')
            modified = True
            print(f"Line {i+1}: Replaced \textwidth with \columnwidth")

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"\n[OK] Fixed figure widths in {filepath}")
        return True
    else:
        print(f"No figure width issues found in {filepath}")
        return False

if __name__ == "__main__":
    # Use path relative to script location
    from pathlib import Path
    filepath = Path(__file__).parent.parent / "reports" / "quadrotor_pinn_report_IEEE.tex"
    fix_figure_widths(filepath)
