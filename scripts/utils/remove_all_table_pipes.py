#!/usr/bin/env python3
"""
Remove ALL pipes from table column specifications.
"""

import re
from pathlib import Path


def remove_table_pipes(input_file, output_file):
    """Remove pipes from all \begin{tabular}{...} specifications."""

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all \begin{tabular}{...} and remove pipes from the column spec
    def remove_pipes(match):
        """Remove pipes from column specification."""
        full_match = match.group(0)  # \begin{tabular}{...}
        col_spec = match.group(1)  # just the column spec part

        # Remove all pipe characters
        col_spec_no_pipes = col_spec.replace("|", "")

        # Reconstruct the command
        return f"\\begin{{tabular}}{{{col_spec_no_pipes}}}"

    # Pattern: \begin{tabular}{anything}
    # Capture group 1 is the column specification
    pattern = r"\\begin\{tabular\}\{([^}]+)\}"
    content = re.sub(pattern, remove_pipes, content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] Removed all pipes from table column specifications")


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    input_file = PROJECT_ROOT / "reports" / "quadrotor_pinn_report_IEEE.tex"
    output_file = PROJECT_ROOT / "reports" / "quadrotor_pinn_report_IEEE_nopipes.tex"

    print("=" * 80)
    print("REMOVING ALL TABLE PIPES")
    print("=" * 80)
    print()

    remove_table_pipes(input_file, output_file)

    # Replace original with fixed version
    import shutil

    shutil.move(str(output_file), str(input_file))
    print(f"[OK] Updated {input_file.name}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
