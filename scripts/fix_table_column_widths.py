#!/usr/bin/env python3
"""
Fix table column widths for IEEE two-column format.
Reduce column widths proportionally to fit within page margins.
"""

import re
from pathlib import Path


def fix_table_widths(input_file, output_file):
    """Scale down table column widths to prevent overflow."""

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    def scale_column_spec(match):
        """Scale down column specification if total width > 0.9."""
        col_spec = match.group(1)

        # Find all p{X\textwidth} patterns
        widths = re.findall(r"p\{([\d.]+)\\textwidth\}", col_spec)

        if not widths:
            return match.group(0)

        # Calculate total width
        total_width = sum(float(w) for w in widths)

        # If total width > 0.88 (leaving margin), scale down proportionally
        if total_width > 0.88:
            scale_factor = 0.85 / total_width  # Target 85% width with margins

            # Replace each width with scaled version
            new_col_spec = col_spec
            for width in widths:
                old_width_str = f"p{{{width}\\textwidth}}"
                new_width = float(width) * scale_factor
                new_width_str = f"p{{{new_width:.3f}\\textwidth}}"
                new_col_spec = new_col_spec.replace(old_width_str, new_width_str, 1)

            print(
                f"  Scaled table: {total_width:.2f} → {total_width*scale_factor:.2f} ({len(widths)} columns)"
            )
            return f"\\begin{{tabular}}{{{new_col_spec}}}"

        return match.group(0)

    # Pattern: \begin{tabular}{...}
    pattern = r"\\begin\{tabular\}\{([^}]+)\}"
    content = re.sub(pattern, scale_column_spec, content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print("[OK] Fixed table column widths")


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    input_file = PROJECT_ROOT / "reports" / "quadrotor_pinn_report_IEEE.tex"
    output_file = PROJECT_ROOT / "reports" / "quadrotor_pinn_report_IEEE_scaled.tex"

    print("=" * 80)
    print("FIXING TABLE COLUMN WIDTHS")
    print("=" * 80)
    print()

    fix_table_widths(input_file, output_file)

    # Replace original
    import os
    import shutil

    if os.path.exists(input_file):
        os.remove(input_file)
    shutil.move(str(output_file), str(input_file))
    print(f"\n[OK] Updated {input_file.name}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
