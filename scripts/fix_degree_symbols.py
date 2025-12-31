#!/usr/bin/env python3
r"""
Fix degree symbols in math mode - replace ° with ^\circ
"""

import re
from pathlib import Path


def fix_degree_symbols(input_file, output_file):
    r"""Replace ° with ^\circ inside math mode $...$"""

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all math mode expressions and replace ° with ^\circ
    def replace_degree_in_math(match):
        """Replace degree symbol in math expression."""
        math_content = match.group(0)
        # Replace ° with ^\circ
        fixed_math = math_content.replace("°", r"^\circ")
        return fixed_math

    # Pattern: $...$ (inline math)
    content = re.sub(r"\$[^$]+\$", replace_degree_in_math, content)

    # Pattern: \(...\) (alternative inline math)
    content = re.sub(r"\\\([^)]+\\\)", replace_degree_in_math, content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] Fixed degree symbols in math mode")
    print(f"[OK] Replaced ° with ^\\circ inside $...$ expressions")


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    input_file = PROJECT_ROOT / "reports" / "quadrotor_pinn_report_IEEE.tex"
    output_file = PROJECT_ROOT / "reports" / "quadrotor_pinn_report_IEEE_degrees.tex"

    print("=" * 80)
    print("FIXING DEGREE SYMBOLS IN MATH MODE")
    print("=" * 80)
    print()

    fix_degree_symbols(input_file, output_file)

    # Replace original
    import os
    import shutil

    if os.path.exists(input_file):
        os.remove(input_file)
    shutil.move(str(output_file), str(input_file))
    print(f"[OK] Updated {input_file.name}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
