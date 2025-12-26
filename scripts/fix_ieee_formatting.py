#!/usr/bin/env python3
"""
Fix IEEE formatting issues in the report.
- Correct table syntax
- Fix figure widths for two-column format
- Improve overall professional appearance
"""

import re
from pathlib import Path


def fix_ieee_formatting(input_file, output_file):
    """Fix formatting issues in IEEE report."""

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # FIX 1: Fix broken table syntax
    # Pattern: \scriptsize{p{...} p{...} p{...}}\n\toprule
    # Should be: \scriptsize\n\begin{tabular}{p{...} p{...} p{...}}\n\toprule
    pattern = r"\\scriptsize\{([p|c|l|r]\{[^}]+\}(?:\s+[p|c|l|r]\{[^}]+\})*)\}\s*\n\s*\\toprule"
    replacement = r"\\scriptsize\n\\begin{tabular}{\1}\n\\toprule"
    content = re.sub(pattern, replacement, content)

    # FIX 2: Add \end{tabular} before \end{table*}
    # Pattern: \bottomrule\n\end{table*}
    # Should be: \bottomrule\n\end{tabular}\n\end{table*}
    content = re.sub(
        r"\\bottomrule\s*\n\s*\\end\{table\*\}",
        r"\\bottomrule\n\\end{tabular}\n\\end{table*}",
        content,
    )

    # FIX 3: Fix figure widths for IEEE two-column format
    # 1.4\textwidth or 1.2\textwidth -> \textwidth (fits both columns)
    content = re.sub(r"width=1\.[0-9]+\\textwidth", r"width=\\textwidth", content)

    # FIX 4: Fix itemize environment parameters (IEEE doesn't support leftmargin)
    content = re.sub(r"\\begin\{itemize\}\[[^\]]+\]", r"\\begin{itemize}", content)

    # FIX 5: Fix enumerate environment parameters
    content = re.sub(r"\\begin\{enumerate\}\[[^\]]+\]", r"\\begin{enumerate}", content)

    # FIX 6: Fix column spec with pipes |p{...}|
    # Change to {p{...}} without pipes for cleaner IEEE style
    content = re.sub(r"\\begin\{tabular\}\{\|([^}]+)\|\}", r"\\begin{tabular}{\1}", content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] Fixed IEEE formatting issues")
    print(f"[OK] Output: {output_file.name}")


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    input_file = PROJECT_ROOT / "reports" / "quadrotor_pinn_report_IEEE.tex"
    output_file = PROJECT_ROOT / "reports" / "quadrotor_pinn_report_IEEE_fixed.tex"

    print("=" * 80)
    print("FIXING IEEE FORMATTING ISSUES")
    print("=" * 80)
    print("\nIssues to fix:")
    print("  1. Broken table syntax (\\scriptsize{...} incorrect)")
    print("  2. Figure widths too large (1.4\\textwidth -> \\textwidth)")
    print("  3. Missing table environment closures")
    print("  4. Itemize parameters not compatible with IEEE")
    print()

    fix_ieee_formatting(input_file, output_file)

    # Replace original with fixed version
    import shutil

    shutil.move(str(output_file), str(input_file))
    print(f"[OK] Replaced {input_file.name} with fixed version")

    print("\n" + "=" * 80)
    print("FORMATTING FIXES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
