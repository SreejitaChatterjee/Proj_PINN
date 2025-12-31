#!/usr/bin/env python3
"""
Detect content anomalies and formatting issues in IEEE report.
"""

import re
from collections import Counter
from pathlib import Path


def detect_anomalies(tex_file):
    """Detect various anomalies in the LaTeX file."""

    with open(tex_file, "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.split("\n")

    issues = []

    # Check 1: Missing references section
    if "\\bibliography" not in content and "\\begin{thebibliography}" not in content:
        issues.append("WARNING: No references/bibliography section found")

    # Check 2: Inconsistent spacing around operators
    inconsistent_spacing = []
    for i, line in enumerate(lines, 1):
        # Check for =, +, -, etc. without spaces
        if re.search(r"[a-zA-Z0-9][=+\-*/][a-zA-Z0-9]", line):
            if not line.strip().startswith("%"):  # Not a comment
                inconsistent_spacing.append(f"Line {i}: {line[:60]}")

    if inconsistent_spacing:
        issues.append(
            f"FORMATTING: Found {len(inconsistent_spacing)} lines with inconsistent operator spacing"
        )

    # Check 3: Long lines that might cause overfull hbox
    long_lines = []
    for i, line in enumerate(lines, 1):
        if len(line) > 200 and "\\includegraphics" not in line:
            long_lines.append(f"Line {i}: {len(line)} chars")

    if long_lines:
        issues.append(f"FORMATTING: Found {len(long_lines)} very long lines")

    # Check 4: Inconsistent decimal precision
    percentages = re.findall(r"(\d+\.\d+)%", content)
    decimals = [len(p.split(".")[1]) for p in percentages]
    if decimals:
        precision_counter = Counter(decimals)
        if len(precision_counter) > 2:
            issues.append(
                f"INCONSISTENCY: Varying decimal precision in percentages: {dict(precision_counter)}"
            )

    # Check 5: Missing captions for figures
    fig_blocks = re.findall(r"\\begin{figure}.*?\\end{figure}", content, re.DOTALL)
    figs_without_caption = sum(1 for block in fig_blocks if "\\caption" not in block)
    if figs_without_caption > 0:
        issues.append(f"CONTENT: {figs_without_caption} figures missing captions")

    # Check 6: Missing labels for figures/tables
    fig_blocks_all = re.findall(r"(\\begin{figure}.*?\\end{figure})", content, re.DOTALL)
    figs_without_label = sum(1 for block in fig_blocks_all if "\\label" not in block)
    if figs_without_label > 0:
        issues.append(f"CONTENT: {figs_without_label} figures missing labels")

    table_blocks = re.findall(r"(\\begin{table\*?}.*?\\end{table\*?})", content, re.DOTALL)
    tables_without_label = sum(1 for block in table_blocks if "\\label" not in block)
    if tables_without_label > 0:
        issues.append(f"CONTENT: {tables_without_label} tables missing labels")

    # Check 7: Inconsistent equation formatting
    inline_math = len(re.findall(r"\$[^$]+\$", content))
    display_math = len(re.findall(r"\\\[.*?\\\]", content, re.DOTALL))
    equation_envs = len(re.findall(r"\\begin{equation}", content))

    issues.append(
        f"INFO: Math usage - Inline: {inline_math}, Display: {display_math}, Equations: {equation_envs}"
    )

    # Check 8: Inconsistent use of \textbf vs \textit
    bold_count = content.count("\\textbf{")
    italic_count = content.count("\\textit{")
    issues.append(f"INFO: Formatting - Bold: {bold_count}, Italic: {italic_count}")

    # Check 9: Check for proper section structure
    sections = len(re.findall(r"\\section{", content))
    subsections = len(re.findall(r"\\subsection{", content))
    subsubsections = len(re.findall(r"\\subsubsection{", content))
    issues.append(
        f"INFO: Document structure - Sections: {sections}, Subsections: {subsections}, Subsubsections: {subsubsections}"
    )

    # Check 10: Abstract present?
    if "\\begin{abstract}" not in content:
        issues.append("WARNING: No abstract found")

    # Check 11: Keywords present?
    if "keywords" not in content.lower() and "index terms" not in content.lower():
        issues.append("WARNING: No keywords/index terms found")

    return issues


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    tex_file = PROJECT_ROOT / "reports" / "quadrotor_pinn_report_IEEE.tex"

    print("=" * 80)
    print("ANOMALY DETECTION REPORT")
    print("=" * 80)
    print()

    issues = detect_anomalies(tex_file)

    for issue in issues:
        if issue.startswith("WARNING"):
            print(f"[!] {issue}")
        elif (
            issue.startswith("CONTENT")
            or issue.startswith("FORMATTING")
            or issue.startswith("INCONSISTENCY")
        ):
            print(f"[X] {issue}")
        else:
            print(f"[i] {issue}")

    print()
    print("=" * 80)
    print(f"TOTAL ISSUES: {len([i for i in issues if not i.startswith('INFO')])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
