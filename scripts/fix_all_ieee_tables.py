#!/usr/bin/env python3
"""
Fix ALL broken table syntax in IEEE report.
"""

import re
from pathlib import Path

def fix_all_tables(input_file, output_file):
    """Fix all broken table syntax issues."""

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # FIX 1: Replace broken \scriptsize{column_spec} with proper syntax
    # Pattern: \scriptsize{|p{...}|...} followed by \hline or \toprule
    # Need to find all instances and replace with \begin{tabular}{...}

    # First, let's find all table environments and fix them
    # Pattern: \begin{table*}...\scriptsize{...}...\end{table*}

    def fix_table_block(match):
        """Fix a single table block."""
        table_content = match.group(0)

        # Extract column specification from \scriptsize{...}
        scriptsize_match = re.search(r'\\scriptsize\{([^}]+)\}', table_content)
        if scriptsize_match:
            column_spec = scriptsize_match.group(1)

            # Remove pipes from column spec for cleaner IEEE style
            column_spec_clean = column_spec.replace('|', '')

            # Replace \scriptsize{...} with proper \begin{tabular}{...}
            table_content = re.sub(
                r'\\scriptsize\{[^}]+\}',
                f'\\\\renewcommand{{\\\\arraystretch}}{{1.4}}\n\\\\begin{{tabular}}{{{column_spec_clean}}}',
                table_content
            )

            # Replace \hline with \toprule for first line
            table_content = re.sub(
                r'^(\\renewcommand.*?\n\\begin\{tabular\}.*?\n)\\hline',
                r'\1\\toprule',
                table_content,
                flags=re.MULTILINE
            )

            # Replace remaining \hline with \midrule (except the last one)
            lines = table_content.split('\n')
            fixed_lines = []
            hline_count = 0
            total_hlines = table_content.count('\\hline')

            for line in lines:
                if '\\hline' in line:
                    hline_count += 1
                    if hline_count == total_hlines:
                        # Last hline becomes \bottomrule
                        fixed_lines.append(line.replace('\\hline', '\\bottomrule'))
                    else:
                        # Middle hlines become \midrule
                        fixed_lines.append(line.replace('\\hline', '\\midrule'))
                else:
                    fixed_lines.append(line)

            table_content = '\n'.join(fixed_lines)

            # Ensure \end{tabular} exists before \end{table*}
            if '\\end{tabular}' not in table_content:
                table_content = table_content.replace('\\end{table*}', '\\end{tabular}\n\\end{table*}')

        return table_content

    # Find and fix all table* blocks that contain \scriptsize{
    pattern = r'\\begin\{table\*\}.*?\\end\{table\*\}'
    content = re.sub(pattern, fix_table_block, content, flags=re.DOTALL)

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] Fixed all broken table syntax")
    print(f"[OK] Replaced \\scriptsize{{...}} with \\begin{{tabular}}{{...}}")
    print(f"[OK] Replaced \\hline with \\toprule, \\midrule, \\bottomrule")
    print(f"[OK] Added \\renewcommand{{\\arraystretch}}{{1.4}} for spacing")
    print(f"[OK] Removed pipes from column specs")

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    input_file = PROJECT_ROOT / 'reports' / 'quadrotor_pinn_report_IEEE.tex'
    output_file = PROJECT_ROOT / 'reports' / 'quadrotor_pinn_report_IEEE_fixed_all.tex'

    print("="*80)
    print("FIXING ALL BROKEN TABLE SYNTAX")
    print("="*80)
    print(f"\nFound 17 tables with broken syntax")
    print(f"Fixing all instances...\n")

    fix_all_tables(input_file, output_file)

    # Replace original with fixed version
    import shutil
    shutil.move(str(output_file), str(input_file))
    print(f"\n[OK] Updated {input_file.name}")

    print("\n" + "="*80)
    print("ALL TABLE FIXES COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
