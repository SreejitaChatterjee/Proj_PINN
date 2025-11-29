#!/usr/bin/env python3
"""
Final cleanup of IEEE tables - remove all pipes and fix spacing.
"""

import re
from pathlib import Path

def clean_tables_final(input_file, output_file):
    """Clean all remaining table issues."""

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # FIX 1: Remove all pipes from tabular column specs
    # Pattern: \begin{tabular}{... with pipes ...}
    def remove_pipes_from_colspec(match):
        """Remove pipes from column specification."""
        col_spec = match.group(1)
        # Remove all pipes
        col_spec_clean = col_spec.replace('|', '')
        return f'\\begin{{tabular}}{{{col_spec_clean}}}'

    content = re.sub(
        r'\\begin\{tabular\}\{([^}]+)\}',
        remove_pipes_from_colspec,
        content
    )

    # FIX 2: Replace excessive \midrule usage - only one after header
    # Pattern: \toprule...header...\midrule...rows with \midrule between each...\bottomrule
    # Replace: \toprule...header...\midrule...rows WITHOUT \midrule...\bottomrule

    def fix_midrule_usage(match):
        """Fix excessive midrule usage in tables."""
        table_content = match.group(0)

        # Count midrules
        midrule_count = table_content.count('\\midrule')

        if midrule_count > 1:
            # Keep only the first \midrule (after header)
            # Replace all others with just a newline
            lines = table_content.split('\n')
            fixed_lines = []
            midrule_seen = False

            for line in lines:
                if '\\midrule' in line:
                    if not midrule_seen:
                        # Keep first midrule
                        fixed_lines.append(line)
                        midrule_seen = True
                    # else: skip subsequent midrules (don't add them)
                else:
                    fixed_lines.append(line)

            return '\n'.join(fixed_lines)
        else:
            return table_content

    # Apply to all table blocks
    content = re.sub(
        r'\\begin\{tabular\}.*?\\end\{tabular\}',
        fix_midrule_usage,
        content,
        flags=re.DOTALL
    )

    # FIX 3: Fix any typos like "	extwidth" (should be "\textwidth")
    content = re.sub(r'	extwidth', r'\\textwidth', content)

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] Removed all pipes from column specs")
    print(f"[OK] Fixed excessive \\midrule usage")
    print(f"[OK] Fixed typos in column specifications")

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    input_file = PROJECT_ROOT / 'reports' / 'quadrotor_pinn_report_IEEE.tex'
    output_file = PROJECT_ROOT / 'reports' / 'quadrotor_pinn_report_IEEE_clean.tex'

    print("="*80)
    print("FINAL TABLE CLEANUP")
    print("="*80)
    print()

    clean_tables_final(input_file, output_file)

    # Replace original with cleaned version
    import shutil
    shutil.move(str(output_file), str(input_file))
    print(f"[OK] Updated {input_file.name}")

    print("\n" + "="*80)
    print("CLEANUP COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
