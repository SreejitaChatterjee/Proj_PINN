#!/usr/bin/env python3
"""
Simple line-by-line fix for table pipes.
"""

from pathlib import Path

def fix_table_pipes_simple(input_file, output_file):
    """Fix pipes in \begin{tabular}{...} lines only."""

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fixed_lines = []
    changes = 0

    for line in lines:
        if '\\begin{tabular}{' in line:
            # This line contains a tabular spec - remove pipes from the column spec
            # Find the column spec part
            if '{' in line and '}' in line:
                start = line.find('\\begin{tabular}{')
                if start != -1:
                    # Find the closing brace for the column spec
                    brace_start = line.find('{', start + len('\\begin{tabular}'))
                    if brace_start != -1:
                        brace_end = line.find('}', brace_start)
                        if brace_end != -1:
                            # Extract column spec
                            col_spec = line[brace_start+1:brace_end]
                            if '|' in col_spec:
                                # Remove pipes
                                col_spec_fixed = col_spec.replace('|', '')
                                # Reconstruct line
                                line = line[:brace_start+1] + col_spec_fixed + line[brace_end:]
                                changes += 1

        fixed_lines.append(line)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

    print(f"[OK] Fixed {changes} table column specifications")

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    input_file = PROJECT_ROOT / 'reports' / 'quadrotor_pinn_report_IEEE.tex'
    output_file = PROJECT_ROOT / 'reports' / 'quadrotor_pinn_report_IEEE_temp.tex'

    print("="*80)
    print("SIMPLE TABLE PIPE FIX")
    print("="*80)
    print()

    fix_table_pipes_simple(input_file, output_file)

    # Replace original
    import shutil
    import os
    if os.path.exists(input_file):
        os.remove(input_file)
    shutil.move(str(output_file), str(input_file))
    print(f"[OK] Updated {input_file.name}")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
