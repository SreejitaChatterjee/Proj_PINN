#!/usr/bin/env python3
"""
Fix figure widths to prevent overflow in IEEE two-column format.
"""

import re
from pathlib import Path

def fix_figure_widths(input_file, output_file):
    """Reduce figure widths to fit within column margins."""

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    changes = 0

    # Replace 0.95\textwidth with 0.9\textwidth (safer margin)
    old_pattern = r'width=0\.95\\textwidth'
    new_value = r'width=0.9\\textwidth'
    content, count = re.subn(old_pattern, new_value, content)
    changes += count
    print(f"[OK] Reduced 0.95\\textwidth to 0.9\\textwidth: {count} instances")

    # Also check for 1.0\textwidth figures and make them slightly smaller
    old_pattern2 = r'width=1\.0\\textwidth'
    new_value2 = r'width=0.95\\textwidth'
    content, count2 = re.subn(old_pattern2, new_value2, content)
    changes += count2
    if count2 > 0:
        print(f"[OK] Reduced 1.0\\textwidth to 0.95\\textwidth: {count2} instances")

    # Check for plain \textwidth without multiplier
    # Only modify if it's for PNG/JPG (not PDF schematics which are designed for full width)
    def reduce_textwidth_for_images(match):
        """Reduce \textwidth for image files."""
        full_match = match.group(0)
        # Check if it's a PNG or JPG
        if '.png' in full_match or '.jpg' in full_match or '.jpeg' in full_match:
            return full_match.replace('width=\\textwidth', 'width=0.9\\textwidth')
        return full_match

    # Find all \includegraphics commands
    pattern = r'\\includegraphics\[width=\\textwidth[^\]]*\]\{[^}]+\}'
    original_content = content
    content = re.sub(pattern, reduce_textwidth_for_images, content)

    if content != original_content:
        count3 = len(re.findall(pattern, original_content)) - len(re.findall(pattern, content))
        changes += count3
        print(f"[OK] Reduced \\textwidth to 0.9\\textwidth for images: {count3} instances")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] Total changes: {changes}")
    return changes

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    input_file = PROJECT_ROOT / 'reports' / 'quadrotor_pinn_report_IEEE.tex'
    output_file = PROJECT_ROOT / 'reports' / 'quadrotor_pinn_report_IEEE_fixed_widths.tex'

    print("="*80)
    print("FIXING FIGURE WIDTHS FOR IEEE TWO-COLUMN FORMAT")
    print("="*80)
    print()

    changes = fix_figure_widths(input_file, output_file)

    if changes > 0:
        # Replace original
        import shutil
        import os
        if os.path.exists(input_file):
            os.remove(input_file)
        shutil.move(str(output_file), str(input_file))
        print(f"\n[OK] Updated {input_file.name}")
    else:
        print("\n[INFO] No changes needed")
        if output_file.exists():
            output_file.unlink()

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
