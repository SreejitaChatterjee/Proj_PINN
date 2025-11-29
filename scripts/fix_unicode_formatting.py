#!/usr/bin/env python3
"""
Fix Unicode characters in IEEE LaTeX file for proper PDF formatting
"""
import re

def fix_unicode_characters(filepath):
    """Replace Unicode characters with proper LaTeX equivalents"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Fix arrows
    content = content.replace('→', r'$\rightarrow$')

    # Fix em-dashes (but not in math mode or comments)
    # Keep -- for number ranges (e.g., 45--60)
    content = re.sub(r'(?<!-)—(?!-)', r'---', content)  # em-dash to LaTeX em-dash

    # Fix multiplication symbols (but not in math mode)
    # Only replace × when NOT between $ signs
    def replace_times(match):
        if '$' in match.group(0):
            return match.group(0)
        return match.group(0).replace('×', r'$\times$')

    # This is tricky - let's be more conservative
    # Replace × only when it appears between numbers
    content = re.sub(r'(\d+)×(\d+)', r'\1$\times$\2', content)
    content = re.sub(r'(\d+)× ', r'\1$\times$ ', content)
    content = re.sub(r' ×(\d+)', r' $\times$\1', content)

    # Fix any remaining standalone ×
    content = content.replace(' × ', r' $\times$ ')

    changes_made = content != original_content

    if changes_made:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[OK] Fixed Unicode characters in {filepath}")
        print(f"  - Replaced arrows with $\\rightarrow$")
        print(f"  - Replaced em-dashes with ---")
        print(f"  - Replaced multiplication symbols with $\\times$")
        return True
    else:
        print(f"No Unicode character issues found in {filepath}")
        return False

if __name__ == "__main__":
    filepath = r"C:\Users\sreej\OneDrive\Documents\GitHub\Proj_PINN\reports\quadrotor_pinn_report_IEEE.tex"
    fix_unicode_characters(filepath)
