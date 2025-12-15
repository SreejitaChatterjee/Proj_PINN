#!/usr/bin/env python3
"""
Setup script for pinn-dynamics package.

Install in development mode:
    pip install -e .

Install from PyPI (when published):
    pip install pinn-dynamics
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="pinn-dynamics",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Physics-Informed Neural Networks for Dynamics Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Proj_PINN",
    license="MIT",

    # Package discovery
    packages=find_packages(include=["scripts", "scripts.*"]),
    py_modules=["demo"],

    # Dependencies
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "joblib>=1.0.0",
        "tqdm>=4.50.0",
    ],

    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
        ],
    },

    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "pinn-demo=demo:main",
        ],
    },

    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],

    # Include non-Python files
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.pth", "*.pkl"],
    },
)
