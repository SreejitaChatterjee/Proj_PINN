"""
MATLAB .mat File Reader for Quadrotor Failure Data

Installation:
    pip install pymatreader tabulate numpy pandas

Usage:
    python mat_reader.py <path_to_mat_file>

    Example:
    python mat_reader.py ../data/quad_failure/m1.mat
"""

import os
import sys

import numpy as np
import pandas as pd
from pymatreader import read_mat
from tabulate import tabulate


def display_mat_contents(mat_data, max_rows=10):
    """
    Nicely display contents of a .mat file
    """
    for key, value in mat_data.items():
        print(f"\n=== Variable: {key} ===")

        # Case 1: NumPy array
        if isinstance(value, np.ndarray):
            print(f"Shape: {value.shape}, dtype: {value.dtype}")

            if value.ndim == 1:
                table = [[i, v] for i, v in enumerate(value[:max_rows])]
                print(tabulate(table, headers=["Index", "Value"], tablefmt="grid"))
                if len(value) > max_rows:
                    print(f"... ({len(value) - max_rows} more rows)")

            elif value.ndim == 2:
                df = pd.DataFrame(value)
                print(tabulate(df.head(max_rows), headers="keys", tablefmt="grid"))
                if len(df) > max_rows:
                    print(f"... ({len(df) - max_rows} more rows)")

            else:
                print(f"Array with shape {value.shape} (too many dimensions to display)")

        # Case 2: Dictionary / MATLAB struct
        elif isinstance(value, dict):
            table = [
                [
                    k,
                    str(type(v).__name__),
                    v.shape if isinstance(v, np.ndarray) else "N/A",
                ]
                for k, v in value.items()
            ]
            print(tabulate(table, headers=["Field", "Type", "Shape"], tablefmt="grid"))

        # Case 3: Scalar or other type
        else:
            print(value)


def print_dict_recursive(d, indent=0):
    """Recursively print dictionary structure"""
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(" " * indent + f"- {k}: ndarray{v.shape}")
        else:
            print(" " * indent + f"- {k}: {type(v).__name__}")
        if isinstance(v, dict):
            print_dict_recursive(v, indent + 2)


def extract_timeseries(mat_data):
    """
    Extract time series data suitable for PINN training.
    Returns a DataFrame with time, states, and controls if available.
    """
    # Common variable names in quadrotor data
    possible_time = ["t", "time", "Time", "timestamp"]
    possible_states = ["x", "state", "states", "X", "pos", "position"]

    result = {}

    for key, value in mat_data.items():
        if isinstance(value, np.ndarray):
            result[key] = value
        elif isinstance(value, dict):
            for subkey, subval in value.items():
                if isinstance(subval, np.ndarray):
                    result[f"{key}.{subkey}"] = subval

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mat_file_path = sys.argv[1]
    else:
        # Default path - adjust as needed
        mat_file_path = "data/quad_failure/m1.mat"

    if not os.path.exists(mat_file_path):
        print(f"Error: File not found: {mat_file_path}")
        print("\nUsage: python mat_reader.py <path_to_mat_file>")
        sys.exit(1)

    print(f"Loading: {mat_file_path}")
    mat_data = read_mat(mat_file_path)

    print("\n" + "=" * 60)
    print("STRUCTURE OVERVIEW")
    print("=" * 60)
    print_dict_recursive(mat_data)

    print("\n" + "=" * 60)
    print("DETAILED CONTENTS")
    print("=" * 60)
    display_mat_contents(mat_data)

    print("\n" + "=" * 60)
    print("EXTRACTED TIME SERIES")
    print("=" * 60)
    extracted = extract_timeseries(mat_data)
    for name, arr in extracted.items():
        print(f"  {name}: shape={arr.shape}")
