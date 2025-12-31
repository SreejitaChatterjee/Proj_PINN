"""
Precompute PADRE to PINN conversion and save to disk.

This converts all PADRE files to PINN-compatible format and saves them
as .npz files for fast loading during training.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pinn_dynamics.data.padre import PADREtoPINNConverter


def denormalize_padre_data(data, accel_range_g=16, gyro_range_dps=1000):
    """Denormalize PADRE data from [-1, 1] to physical units."""
    g = 9.81
    data_out = data.copy()
    for motor_idx in range(4):
        base = motor_idx * 6
        data_out[:, base : base + 3] *= accel_range_g * g
        data_out[:, base + 3 : base + 6] *= gyro_range_dps * (np.pi / 180)
    return data_out


def main():
    import re

    data_dir = Path("data/PADRE_dataset/Parrot_Bebop_2/Normalized_data")
    output_dir = Path("data/PADRE_PINN_converted")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} files to convert")

    converter = PADREtoPINNConverter(
        dt=0.002,
        mass=0.5,
        Jxx=0.005,
        Jyy=0.005,
        Jzz=0.009,
        complementary_alpha=0.98,
        drag_coeff=0.01,
    )

    window_size = 128
    stride = 64
    max_samples = 20000  # Limit per file for speed

    all_results = []

    for csv_file in tqdm(csv_files, desc="Converting"):
        # Parse fault label
        match = re.search(r"normalized_(\d{4})\.csv$", csv_file.name)
        if not match:
            continue

        codes = match.group(1)
        motor_faults = {
            "A": int(codes[0]),
            "B": int(codes[1]),
            "C": int(codes[2]),
            "D": int(codes[3]),
        }
        is_faulty = 1 if any(f > 0 for f in motor_faults.values()) else 0

        # Load and denormalize
        df = pd.read_csv(csv_file)
        padre_data = df.values[:max_samples].astype(np.float32)
        padre_data = denormalize_padre_data(padre_data)

        # Convert
        try:
            X, Y = converter.convert_windowed(padre_data, window_size, stride)

            all_results.append(
                {
                    "file": csv_file.name,
                    "X": X,
                    "Y": Y,
                    "fault_label": is_faulty,
                    "motor_faults": codes,
                }
            )

            print(f"  {csv_file.name}: {X.shape[0]} windows, fault={is_faulty}")

        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")

    # Combine all data
    print("\nCombining data...")
    X_all = np.concatenate([r["X"] for r in all_results], axis=0)
    Y_all = np.concatenate([r["Y"] for r in all_results], axis=0)
    labels = np.concatenate([np.full(r["X"].shape[0], r["fault_label"]) for r in all_results])

    print(f"\nTotal windows: {X_all.shape[0]}")
    print(f"X shape: {X_all.shape}")
    print(f"Y shape: {Y_all.shape}")
    print(f"Normal samples: {(labels == 0).sum()}")
    print(f"Faulty samples: {(labels == 1).sum()}")

    # Flatten for single-step prediction
    n_windows, seq_len, input_dim = X_all.shape
    X_flat = X_all.reshape(-1, input_dim)
    Y_flat = Y_all.reshape(-1, Y_all.shape[-1])
    labels_flat = np.repeat(labels, seq_len)

    print(f"\nFlattened: {X_flat.shape[0]} samples")

    # Save
    output_file = output_dir / "padre_pinn_data.npz"
    np.savez_compressed(
        output_file, X=X_flat, Y=Y_flat, labels=labels_flat, window_size=window_size, stride=stride
    )

    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
