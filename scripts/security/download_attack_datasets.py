"""
Download and prepare attack datasets for PINN-based anomaly detection.

Supports:
1. Drone Sensor Fusion Dataset (IEEE DataPort, 2025)
2. Cyber-Physical UAV Dataset (IEEE DataPort, 2025)
3. ALFA Dataset (AirLab, 2020)

Usage:
    python scripts/security/download_attack_datasets.py --dataset drone_fusion --output data/attack_datasets
"""

import argparse
import os
import requests
from pathlib import Path
from tqdm import tqdm


DATASETS = {
    "drone_fusion": {
        "name": "Drone Sensor Fusion Dataset",
        "url": "https://ieee-dataport.org/documents/drone-sensor-fusion-dataset",
        "description": "GPS + IMU + Mag with simulated attacks (March 2025)",
        "size_gb": 2.5,
        "doi": "10.21227/drone-fusion-2025",
    },
    "cyber_physical": {
        "name": "Cyber-Physical UAV Dataset",
        "url": "https://ieee-dataport.org/documents/optimized-cyber-physical-dataset-attack-detection-internet-drones",
        "description": "DoS, MITM, GPS spoofing + telemetry (May 2025)",
        "size_gb": 1.8,
        "doi": "10.21227/uav-cyber-2025",
    },
    "alfa": {
        "name": "ALFA Dataset (AirLab)",
        "url": "https://theairlab.org/alfa-dataset/",
        "description": "Comprehensive UAV failure & anomaly dataset (2020)",
        "size_gb": 18.0,
        "doi": None,
    },
}


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=output_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser(description="Download UAV attack datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()) + ["all"],
        default="drone_fusion",
        help="Dataset to download",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pinn_dynamics/data/attack_datasets",
        help="Output directory",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if dataset already exists",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to download
    datasets_to_download = DATASETS.keys() if args.dataset == "all" else [args.dataset]

    print("=" * 60)
    print("UAV Attack Dataset Downloader")
    print("=" * 60)

    for ds_key in datasets_to_download:
        ds_info = DATASETS[ds_key]
        print(f"\n📦 {ds_info['name']}")
        print(f"   Size: ~{ds_info['size_gb']} GB")
        print(f"   Description: {ds_info['description']}")
        print(f"   URL: {ds_info['url']}")

        # Check if already exists
        ds_dir = output_dir / ds_key
        if ds_dir.exists() and args.skip_existing:
            print(f"   ✓ Already exists, skipping...")
            continue

        # Manual download instructions (IEEE DataPort requires login)
        if "ieee-dataport.org" in ds_info['url']:
            print(f"\n   ⚠️  IEEE DataPort datasets require manual download:")
            print(f"   1. Visit: {ds_info['url']}")
            print(f"   2. Sign in (free IEEE account required)")
            print(f"   3. Download dataset")
            print(f"   4. Extract to: {ds_dir.absolute()}")
            if ds_info['doi']:
                print(f"   DOI: {ds_info['doi']}")
        else:
            # Direct download for public datasets (ALFA)
            print(f"\n   Downloading...")
            ds_dir.mkdir(exist_ok=True)
            # Note: ALFA requires custom download logic (ROS bags)
            print(f"   ⚠️  Please visit {ds_info['url']} for download instructions")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Download datasets manually (see URLs above)")
    print("2. Run: python scripts/security/preprocess_attacks.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
