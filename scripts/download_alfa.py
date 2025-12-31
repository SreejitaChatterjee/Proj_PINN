"""
Download and preprocess ALFA dataset for reproducibility.

ALFA Dataset: https://theairlab.org/alfa-dataset/
Citation: Keipour et al., "ALFA: A dataset for UAV fault and anomaly detection," IJRR 2021

This script documents the exact steps to reproduce the ALFA evaluation.
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Configuration
ALFA_URL = "https://theairlab.org/alfa-dataset/"
DATA_DIR = Path("data/alfa")
PROCESSED_DIR = DATA_DIR / "processed"


def print_instructions():
    """Print manual download instructions."""
    print("=" * 70)
    print("ALFA DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print()
    print("The ALFA dataset must be downloaded manually from CMU AirLab.")
    print()
    print("Steps:")
    print("1. Visit: https://theairlab.org/alfa-dataset/")
    print("2. Download the ROS bag files")
    print("3. Extract to: data/alfa/raw/")
    print()
    print("Expected structure:")
    print("  data/alfa/")
    print("  +-- raw/")
    print("  |   +-- normal/")
    print("  |   |   +-- flight_01.bag")
    print("  |   |   +-- ...")
    print("  |   +-- engine_failure/")
    print("  |   |   +-- flight_01.bag")
    print("  |   |   +-- ...")
    print("  |   +-- ...")
    print("  +-- processed/")
    print("      +-- alfa_processed.csv")
    print()
    print("Flight breakdown (47 total):")
    print("  - Normal: 10 flights")
    print("  - Engine Failure: 23 flights")
    print("  - Rudder Stuck: 3 flights")
    print("  - Aileron Stuck: 8 flights")
    print("  - Elevator Stuck: 2 flights")
    print("  - Unknown Fault: 1 flight")
    print()
    print("=" * 70)


def check_data_exists():
    """Check if ALFA data exists."""
    if DATA_DIR.exists():
        files = list(DATA_DIR.rglob("*.bag")) + list(DATA_DIR.rglob("*.csv"))
        if files:
            print(f"Found {len(files)} ALFA data files")
            return True
    print("ALFA data NOT found")
    return False


def create_placeholder():
    """Create placeholder structure."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Create README
    readme = DATA_DIR / "README.md"
    readme.write_text(
        """# ALFA Dataset

**Status:** NOT DOWNLOADED

## Download Instructions

1. Visit: https://theairlab.org/alfa-dataset/
2. Download ROS bag files
3. Extract to `raw/` subdirectory
4. Run preprocessing: `python scripts/preprocess_alfa.py`

## Citation

```bibtex
@article{keipour2021alfa,
  title={ALFA: A dataset for UAV fault and anomaly detection},
  author={Keipour, Azarakhsh and others},
  journal={International Journal of Robotics Research},
  year={2021}
}
```

## Expected Files

- 47 flight recordings (ROS bags)
- 10 normal + 37 fault flights
- ~5,506 samples after preprocessing
"""
    )

    print(f"Created placeholder at {DATA_DIR}")


def create_reproducibility_config():
    """Create config for reproducible evaluation."""
    config = {
        "dataset": "ALFA",
        "version": "1.0",
        "download_url": ALFA_URL,
        "citation": "Keipour et al., IJRR 2021",
        "preprocessing": {
            "script": "scripts/preprocess_alfa.py",
            "output": "data/alfa/processed/alfa_processed.csv",
        },
        "evaluation": {
            "random_seed": 42,
            "cv_method": "leave-one-flight-out",
            "n_flights": 47,
            "contamination": 0.05,
            "threshold_method": "grid_search_balanced_accuracy",
        },
        "hardware": {"note": "Document your hardware here when running evaluation"},
        "created": datetime.now().isoformat(),
    }

    config_path = DATA_DIR / "reproducibility_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created reproducibility config at {config_path}")
    return config


def main():
    print_instructions()

    if not check_data_exists():
        print("\nCreating placeholder structure...")
        create_placeholder()
        create_reproducibility_config()
        print("\nPlease download ALFA data manually and re-run.")
        return False

    print("\nALFA data found. Ready for preprocessing.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
