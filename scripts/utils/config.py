"""
Configuration Management

Load and merge YAML configuration files.

Usage:
    from scripts.config import load_config

    cfg = load_config("configs/quadrotor.yaml")
    print(cfg.model.hidden_size)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigDict(dict):
    """Dict that allows attribute access."""

    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


def load_config(config_path: str, defaults_path: str = None) -> ConfigDict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file
        defaults_path: Optional path to defaults file (merged first)

    Returns:
        ConfigDict with configuration values
    """
    config = {}

    # Load defaults first
    if defaults_path is None:
        defaults_path = Path(__file__).parent.parent / "configs" / "default.yaml"

    if Path(defaults_path).exists():
        with open(defaults_path, "r") as f:
            config = yaml.safe_load(f) or {}

    # Load and merge specific config
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            specific = yaml.safe_load(f) or {}
        config = deep_merge(config, specific)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return ConfigDict(config)


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: dict, path: str):
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(dict(config), f, default_flow_style=False)


def get_device(device_str: str = "auto") -> str:
    """Get compute device."""
    import torch

    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


# Example usage
if __name__ == "__main__":
    cfg = load_config("configs/quadrotor.yaml")
    print(f"Model: {cfg.model.name}")
    print(f"Hidden size: {cfg.model.hidden_size}")
    print(f"Learning rate: {cfg.training.learning_rate}")
    print(f"Physics loss weight: {cfg.training.loss_weights.physics}")
