"""
Configuration management.

Load and merge YAML configuration files with attribute access.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """
    Dict that allows attribute access.

    Example:
        cfg = ConfigDict({'model': {'hidden_size': 256}})
        print(cfg.model.hidden_size)  # 256
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        self[name] = value

    def to_dict(self) -> dict:
        """Convert to regular dict."""
        return dict(self)


def load_config(
    config_path: str,
    defaults_path: Optional[str] = None,
) -> ConfigDict:
    """
    Load configuration from YAML file.

    Supports hierarchical configs: loads defaults first, then merges
    specific config on top.

    Args:
        config_path: Path to config YAML file
        defaults_path: Optional path to defaults file (merged first)

    Returns:
        ConfigDict with configuration values

    Example:
        cfg = load_config('configs/quadrotor.yaml')
        print(cfg.model.hidden_size)
        print(cfg.training.learning_rate)
    """
    config = {}

    # Find defaults
    if defaults_path is None:
        # Look relative to config_path or in standard location
        config_dir = Path(config_path).parent
        for candidate in [
            config_dir / "default.yaml",
            config_dir.parent / "configs" / "default.yaml",
            Path("configs") / "default.yaml",
        ]:
            if candidate.exists():
                defaults_path = str(candidate)
                break

    # Load defaults
    if defaults_path and Path(defaults_path).exists():
        with open(defaults_path, "r") as f:
            config = yaml.safe_load(f) or {}
        logger.debug(f"Loaded defaults from {defaults_path}")

    # Load and merge specific config
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            specific = yaml.safe_load(f) or {}
        config = deep_merge(config, specific)
        logger.debug(f"Loaded config from {config_path}")
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return ConfigDict(config)


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.

    Values in override take precedence. Nested dicts are merged recursively.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: dict, path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dict
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(dict(config), f, default_flow_style=False)
    logger.info(f"Config saved to {path}")


def get_device(device_str: str = "auto") -> str:
    """
    Get compute device.

    Args:
        device_str: 'auto', 'cpu', or 'cuda'

    Returns:
        Device string ('cpu' or 'cuda')
    """
    import torch

    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str
