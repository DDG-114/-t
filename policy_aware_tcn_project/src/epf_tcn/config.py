"""Configuration utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dirs(config: Dict[str, Any]) -> None:
    """Create output directories defined in config."""
    for key in ["model_dir", "prediction_dir", "report_dir"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["processed_features"]).parent.mkdir(parents=True, exist_ok=True)
