from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PREPROCESSING_CONFIG = "preprocessing_config_template.yaml"
PREPROCESSING_CONFIG_PACKAGE = "xrd_preprocessing.configs"
REQUIRED_PREPROCESSING_CONFIG_SECTIONS = (
    "raw_data",
    "metadata",
    "filters",
    "labels",
    "one_to_many",
    "one_to_one",
    "integration",
    "snr",
    "normalization",
    "profile_gate",
)


def available_preprocessing_configs() -> list[str]:
    """Return bundled preprocessing config contract names."""
    root = files(PREPROCESSING_CONFIG_PACKAGE)
    return sorted(item.name for item in root.iterdir() if item.name.endswith(".yaml"))


def preprocessing_config_path(name: str = DEFAULT_PREPROCESSING_CONFIG) -> Path:
    """Return filesystem path to a bundled preprocessing config contract."""
    resource = files(PREPROCESSING_CONFIG_PACKAGE).joinpath(name)
    if not resource.is_file():
        raise FileNotFoundError(f"Unknown bundled preprocessing config: {name}")
    return Path(str(resource))


def load_preprocessing_config(source: str | Path | dict[str, Any] | None = None) -> dict:
    """Load a preprocessing YAML config from path, bundled name, or mapping."""
    if source is None:
        source = DEFAULT_PREPROCESSING_CONFIG
    if isinstance(source, dict):
        config = deepcopy(source)
    elif _looks_like_existing_path(source):
        config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    else:
        name = str(source)
        resource = files(PREPROCESSING_CONFIG_PACKAGE).joinpath(name)
        if not resource.is_file():
            raise FileNotFoundError(f"Preprocessing config not found: {source}")
        config = yaml.safe_load(resource.read_text(encoding="utf-8"))
    validate_preprocessing_config(config)
    return config


def validate_preprocessing_config(config: dict[str, Any]) -> None:
    """Validate the minimal preprocessing config contract structure."""
    if not isinstance(config, dict):
        raise TypeError("Preprocessing config must be a mapping.")
    missing = [
        section
        for section in REQUIRED_PREPROCESSING_CONFIG_SECTIONS
        if section not in config
    ]
    if missing:
        raise ValueError(f"Missing preprocessing config sections: {missing}")
    source = str(config["raw_data"].get("source", "")).lower()
    allowed = {str(item).lower() for item in config["raw_data"].get("allowed_sources", [])}
    if allowed and source not in allowed:
        raise ValueError(f"raw_data.source={source!r} is not in allowed_sources.")
    thickness = config["integration"].get("thickness_correction", {})
    filters = config["filters"].get("thickness", {})
    sample_filter = filters.get("sample", {})
    calibrant_filter = filters.get("calibrant", {})
    if sample_filter.get("column") not in {
        None,
        thickness.get("sample_thickness_column"),
    }:
        raise ValueError("Sample-thickness filter column differs from integration column.")
    if calibrant_filter.get("column") not in {
        None,
        thickness.get("calibrant_thickness_column"),
    }:
        raise ValueError(
            "Calibrant-thickness filter column differs from integration column."
        )


def _looks_like_existing_path(source: str | Path) -> bool:
    try:
        return Path(source).exists()
    except (OSError, TypeError, ValueError):
        return False
