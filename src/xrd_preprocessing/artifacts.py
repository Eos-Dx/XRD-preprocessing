"""Joblib artifacts for preprocessing outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml


PREPROCESSING_ARTIFACT_KIND = "xrd_preprocessing_dataframe"
PREPROCESSING_ARTIFACT_VERSION = "0.1"


def build_preprocessing_artifact(
    dataframe: pd.DataFrame,
    *,
    preprocessing_config: dict[str, Any] | None = None,
    preprocessing_config_text: str | None = None,
    preprocessing_config_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a portable preprocessing artifact containing data and config."""
    config_text = preprocessing_config_text
    if config_text is None and preprocessing_config is not None:
        config_text = yaml.safe_dump(preprocessing_config, sort_keys=False)

    artifact = {
        "kind": PREPROCESSING_ARTIFACT_KIND,
        "version": PREPROCESSING_ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataframe": dataframe,
        "preprocessing_config": preprocessing_config,
        "preprocessing_config_text": config_text,
        "preprocessing_config_path": (
            str(preprocessing_config_path) if preprocessing_config_path is not None else None
        ),
        "preprocessing_config_sha256": (
            sha256(config_text.encode("utf-8")).hexdigest()
            if config_text is not None
            else None
        ),
        "metadata": dict(metadata or {}),
    }
    return artifact


def save_preprocessing_artifact(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    *,
    preprocessing_config: dict[str, Any] | None = None,
    preprocessing_config_text: str | None = None,
    preprocessing_config_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a preprocessing artifact joblib and return the written object."""
    artifact = build_preprocessing_artifact(
        dataframe,
        preprocessing_config=preprocessing_config,
        preprocessing_config_text=preprocessing_config_text,
        preprocessing_config_path=preprocessing_config_path,
        metadata=metadata,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return artifact


def load_preprocessing_artifact(path: str | Path) -> dict[str, Any]:
    """Load artifact joblib; wrap legacy DataFrame joblibs in artifact shape."""
    obj = joblib.load(path)
    if isinstance(obj, pd.DataFrame):
        return build_preprocessing_artifact(obj, metadata={"legacy_dataframe_joblib": True})
    if not isinstance(obj, dict) or "dataframe" not in obj:
        raise TypeError("Preprocessing joblib must contain a DataFrame or artifact dict.")
    return obj


def load_preprocessing_dataframe(path: str | Path) -> pd.DataFrame:
    """Load only the DataFrame from a preprocessing artifact joblib."""
    artifact = load_preprocessing_artifact(path)
    dataframe = artifact["dataframe"]
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Preprocessing artifact field 'dataframe' is not a DataFrame.")
    return dataframe
