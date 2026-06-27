"""H5 selection and reader transformers for product pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .._compat import BaseEstimator, TransformerMixin
from ..h5 import (
    H5SessionFilter,
    filter_h5_session_df,
    h5_to_df,
    list_h5_measurement_stage_sets,
    list_h5_sessions,
)


def _manifest_from_input(X: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(X, Mapping):
        return dict(X)
    return {"archive_path": Path(X)}


def _archive_path_from_manifest(manifest: Mapping[str, Any]) -> Path:
    return Path(manifest["archive_path"])


def _manifest_value(
    manifest: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    return default if default is not None else manifest.get(key)


class H5SessionSelectorTransformer(TransformerMixin, BaseEstimator):
    """Select H5 sessions before detector-frame loading."""

    def __init__(
        self,
        *,
        filters: Sequence[H5SessionFilter | dict[str, Any]] | None = None,
        session_category: str | Sequence[str] | set[str] | None = "SAMPLE",
        session_started_at_min: str | pd.Timestamp | None = None,
        max_sessions: int | None = None,
    ) -> None:
        self.filters = filters
        self.session_category = session_category
        self.session_started_at_min = session_started_at_min
        self.max_sessions = max_sessions
        self.all_session_df_: pd.DataFrame | None = None
        self.session_df_: pd.DataFrame | None = None
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: str | Path, y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: str | Path | Mapping[str, Any]) -> dict[str, Any]:
        manifest = _manifest_from_input(X)
        archive_path = _archive_path_from_manifest(manifest)
        all_session_df = (
            manifest["all_session_df"]
            if "all_session_df" in manifest
            else list_h5_sessions(archive_path)
        )
        session_df = filter_h5_session_df(
            all_session_df,
            self.filters,
            session_category=self.session_category,
            session_started_at_min=self.session_started_at_min,
            max_sessions=self.max_sessions,
        )
        self.all_session_df_ = all_session_df
        self.session_df_ = session_df
        self.stats_ = {
            "filter_type": "h5_session_selector",
            "sessions_in": int(len(all_session_df)),
            "sessions_pass": int(len(session_df)),
            "sessions_dropped": int(len(all_session_df) - len(session_df)),
            "session_category": self.session_category,
        }
        manifest.update(
            {
                "archive_path": archive_path,
                "all_session_df": all_session_df,
                "session_df": session_df,
                "h5_filters": self.filters,
                "h5_session_selector_stats": self.stats_,
            }
        )
        return manifest


class H5MeasurementSetAuditTransformer(TransformerMixin, BaseEstimator):
    """Build H5 measurement-set audit stages without detector-frame loading."""

    def __init__(
        self,
        *,
        stage_filters: Mapping[str, Sequence[H5SessionFilter | dict[str, Any]]],
        session_category: str | Sequence[str] | set[str] | None = "SAMPLE",
        set_category: str | Sequence[str] | set[str] | None = "SAMPLE",
        max_sessions_by_stage: Mapping[str, int | None] | None = None,
    ) -> None:
        self.stage_filters = dict(stage_filters)
        self.session_category = session_category
        self.set_category = set_category
        self.max_sessions_by_stage = (
            dict(max_sessions_by_stage) if max_sessions_by_stage is not None else None
        )
        self.stage_frames_: dict[str, pd.DataFrame] | None = None
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: str | Path | Mapping[str, Any], y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: str | Path | Mapping[str, Any]) -> dict[str, Any]:
        manifest = _manifest_from_input(X)
        archive_path = _archive_path_from_manifest(manifest)
        session_df = manifest.get("all_session_df")
        frames = list_h5_measurement_stage_sets(
            archive_path,
            session_df=session_df,
            stage_filters=self.stage_filters,
            session_category=self.session_category,
            set_category=self.set_category,
            max_sessions_by_stage=self.max_sessions_by_stage,
        )
        self.stage_frames_ = frames
        self.stats_ = {
            "filter_type": "h5_measurement_set_audit",
            "stages": {
                stage_name: int(len(frame)) for stage_name, frame in frames.items()
            },
        }
        manifest.update(
            {
                "h5_stage_frames": frames,
                "h5_measurement_set_audit_stats": self.stats_,
            }
        )
        return manifest


class H5ToDataFrameTransformer(TransformerMixin, BaseEstimator):
    """Read an H5 container into measurement rows as a sklearn transformer."""

    def __init__(
        self,
        *,
        data_preference: str = "gfrm",
        raw_root: str | Path | None = None,
        convert_gfrm: bool = True,
        require_clinical_ids: bool = True,
        drop_missing_sample_thickness: bool = False,
        h5_session_df: pd.DataFrame | None = None,
        h5_filters: Sequence[H5SessionFilter | dict[str, Any]] | None = None,
        measurement_filters: Sequence[H5SessionFilter | dict[str, Any]] | None = None,
        max_sessions: int | None = None,
        session_category: str | Sequence[str] | set[str] | None = None,
        session_started_at_min: str | pd.Timestamp | None = None,
        set_category: str | Sequence[str] | set[str] | None = None,
        reader: Callable[..., tuple[pd.DataFrame, pd.DataFrame]] = h5_to_df,
    ) -> None:
        self.data_preference = data_preference
        self.raw_root = raw_root
        self.convert_gfrm = convert_gfrm
        self.require_clinical_ids = require_clinical_ids
        self.drop_missing_sample_thickness = drop_missing_sample_thickness
        self.h5_session_df = h5_session_df
        self.h5_filters = h5_filters
        self.measurement_filters = measurement_filters
        self.max_sessions = max_sessions
        self.session_category = session_category
        self.session_started_at_min = session_started_at_min
        self.set_category = set_category
        self.reader = reader
        self.calibration_df_: pd.DataFrame | None = None
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: str | Path | Mapping[str, Any], y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: str | Path | Mapping[str, Any]) -> pd.DataFrame:
        archive_path = X
        manifest: Mapping[str, Any] = {}
        if isinstance(X, Mapping):
            manifest = X
            archive_path = X["archive_path"]
        calibration_df, measurement_df = self.reader(
            archive_path,
            data_preference=self.data_preference,
            raw_root=self.raw_root,
            convert_gfrm=self.convert_gfrm,
            require_clinical_ids=self.require_clinical_ids,
            drop_missing_sample_thickness=self.drop_missing_sample_thickness,
            h5_session_df=_manifest_value(
                manifest,
                "session_df",
                self.h5_session_df,
            ),
            h5_filters=_manifest_value(manifest, "h5_filters", self.h5_filters),
            measurement_filters=self.measurement_filters,
            max_sessions=_manifest_value(manifest, "max_sessions", self.max_sessions),
            session_category=self.session_category,
            session_started_at_min=self.session_started_at_min,
            set_category=self.set_category,
        )
        self.calibration_df_ = calibration_df
        self.stats_ = {
            "filter_type": "h5_to_dataframe",
            "calibration_rows": int(len(calibration_df)),
            "measurement_rows": int(len(measurement_df)),
            "dropped_missing_sample_thickness": int(
                measurement_df.attrs.get("dropped_missing_sample_thickness", 0)
            ),
        }
        return measurement_df


class H5BlobDataFrameTransformer(TransformerMixin, BaseEstimator):
    """Read a simple H5 blob layout into a measurement-level DataFrame.

    This supports small test fixtures or preprocessed 2D-array containers where
    each child group contains attrs plus a raw/processed 2D dataset.
    """

    def __init__(
        self,
        *,
        source: str = "npy",
        root_group: str = "measurements",
        dataset_candidates: Sequence[str] = ("raw/data", "processed/data"),
    ) -> None:
        self.source = str(source).lower()
        self.root_group = root_group
        self.dataset_candidates = tuple(dataset_candidates)
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: str | Path, y: Any = None):
        _ = X
        _ = y
        return self

    @staticmethod
    def _decode(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _read_dataset(self, group: h5py.Group) -> tuple[np.ndarray, str]:
        for candidate in self.dataset_candidates:
            if candidate not in group:
                continue
            dataset = group[candidate]
            data = np.asarray(dataset)
            if data.dtype.kind in {"u", "i", "f"} and data.ndim == 2:
                return data.astype(float), f"{self.source}:{candidate}"
            if self.source == "npy":
                return np.load(BytesIO(data.tobytes())).astype(float), (
                    f"{self.source}:{candidate}"
                )
            raise ValueError(
                f"Dataset '{candidate}' for source '{self.source}' is not a "
                "numeric 2D array."
            )
        raise ValueError(f"No H5 dataset found for source={self.source!r}.")

    def transform(self, X: str | Path) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        with h5py.File(X, "r") as h5:
            root = h5[self.root_group] if self.root_group in h5 else h5
            for name in sorted(root):
                group = root[name]
                if not isinstance(group, h5py.Group):
                    continue
                row = {key: self._decode(value) for key, value in group.attrs.items()}
                row.setdefault("id", name)
                row.setdefault("meas_name", name)
                row.setdefault("set_name", name)
                row["measurement_data"], row["measurement_data_source"] = (
                    self._read_dataset(group)
                )
                rows.append(row)
        frame = pd.DataFrame(rows)
        for column in (
            "sample_thickness_mm",
            "calibrant_thickness_mm",
            "poni_q_max_nm_inv",
        ):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        self.stats_ = {
            "filter_type": "h5_blob_to_dataframe",
            "rows": int(len(frame)),
            "source": self.source,
            "root_group": self.root_group,
        }
        return frame
