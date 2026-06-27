"""Eos-Dx H5 container metadata selection and DataFrame loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd

from .azimuthal import estimate_poni_q_range_nm_inv
from .gfrm import gfrm_to_photons


PROMOTED_ARCHIVE_GROUP_FIELDS = (
    "calibrant_thickness_mm",
    "calibrant_thickness_source",
    "calibrant_thickness_rule",
    "calibrant_thickness_effective_date",
    "calibrant_thickness_backfilled_at",
    "kbeta_absent",
    "xray_spectrum",
    "product_batch_usable",
    "product_batch_id",
    "human1_data_batch",
    "product_protocol_version",
)

CALIBRANT_THICKNESS_COLUMN = "calibrant_thickness_mm"
CALIBRANT_THICKNESS_MIN_MM = 10.0
CALIBRANT_THICKNESS_MAX_MM = 40.0


@dataclass(frozen=True)
class H5SessionFilter:
    """Filter H5 session metadata before frame loading and DataFrame creation."""

    column: str
    op: str = "=="
    value: Any = None
    values: Sequence[Any] | set[Any] | None = None
    lower: Any = None
    upper: Any = None


def calibrant_thickness_h5_filters(
    *,
    column: str = CALIBRANT_THICKNESS_COLUMN,
    min_mm: float = CALIBRANT_THICKNESS_MIN_MM,
    max_mm: float = CALIBRANT_THICKNESS_MAX_MM,
) -> list[H5SessionFilter]:
    """H5-level filters for calibrant thickness metadata before frame loading."""
    return [
        H5SessionFilter(column=column, op="notna"),
        H5SessionFilter(column=column, op=">=", value=float(min_mm)),
        H5SessionFilter(column=column, op="<=", value=float(max_mm)),
    ]


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_decode(v) for v in value.tolist()]
    return value


def _attrs(obj: h5py.Group | h5py.Dataset, prefix: str = "") -> dict[str, Any]:
    return {f"{prefix}{key}": _decode(value) for key, value in obj.attrs.items()}


def _json_dataset(group: h5py.Group, name: str, default: Any = None) -> Any:
    if name not in group:
        return default
    return json.loads(_decode(group[name][()]))


def _scalar_fields(group: h5py.Group, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, obj in group.items():
        if isinstance(obj, h5py.Dataset) and obj.shape == ():
            out[f"{prefix}{name}"] = _decode(obj[()])
    return out


def _first_child_by_prefix(group: h5py.Group | None, prefix: str):
    if group is None:
        return None
    for name in sorted(group):
        if name.startswith(prefix):
            return group[name]
    return None


def _set_index(set_name: str) -> int:
    parts = set_name.split("_", 2)
    if len(parts) < 2 or parts[0] != "set":
        raise ValueError(f"Cannot parse set index from {set_name!r}")
    return int(parts[1])


def _allowed_values(values: str | list[str] | tuple[str, ...] | set[str] | None):
    if values is None:
        return None
    if isinstance(values, str):
        return {values.upper()}
    return {str(value).upper() for value in values}


def _matches_allowed(value: Any, allowed: set[str] | None) -> bool:
    if allowed is None:
        return True
    return str(value or "").upper() in allowed


def _timestamp(value: Any) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp


def _matches_min_timestamp(value: Any, minimum: Any | None) -> bool:
    if minimum is None:
        return True
    timestamp = _timestamp(value)
    min_timestamp = _timestamp(minimum)
    if timestamp is None or min_timestamp is None:
        return False
    return timestamp >= min_timestamp


def _positive_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def _coerce_filter(spec: H5SessionFilter | dict[str, Any]) -> H5SessionFilter:
    if isinstance(spec, H5SessionFilter):
        return spec
    if isinstance(spec, dict):
        return H5SessionFilter(**spec)
    raise TypeError(f"Expected H5SessionFilter or dict, got {type(spec).__name__}.")


def _sequence_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, set | tuple | list):
        return list(value)
    return [value]


def _filter_values(filter_spec: H5SessionFilter) -> list[Any]:
    if filter_spec.values is not None:
        return list(filter_spec.values)
    return _sequence_values(filter_spec.value)


def _date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def _date_values(values: Sequence[Any]) -> set[Any]:
    timestamps = pd.to_datetime(list(values), errors="coerce")
    if pd.isna(timestamps).any():
        invalid = [
            value
            for value, timestamp in zip(values, timestamps, strict=False)
            if pd.isna(timestamp)
        ]
        raise ValueError(f"Invalid H5 date filter values: {invalid!r}.")
    return set(timestamps.date)


def _filter_mask(df: pd.DataFrame, filter_spec: H5SessionFilter) -> pd.Series:
    if filter_spec.column not in df.columns:
        raise ValueError(f"H5 session metadata is missing column {filter_spec.column!r}.")

    series = df[filter_spec.column]
    op = filter_spec.op.lower()
    if op in {"==", "eq"}:
        return series == filter_spec.value
    if op in {"!=", "ne"}:
        return series != filter_spec.value
    if op == "in":
        return series.isin(_filter_values(filter_spec))
    if op in {"not in", "not_in"}:
        return ~series.isin(_filter_values(filter_spec))
    if op == "contains":
        return series.astype(str).str.contains(str(filter_spec.value), na=False)
    if op == "startswith":
        return series.astype(str).str.startswith(str(filter_spec.value), na=False)
    if op == "endswith":
        return series.astype(str).str.endswith(str(filter_spec.value), na=False)
    if op == "isna":
        return series.isna()
    if op == "notna":
        return series.notna()
    if op in {">", ">=", "<", "<="}:
        if op == ">":
            return series > filter_spec.value
        if op == ">=":
            return series >= filter_spec.value
        if op == "<":
            return series < filter_spec.value
        return series <= filter_spec.value

    if op in {"date>", "date>=", "date<", "date<="}:
        timestamps = pd.to_datetime(series, errors="coerce")
        threshold = pd.to_datetime(filter_spec.value, errors="coerce")
        if pd.isna(threshold):
            raise ValueError(f"Invalid H5 date filter value: {filter_spec.value!r}.")
        if op == "date>":
            return timestamps > threshold
        if op == "date>=":
            return timestamps >= threshold
        if op == "date<":
            return timestamps < threshold
        return timestamps <= threshold

    if op in {"date in", "date_in", "date not in", "date_not_in"}:
        values = _filter_values(filter_spec)
        if not values:
            raise ValueError(f"{filter_spec.op!r} requires value or values.")
        allowed_dates = _date_values(values)
        mask = _date_series(series).isin(allowed_dates)
        if op in {"date not in", "date_not_in"}:
            return ~mask
        return mask

    if op in {"between", "date_between"}:
        lower = filter_spec.lower
        upper = filter_spec.upper
        if lower is None or upper is None:
            values = _sequence_values(filter_spec.value)
            if len(values) != 2:
                raise ValueError(
                    f"{filter_spec.op!r} requires lower/upper or two values."
                )
            lower, upper = values
        if op == "date_between":
            values = pd.to_datetime(series, errors="coerce")
            lower = pd.to_datetime(lower, errors="coerce")
            upper = pd.to_datetime(upper, errors="coerce")
        else:
            values = series
        return (values >= lower) & (values <= upper)

    raise ValueError(f"Unsupported H5 session filter op: {filter_spec.op!r}.")


def _row_matches_filters(
    row: dict[str, Any],
    filters: Sequence[H5SessionFilter | dict[str, Any]] | None,
) -> bool:
    if not filters:
        return True
    frame = pd.DataFrame([row])
    for filter_spec in [_coerce_filter(spec) for spec in filters]:
        if not bool(_filter_mask(frame, filter_spec).iloc[0]):
            return False
    return True


def _set_category(set_group: h5py.Group) -> str:
    return str(
        _decode(
            set_group.attrs.get(
                "measurement_type_category",
                set_group.attrs.get("category", ""),
            )
        )
        or ""
    ).upper()


def _set_sample_thickness_mm(set_group: h5py.Group) -> float | None:
    for key in ("sample_thickness_mm", "sample_thickness", "thickness_raw_mm"):
        value = _positive_float(set_group.attrs.get(key))
        if value is not None:
            return value
    acquisition = set_group.get("acquisition")
    if isinstance(acquisition, h5py.Group):
        acquisition_fields = _scalar_fields(acquisition)
        for key in ("sample_thickness_mm", "sample_thickness", "thickness_raw_mm"):
            value = _positive_float(acquisition_fields.get(key))
            if value is not None:
                return value
    metadata = _json_dataset(set_group, "metadata", default={})
    if isinstance(metadata, dict):
        for key in ("sample_thickness_mm", "sample_thickness", "thickness_raw_mm"):
            value = _positive_float(metadata.get(key))
            if value is not None:
                return value
    return None


def _session_sample_set_summary(session: h5py.Group) -> dict[str, Any]:
    sets = session.get("sets")
    if not isinstance(sets, h5py.Group):
        return {}

    q_min_values = []
    q_max_values = []
    distance_values = []
    sample_set_count = 0
    sample_poni_count = 0
    sample_thickness_count = 0
    sample_thickness_values = []
    for set_name in sorted(sets):
        set_group = sets[set_name]
        if not isinstance(set_group, h5py.Group):
            continue
        category = _set_category(set_group)
        if category and category != "SAMPLE":
            continue
        sample_set_count += 1
        sample_thickness = _set_sample_thickness_mm(set_group)
        if sample_thickness is not None:
            sample_thickness_count += 1
            sample_thickness_values.append(sample_thickness)
        poni_text = _text_dataset(set_group, "artifacts/poni")
        if poni_text is None:
            continue
        sample_poni_count += 1
        try:
            q_min, q_max, distance_m = estimate_poni_q_range_nm_inv(str(poni_text))
        except Exception:
            q_min, q_max, distance_m = np.nan, np.nan, np.nan
        q_min_values.append(q_min)
        q_max_values.append(q_max)
        distance_values.append(distance_m)

    finite_q_min = [value for value in q_min_values if np.isfinite(value)]
    finite_q_max = [value for value in q_max_values if np.isfinite(value)]
    finite_distance = [value for value in distance_values if np.isfinite(value)]
    finite_thickness = [
        value for value in sample_thickness_values if np.isfinite(value)
    ]
    return {
        "h5_sample_set_count": int(sample_set_count),
        "h5_sample_poni_count": int(sample_poni_count),
        "h5_sample_all_sets_have_poni": bool(
            sample_set_count > 0 and sample_set_count == sample_poni_count
        ),
        "h5_sample_thickness_count": int(sample_thickness_count),
        "h5_sample_all_sets_have_thickness": bool(
            sample_set_count > 0 and sample_set_count == sample_thickness_count
        ),
        "sample_thickness_mm_min": (
            float(np.nanmin(finite_thickness)) if finite_thickness else np.nan
        ),
        "sample_thickness_mm_max": (
            float(np.nanmax(finite_thickness)) if finite_thickness else np.nan
        ),
        "poni_q_min_nm_inv": float(np.nanmin(finite_q_min)) if finite_q_min else np.nan,
        "poni_q_max_nm_inv": float(np.nanmin(finite_q_max)) if finite_q_max else np.nan,
        "poni_q_max_nm_inv_max": (
            float(np.nanmax(finite_q_max)) if finite_q_max else np.nan
        ),
        "poni_calculated_distance_m": (
            float(np.nanmax(finite_distance)) if finite_distance else np.nan
        ),
    }


def _session_metadata_row(
    *,
    file_path: Path,
    container_attrs: dict[str, Any],
    session: h5py.Group,
    archive_group_name: str | None = None,
    archive_group_attrs: dict[str, Any] | None = None,
    archive_session_name: str | None = None,
) -> dict[str, Any]:
    row = {
        "source_file": str(file_path),
        "schema_version": container_attrs.get("schema_version"),
        "container_format": container_attrs.get("format"),
        "session_path": session.name,
    }
    row.update(_attrs(session))
    row.setdefault("format", row["container_format"])
    sample = session.get("sample")
    if isinstance(sample, h5py.Group):
        sample_attrs = _attrs(sample)
        for key, value in sample_attrs.items():
            row[f"sample_{key}"] = value
        for key in (
            "additional_info",
            "age",
            "biopsy",
            "birads",
            "birads_category",
            "breast_density",
            "mri",
            "race_ethnicity",
            "side",
            "specimen_status",
        ):
            if key in sample_attrs:
                row[key] = sample_attrs[key]
        patient_name = _text_dataset(sample, "patient_name")
        specimen_name = _text_dataset(sample, "name")
        sample_type = _text_dataset(sample, "sample_type")
        if patient_name is not None:
            row["patientId"] = patient_name
            row["patient_name"] = patient_name
        if specimen_name is not None:
            row["specimenId"] = specimen_name
            row["name"] = specimen_name
        if sample_type is not None:
            row["sample_type"] = sample_type

    if archive_group_name is not None:
        row["archive_group"] = archive_group_name
        row["archive_session_name"] = archive_session_name
        row["archive_session_path"] = session.name
        if archive_group_attrs:
            row["calibration_session_uid"] = archive_group_attrs.get(
                "calibration_session_uid"
            )
            for key, value in archive_group_attrs.items():
                row[f"archive_group_{key}"] = value
                if key in PROMOTED_ARCHIVE_GROUP_FIELDS:
                    row[key] = value
    else:
        row["archive_session_path"] = session.name

    row["session_category"] = row.get("category")
    row.update(_session_sample_set_summary(session))
    return row


def list_h5_sessions(file_path: str | Path) -> pd.DataFrame:
    """List H5 session attrs without reading detector frames."""
    file_path = Path(file_path)
    stat = file_path.stat()
    return _list_h5_sessions_cached(
        str(file_path),
        stat.st_mtime_ns,
        stat.st_size,
    ).copy(deep=True)


@lru_cache(maxsize=4)
def _list_h5_sessions_cached(
    file_path: str,
    mtime_ns: int,
    size_bytes: int,
) -> pd.DataFrame:
    _ = mtime_ns
    _ = size_bytes
    file_path = Path(file_path)
    rows: list[dict[str, Any]] = []
    with h5py.File(file_path, "r") as h5:
        container_attrs = _attrs(h5)
        version = container_attrs.get("schema_version")
        fmt = container_attrs.get("format")
        if version != "0.3" or fmt not in {"xrd-session", "xrd-session-archive"}:
            raise ValueError(
                f"Unsupported container format: schema_version={version!r}, format={fmt!r}"
            )
        if fmt == "xrd-session":
            session = h5.get("session")
            if isinstance(session, h5py.Group):
                rows.append(
                    _session_metadata_row(
                        file_path=file_path,
                        container_attrs=container_attrs,
                        session=session,
                    )
                )
        else:
            for archive_group_name in sorted(h5):
                archive_group = h5[archive_group_name]
                if not isinstance(archive_group, h5py.Group):
                    continue
                archive_group_attrs = _attrs(archive_group)
                for session_name in sorted(archive_group):
                    session = archive_group[session_name]
                    if not isinstance(session, h5py.Group):
                        continue
                    rows.append(
                        _session_metadata_row(
                            file_path=file_path,
                            container_attrs=container_attrs,
                            session=session,
                            archive_group_name=archive_group_name,
                            archive_group_attrs=archive_group_attrs,
                            archive_session_name=session_name,
                        )
                    )
    return pd.DataFrame(rows)


def filter_h5_sessions(
    file_path: str | Path,
    filters: Sequence[H5SessionFilter | dict[str, Any]] | None = None,
    *,
    session_category: str | list[str] | tuple[str, ...] | set[str] | None = None,
    session_started_at_min: str | pd.Timestamp | None = None,
    max_sessions: int | None = None,
) -> pd.DataFrame:
    """Filter H5 session attrs before frame loading and DataFrame creation."""
    session_df = list_h5_sessions(file_path)
    return filter_h5_session_df(
        session_df,
        filters=filters,
        session_category=session_category,
        session_started_at_min=session_started_at_min,
        max_sessions=max_sessions,
    )


def filter_h5_session_df(
    session_df: pd.DataFrame,
    filters: Sequence[H5SessionFilter | dict[str, Any]] | None = None,
    *,
    session_category: str | list[str] | tuple[str, ...] | set[str] | None = None,
    session_started_at_min: str | pd.Timestamp | None = None,
    max_sessions: int | None = None,
) -> pd.DataFrame:
    """Filter an already loaded H5 session metadata table."""
    session_df = session_df.copy()
    filter_specs = [_coerce_filter(spec) for spec in filters or []]

    if session_category is not None:
        filter_specs.append(
            H5SessionFilter(
                "category",
                op="in",
                values=list(_allowed_values(session_category) or []),
            )
        )
        if "category" in session_df.columns:
            session_df = session_df.assign(
                category=session_df["category"].astype(str).str.upper()
            )
    if session_started_at_min is not None:
        filter_specs.append(
            H5SessionFilter(
                "started_at",
                op="date>=",
                value=session_started_at_min,
            )
        )

    for filter_spec in filter_specs:
        session_df = session_df.loc[_filter_mask(session_df, filter_spec)].copy()
    if max_sessions is not None:
        session_df = session_df.head(max_sessions).copy()
    return session_df.reset_index(drop=True)


def list_h5_measurement_stage_sets(
    file_path: str | Path,
    *,
    stage_filters: dict[str, Sequence[H5SessionFilter | dict[str, Any]]],
    session_df: pd.DataFrame | None = None,
    session_category: str | list[str] | tuple[str, ...] | set[str] | None = "SAMPLE",
    set_category: str | list[str] | tuple[str, ...] | set[str] | None = "SAMPLE",
    max_sessions_by_stage: dict[str, int | None] | None = None,
) -> dict[str, pd.DataFrame]:
    """List measurement-set metadata once, then split it into filter stages."""
    if session_df is None:
        session_df = list_h5_sessions(file_path)
    base_sessions = filter_h5_session_df(session_df, session_category=session_category)
    all_sets = list_h5_measurement_sets(
        file_path,
        session_df=base_sessions,
        set_category=set_category,
    )
    max_sessions_by_stage = max_sessions_by_stage or {}
    frames: dict[str, pd.DataFrame] = {}
    for stage_name, filters in stage_filters.items():
        stage_sessions = filter_h5_session_df(
            base_sessions,
            filters=filters,
            max_sessions=max_sessions_by_stage.get(stage_name),
        )
        if "session_path" not in stage_sessions.columns or "session_path" not in all_sets:
            frames[stage_name] = all_sets.iloc[0:0].copy()
            continue
        paths = set(stage_sessions["session_path"].astype(str))
        frames[stage_name] = all_sets.loc[
            all_sets["session_path"].astype(str).isin(paths)
        ].copy()
    return frames


def list_h5_measurement_sets(
    file_path: str | Path,
    *,
    session_df: pd.DataFrame | None = None,
    session_filters: Sequence[H5SessionFilter | dict[str, Any]] | None = None,
    session_category: str | list[str] | tuple[str, ...] | set[str] | None = None,
    session_started_at_min: str | pd.Timestamp | None = None,
    set_category: str | list[str] | tuple[str, ...] | set[str] | None = "SAMPLE",
    drop_missing_sample_thickness: bool = False,
    max_sessions: int | None = None,
) -> pd.DataFrame:
    """List H5 measurement-set metadata without reading detector frames.

    Returned rows are one row per session set. No count is multiplied by
    ``h5_sample_set_count``; measurement statistics should use ``len(df)``.
    """
    file_path = Path(file_path)
    if session_df is None:
        session_df = filter_h5_sessions(
            file_path,
            filters=session_filters,
            session_category=session_category,
            session_started_at_min=session_started_at_min,
            max_sessions=max_sessions,
        )
    elif max_sessions is not None:
        session_df = session_df.head(max_sessions).copy()

    rows: list[dict[str, Any]] = []
    set_allowed = _allowed_values(set_category)
    with h5py.File(file_path, "r") as h5:
        for _, session_row in session_df.iterrows():
            session_path = session_row.get("session_path")
            if not isinstance(session_path, str) or session_path not in h5:
                continue
            session = h5[session_path]
            if not isinstance(session, h5py.Group):
                continue
            sets = session.get("sets")
            if not isinstance(sets, h5py.Group):
                continue

            session_meta = session_row.to_dict()
            for set_name in sorted(sets):
                set_group = sets[set_name]
                if not isinstance(set_group, h5py.Group):
                    continue
                category = _set_category(set_group)
                if not _matches_allowed(category, set_allowed):
                    continue

                row = {
                    **session_meta,
                    "id": set_name,
                    "meas_name": set_name,
                    "set_name": set_name,
                    "set_path": set_group.name,
                    "measurement_type_category": category,
                    **_attrs(set_group),
                }
                acquisition = set_group.get("acquisition")
                if isinstance(acquisition, h5py.Group):
                    row.update(_scalar_fields(acquisition))
                _standardize_clinical_ids(row)

                sample_thickness = _set_sample_thickness_mm(set_group)
                row["sample_thickness_mm"] = sample_thickness
                if drop_missing_sample_thickness and sample_thickness is None:
                    continue

                poni_text = _text_dataset(set_group, "artifacts/poni")
                if poni_text is not None:
                    row["ponifile"] = poni_text
                    try:
                        q_min, q_max, distance_m = estimate_poni_q_range_nm_inv(
                            str(poni_text)
                        )
                    except Exception:
                        q_min, q_max, distance_m = np.nan, np.nan, np.nan
                    row["poni_q_min_nm_inv"] = q_min
                    row["poni_q_max_nm_inv"] = q_max
                    row["poni_calculated_distance_m"] = distance_m
                rows.append(row)

    return pd.DataFrame(rows)


def filter_h5_measurement_sets(
    file_path: str | Path,
    session_filters: Sequence[H5SessionFilter | dict[str, Any]] | None = None,
    *,
    measurement_filters: Sequence[H5SessionFilter | dict[str, Any]] | None = None,
    session_category: str | list[str] | tuple[str, ...] | set[str] | None = None,
    session_started_at_min: str | pd.Timestamp | None = None,
    set_category: str | list[str] | tuple[str, ...] | set[str] | None = "SAMPLE",
    drop_missing_sample_thickness: bool = False,
    max_sessions: int | None = None,
) -> pd.DataFrame:
    """Filter H5 metadata and return one row per passing measurement set."""
    measurement_df = list_h5_measurement_sets(
        file_path,
        session_filters=session_filters,
        session_category=session_category,
        session_started_at_min=session_started_at_min,
        set_category=set_category,
        drop_missing_sample_thickness=drop_missing_sample_thickness,
        max_sessions=max_sessions,
    )
    for filter_spec in [_coerce_filter(spec) for spec in measurement_filters or []]:
        measurement_df = measurement_df.loc[
            _filter_mask(measurement_df, filter_spec)
        ].copy()
    return measurement_df.reset_index(drop=True)


def h5_measurement_set_counts(
    df: pd.DataFrame,
    *,
    diagnosis_column: str = "specimen_status",
) -> dict[str, Any]:
    """Count measurement-set rows, patients, specimens, and diagnosis values."""
    diagnosis = (
        df[diagnosis_column].fillna("NA").astype(str).str.strip().str.upper()
        if diagnosis_column in df.columns
        else pd.Series([], dtype="object")
    )
    diagnosis = diagnosis.replace("", "NA")
    return {
        "measurements": int(len(df)),
        "patients": int(df["patientId"].nunique()) if "patientId" in df else 0,
        "specimens": int(df["specimenId"].nunique()) if "specimenId" in df else 0,
        "diagnosis_variants": int(diagnosis.nunique()),
        "diagnosis_counts": diagnosis.value_counts().to_dict(),
    }


def h5_filter_statistics(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    *,
    diagnosis_column: str = "specimen_status",
) -> dict[str, Any]:
    """Return before/after measurement-set counts for one H5 filter step."""
    before = h5_measurement_set_counts(
        before_df,
        diagnosis_column=diagnosis_column,
    )
    after = h5_measurement_set_counts(
        after_df,
        diagnosis_column=diagnosis_column,
    )
    return {
        "before": before,
        "after": after,
        "dropped": {
            key: before[key] - after[key]
            for key in ("measurements", "patients", "specimens", "diagnosis_variants")
        },
    }


def _dataset(group: h5py.Group, rel_path: str) -> np.ndarray | None:
    if rel_path not in group:
        return None
    obj = group[rel_path]
    if isinstance(obj, h5py.Dataset):
        return obj[()]
    return None


def _text_dataset(group: h5py.Group, rel_path: str) -> str | None:
    data = _dataset(group, rel_path)
    if data is None:
        return None
    return _decode(data)


def _open_container_reader(file_path: Path):
    try:
        from container import open_container
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "eosdx-container is required for v0.3 H5 reading. "
            "Install /Users/sad/dev/container into the active environment."
        ) from exc
    return open_container(file_path, validate=False)


def _detector_catalog(session: h5py.Group) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    root = session.get("instrument/detector_sets")
    if root is None:
        return out
    for detector_set_name in sorted(root):
        detector_set = root[detector_set_name]
        detectors = detector_set.get("detectors")
        if detectors is None:
            continue
        for detector_name in sorted(detectors):
            det = detectors[detector_name]
            det_id = int(det.attrs["detector_id"])
            out[det_id] = {
                "detector_name": detector_name,
                "detector_set_name": detector_set_name,
                **_attrs(det),
                **_scalar_fields(det),
            }
    return out


def _measurement_payloads(
    set_group: h5py.Group,
    detector_catalog: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    measurements = set_group.get("measurements")
    if measurements is None:
        return []

    rows = []
    for name in sorted(measurements):
        meas = measurements[name]
        if not isinstance(meas, h5py.Group):
            continue
        detector_id = int(meas.attrs.get("detector_id", -1))
        row = {
            "measurement_name": name,
            **_attrs(meas),
            **detector_catalog.get(detector_id, {}),
        }
        data = _dataset(meas, "data")
        if data is not None:
            row["detector_data"] = data
        mask = _dataset(meas, "mask")
        if mask is not None:
            row["detector_mask"] = mask
        rows.append(row)
    return rows


def _resolve_file_path(
    value: str | None,
    *,
    container_path: Path,
    raw_root: str | Path | None,
) -> Path | None:
    if not value:
        return None

    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    roots = []
    if raw_root is not None:
        roots.append(Path(raw_root))
    roots.append(container_path.parent)

    for root in roots:
        direct = root / candidate
        if direct.exists():
            return direct
        by_name = next(root.rglob(candidate.name), None) if root.exists() else None
        if by_name is not None and by_name.exists():
            return by_name
    return None


def _first_gfrm_path_from_measurements(
    measurements: list[dict[str, Any]],
    *,
    container_path: Path,
    raw_root: str | Path | None,
) -> Path | None:
    for measurement in measurements:
        source = measurement.get("file_path")
        if source and str(source).lower().endswith(".gfrm"):
            resolved = _resolve_file_path(
                str(source),
                container_path=container_path,
                raw_root=raw_root,
            )
            if resolved is not None:
                return resolved
    return None


def _first_gfrm_path(
    set_group: h5py.Group,
    *,
    container_path: Path,
    raw_root: str | Path | None,
) -> Path | None:
    measurements = set_group.get("measurements")
    if measurements is None:
        return None
    for name in sorted(measurements):
        meas = measurements[name]
        if not isinstance(meas, h5py.Group):
            continue
        source = _decode(meas.attrs.get("file_path"))
        if source and str(source).lower().endswith(".gfrm"):
            resolved = _resolve_file_path(
                str(source),
                container_path=container_path,
                raw_root=raw_root,
            )
            if resolved is not None:
                return resolved
    return None


def _set_dataset_from_file(
    file_path: Path,
    set_idx: int,
    rel_path: str,
) -> np.ndarray | str | None:
    with h5py.File(file_path, "r") as h5:
        session = h5.get("session")
        if session is None:
            return None
        sets = session.get("sets")
        set_group = _first_child_by_prefix(sets, f"set_{set_idx:03d}_")
        if set_group is None or rel_path not in set_group:
            return None
        obj = set_group[rel_path]
        if isinstance(obj, h5py.Dataset):
            return _decode(obj[()])
        return None


def _first_measurement_dataset_from_file(
    file_path: Path,
    set_idx: int,
    dataset_name: str,
) -> np.ndarray | str | None:
    with h5py.File(file_path, "r") as h5:
        session = h5.get("session")
        if session is None:
            return None
        sets = session.get("sets")
        set_group = _first_child_by_prefix(sets, f"set_{set_idx:03d}_")
        if set_group is None:
            return None
        measurements = set_group.get("measurements")
        if measurements is None:
            return None
        for measurement_name in sorted(measurements):
            measurement = measurements[measurement_name]
            if dataset_name in measurement and isinstance(
                measurement[dataset_name], h5py.Dataset
            ):
                return _decode(measurement[dataset_name][()])
    return None


def _json_set_dataset_from_file(
    file_path: Path,
    set_idx: int,
    rel_path: str,
    default: Any = None,
) -> Any:
    data = _set_dataset_from_file(file_path, set_idx, rel_path)
    if data is None:
        return default
    return json.loads(data)


def _gfrm_to_photons_from_embedded_raw_file(
    file_path: Path,
    set_idx: int,
    source_path: str | None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    raw_file = _first_measurement_dataset_from_file(file_path, set_idx, "raw_file")
    if raw_file is None:
        return None
    raw_bytes = np.asarray(raw_file, dtype=np.uint8).tobytes()
    with tempfile.NamedTemporaryFile(suffix=".gfrm") as handle:
        handle.write(raw_bytes)
        handle.flush()
        photons, metadata = gfrm_to_photons(handle.name)
    metadata.update(
        {
            "source": "embedded measurement raw_file",
            "source_file": source_path,
            "source_path": source_path,
            "embedded_raw_file": True,
        }
    )
    return photons, metadata


def _archive_sessions(
    file_path: Path,
    *,
    h5_filters: Sequence[H5SessionFilter | dict[str, Any]] | None,
    h5_session_df: pd.DataFrame | None,
    session_category: str | list[str] | tuple[str, ...] | set[str] | None,
    session_started_at_min: str | pd.Timestamp | None,
    max_sessions: int | None,
) -> list[dict[str, Any]]:
    if h5_session_df is not None:
        return filter_h5_session_df(
            h5_session_df,
            h5_filters,
            session_category=session_category,
            session_started_at_min=session_started_at_min,
            max_sessions=max_sessions,
        ).to_dict("records")
    return filter_h5_sessions(
        file_path,
        h5_filters,
        session_category=session_category,
        session_started_at_min=session_started_at_min,
        max_sessions=max_sessions,
    ).to_dict("records")


def _copy_archive_session_to_file(
    archive_path: Path,
    session_path: str,
    target_path: Path,
) -> None:
    with h5py.File(archive_path, "r") as src, h5py.File(target_path, "w") as dst:
        session = src[session_path]
        src.copy(session, dst, name="session")
        for key, value in session.attrs.items():
            dst.attrs[key] = value


def _set_row(
    file_path: Path,
    root_attrs: dict[str, Any],
    session_meta: dict[str, Any],
    set_name: str,
    set_group: h5py.Group,
    detector_catalog: dict[int, dict[str, Any]],
    data_preference: str,
    raw_root: str | Path | None,
    convert_gfrm: bool,
) -> dict[str, Any]:
    acq = set_group.get("acquisition")
    row = {
        "source_file": str(file_path),
        "id": set_name,
        "meas_name": set_name,
        **root_attrs,
        **session_meta,
        **_attrs(set_group),
    }
    if acq is not None:
        row.update(_scalar_fields(acq))

    raw = _dataset(set_group, "raw/data")
    processed = _dataset(set_group, "processed/data")
    if data_preference != "gfrm":
        raise ValueError("Product h5_to_df requires data_preference='gfrm'.")
    if not convert_gfrm:
        raise ValueError("Product h5_to_df requires convert_gfrm=True.")

    gfrm_path = _first_gfrm_path(
        set_group,
        container_path=file_path,
        raw_root=raw_root,
    )
    if gfrm_path is None:
        raise FileNotFoundError(
            f"Missing required RAW GFRM artifact for container set '{set_name}'."
        )
    gfrm_data, gfrm_metadata = gfrm_to_photons(gfrm_path)

    row["measurement_data"] = gfrm_data
    row["measurement_data_source"] = "gfrm_to_photons"
    row["raw_data"] = raw
    row["processed_data"] = processed
    row["gfrm_data"] = gfrm_data
    row["gfrm_path"] = str(gfrm_path) if gfrm_path is not None else None
    row["gfrm_conversion_metadata"] = gfrm_metadata

    poni_text = _text_dataset(set_group, "artifacts/poni")
    if poni_text is not None:
        row["ponifile"] = poni_text

    if "integration" in set_group:
        integration = set_group["integration"]
        if "q" in integration:
            row["q_range"] = integration["q"][()]
        if "i" in integration:
            row["radial_profile_data"] = integration["i"][()]
        if "sigma" in integration:
            row["radial_profile_sigma"] = integration["sigma"][()]
        row.update(_attrs(integration, prefix="integration_"))

    row["metadata"] = _json_dataset(set_group, "metadata", default={})
    processing = set_group.get("processing")
    if processing is not None:
        row["processing_config"] = _json_dataset(processing, "config", default={})
    else:
        row["processing_config"] = {}
    row["detector_measurements"] = _measurement_payloads(set_group, detector_catalog)
    return row


def _standardize_clinical_ids(row: dict[str, Any]) -> None:
    if "patientId" not in row:
        row["patientId"] = (
            row.get("patient_clinical_name")
            or row.get("patient_name")
            or row.get("sample_patient_name")
            or row.get("patient_id")
            or row.get("patientID")
        )
    if "specimenId" not in row:
        row["specimenId"] = (
            row.get("sample_clinical_name")
            or row.get("name")
            or row.get("clinical_sample_name")
            or row.get("specimen_id")
            or row.get("specimenID")
        )
    for thickness_key in (
        "sample_thickness_mm",
        "sample_thickness",
        "thickness_raw_mm",
    ):
        sample_thickness_mm = _positive_float(row.get(thickness_key))
        if sample_thickness_mm is not None:
            row["sample_thickness_mm"] = sample_thickness_mm
            break
    else:
        row["sample_thickness_mm"] = np.nan


def _has_valid_sample_thickness(row: dict[str, Any]) -> bool:
    return _positive_float(row.get("sample_thickness_mm")) is not None


def _validate_clinical_ids(df: pd.DataFrame, *, frame_name: str) -> None:
    required = ["patientId", "specimenId"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")
    invalid = []
    for column in required:
        values = df[column]
        bad = values.isna() | values.astype(str).str.strip().eq("")
        if bool(bad.any()):
            invalid.append(column)
    if invalid:
        raise ValueError(f"{frame_name} has empty required clinical IDs: {invalid}")


def _rows_from_container_reader(
    file_path: Path,
    *,
    data_preference: str,
    raw_root: str | Path | None,
    convert_gfrm: bool,
    session_started_at_min: str | pd.Timestamp | None,
    set_category: str | list[str] | tuple[str, ...] | set[str] | None,
    measurement_filters: Sequence[H5SessionFilter | dict[str, Any]] | None = None,
    archive_meta: dict[str, Any] | None = None,
    drop_missing_sample_thickness: bool = False,
    drop_stats: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    container = _open_container_reader(file_path)
    session_meta = container.session_meta()
    if not _matches_min_timestamp(
        session_meta.get("started_at"),
        session_started_at_min,
    ):
        return [], []
    detector_catalog = {
        int(detector["detector_id"]): detector
        for detector in container.detectors()
        if "detector_id" in detector
    }
    set_allowed = _allowed_values(set_category)
    calibration_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []

    for set_info in container.sets():
        set_name = set_info["name"]
        set_idx = _set_index(set_name)
        measurement_type_category = set_info.get("measurement_type_category")
        if not _matches_allowed(measurement_type_category, set_allowed):
            continue

        row = {
            "source_file": str(file_path),
            "id": set_name,
            "meas_name": set_name,
            "set_name": set_name,
            **session_meta,
            **{key: value for key, value in set_info.items() if key != "name"},
        }
        if archive_meta:
            row.update(archive_meta)

        acquisition = set_info.get("acquisition")
        if isinstance(acquisition, dict):
            row.update(acquisition)
        _standardize_clinical_ids(row)

        if not _row_matches_filters(row, measurement_filters):
            continue

        category = str(measurement_type_category or row.get("category") or "").upper()
        if (
            drop_missing_sample_thickness
            and category != "CALIBRATION"
            and not _has_valid_sample_thickness(row)
        ):
            if drop_stats is not None:
                drop_stats["missing_sample_thickness"] = (
                    drop_stats.get("missing_sample_thickness", 0) + 1
                )
            continue

        measurements = container.measurements(set_idx)
        detector_measurements = []
        for measurement in measurements:
            detector_id = measurement.get("detector_id")
            detector = detector_catalog.get(int(detector_id)) if detector_id else {}
            detector_measurements.append({**measurement, **detector})

        raw_data = container.raw(set_idx)
        processed_data = container.processed(set_idx)

        normalized_preference = data_preference.lower()
        if normalized_preference == "gfrm":
            if not convert_gfrm:
                raise ValueError("Product h5_to_df requires convert_gfrm=True.")
            gfrm_path = _first_gfrm_path_from_measurements(
                measurements,
                container_path=file_path,
                raw_root=raw_root,
            )
            if gfrm_path is not None:
                measurement_data, gfrm_metadata = gfrm_to_photons(gfrm_path)
                row["measurement_data_source"] = "gfrm_to_photons"
                row["gfrm_path"] = str(gfrm_path)
            else:
                source_path = measurements[0].get("file_path") if measurements else None
                embedded = _gfrm_to_photons_from_embedded_raw_file(
                    file_path,
                    set_idx,
                    str(source_path) if source_path is not None else None,
                )
                if embedded is not None:
                    measurement_data, gfrm_metadata = embedded
                    row["measurement_data_source"] = "embedded_raw_file_gfrm_to_photons"
                    row["gfrm_path"] = source_path
                else:
                    measurement_data = None
                    gfrm_metadata = None
            if measurement_data is None or gfrm_metadata is None:
                raise FileNotFoundError(
                    f"Missing required RAW GFRM artifact for container set "
                    f"'{set_name}'."
                )
            row["gfrm_data"] = measurement_data
            row["gfrm_conversion_metadata"] = gfrm_metadata
        elif normalized_preference in {"raw", "container_raw", "fabio"}:
            measurement_data = raw_data
            if measurement_data is None and measurements:
                detector_id = int(measurements[0]["detector_id"])
                measurement_data = container.frame(set_idx, detector_id)
            if measurement_data is None:
                raise ValueError(f"Missing container raw data for set '{set_name}'.")
            row["measurement_data_source"] = "container_raw_data"
            row["gfrm_path"] = measurements[0].get("file_path") if measurements else None
            row["gfrm_conversion_metadata"] = {
                "source": "container raw/data",
                "decoder": "container backfill Fabio decode",
            }
        else:
            raise ValueError(
                "data_preference must be one of: 'gfrm', 'raw', "
                "'container_raw', 'fabio'."
            )

        row["measurement_data"] = measurement_data
        row["raw_data"] = raw_data
        row["processed_data"] = processed_data

        integration = container.integration(set_idx)
        if integration is not None:
            row["q_range"], row["radial_profile_data"] = integration
        sigma = _set_dataset_from_file(file_path, set_idx, "integration/sigma")
        if sigma is not None:
            row["radial_profile_sigma"] = sigma
        poni_text = _set_dataset_from_file(file_path, set_idx, "artifacts/poni")
        if poni_text is not None:
            row["ponifile"] = poni_text

        row["metadata"] = _json_set_dataset_from_file(
            file_path,
            set_idx,
            "metadata",
            default={},
        )
        row["processing_config"] = container.processing_config(set_idx) or {}
        row["detector_measurements"] = detector_measurements

        if category == "CALIBRATION":
            row["cal_name"] = set_name
            calibration_rows.append(row)
        else:
            measurement_rows.append(row)

    return calibration_rows, measurement_rows


def h5_to_df(
    file_path: str | Path,
    *,
    data_preference: str = "gfrm",
    raw_root: str | Path | None = None,
    convert_gfrm: bool = True,
    require_clinical_ids: bool = True,
    drop_missing_sample_thickness: bool = False,
    h5_session_df: pd.DataFrame | None = None,
    h5_filters: Sequence[H5SessionFilter | dict[str, Any]] | None = None,
    max_sessions: int | None = None,
    session_category: str | list[str] | tuple[str, ...] | set[str] | None = None,
    session_started_at_min: str | pd.Timestamp | None = None,
    set_category: str | list[str] | tuple[str, ...] | set[str] | None = None,
    measurement_filters: Sequence[H5SessionFilter | dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read Eos-Dx container v0.3 into xrd-analysis-style dataframes.

    Returns ``(calibration_df, measurement_df)``.
    Standalone ``xrd-session`` files are opened through the ``eosdx-container``
    reader. Combined ``xrd-session-archive`` files are read by copying each
    grafted session to a temporary standalone container and using the same
    reader. Rows are one row per session set.

    ``data_preference="gfrm"`` decodes an external ``.gfrm`` path with FabIO.
    ``data_preference="raw"`` uses the container's embedded ``raw/data`` matrix,
    which the EOSCAN backfill writes from FabIO-decoded GFRM frames.
    ``h5_filters`` filters session attrs before detector frames are loaded.
    ``h5_session_df`` can provide an already selected session manifest.
    ``measurement_filters`` filters set metadata before detector frames are loaded.
    ``drop_missing_sample_thickness`` excludes measurement sets without positive
    numeric sample thickness before frame loading.
    """
    file_path = Path(file_path)
    calibration_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    drop_stats: dict[str, int] = {"missing_sample_thickness": 0}

    with h5py.File(file_path, "r") as h5:
        version = _decode(h5.attrs.get("schema_version"))
        fmt = _decode(h5.attrs.get("format"))
        if version != "0.3" or fmt not in {"xrd-session", "xrd-session-archive"}:
            raise ValueError(
                f"Unsupported container format: schema_version={version!r}, format={fmt!r}"
            )

    if fmt == "xrd-session":
        if h5_session_df is None:
            sessions = filter_h5_sessions(
                file_path,
                h5_filters,
                session_category=session_category,
                session_started_at_min=session_started_at_min,
                max_sessions=max_sessions,
            )
        else:
            sessions = filter_h5_session_df(
                h5_session_df,
                h5_filters,
                session_category=session_category,
                session_started_at_min=session_started_at_min,
                max_sessions=max_sessions,
            )
        if not sessions.empty:
            calibration_rows, measurement_rows = _rows_from_container_reader(
                file_path,
                data_preference=data_preference,
                raw_root=raw_root,
                convert_gfrm=convert_gfrm,
                session_started_at_min=session_started_at_min,
                set_category=set_category,
                measurement_filters=measurement_filters,
                drop_missing_sample_thickness=drop_missing_sample_thickness,
                drop_stats=drop_stats,
            )
    elif fmt == "xrd-session-archive":
        sessions = _archive_sessions(
            file_path,
            h5_filters=h5_filters,
            h5_session_df=h5_session_df,
            session_category=session_category,
            session_started_at_min=session_started_at_min,
            max_sessions=max_sessions,
        )
        with tempfile.TemporaryDirectory(prefix="xrd_preprocessing_h5_") as temp_root:
            temp_root_path = Path(temp_root)
            for idx, session_meta in enumerate(sessions, start=1):
                temp_path = temp_root_path / f"session_{idx:04d}.nxs.h5"
                _copy_archive_session_to_file(
                    file_path,
                    session_meta["archive_session_path"],
                    temp_path,
                )
                cal_rows, meas_rows = _rows_from_container_reader(
                    temp_path,
                    data_preference=data_preference,
                    raw_root=raw_root,
                    convert_gfrm=convert_gfrm,
                    session_started_at_min=session_started_at_min,
                    set_category=set_category,
                    measurement_filters=measurement_filters,
                    drop_missing_sample_thickness=drop_missing_sample_thickness,
                    drop_stats=drop_stats,
                    archive_meta={
                        **session_meta,
                        "source_file": str(file_path),
                    },
                )
                for row in cal_rows + meas_rows:
                    row["source_file"] = str(file_path)
                calibration_rows.extend(cal_rows)
                measurement_rows.extend(meas_rows)
    else:
        raise AssertionError(f"Unhandled format: {fmt!r}")

    calibration_df = pd.DataFrame(calibration_rows)
    measurement_df = pd.DataFrame(measurement_rows)
    measurement_df.attrs["dropped_missing_sample_thickness"] = drop_stats[
        "missing_sample_thickness"
    ]
    if require_clinical_ids and not measurement_df.empty:
        _validate_clinical_ids(measurement_df, frame_name="measurement_df")
    return calibration_df, measurement_df


def _legacy_h5_to_df(
    file_path: str | Path,
    *,
    data_preference: str = "gfrm",
    raw_root: str | Path | None = None,
    convert_gfrm: bool = True,
    require_clinical_ids: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deprecated pre-reader implementation kept for fixture comparison."""
    file_path = Path(file_path)
    calibration_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []

    with h5py.File(file_path, "r") as h5:
        version = _decode(h5.attrs.get("schema_version"))
        fmt = _decode(h5.attrs.get("format"))
        if version != "0.3" or fmt != "xrd-session":
            raise ValueError(
                f"Unsupported container format: schema_version={version!r}, format={fmt!r}"
            )

        root_attrs = _attrs(h5)
        session = h5["session"]
        session_meta = _attrs(session)
        if "sample" in session:
            session_meta.update(_scalar_fields(session["sample"], prefix="sample_"))
            session_meta["patientId"] = _text_dataset(session["sample"], "patient_name")
            session_meta["specimenId"] = _text_dataset(session["sample"], "name")
        if "instrument" in session:
            session_meta.update(_attrs(session["instrument"], prefix="instrument_"))
            session_meta.update(_scalar_fields(session["instrument"]))

        detector_catalog = _detector_catalog(session)
        sets = session.get("sets")
        if sets is None:
            return pd.DataFrame(), pd.DataFrame()

        for set_name in sorted(sets):
            row = _set_row(
                file_path,
                root_attrs,
                session_meta,
                set_name,
                sets[set_name],
                detector_catalog,
                data_preference,
                raw_root,
                convert_gfrm,
            )
            _standardize_clinical_ids(row)
            category = str(
                row.get("measurement_type_category")
                or row.get("category")
                or ""
            ).upper()
            if category == "CALIBRATION":
                row["cal_name"] = set_name
                calibration_rows.append(row)
            else:
                measurement_rows.append(row)

    calibration_df = pd.DataFrame(calibration_rows)
    measurement_df = pd.DataFrame(measurement_rows)
    if require_clinical_ids and not measurement_df.empty:
        _validate_clinical_ids(measurement_df, frame_name="measurement_df")
    return calibration_df, measurement_df
