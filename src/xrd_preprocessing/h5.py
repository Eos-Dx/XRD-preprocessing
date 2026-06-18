from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .gfrm import gfrm_to_photons


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
            or row.get("sample_patient_name")
            or row.get("patient_id")
            or row.get("patientID")
        )
    if "specimenId" not in row:
        row["specimenId"] = (
            row.get("sample_clinical_name")
            or row.get("clinical_sample_name")
            or row.get("specimen_id")
            or row.get("specimenID")
        )


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


def h5_to_df(
    file_path: str | Path,
    *,
    data_preference: str = "gfrm",
    raw_root: str | Path | None = None,
    convert_gfrm: bool = True,
    require_clinical_ids: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read Eos-Dx container v0.3 into xrd-analysis-style dataframes.

    Returns ``(calibration_df, measurement_df)``.
    v0.3 rows are one row per ``/session/sets/*`` capture. ``measurement_data``
    requires RAW GFRM semantics. NumPy arrays stored in the container are kept
    as decoded products, but ``measurement_data`` is always produced from the
    referenced GFRM artifact.
    """
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
