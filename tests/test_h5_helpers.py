from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import xrd_preprocessing.h5 as h5_module
from xrd_preprocessing import H5SessionFilter


class FakeContainer:
    def __init__(self, *, started_at="2026-01-01T10:00:00", raw_data=None):
        self._started_at = started_at
        self._raw_data = raw_data

    def session_meta(self):
        return {
            "started_at": self._started_at,
            "category": "SAMPLE",
            "patient_clinical_name": "Nova_001",
            "sample_clinical_name": "Nova_001_Left",
            "specimen_status": "BENIGN",
        }

    def detectors(self):
        return [{"detector_id": 7, "detector_name": "fake"}]

    def sets(self):
        return [
            {
                "name": "set_001_sample_main",
                "measurement_type_category": "SAMPLE",
                "acquisition": {"sample_thickness_mm": 1.5},
            },
            {
                "name": "set_002_calibration_main",
                "measurement_type_category": "CALIBRATION",
                "acquisition": {"sample_thickness_mm": 2.0},
            },
        ]

    def measurements(self, set_idx):
        return [
            {
                "detector_id": 7,
                "file_path": f"raw/set_{set_idx:03d}.gfrm",
            }
        ]

    def raw(self, set_idx):
        if self._raw_data is None:
            return np.full((2, 2), set_idx, dtype=np.float32)
        return self._raw_data

    def processed(self, set_idx):
        return np.full((2, 2), set_idx + 10, dtype=np.float32)

    def frame(self, set_idx, detector_id):
        return np.full((2, 2), set_idx + detector_id, dtype=np.float32)

    def integration(self, set_idx):
        return np.asarray([1.0, 2.0]), np.asarray([set_idx, set_idx + 1.0])

    def processing_config(self, set_idx):
        return {"set_idx": set_idx}


def _write_reader_support_h5(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        session = h5.create_group("session")
        sets = session.create_group("sets")
        for idx, name in [(1, "set_001_sample_main"), (2, "set_002_calibration_main")]:
            group = sets.create_group(name)
            integration = group.create_group("integration")
            integration.create_dataset("sigma", data=np.asarray([0.1, 0.2]))
            artifacts = group.create_group("artifacts")
            artifacts.create_dataset("poni", data=np.bytes_("Distance: 0.1\n"))
            group.create_dataset("metadata", data=np.bytes_(f'{{"idx": {idx}}}'))
            measurements = group.create_group("measurements")
            measurement = measurements.create_group("m1")
            measurement.attrs["detector_id"] = 7
            measurement.create_dataset("raw_file", data=np.asarray([1, 2, 3], dtype=np.uint8))


def test_h5_small_helpers_and_filter_errors(tmp_path):
    with h5py.File(tmp_path / "helpers.h5", "w") as h5:
        group = h5.create_group("group")
        group.attrs["bytes"] = np.bytes_("text")
        group.attrs["scalar"] = np.int64(3)
        group.attrs["array"] = np.asarray([np.bytes_("a"), np.bytes_("b")])
        group.create_dataset("json", data=np.bytes_('{"a": 1}'))
        group.create_dataset("scalar_dataset", data=np.bytes_("value"))
        group.create_group("set_002_b")
        group.create_group("set_001_a")

        assert h5_module._attrs(group)["bytes"] == "text"
        assert h5_module._attrs(group)["scalar"] == 3
        assert h5_module._attrs(group)["array"] == ["a", "b"]
        assert h5_module._json_dataset(group, "missing", default={}) == {}
        assert h5_module._json_dataset(group, "json") == {"a": 1}
        assert h5_module._scalar_fields(group)["scalar_dataset"] == "value"
        assert h5_module._first_child_by_prefix(group, "set_").name.endswith("set_001_a")

    assert h5_module._set_index("set_123_main") == 123
    with pytest.raises(ValueError, match="Cannot parse set index"):
        h5_module._set_index("bad")
    assert h5_module._allowed_values("sample") == {"SAMPLE"}
    assert h5_module._matches_allowed("sample", {"SAMPLE"}) is True
    assert h5_module._timestamp("bad") is None
    assert h5_module._matches_min_timestamp("bad", "2026-01-01") is False
    assert h5_module._positive_float("0") is None
    assert np.isnan(h5_module._poni_q_range_or_nan("bad poni")[0])
    with pytest.raises(TypeError, match="Expected H5SessionFilter"):
        h5_module._coerce_filter("bad")


@pytest.mark.parametrize(
    ("op", "kwargs", "expected"),
    [
        ("==", {"value": "Nova_1"}, [True, False, False]),
        ("!=", {"value": "Nova_1"}, [False, True, True]),
        ("contains", {"value": "Nova"}, [True, True, False]),
        ("startswith", {"value": "Nova"}, [True, True, False]),
        ("endswith", {"value": "2"}, [False, True, False]),
        ("isna", {}, [False, False, True]),
        ("notna", {}, [True, True, False]),
        ("not in", {"values": ["Nova_1"]}, [False, True, True]),
        (">", {"value": 1}, [False, True, False]),
        ("<", {"value": 2}, [True, False, False]),
        ("<=", {"value": 2}, [True, True, False]),
        ("between", {"value": [1, 2]}, [True, True, False]),
        ("between", {"lower": 1, "upper": 2}, [True, True, False]),
        ("date>", {"value": "2026-01-01"}, [False, True, False]),
        ("date<", {"value": "2026-01-02"}, [True, False, False]),
        ("date<=", {"value": "2026-01-01"}, [True, False, False]),
        ("date not in", {"values": ["2026-01-01"]}, [False, True, True]),
        ("date_between", {"lower": "2026-01-01", "upper": "2026-01-02"}, [True, True, False]),
    ],
)
def test_h5_filter_mask_ops(op, kwargs, expected):
    frame = pd.DataFrame(
        {
            "name": ["Nova_1", "Nova_2", None],
            "number": [1, 2, np.nan],
            "day": ["2026-01-01", "2026-01-02", "bad"],
        }
    )
    column = (
        "day"
        if op.startswith("date")
        else "number"
        if op in {">", "<", "<=", "between"}
        else "name"
    )

    mask = h5_module._filter_mask(frame, H5SessionFilter(column, op=op, **kwargs))

    assert mask.tolist() == expected


def test_h5_filter_mask_invalid_date_and_missing_column():
    frame = pd.DataFrame({"day": ["2026-01-01"]})

    with pytest.raises(ValueError, match="Invalid H5 date filter value"):
        h5_module._filter_mask(frame, H5SessionFilter("day", op="date>=", value="bad"))

    with pytest.raises(ValueError, match="Invalid H5 date filter values"):
        h5_module._filter_mask(frame, H5SessionFilter("day", op="date_in", values=["bad"]))

    with pytest.raises(ValueError, match="requires value or values"):
        h5_module._filter_mask(frame, H5SessionFilter("day", op="date_in"))

    with pytest.raises(ValueError, match="requires lower/upper"):
        h5_module._filter_mask(frame, H5SessionFilter("day", op="between", value=[1]))

    with pytest.raises(ValueError, match="missing column"):
        h5_module._filter_mask(frame, H5SessionFilter("missing", op="==", value=1))

    assert h5_module._filter_mask(
        frame,
        H5SessionFilter(
            "missing",
            op="==",
            value="ok",
            fallback=H5SessionFilter("day", op="==", value="2026-01-01"),
        ),
    ).tolist() == [True]

    with pytest.raises(ValueError, match="Unsupported H5 session filter op"):
        h5_module._filter_mask(frame, H5SessionFilter("day", op="bad"))

    assert h5_module._row_matches_filters({"day": "2026-01-01"}, None) is True
    assert (
        h5_module._row_matches_filters(
            {"day": "2026-01-01"},
            [H5SessionFilter("day", op="date>=", value="2026-01-02")],
        )
        is False
    )


def test_h5_measurement_set_counts_and_filter_statistics():
    before = pd.DataFrame(
        {
            "patientId": ["P1", "P1", "P2"],
            "specimenId": ["P1_L", "P1_R", "P2_L"],
            "specimen_status": ["BENIGN", "CANCER", ""],
        }
    )
    after = before.iloc[:2].copy()

    counts = h5_module.h5_measurement_set_counts(before)
    stats = h5_module.h5_filter_statistics(before, after)

    assert counts["measurements"] == 3
    assert counts["patients"] == 2
    assert counts["specimens"] == 3
    assert counts["diagnosis_counts"] == {"BENIGN": 1, "CANCER": 1, "NA": 1}
    assert stats["before"]["measurements"] == 3
    assert stats["after"]["measurements"] == 2
    assert stats["dropped"]["measurements"] == 1
    assert stats["dropped"]["patients"] == 1


def test_h5_file_resolution_and_dataset_helpers(tmp_path):
    container_path = tmp_path / "container.h5"
    nested = tmp_path / "raw" / "nested"
    nested.mkdir(parents=True)
    gfrm = nested / "frame.gfrm"
    gfrm.write_bytes(b"fake")
    _write_reader_support_h5(container_path)

    assert h5_module._resolve_file_path(str(gfrm), container_path=container_path, raw_root=None) == gfrm
    assert h5_module._resolve_file_path("frame.gfrm", container_path=container_path, raw_root=tmp_path / "raw") == gfrm
    assert h5_module._resolve_file_path("missing.gfrm", container_path=container_path, raw_root=tmp_path) is None
    assert h5_module._first_gfrm_path_from_measurements(
        [{"file_path": "frame.gfrm"}],
        container_path=container_path,
        raw_root=tmp_path / "raw",
    ) == gfrm
    assert h5_module._set_dataset_from_file(container_path, 1, "artifacts/poni") == "Distance: 0.1\n"
    np.testing.assert_array_equal(
        h5_module._first_measurement_dataset_from_file(container_path, 1, "raw_file"),
        np.asarray([1, 2, 3], dtype=np.uint8),
    )
    assert h5_module._json_set_dataset_from_file(container_path, 1, "metadata") == {"idx": 1}
    assert h5_module._set_dataset_from_file(container_path, 9, "artifacts/poni") is None
    assert h5_module._first_measurement_dataset_from_file(container_path, 9, "raw_file") is None


def test_h5_rows_from_fake_container_reader_gfrm_and_raw(monkeypatch, tmp_path):
    h5_path = tmp_path / "reader.h5"
    _write_reader_support_h5(h5_path)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    for idx in (1, 2):
        (raw_root / f"set_{idx:03d}.gfrm").write_bytes(b"fake")

    monkeypatch.setattr(h5_module, "_open_container_reader", lambda _path: FakeContainer())
    monkeypatch.setattr(
        h5_module,
        "gfrm_to_photons",
        lambda _path: (np.ones((2, 2)), {"decoder": "fake"}),
    )

    calibration_rows, measurement_rows = h5_module._rows_from_container_reader(
        h5_path,
        data_preference="gfrm",
        raw_root=raw_root,
        convert_gfrm=True,
        session_started_at_min=None,
        set_category=None,
        archive_meta={"archive_group": "calib"},
    )

    assert len(calibration_rows) == 1
    assert len(measurement_rows) == 1
    assert measurement_rows[0]["measurement_data_source"] == "gfrm_to_photons"
    assert measurement_rows[0]["archive_group"] == "calib"
    assert measurement_rows[0]["patientId"] == "Nova_001"
    assert measurement_rows[0]["specimenId"] == "Nova_001_Left"
    assert measurement_rows[0]["metadata"] == {"idx": 1}
    assert measurement_rows[0]["processing_config"] == {"set_idx": 1}
    assert measurement_rows[0]["detector_measurements"][0]["detector_name"] == "fake"

    _, raw_rows = h5_module._rows_from_container_reader(
        h5_path,
        data_preference="raw",
        raw_root=None,
        convert_gfrm=True,
        session_started_at_min=None,
        set_category="SAMPLE",
    )

    assert len(raw_rows) == 1
    assert raw_rows[0]["measurement_data_source"] == "container_raw_data"


def test_h5_rows_from_fake_container_reader_error_and_drop_branches(monkeypatch, tmp_path):
    h5_path = tmp_path / "reader.h5"
    _write_reader_support_h5(h5_path)
    monkeypatch.setattr(
        h5_module,
        "_open_container_reader",
        lambda _path: FakeContainer(started_at="2025-01-01T10:00:00"),
    )

    assert h5_module._rows_from_container_reader(
        h5_path,
        data_preference="raw",
        raw_root=None,
        convert_gfrm=True,
        session_started_at_min="2026-01-01",
        set_category=None,
    ) == ([], [])

    monkeypatch.setattr(h5_module, "_open_container_reader", lambda _path: FakeContainer(raw_data=None))
    with pytest.raises(ValueError, match="data_preference must be one of"):
        h5_module._rows_from_container_reader(
            h5_path,
            data_preference="bad",
            raw_root=None,
            convert_gfrm=True,
            session_started_at_min=None,
            set_category="SAMPLE",
        )

    with pytest.raises(ValueError, match="requires convert_gfrm"):
        h5_module._rows_from_container_reader(
            h5_path,
            data_preference="gfrm",
            raw_root=None,
            convert_gfrm=False,
            session_started_at_min=None,
            set_category="SAMPLE",
        )

    drop_stats = {}
    _, rows = h5_module._rows_from_container_reader(
        h5_path,
        data_preference="raw",
        raw_root=None,
        convert_gfrm=True,
        session_started_at_min=None,
        set_category="SAMPLE",
        measurement_filters=[H5SessionFilter("specimen_status", op="in", values=["CANCER"])],
        drop_missing_sample_thickness=True,
        drop_stats=drop_stats,
    )
    assert rows == []
    assert drop_stats == {}
