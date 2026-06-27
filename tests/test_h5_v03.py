import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import h5py

sys.path.insert(0, "/Users/sad/dev/container/src")
sys.path.insert(0, "/Users/sad/dev/container/tests/v0_3")

from xrd_preprocessing import (  # noqa: E402
    H5SessionFilter,
    calibrant_thickness_h5_filters,
    filter_h5_session_df,
    filter_h5_sessions,
    h5_to_df,
    list_h5_measurement_stage_sets,
    list_h5_sessions,
)

GFRM_ARCHIVE = Path("examples/data/gfrm_measurements.tar.gz")
WATER_20MM_REL = (
    Path("GFRM_measurements")
    / "Water"
    / "20260608_112438_Water_20mm"
    / "20260608_112438_Water_20mm_Main.gfrm"
)


def fake_poni(distance_m: float = 0.1) -> str:
    return f"""# Fake PONI for tests
poni_version: 2.1
Detector: Detector
Detector_config: {{"pixel1": 0.0001, "pixel2": 0.0001, "max_shape": [16, 16], "orientation": 3}}
Distance: {distance_m}
Poni1: 0.0008
Poni2: 0.0008
Rot1: 0
Rot2: 0
Rot3: 0
Wavelength: 1e-10
"""


def test_h5_to_df_v03_roundtrip(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_measurement, make_session, make_set
    from container.v0_3 import build_session_container
    from xrd_preprocessing import extract_gfrm_archive, gfrm_to_photons

    processed = np.arange(16, dtype=np.float32).reshape(4, 4)
    raw = processed + 1
    raw_root = extract_gfrm_archive(GFRM_ARCHIVE, tmp_path / "raw")
    expected, _ = gfrm_to_photons(raw_root / WATER_20MM_REL)
    measurement = make_measurement(
        file_path=str(WATER_20MM_REL),
        data=None,
    )
    _, _, path = build_session_container(
        make_session(sets=[make_set(measurements=[measurement], raw=raw, processed=processed)]),
        tmp_path / "container",
    )

    calib_df, meas_df = h5_to_df(Path(path), raw_root=raw_root)

    assert calib_df.empty
    assert len(meas_df) == 1
    row = meas_df.iloc[0]
    assert row["schema_version"] == "0.3"
    assert row["format"] == "xrd-session"
    assert row["patientId"] == "PAT001"
    assert row["specimenId"] == "PAT001-S01"
    assert row["measurement_data_source"] == "gfrm_to_photons"
    np.testing.assert_allclose(row["measurement_data"], expected, rtol=0, atol=0, equal_nan=True)
    assert np.array_equal(row["raw_data"], raw)
    assert np.array_equal(row["processed_data"], processed)
    assert row["ponifile"] == "Distance: 0.17\n"
    assert len(row["q_range"]) == 2000
    assert len(row["detector_measurements"]) == 1


def test_h5_to_df_rejects_unsupported_container(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "0.2"
        h5.attrs["format"] = "xrd-session"

    with pytest.raises(ValueError, match="Unsupported container format"):
        h5_to_df(path)


def test_h5_to_df_requires_gfrm_artifact(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_session, make_set
    from container.v0_3 import build_session_container

    processed = np.arange(16, dtype=np.float32).reshape(4, 4)
    raw = processed + 1
    _, _, path = build_session_container(
        make_session(sets=[make_set(raw=raw, processed=processed)]),
        tmp_path,
    )

    with pytest.raises(FileNotFoundError, match="Missing required RAW GFRM artifact"):
        h5_to_df(Path(path))


def test_h5_to_df_can_use_container_raw_data(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_measurement, make_session, make_set
    from container.v0_3 import build_session_container

    raw = np.arange(16, dtype=np.float32).reshape(4, 4)
    measurement = make_measurement(file_path="sample.gfrm", data=raw)
    _, _, path = build_session_container(
        make_session(sets=[make_set(measurements=[measurement], raw=raw)]),
        tmp_path / "container",
    )

    _, meas_df = h5_to_df(Path(path), data_preference="raw")

    row = meas_df.iloc[0]
    assert row["measurement_data_source"] == "container_raw_data"
    assert row["gfrm_path"] == "sample.gfrm"
    np.testing.assert_array_equal(row["measurement_data"], raw)


def test_h5_to_df_can_convert_embedded_raw_file_gfrm(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_measurement, make_session, make_set
    from container.v0_3 import build_session_container
    from xrd_preprocessing import extract_gfrm_archive, gfrm_to_photons

    raw_root = extract_gfrm_archive(GFRM_ARCHIVE, tmp_path / "raw")
    gfrm_path = raw_root / WATER_20MM_REL
    expected, _ = gfrm_to_photons(gfrm_path)
    measurement = make_measurement(
        file_path=str(WATER_20MM_REL),
        data=np.ones((4, 4), dtype=np.int64),
    )
    _, _, path = build_session_container(
        make_session(sets=[make_set(measurements=[measurement], raw=None)]),
        tmp_path / "container",
    )
    with h5py.File(path, "a") as h5:
        measurements = h5["session/sets/set_001_sample_main/measurements"]
        measurement_group = measurements[sorted(measurements)[0]]
        measurement_group.create_dataset(
            "raw_file",
            data=np.frombuffer(gfrm_path.read_bytes(), dtype=np.uint8),
        )

    _, meas_df = h5_to_df(Path(path), data_preference="gfrm")

    row = meas_df.iloc[0]
    assert row["measurement_data_source"] == "embedded_raw_file_gfrm_to_photons"
    assert row["gfrm_conversion_metadata"]["embedded_raw_file"] is True
    np.testing.assert_allclose(
        row["measurement_data"],
        expected,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_h5_to_df_can_drop_missing_sample_thickness_before_output(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_measurement, make_session, make_set
    from container.v0_3 import build_session_container

    raw = np.arange(16, dtype=np.float32).reshape(4, 4)
    measurement = make_measurement(file_path="sample.gfrm", data=raw)
    _, _, path = build_session_container(
        make_session(
            sets=[
                make_set(
                    pk=1,
                    measurements=[measurement],
                    raw=raw,
                    sample_thickness_mm=None,
                ),
                make_set(
                    pk=2,
                    measurements=[measurement],
                    raw=raw,
                    sample_thickness_mm=1.2,
                ),
            ]
        ),
        tmp_path / "container",
    )

    _, unfiltered_df = h5_to_df(Path(path), data_preference="raw")
    _, filtered_df = h5_to_df(
        Path(path),
        data_preference="raw",
        drop_missing_sample_thickness=True,
    )

    assert len(unfiltered_df) == 2
    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]["sample_thickness_mm"] == 1.2
    assert filtered_df.attrs["dropped_missing_sample_thickness"] == 1


def test_h5_to_df_reads_combined_archive_sessions(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_measurement, make_session, make_set
    from container.v0_3 import build_session_container

    raw = np.arange(16, dtype=np.float32).reshape(4, 4)
    measurement = make_measurement(file_path="sample.gfrm", data=raw)
    _, _, session_path = build_session_container(
        make_session(sets=[make_set(measurements=[measurement], raw=raw)]),
        tmp_path / "container",
    )
    archive_path = tmp_path / "combined_archive.h5"
    with h5py.File(session_path, "r") as src, h5py.File(archive_path, "w") as dst:
        dst.attrs["schema_version"] = "0.3"
        dst.attrs["format"] = "xrd-session-archive"
        dst.attrs["grouped_by"] = "calibration"
        group = dst.create_group("calib_test")
        group.attrs["calibration_session_uid"] = "calib-uid"
        group.attrs["calibrant_thickness_mm"] = 10.0
        group.attrs["calibrant_thickness_source"] = "test"
        src.copy(src["session"], group, name="sample_01_PAT001")
        for key, value in src.attrs.items():
            group["sample_01_PAT001"].attrs[key] = value

    _, meas_df = h5_to_df(
        archive_path,
        data_preference="raw",
        session_category="SAMPLE",
        set_category="SAMPLE",
        max_sessions=1,
    )

    row = meas_df.iloc[0]
    assert row["archive_group"] == "calib_test"
    assert row["archive_session_name"] == "sample_01_PAT001"
    assert row["calibration_session_uid"] == "calib-uid"
    assert row["calibrant_thickness_mm"] == 10.0
    assert row["archive_group_calibrant_thickness_mm"] == 10.0
    assert row["calibrant_thickness_source"] == "test"
    assert row["archive_group_calibrant_thickness_source"] == "test"
    assert row["patientId"] == "PAT001"
    assert row["specimenId"] == "PAT001-S01"
    np.testing.assert_array_equal(row["measurement_data"], raw)


def test_calibrant_thickness_h5_filters():
    filters = calibrant_thickness_h5_filters()
    assert [(item.column, item.op, item.value) for item in filters] == [
        ("calibrant_thickness_mm", "notna", None),
        ("calibrant_thickness_mm", ">=", 10.0),
        ("calibrant_thickness_mm", "<=", 40.0),
    ]


def test_list_and_filter_h5_sessions_from_archive_attrs_only(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_measurement, make_session, make_set
    from container.v0_3 import build_session_container

    raw = np.arange(16, dtype=np.float32).reshape(4, 4)
    measurement = make_measurement(file_path="sample.gfrm", data=raw)
    _, _, old_session_path = build_session_container(
        make_session(
            completed_at="2025-12-31 10:00:00",
            started_at="2025-12-31 10:00:00",
            sets=[make_set(measurements=[measurement], raw=raw)],
        ),
        tmp_path / "old_container",
    )
    _, _, new_session_path = build_session_container(
        make_session(
            completed_at="2026-01-02 10:00:00",
            started_at="2026-01-02 10:00:00",
            sets=[make_set(measurements=[measurement], raw=raw)],
        ),
        tmp_path / "new_container",
    )
    archive_path = tmp_path / "combined_archive.h5"
    with h5py.File(archive_path, "w") as dst:
        dst.attrs["schema_version"] = "0.3"
        dst.attrs["format"] = "xrd-session-archive"
        dst.attrs["grouped_by"] = "calibration"
        group = dst.create_group("calib_test")
        group.attrs["calibration_session_uid"] = "calib-uid"
        for name, path in [
            ("sample_01_old", old_session_path),
            ("sample_02_new", new_session_path),
        ]:
            with h5py.File(path, "r") as src:
                src.copy(src["session"], group, name=name)
                for key, value in src.attrs.items():
                    group[name].attrs[key] = value
                group[name]["sample"].attrs["specimen_status"] = (
                    "NORMAL" if name == "sample_01_old" else "BENIGN"
                )

    session_df = list_h5_sessions(archive_path)
    selected_df = filter_h5_sessions(
        archive_path,
        [H5SessionFilter("started_at", op="date>=", value="2026-01-01")],
        session_category="SAMPLE",
    )
    selected_by_dates_df = filter_h5_sessions(
        archive_path,
        [
            H5SessionFilter(
                "started_at",
                op="date in",
                values=["2025-12-31", pd.Timestamp("2026-01-02")],
            )
        ],
        session_category="SAMPLE",
    )
    rejected_by_dates_df = filter_h5_sessions(
        archive_path,
        [H5SessionFilter("started_at", op="date not in", value=["2025-12-31"])],
        session_category="SAMPLE",
    )
    selected_by_diagnosis_df = filter_h5_sessions(
        archive_path,
        [H5SessionFilter("specimen_status", op="in", values=["BENIGN", "CANCER"])],
        session_category="SAMPLE",
    )

    assert len(session_df) == 2
    assert "measurement_data" not in session_df.columns
    assert "raw_data" not in session_df.columns
    assert "specimen_status" in session_df.columns
    assert selected_df["archive_session_name"].tolist() == ["sample_02_new"]
    assert selected_by_dates_df["archive_session_name"].tolist() == [
        "sample_01_old",
        "sample_02_new",
    ]
    assert rejected_by_dates_df["archive_session_name"].tolist() == ["sample_02_new"]
    assert selected_by_diagnosis_df["archive_session_name"].tolist() == [
        "sample_02_new"
    ]
    assert selected_df["calibration_session_uid"].tolist() == ["calib-uid"]

    selected_from_df = filter_h5_session_df(
        session_df,
        [H5SessionFilter("started_at", op="date>=", value="2026-01-01")],
        session_category="SAMPLE",
    )
    stage_frames = list_h5_measurement_stage_sets(
        archive_path,
        session_df=session_df,
        stage_filters={
            "before": [],
            "after_date": [
                H5SessionFilter("started_at", op="date>=", value="2026-01-01")
            ],
            "after_diagnosis": [
                H5SessionFilter("specimen_status", op="in", values=["BENIGN"])
            ],
        },
    )

    assert selected_from_df["archive_session_name"].tolist() == ["sample_02_new"]
    assert stage_frames["before"]["archive_session_name"].tolist() == [
        "sample_01_old",
        "sample_02_new",
    ]
    assert stage_frames["after_date"]["archive_session_name"].tolist() == [
        "sample_02_new"
    ]
    assert stage_frames["after_diagnosis"]["archive_session_name"].tolist() == [
        "sample_02_new"
    ]


def test_h5_session_filters_can_use_poni_q_range_before_frame_loading(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_measurement, make_session, make_set
    from container.v0_3 import build_session_container

    raw = np.arange(16, dtype=np.float32).reshape(4, 4)
    measurement = make_measurement(file_path="sample.gfrm", data=raw)
    _, _, short_distance_path = build_session_container(
        make_session(
            sets=[
                make_set(
                    measurements=[measurement],
                    raw=raw,
                    poni_text=fake_poni(0.1),
                )
            ],
        ),
        tmp_path / "short_distance",
    )
    _, _, long_distance_path = build_session_container(
        make_session(
            sets=[
                make_set(
                    measurements=[measurement],
                    raw=raw,
                    poni_text=fake_poni(1.0),
                )
            ],
        ),
        tmp_path / "long_distance",
    )
    archive_path = tmp_path / "combined_archive.h5"
    with h5py.File(archive_path, "w") as dst:
        dst.attrs["schema_version"] = "0.3"
        dst.attrs["format"] = "xrd-session-archive"
        dst.attrs["grouped_by"] = "calibration"
        group = dst.create_group("calib_test")
        group.attrs["calibration_session_uid"] = "calib-uid"
        for name, path in [
            ("sample_01_short", short_distance_path),
            ("sample_02_long", long_distance_path),
        ]:
            with h5py.File(path, "r") as src:
                src.copy(src["session"], group, name=name)
                for key, value in src.attrs.items():
                    group[name].attrs[key] = value

    session_df = list_h5_sessions(archive_path)
    selected_df = filter_h5_sessions(
        archive_path,
        [H5SessionFilter("poni_q_max_nm_inv", op=">=", value=0.5)],
        session_category="SAMPLE",
    )

    assert "measurement_data" not in session_df.columns
    assert "poni_q_max_nm_inv" in session_df.columns
    assert "h5_sample_all_sets_have_thickness" in session_df.columns
    assert session_df["h5_sample_all_sets_have_thickness"].tolist() == [True, True]
    assert selected_df["archive_session_name"].tolist() == ["sample_01_short"]


def test_h5_session_filters_can_require_sample_thickness_before_frame_loading(
    tmp_path,
):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_measurement, make_session, make_set
    from container.v0_3 import build_session_container

    raw = np.arange(16, dtype=np.float32).reshape(4, 4)
    measurement = make_measurement(file_path="sample.gfrm", data=raw)
    _, _, missing_thickness_path = build_session_container(
        make_session(
            sets=[
                make_set(
                    measurements=[measurement],
                    raw=raw,
                    sample_thickness_mm=None,
                )
            ],
        ),
        tmp_path / "missing_thickness",
    )
    _, _, valid_thickness_path = build_session_container(
        make_session(
            sets=[
                make_set(
                    measurements=[measurement],
                    raw=raw,
                    sample_thickness_mm=1.2,
                )
            ],
        ),
        tmp_path / "valid_thickness",
    )
    archive_path = tmp_path / "combined_archive.h5"
    with h5py.File(archive_path, "w") as dst:
        dst.attrs["schema_version"] = "0.3"
        dst.attrs["format"] = "xrd-session-archive"
        dst.attrs["grouped_by"] = "calibration"
        group = dst.create_group("calib_test")
        group.attrs["calibration_session_uid"] = "calib-uid"
        for name, path in [
            ("sample_01_missing", missing_thickness_path),
            ("sample_02_valid", valid_thickness_path),
        ]:
            with h5py.File(path, "r") as src:
                src.copy(src["session"], group, name=name)
                for key, value in src.attrs.items():
                    group[name].attrs[key] = value

    selected_df = filter_h5_sessions(
        archive_path,
        [H5SessionFilter("h5_sample_all_sets_have_thickness", op="==", value=True)],
        session_category="SAMPLE",
    )

    assert selected_df["archive_session_name"].tolist() == ["sample_02_valid"]


def test_h5_to_df_filters_archive_sessions_before_loading(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_measurement, make_session, make_set
    from container.v0_3 import build_session_container

    raw = np.arange(16, dtype=np.float32).reshape(4, 4)
    measurement = make_measurement(file_path="sample.gfrm", data=raw)
    _, _, old_session_path = build_session_container(
        make_session(
            completed_at="2025-12-31 10:00:00",
            started_at="2025-12-31 10:00:00",
            sets=[make_set(measurements=[measurement], raw=raw)],
        ),
        tmp_path / "old_container",
    )
    _, _, new_session_path = build_session_container(
        make_session(
            completed_at="2026-01-02 10:00:00",
            started_at="2026-01-02 10:00:00",
            sets=[make_set(measurements=[measurement], raw=raw)],
        ),
        tmp_path / "new_container",
    )
    archive_path = tmp_path / "combined_archive.h5"
    with h5py.File(archive_path, "w") as dst:
        dst.attrs["schema_version"] = "0.3"
        dst.attrs["format"] = "xrd-session-archive"
        dst.attrs["grouped_by"] = "calibration"
        group = dst.create_group("calib_test")
        group.attrs["calibration_session_uid"] = "calib-uid"
        for name, path in [
            ("sample_01_old", old_session_path),
            ("sample_02_new", new_session_path),
        ]:
            with h5py.File(path, "r") as src:
                src.copy(src["session"], group, name=name)
                for key, value in src.attrs.items():
                    group[name].attrs[key] = value
                group[name]["sample"].attrs["specimen_status"] = (
                    "NORMAL" if name == "sample_01_old" else "CANCER"
                )

    _, meas_df = h5_to_df(
        archive_path,
        data_preference="raw",
        h5_filters=[
            H5SessionFilter("started_at", op="date_in", values=["2026-01-02"]),
            H5SessionFilter("specimen_status", op="in", values=["BENIGN", "CANCER"]),
        ],
        session_category="SAMPLE",
        set_category="SAMPLE",
    )

    assert len(meas_df) == 1
    assert meas_df.iloc[0]["started_at"] == "2026-01-02 10:00:00"
    assert meas_df.iloc[0]["specimen_status"] == "CANCER"


def test_h5_to_df_rejects_non_gfrm_data_preference(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_session, make_set
    from container.v0_3 import build_session_container

    _, _, path = build_session_container(
        make_session(sets=[make_set(raw=np.ones((2, 2), dtype=np.float32))]),
        tmp_path,
    )

    with pytest.raises(ValueError, match="data_preference must be one of"):
        h5_to_df(Path(path), data_preference="processed")


def test_h5_to_df_can_convert_raw_gfrm_path(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_measurement, make_session, make_set
    from container.v0_3 import build_session_container
    from xrd_preprocessing import extract_gfrm_archive, gfrm_to_photons

    raw_root = extract_gfrm_archive(GFRM_ARCHIVE, tmp_path / "raw")
    expected, _ = gfrm_to_photons(raw_root / WATER_20MM_REL)
    measurement = make_measurement(
        file_path=str(WATER_20MM_REL),
        data=None,
    )
    _, _, path = build_session_container(
        make_session(sets=[make_set(measurements=[measurement], raw=None, processed=None)]),
        tmp_path / "container",
    )

    _, meas_df = h5_to_df(Path(path), raw_root=raw_root)

    row = meas_df.iloc[0]
    assert row["measurement_data_source"] == "gfrm_to_photons"
    assert row["gfrm_path"].endswith("20260608_112438_Water_20mm_Main.gfrm")
    assert row["gfrm_conversion_metadata"]["baseline_adu"] == 64.0
    np.testing.assert_allclose(row["measurement_data"], expected, rtol=0, atol=0, equal_nan=True)


def test_h5_to_df_requires_patient_and_specimen_ids(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_measurement, make_session, make_set
    from container.v0_3 import build_session_container
    from xrd_preprocessing import extract_gfrm_archive

    raw_root = extract_gfrm_archive(GFRM_ARCHIVE, tmp_path / "raw")
    measurement = make_measurement(
        file_path=str(WATER_20MM_REL),
        data=None,
    )
    _, _, path = build_session_container(
        make_session(
            sets=[make_set(measurements=[measurement], raw=None, processed=None)],
            patient_clinical_name=None,
            sample_clinical_name=None,
        ),
        tmp_path / "container",
    )

    with pytest.raises(ValueError, match="empty required clinical IDs"):
        h5_to_df(Path(path), raw_root=raw_root)
