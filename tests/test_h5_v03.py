import sys
from pathlib import Path

import numpy as np
import pytest
import h5py

sys.path.insert(0, "/Users/sad/dev/container/src")
sys.path.insert(0, "/Users/sad/dev/container/tests/v0_3")

from xrd_preprocessing import h5_to_df  # noqa: E402

GFRM_ARCHIVE = Path("examples/data/gfrm_measurements.tar.gz")
WATER_20MM_REL = (
    Path("GFRM_measurements")
    / "Water"
    / "20260608_112438_Water_20mm"
    / "20260608_112438_Water_20mm_Main.gfrm"
)


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


def test_h5_to_df_rejects_non_gfrm_data_preference(tmp_path):
    pytest.importorskip("container.v0_3")
    from _factory_v0_3 import make_session, make_set
    from container.v0_3 import build_session_container

    _, _, path = build_session_container(
        make_session(sets=[make_set(raw=np.ones((2, 2), dtype=np.float32))]),
        tmp_path,
    )

    with pytest.raises(ValueError, match="requires data_preference='gfrm'"):
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
