from pathlib import Path

import pandas as pd
import pytest

from xrd_preprocessing import (
    ColumnValueFilter,
    ConstantQRangeTransformer,
    DropColumnsTransformer,
    H5BlobDataFrameTransformer,
    H5ToDataFrameTransformer,
    JoblibWriterTransformer,
    PairedGroupFilter,
    ProductColumnBuilder,
    ProductStatusGroupFilter,
    RequiredColumnsTransformer,
    SimpleRadialProfileTransformer,
)


def _fake_h5_reader(path, **kwargs):
    _fake_h5_reader.kwargs = kwargs
    calibration = pd.DataFrame({"cal_id": ["agbh_1"]})
    measurement = pd.DataFrame(
        {
            "id": [Path(path).name],
            "specimen_status": ["BENIGN"],
            "sample_thickness_mm": [11.0],
        }
    )
    measurement.attrs["dropped_missing_sample_thickness"] = 2
    return calibration, measurement


def test_h5_to_dataframe_transformer_returns_measurements_and_keeps_calibration():
    transformer = H5ToDataFrameTransformer(
        reader=_fake_h5_reader,
        measurement_filters=[{"column": "position", "op": "in", "values": ["P1"]}],
    )

    out = transformer.fit_transform("/tmp/synthetic.h5")

    assert out["id"].tolist() == ["synthetic.h5"]
    assert transformer.calibration_df_["cal_id"].tolist() == ["agbh_1"]
    assert transformer.stats_["measurement_rows"] == 1
    assert transformer.stats_["dropped_missing_sample_thickness"] == 2
    assert _fake_h5_reader.kwargs["measurement_filters"] == [
        {"column": "position", "op": "in", "values": ["P1"]}
    ]


def test_product_column_builder_groups_status_at_specimen_level():
    df = pd.DataFrame(
        {
            "patientId": ["p1", "p1", "p2"],
            "specimen_status": ["BENIGN", "ATYPICAL", "NORMAL"],
            "started_at": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "sample_thickness": [11, 12, 13],
            "calibrant_thickness_mm": [40, 40, 10],
        }
    )

    out = ProductColumnBuilder().fit_transform(df)

    assert out["product_status_group"].tolist() == ["BENIGN", "CANCER", "NORMAL"]
    assert out["product_diagnosis"].tolist()[:2] == ["BENIGN", "CANCER"]
    assert out["patient_product_diagnosis"].tolist()[:2] == ["CANCER", "CANCER"]
    assert out["sample_thickness_mm"].tolist() == [11, 12, 13]


def test_product_status_and_paired_group_filters_are_transformers():
    df = ProductColumnBuilder().fit_transform(
        pd.DataFrame(
            {
                "patientId": ["p1", "p1", "p2", "p2", "p3", "p3"],
                "specimenId": ["l", "r", "l", "r", "l", "r"],
                "specimen_status": [
                    "BENIGN",
                    "CANCER",
                    "BENIGN",
                    "NORMAL",
                    "CANCER",
                    "CANCER",
                ],
            }
        )
    )

    grouped = ProductStatusGroupFilter(["BENIGN", "CANCER", "NORMAL"]).fit_transform(df)
    paired = PairedGroupFilter().fit_transform(grouped)

    assert grouped["product_status_group"].tolist() == [
        "BENIGN",
        "CANCER",
        "BENIGN",
        "NORMAL",
        "CANCER",
        "CANCER",
    ]
    assert paired["patientId"].tolist() == ["p1", "p1", "p2", "p2"]
    assert set(paired["one_to_one_pair_type"]) == {
        "BENIGN__CANCER",
        "BENIGN__NORMAL",
    }


def test_q_range_drop_columns_and_required_columns_transformers():
    df = pd.DataFrame(
        {
            "sample_thickness_mm": [11.0],
            "calibrant_thickness_mm": [40.0],
            "measurement_data": [[1, 2, 3]],
        }
    )
    checked = RequiredColumnsTransformer(
        {
            "sample_thickness_mm": (0.0, None),
            "calibrant_thickness_mm": (10.0, 40.0),
        }
    ).fit_transform(df)
    ranged = ConstantQRangeTransformer(q_min=2.0, q_max=23.0).fit_transform(checked)
    dropper = DropColumnsTransformer(["measurement_data", "missing"])
    out = dropper.fit_transform(ranged)

    assert out["interpolation_q_range"].tolist() == [(2.0, 23.0)]
    assert "measurement_data" not in out.columns
    assert dropper.dropped_columns_ == ["measurement_data"]


def test_required_columns_transformer_raises_for_invalid_calibrant_thickness():
    df = pd.DataFrame({"calibrant_thickness_mm": [41.0]})

    with pytest.raises(ValueError, match="Invalid values"):
        RequiredColumnsTransformer({"calibrant_thickness_mm": (10.0, 40.0)}).transform(df)


def test_h5_blob_simple_radial_profile_and_joblib_writer(tmp_path):
    import h5py
    import joblib
    import numpy as np

    h5_path = tmp_path / "simple_blob.h5"
    with h5py.File(h5_path, "w") as h5:
        group = h5.require_group("measurements/m1")
        group.attrs["started_at"] = "2026-05-01 10:00:00"
        group.attrs["patientId"] = "P1"
        group.attrs["specimenId"] = "P1_LEFT"
        group.attrs["specimen_status"] = "BENIGN"
        raw = group.require_group("raw")
        raw.create_dataset("data", data=np.ones((16, 16), dtype=float))

    reader = H5BlobDataFrameTransformer(
        source="npy",
        dataset_candidates=("raw/data",),
    )
    radial = SimpleRadialProfileTransformer(npt=12, q_min=2.0, q_max=23.0)
    writer = JoblibWriterTransformer(tmp_path / "out.joblib")

    df = reader.fit_transform(h5_path)
    df = ColumnValueFilter(
        "started_at",
        op="date in",
        values=["2026-05-01"],
    ).fit_transform(df)
    df = radial.fit_transform(df)
    out = writer.fit_transform(df)
    loaded = joblib.load(tmp_path / "out.joblib")

    assert len(out) == 1
    assert out["measurement_data"].iloc[0].shape == (16, 16)
    assert len(out["q_range"].iloc[0]) == 12
    pd.testing.assert_frame_equal(out, loaded)
