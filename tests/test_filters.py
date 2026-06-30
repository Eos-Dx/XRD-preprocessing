import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from xrd_preprocessing import (
    ColumnValueFilter,
    MetadataFilter,
    PatientFilter,
    PatientSpecimenValidityFilter,
    PoniQRangeFilter,
    RadialProfileValueFilter,
    SNRFilter,
    SpecimenValidityFilter,
    estimate_poni_q_range_nm_inv,
)


def test_column_value_filter_keeps_allowed_metadata_values():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "diagnosis": ["BENIGN", "CANCER", "CONTROL"],
        }
    )

    out = ColumnValueFilter("diagnosis", values=["BENIGN", "CANCER"]).transform(df)

    assert out["sample_id"].tolist() == ["a", "b"]


def test_column_value_filter_numeric_comparison():
    df = pd.DataFrame({"sample_id": ["a", "b", "c"], "score": [19.0, 20.0, 25.0]})

    out = ColumnValueFilter("score", op=">=", value=20.0).transform(df)

    assert out["sample_id"].tolist() == ["b", "c"]


def test_column_value_filter_between_and_contains():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "score": [0.2, 0.5, 0.9],
            "comment": ["normal", "needs biopsy", "mri followup"],
        }
    )

    between = ColumnValueFilter("score", op="between", lower=0.3, upper=0.8)
    contains = ColumnValueFilter("comment", op="contains", value="biopsy")

    assert between.transform(df)["sample_id"].tolist() == ["b"]
    assert contains.transform(df)["sample_id"].tolist() == ["b"]


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("not_in", ["bad"]),
        ("!=", ["edge", "new", "bad"]),
        (">", ["new"]),
        ("<", ["old"]),
        ("<=", ["old", "edge"]),
        ("date_after", ["new"]),
        ("date_before", ["old"]),
        ("date<=", ["old", "edge"]),
        ("date_between", ["edge", "new"]),
        ("date in", ["edge", "new"]),
        ("isna", ["bad"]),
        ("notna", ["old", "edge", "new"]),
    ],
)
def test_column_value_filter_branch_ops(op, expected):
    df = pd.DataFrame(
        {
            "sample_id": ["old", "edge", "new", "bad"],
            "score": [0.2, 0.5, 0.9, np.nan],
            "measurementDate": ["2026-01-10", "2026-06-01", "2026-06-02", None],
        }
    )
    kwargs = {
        "not_in": {"values": ["old", "edge", "new"]},
        "!=": {"value": "old"},
        ">": {"value": 0.5},
        "<": {"value": 0.5},
        "<=": {"value": 0.5},
        "date_after": {"value": "2026-06-01"},
        "date_before": {"value": "2026-06-01"},
        "date<=": {"value": "2026-06-01"},
        "date_between": {"lower": "2026-06-01", "upper": "2026-06-02"},
        "date in": {"values": ["2026-06-01", "2026-06-02"]},
        "isna": {},
        "notna": {},
    }[op]
    column = "score" if op in {">", "<", "<=", "isna", "notna"} else "measurementDate"
    if op in {"not_in", "!="}:
        column = "sample_id"

    out = ColumnValueFilter(column, op=op, **kwargs).transform(df)

    assert out["sample_id"].tolist() == expected


@pytest.mark.parametrize(
    ("op", "kwargs", "message"),
    [
        ("in", {}, "values must be provided"),
        ("not_in", {}, "values must be provided"),
        ("between", {}, "lower and upper"),
        ("date_between", {}, "lower and upper"),
        ("date in", {}, "values must be provided"),
        ("unsupported", {}, "Unsupported"),
    ],
)
def test_column_value_filter_rejects_invalid_ops(op, kwargs, message):
    df = pd.DataFrame({"value": [1.0]})

    with pytest.raises(ValueError, match=message):
        ColumnValueFilter("value", op=op, **kwargs).transform(df)


def test_column_value_filter_keep_na_and_no_reset_index():
    df = pd.DataFrame({"value": [1.0, np.nan, 3.0]}, index=[10, 11, 12])

    out = ColumnValueFilter(
        "value",
        op=">",
        value=2.0,
        keep_na=True,
        reset_index=False,
    ).transform(df)

    assert out.index.tolist() == [11, 12]


def test_column_value_filter_date_cutoff():
    df = pd.DataFrame(
        {
            "sample_id": ["old", "edge", "new", "bad"],
            "measurementDate": ["2026-01-10", "2026-06-01", "2026-06-02", None],
        }
    )

    out = ColumnValueFilter(
        "measurementDate",
        op="date>=",
        value="2026-06-01",
    ).transform(df)

    assert out["sample_id"].tolist() == ["edge", "new"]


def test_metadata_filter_uses_one_column_rule():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "diagnosis": ["BENIGN", "CANCER", "CONTROL"],
        }
    )

    filt = MetadataFilter(
        "diagnosis",
        op="in",
        values=["BENIGN", "CANCER"],
    )
    out = filt.transform(df)

    assert out["sample_id"].tolist() == ["a", "b"]
    assert filt.stats_["filter_type"] == "metadata"
    assert filt.stats_["rows_in"] == 3
    assert filt.stats_["rows_pass"] == 2
    assert filt.stats_["rows_fail"] == 1


def test_multiple_column_filters_are_composed_as_pipeline():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "diagnosis": ["BENIGN", "CANCER", "CANCER", "CONTROL"],
            "scan_type": ["water", "water", "calibrant", "water"],
        }
    )

    pipeline = Pipeline(
        [
            (
                "diagnosis_filter",
                MetadataFilter("diagnosis", values=["BENIGN", "CANCER"]),
            ),
            (
                "scan_type_filter",
                ColumnValueFilter("scan_type", op="==", value="water"),
            ),
        ]
    )
    out = pipeline.fit_transform(df)

    assert out["sample_id"].tolist() == ["a", "b"]


def test_patient_filter_uses_metadata_rule():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "diagnosis": ["BENIGN", "CANCER"],
        }
    )

    out = PatientFilter("diagnosis", op="==", value="CANCER").transform(df)

    assert out["sample_id"].tolist() == ["b"]


def test_metadata_filter_missing_column_raises_key_error():
    df = pd.DataFrame({"diagnosis": ["BENIGN"]})

    with pytest.raises(KeyError, match="missing"):
        MetadataFilter("missing", values=["BENIGN"]).transform(df)


def test_snr_filter_accepts_custom_snr_column():
    df = pd.DataFrame(
        {
            "sample_id": ["low", "ok"],
            "snr_db": [19.0, 21.0],
        }
    )

    filt = SNRFilter(snr_column="snr_db", min_snr_db=20.0)
    out = filt.transform(df)

    assert out["sample_id"].tolist() == ["ok"]
    assert filt.stats_["failed_ids"] == ["low"]


def test_snr_filter_default_column_is_snr_db():
    df = pd.DataFrame({"sample_id": ["low", "ok"], "snr_db": [19.0, 21.0]})

    out = SNRFilter().transform(df)

    assert out["sample_id"].tolist() == ["ok"]
    assert SNRFilter().snr_column == "snr_db"


def test_patient_specimen_validity_filter_keeps_specimens_with_two_measurements():
    df = pd.DataFrame(
        {
            "patientId": ["p1", "p1", "p1", "p2"],
            "specimenId": ["left", "left", "right", "left"],
            "scan_id": ["a", "b", "c", "d"],
        }
    )

    filt = PatientSpecimenValidityFilter()
    out = filt.transform(df)

    assert out["scan_id"].tolist() == ["a", "b"]
    assert out["specimen_measurement_count"].tolist() == [2, 2]
    assert out["patient_valid_specimen_count"].tolist() == [1, 1]
    assert filt.stats_["rows_in"] == 4
    assert filt.stats_["rows_pass"] == 2


def test_patient_specimen_validity_filter_can_require_two_specimens_per_patient():
    df = pd.DataFrame(
        {
            "patientId": ["p1", "p1", "p1", "p1", "p2", "p2"],
            "specimenId": ["left", "left", "right", "right", "left", "left"],
            "scan_id": ["a", "b", "c", "d", "e", "f"],
        }
    )

    out = PatientSpecimenValidityFilter(min_specimens_per_patient=2).transform(df)

    assert out["scan_id"].tolist() == ["a", "b", "c", "d"]


def test_patient_specimen_validity_filter_marks_reasons_when_not_dropping():
    df = pd.DataFrame(
        {
            "patientId": ["p1", "p1", "p2", None],
            "specimenId": ["left", "left", "left", "left"],
            "scan_id": ["a", "b", "c", "d"],
        }
    )

    out = PatientSpecimenValidityFilter(drop=False).transform(df)

    assert out["patient_specimen_valid"].tolist() == [True, True, False, False]
    assert out["patient_specimen_validity_reason"].tolist() == [
        "valid",
        "valid",
        "specimen_measurements_below_minimum",
        "missing_patient_or_specimen_id",
    ]


def test_patient_specimen_validity_filter_requires_standard_id_columns():
    df = pd.DataFrame({"patient_id": ["p1"], "specimen_id": ["left"]})

    with pytest.raises(KeyError, match="patientId"):
        PatientSpecimenValidityFilter().transform(df)


def test_patient_specimen_validity_filter_rejects_invalid_minimums():
    df = pd.DataFrame({"patientId": ["p1"], "specimenId": ["s1"]})

    with pytest.raises(ValueError, match="min_measurements"):
        PatientSpecimenValidityFilter(min_measurements_per_specimen=0).transform(df)
    with pytest.raises(ValueError, match="min_specimens"):
        PatientSpecimenValidityFilter(min_specimens_per_patient=0).transform(df)


def test_specimen_validity_filter_uses_specimen_id_only():
    df = pd.DataFrame(
        {
            "patientId": ["p1", "p2", "p3", "p4"],
            "specimenId": ["left", "left", "right", ""],
            "scan_id": ["a", "b", "c", "d"],
        }
    )

    filt = SpecimenValidityFilter(min_measurements_per_specimen=2)
    out = filt.transform(df)

    assert out["scan_id"].tolist() == ["a", "b"]
    assert out["specimen_measurement_count"].tolist() == [2, 2]
    assert filt.stats_["specimens_pass"] == 1
    assert filt.stats_["rows_pass"] == 2
    assert "patient_valid_specimen_count" not in out.columns


def test_specimen_validity_filter_marks_reasons_when_not_dropping():
    df = pd.DataFrame(
        {
            "specimenId": ["left", "left", "right", None],
            "scan_id": ["a", "b", "c", "d"],
        }
    )

    out = SpecimenValidityFilter(
        min_measurements_per_specimen=2,
        drop=False,
    ).transform(df)

    assert out["specimen_valid"].tolist() == [True, True, False, False]
    assert out["specimen_validity_reason"].tolist() == [
        "valid",
        "valid",
        "specimen_measurements_below_minimum",
        "missing_specimen_id",
    ]


def test_specimen_validity_filter_rejects_missing_column_and_invalid_minimum():
    with pytest.raises(KeyError, match="specimenId"):
        SpecimenValidityFilter().transform(pd.DataFrame({"id": ["s1"]}))
    with pytest.raises(ValueError, match="min_measurements"):
        SpecimenValidityFilter(min_measurements_per_specimen=0).transform(
            pd.DataFrame({"specimenId": ["s1"]})
        )


def _fake_poni(distance_m: float = 0.1) -> str:
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


def test_estimate_poni_q_range_uses_poni_geometry():
    _q_min, q_max, distance = estimate_poni_q_range_nm_inv(
        _fake_poni(0.1),
        shape=(16, 16),
    )

    assert q_max > 0.6
    assert distance == pytest.approx(0.1)


def test_poni_q_range_filter_keeps_rows_that_cover_requested_q_max():
    image = np.zeros((16, 16), dtype=np.float32)
    df = pd.DataFrame(
        {
            "sample_id": ["short", "long"],
            "measurement_data": [image, image],
            "ponifile": [_fake_poni(0.1), _fake_poni(1.0)],
        }
    )

    filt = PoniQRangeFilter(required_q_max_nm_inv=0.5)
    out = filt.transform(df)

    assert out["sample_id"].tolist() == ["short"]
    assert out["poni_q_range_pass"].tolist() == [True]
    assert filt.stats_["rows_in"] == 2
    assert filt.stats_["rows_pass"] == 1
    assert filt.stats_["failed_ids"] == ["long"]


def test_poni_q_range_filter_can_use_thickness_adjusted_distance():
    image = np.zeros((16, 16), dtype=np.float32)
    df = pd.DataFrame(
        {
            "sample_id": ["without_adjustment", "with_adjustment"],
            "measurement_data": [image, image],
            "ponifile": [_fake_poni(0.1), _fake_poni(0.1)],
            "sample_thickness_mm": [11.0, 111.0],
        }
    )

    filt = PoniQRangeFilter(
        required_q_max_nm_inv=1.2,
        thickness_adjustment=True,
        thickness_reference_mm=11.0,
    )
    out = filt.transform(df)

    assert out["sample_id"].tolist() == ["with_adjustment"]
    assert out["poni_calculated_distance_m"].iloc[0] == pytest.approx(0.05)


def test_radial_profile_value_filter_keeps_profiles_above_threshold_near_q():
    q = np.array([13.8, 14.05, 14.3])
    df = pd.DataFrame(
        {
            "sample_id": ["air", "signal"],
            "q_range": [q, q],
            "radial_profile_data": [
                np.array([0.5, 1.2, 0.7]),
                np.array([0.5, 2.4, 0.7]),
            ],
        }
    )

    filt = RadialProfileValueFilter(q_value_nm_inv=14.0, threshold=2.0, op=">")
    out = filt.transform(df)

    assert out["sample_id"].tolist() == ["signal"]
    assert out["radial_profile_nearest_q_nm_inv"].iloc[0] == pytest.approx(14.05)
    assert out["radial_profile_value_at_q"].iloc[0] == pytest.approx(2.4)
    assert filt.stats_["rows_in"] == 2
    assert filt.stats_["rows_pass"] == 1
    assert filt.stats_["failed_ids"] == ["air"]


def test_radial_profile_value_filter_can_require_nearby_q_point():
    df = pd.DataFrame(
        {
            "sample_id": ["far"],
            "q_range": [np.array([13.0, 13.2])],
            "radial_profile_data": [np.array([10.0, 10.0])],
        }
    )

    filt = RadialProfileValueFilter(
        q_value_nm_inv=14.0,
        threshold=2.0,
        max_q_delta_nm_inv=0.2,
    )
    out = filt.transform(df)

    assert out.empty
    assert filt.stats_["rows_fail"] == 1


def test_radial_profile_value_filter_error_branches():
    with pytest.raises(KeyError, match="Missing required radial profile"):
        RadialProfileValueFilter(q_value_nm_inv=14.0, threshold=2.0).transform(
            pd.DataFrame({"q_range": [[1, 2]]})
        )
    with pytest.raises(ValueError, match="same shape"):
        RadialProfileValueFilter(q_value_nm_inv=14.0, threshold=2.0).transform(
            pd.DataFrame(
                {
                    "q_range": [np.array([1.0, 2.0])],
                    "radial_profile_data": [np.array([1.0])],
                }
            )
        )
    with pytest.raises(ValueError, match="Unsupported"):
        RadialProfileValueFilter(
            q_value_nm_inv=14.0,
            threshold=2.0,
            op="bad",
        ).transform(
            pd.DataFrame(
                {
                    "q_range": [np.array([14.0])],
                    "radial_profile_data": [np.array([1.0])],
                }
            )
        )


@pytest.mark.parametrize("op", [">=", "<", "<="])
def test_radial_profile_value_filter_compare_ops(op):
    q = np.array([14.0])
    df = pd.DataFrame(
        {
            "sample_id": ["row"],
            "q_range": [q],
            "radial_profile_data": [np.array([2.0])],
        }
    )

    out = RadialProfileValueFilter(q_value_nm_inv=14.0, threshold=2.0, op=op).transform(
        df
    )

    assert len(out) == (1 if op in {">=", "<="} else 0)


def test_snr_filter_missing_column_and_no_finite_values():
    with pytest.raises(KeyError, match="snr_db"):
        SNRFilter().transform(pd.DataFrame({"id": ["a"]}))

    out = SNRFilter(drop=False).transform(
        pd.DataFrame({"sample_id": ["a"], "snr_db": [np.nan]})
    )

    assert out["snr_pass"].tolist() == [False]
    assert np.isnan(SNRFilter(drop=False).fit_transform(out)["snr_db"].iloc[0])
