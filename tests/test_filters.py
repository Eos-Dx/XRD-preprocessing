import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from xrd_preprocessing import (
    ColumnValueFilter,
    MetadataFilter,
    PatientFilter,
    PatientSpecimenValidityFilter,
    SNRFilter,
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


def test_metadata_filter_is_one_column_alias():
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


def test_patient_filter_is_metadata_filter_alias():
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
