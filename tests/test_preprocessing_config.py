from __future__ import annotations

import pytest
import pandas as pd

from xrd_preprocessing import (
    DEFAULT_PREPROCESSING_CONFIG,
    H5SessionFilter,
    available_preprocessing_configs,
    filter_h5_session_df,
    load_preprocessing_config,
    preprocessing_config_path,
    validate_preprocessing_config,
)


def test_bundled_preprocessing_template_contract():
    assert DEFAULT_PREPROCESSING_CONFIG in available_preprocessing_configs()
    path = preprocessing_config_path(DEFAULT_PREPROCESSING_CONFIG)
    assert path.name == DEFAULT_PREPROCESSING_CONFIG

    config = load_preprocessing_config()

    assert config["xrd_preprocessing"]["release_tag"] == "v0.1.4-beta"
    assert config["preprocessing"]["version"] == "0.1-template"
    assert config["raw_data"]["source"] == "gfrm"
    assert "io" in config
    assert config["raw_data"]["allowed_sources"] == ["gfrm", "npy", "tiff"]
    assert config["integration"]["npt"] == 100
    assert config["snr"]["min_snr_db"] == 18.0
    assert config["filters"]["thickness"]["calibrant"]["column"] == (
        "calibrant_thickness_mm"
    )
    assert config["integration"]["thickness_correction"][
        "calibrant_thickness_column"
    ] == "calibrant_thickness_mm"


def test_bundled_branch_preprocessing_template_contract():
    assert "preprocessing_branch_config_template.yaml" in available_preprocessing_configs()
    path = preprocessing_config_path("preprocessing_branch_config_template.yaml")
    assert path.name == "preprocessing_branch_config_template.yaml"

    config = load_preprocessing_config("preprocessing_branch_config_template.yaml")

    assert config["preprocessing"]["version"] == "0.1-branch-template"
    assert config["aramis_preprocessing"]["branch"] == "one_to_one"
    assert "branch_settings" in config
    assert "one_to_one" not in config
    assert "one_to_many" not in config
    assert config["filters"]["require_biopsy"] is False
    assert config["filters"]["biopsy_column"] == "biopsy"
    assert "quality_exclusions" in config["filters"]
    assert config["integration"]["npt"] == 100
    assert config["integration"]["thickness_correction"][
        "calibrant_thickness_column"
    ] == "calibrant_thickness_mm"


def test_preprocessing_config_validation_rejects_column_mismatch():
    config = load_preprocessing_config()
    config["filters"]["thickness"]["calibrant"]["column"] = "agbh_thickness_mm"

    with pytest.raises(ValueError, match="Calibrant-thickness"):
        validate_preprocessing_config(config)


def test_preprocessing_config_validation_accepts_branch_specific_contract():
    config = load_preprocessing_config()
    config.pop("one_to_one")
    config.pop("one_to_many")
    config["aramis_preprocessing"] = {"branch": "one_to_one"}
    config["branch_settings"] = {
        "specimen_status_keep": ["BENIGN", "CANCER", "NORMAL"],
        "min_measurements_per_specimen_after_snr": 1,
        "min_specimens_per_patient_after_snr": 2,
    }

    validate_preprocessing_config(config)


def test_preprocessing_config_can_extend_bundled_template():
    config = load_preprocessing_config(
        {
            "extends": "preprocessing_branch_config_template.yaml",
            "aramis_preprocessing": {
                "branch": "one_to_many",
                "decision_unit": "specimenId",
            },
            "filters": {"require_biopsy": True},
            "branch_settings": {
                "specimen_status_keep": ["BENIGN", "CANCER"],
                "min_measurements_per_specimen_after_snr": 1,
            },
        }
    )

    assert config["aramis_preprocessing"]["branch"] == "one_to_many"
    assert config["filters"]["require_biopsy"] is True
    assert config["raw_data"]["source"] == "gfrm"
    assert config["integration"]["npt"] == 100


def test_h5_session_filter_uses_fallback_when_primary_column_is_missing():
    frame = pd.DataFrame(
        {
            "started_at": ["2026-03-16T09:00:00", "2026-03-17T09:00:00"],
            "category": ["SAMPLE", "SAMPLE"],
        }
    )
    filters = [
        H5SessionFilter(
            column="linked_agbh_session_uid",
            op="not in",
            values=["bad-session"],
            fallback={
                "column": "started_at",
                "op": "date not in",
                "values": ["2026-03-16"],
            },
        )
    ]

    filtered = filter_h5_session_df(frame, filters=filters, session_category="SAMPLE")

    assert filtered["started_at"].tolist() == ["2026-03-17T09:00:00"]
