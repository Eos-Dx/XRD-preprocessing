from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd
import yaml

import xrd_preprocessing.config as config_module
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

    assert config["xrd_preprocessing"]["release_tag"] == "v0.1.5-beta"
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


def test_preprocessing_config_can_extend_local_yaml(tmp_path: Path):
    base = load_preprocessing_config("preprocessing_branch_config_template.yaml")
    base["metadata"]["output_columns"] = ["patientId", "radial_profile_data"]
    base_path = tmp_path / "base.yaml"
    child_path = tmp_path / "minimal.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    child_path.write_text(
        yaml.safe_dump(
            {
                "extends": "base.yaml",
                "io": {"output_joblib_path": "minimal.joblib"},
                "metadata": {"output_columns": ["patientId", "specimenId"]},
            }
        ),
        encoding="utf-8",
    )

    config = load_preprocessing_config(child_path)

    assert config["raw_data"]["source"] == "gfrm"
    assert config["metadata"]["output_columns"] == ["patientId", "specimenId"]
    assert config["io"]["output_joblib_path"] == "minimal.joblib"


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


def test_preprocessing_config_rejects_unknown_bundled_name_and_bad_mapping():
    with pytest.raises(FileNotFoundError, match="Unknown bundled preprocessing config"):
        preprocessing_config_path("missing.yaml")

    with pytest.raises(FileNotFoundError, match="Preprocessing config not found"):
        load_preprocessing_config("missing.yaml")

    with pytest.raises(TypeError, match="must be a mapping"):
        validate_preprocessing_config([])


def test_preprocessing_config_validation_rejects_missing_sections_and_raw_source():
    config = load_preprocessing_config()
    config.pop("raw_data")
    with pytest.raises(ValueError, match="Missing preprocessing config sections"):
        validate_preprocessing_config(config)

    config = load_preprocessing_config()
    config.pop("one_to_one")
    config.pop("one_to_many")
    with pytest.raises(ValueError, match="Missing preprocessing branch contract"):
        validate_preprocessing_config(config)

    config = load_preprocessing_config()
    config["aramis_preprocessing"] = {"branch": "bad"}
    config["branch_settings"] = {}
    with pytest.raises(ValueError, match="Unknown branch-specific"):
        validate_preprocessing_config(config)

    config = load_preprocessing_config()
    config["raw_data"]["source"] = "bad"
    with pytest.raises(ValueError, match="not in allowed_sources"):
        validate_preprocessing_config(config)


def test_preprocessing_config_validation_rejects_sample_column_mismatch():
    config = load_preprocessing_config()
    config["filters"]["thickness"]["sample"]["column"] = "wrong_thickness"

    with pytest.raises(ValueError, match="Sample-thickness"):
        validate_preprocessing_config(config)


def test_preprocessing_config_extend_errors_and_path_edge(monkeypatch, tmp_path):
    child_path = tmp_path / "child.yaml"
    child_path.write_text("extends: missing.yaml\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Unknown preprocessing config template"):
        load_preprocessing_config(child_path)

    monkeypatch.setattr(
        config_module,
        "Path",
        lambda _source: (_ for _ in ()).throw(OSError("bad path")),
    )
    assert config_module._looks_like_existing_path("bad") is False
