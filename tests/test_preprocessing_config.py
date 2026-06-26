from __future__ import annotations

import pytest

from xrd_preprocessing import (
    DEFAULT_PREPROCESSING_CONFIG,
    available_preprocessing_configs,
    load_preprocessing_config,
    preprocessing_config_path,
    validate_preprocessing_config,
)


def test_bundled_preprocessing_template_contract():
    assert DEFAULT_PREPROCESSING_CONFIG in available_preprocessing_configs()
    path = preprocessing_config_path(DEFAULT_PREPROCESSING_CONFIG)
    assert path.name == DEFAULT_PREPROCESSING_CONFIG

    config = load_preprocessing_config()

    assert config["xrd_preprocessing"]["release_tag"] == "v0.1.2-beta"
    assert config["preprocessing"]["version"] == "0.1-template"
    assert config["raw_data"]["source"] == "gfrm"
    assert config["raw_data"]["allowed_sources"] == ["gfrm", "npy", "tiff"]
    assert config["integration"]["npt"] == 100
    assert config["snr"]["min_snr_db"] == 18.0
    assert config["filters"]["thickness"]["calibrant"]["column"] == (
        "calibrant_thickness_mm"
    )
    assert config["integration"]["thickness_correction"][
        "calibrant_thickness_column"
    ] == "calibrant_thickness_mm"


def test_preprocessing_config_validation_rejects_column_mismatch():
    config = load_preprocessing_config()
    config["filters"]["thickness"]["calibrant"]["column"] = "agbh_thickness_mm"

    with pytest.raises(ValueError, match="Calibrant-thickness"):
        validate_preprocessing_config(config)
