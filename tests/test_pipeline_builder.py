from __future__ import annotations

import pytest
import yaml

from xrd_preprocessing import (
    build_pipeline_from_config,
    build_pipeline_steps_from_config,
    load_preprocessing_config,
    transformer_registry,
)
from xrd_preprocessing.transformers import (
    ConstantQRangeTransformer,
    KeepColumnsTransformer,
)


def test_build_pipeline_from_yaml_steps_and_refs():
    config = {
        "metadata": {"output_columns": ["q_range", "radial_profile_data"]},
        "integration": {"q_range_nm_inv": [2.0, 23.0]},
        "pipeline": {
            "steps": [
                {
                    "name": "q_grid",
                    "transformer": "ConstantQRangeTransformer",
                    "params": {
                        "q_min": {"$ref": "integration.q_range_nm_inv.0"},
                        "q_max": {"$ref": "integration.q_range_nm_inv.1"},
                        "output_column": "interpolation_q_range",
                    },
                },
                {
                    "name": "keep_columns",
                    "transformer": "KeepColumnsTransformer",
                    "params": {"columns": {"$ref": "metadata.output_columns"}},
                },
            ]
        },
    }

    pipeline = build_pipeline_from_config(config)

    assert [name for name, _ in pipeline.steps] == ["q_grid", "keep_columns"]
    assert isinstance(pipeline.steps[0][1], ConstantQRangeTransformer)
    assert pipeline.steps[0][1].q_min == 2.0
    assert pipeline.steps[0][1].q_max == 23.0
    assert isinstance(pipeline.steps[1][1], KeepColumnsTransformer)
    assert pipeline.steps[1][1].columns == ("q_range", "radial_profile_data")


def test_transformer_registry_returns_copy_and_nested_list_refs():
    registry = transformer_registry()
    registry.clear()
    config = {
        "columns": ["patientId", "q_range"],
        "pipeline": {
            "steps": [
                {
                    "transformer": "KeepColumnsTransformer",
                    "params": {
                        "columns": [
                            {"$ref": "columns.0"},
                            {"$ref": "columns.1"},
                        ]
                    },
                },
            ]
        },
    }

    steps = build_pipeline_steps_from_config(config)

    assert "KeepColumnsTransformer" in transformer_registry()
    assert steps[0][0] == "step_1"
    assert steps[0][1].columns == ("patientId", "q_range")


def test_pipeline_builder_rejects_empty_missing_unknown_and_skips_disabled():
    with pytest.raises(ValueError, match="requires pipeline.steps"):
        build_pipeline_steps_from_config({})

    with pytest.raises(ValueError, match="missing transformer"):
        build_pipeline_steps_from_config({"pipeline": {"steps": [{"name": "bad"}]}})

    with pytest.raises(ValueError, match="Unknown pipeline transformer"):
        build_pipeline_steps_from_config(
            {"pipeline": {"steps": [{"transformer": "UnknownTransformer"}]}}
        )

    with pytest.raises(ValueError, match="Duplicate pipeline step name"):
        build_pipeline_steps_from_config(
            {
                "pipeline": {
                    "steps": [
                        {
                            "name": "keep",
                            "transformer": "KeepColumnsTransformer",
                            "params": {"columns": []},
                        },
                        {
                            "name": "keep",
                            "transformer": "KeepColumnsTransformer",
                            "params": {"columns": []},
                        },
                    ]
                }
            }
        )


def test_pipeline_builder_resolves_enabled_refs_inside_params():
    config = {
        "flags": {"keep_step": True, "keep_filter": False},
        "pipeline": {
            "steps": [
                {
                    "name": "status_filter",
                    "transformer": "H5ToDataFrameTransformer",
                    "enabled": {"$ref": "flags.keep_step"},
                    "params": {
                        "h5_filters": [
                            {
                                "enabled": {"$ref": "flags.keep_filter"},
                                "column": "biopsy",
                                "op": "==",
                                "value": True,
                            },
                            {"column": "specimen_status", "op": "in", "values": ["BENIGN"]},
                        ]
                    },
                }
            ]
        },
    }

    steps = build_pipeline_steps_from_config(config)

    assert len(steps) == 1
    assert steps[0][1].h5_filters == [
        {"column": "specimen_status", "op": "in", "values": ["BENIGN"]}
    ]

    with pytest.raises(ValueError, match="no enabled pipeline steps"):
        build_pipeline_steps_from_config(
            {
                "pipeline": {
                    "steps": [
                        {
                            "name": "disabled",
                            "transformer": "KeepColumnsTransformer",
                            "enabled": False,
                            "params": {"columns": []},
                        }
                    ]
                }
            }
        )


def test_pipeline_builder_concatenates_config_refs():
    config = {
        "left": ["a", "b"],
        "right": ["c"],
        "pipeline": {
            "steps": [
                {
                    "name": "keep",
                    "transformer": "KeepColumnsTransformer",
                    "params": {
                        "columns": {
                            "$concat": [
                                {"$ref": "left"},
                                {"$ref": "right"},
                                "d",
                            ],
                        },
                    },
                },
            ],
        },
    }

    steps = build_pipeline_steps_from_config(config)

    assert steps[0][1].columns == ("a", "b", "c", "d")


def test_pipeline_builder_reports_invalid_config_ref():
    with pytest.raises(ValueError, match="Invalid config ref 'missing.value'"):
        build_pipeline_steps_from_config(
            {
                "pipeline": {
                    "steps": [
                        {
                            "transformer": "KeepColumnsTransformer",
                            "params": {"columns": {"$ref": "missing.value"}},
                        },
                    ]
                }
            }
        )


def test_preprocessing_config_loads_multiple_local_extends(tmp_path):
    (tmp_path / "base_a.yaml").write_text(
        yaml.safe_dump(
            {
                "raw_data": {
                    "source": "gfrm",
                    "allowed_sources": ["gfrm"],
                },
                "metadata": {"output_columns": ["a"]},
                "filters": {
                    "thickness": {
                        "sample": {"column": "sample_thickness_mm"},
                        "calibrant": {"column": "calibrant_thickness_mm"},
                    },
                },
                "labels": {},
                "integration": {
                    "thickness_correction": {
                        "sample_thickness_column": "sample_thickness_mm",
                        "calibrant_thickness_column": "calibrant_thickness_mm",
                    },
                },
                "snr": {},
                "normalization": {},
                "profile_gate": {},
                "branch_settings": {"output_columns": []},
                "pipeline": {
                    "steps": [
                        {
                            "name": "keep",
                            "transformer": "KeepColumnsTransformer",
                            "params": {"columns": {"$ref": "metadata.output_columns"}},
                        },
                    ],
                },
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "base_b.yaml").write_text(
        yaml.safe_dump({"metadata": {"output_columns": ["b"]}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "extends": ["base_a.yaml", "base_b.yaml"],
                "metadata": {"keep_columns_errors": "raise"},
            },
        ),
        encoding="utf-8",
    )

    config = load_preprocessing_config(config_path)

    assert config["metadata"]["output_columns"] == ["b"]
    assert config["metadata"]["keep_columns_errors"] == "raise"
