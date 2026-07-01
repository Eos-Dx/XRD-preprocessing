"""Build sklearn-style preprocessing pipelines from YAML config."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from sklearn.pipeline import Pipeline

from .azimuthal import AzimuthalIntegration
from .faulty_pixels import FaultyPixelDetector
from .filters import (
    ColumnValueFilter,
    GroupValueFilter,
    MetadataFilter,
    PatientFilter,
    PatientSpecimenValidityFilter,
    PoniQRangeFilter,
    RadialProfileValueFilter,
    SNRFilter,
    SpecimenValidityFilter,
)
from .normalization import QRangeNormalizer, QRangeValueNormalizer
from .snr import SNRTransformer
from .snapshots import RadialProfileSnapshot
from .transformers import (
    ConstantQRangeTransformer,
    DropColumnsTransformer,
    H5BlobDataFrameTransformer,
    H5MeasurementSetAuditTransformer,
    H5SessionSelectorTransformer,
    H5ToDataFrameTransformer,
    JoblibWriterTransformer,
    KeepColumnsTransformer,
    PairedGroupFilter,
    ProductColumnBuilder,
    ProductStatusGroupFilter,
    RequiredColumnsTransformer,
    SimpleRadialProfileTransformer,
)


TRANSFORMER_REGISTRY = {
    "AzimuthalIntegration": AzimuthalIntegration,
    "ColumnValueFilter": ColumnValueFilter,
    "ConstantQRangeTransformer": ConstantQRangeTransformer,
    "DropColumnsTransformer": DropColumnsTransformer,
    "FaultyPixelDetector": FaultyPixelDetector,
    "GroupValueFilter": GroupValueFilter,
    "H5BlobDataFrameTransformer": H5BlobDataFrameTransformer,
    "H5MeasurementSetAuditTransformer": H5MeasurementSetAuditTransformer,
    "H5SessionSelectorTransformer": H5SessionSelectorTransformer,
    "H5ToDataFrameTransformer": H5ToDataFrameTransformer,
    "JoblibWriterTransformer": JoblibWriterTransformer,
    "KeepColumnsTransformer": KeepColumnsTransformer,
    "MetadataFilter": MetadataFilter,
    "PairedGroupFilter": PairedGroupFilter,
    "PatientFilter": PatientFilter,
    "PatientSpecimenValidityFilter": PatientSpecimenValidityFilter,
    "PoniQRangeFilter": PoniQRangeFilter,
    "ProductColumnBuilder": ProductColumnBuilder,
    "ProductStatusGroupFilter": ProductStatusGroupFilter,
    "QRangeNormalizer": QRangeNormalizer,
    "QRangeValueNormalizer": QRangeValueNormalizer,
    "RadialProfileSnapshot": RadialProfileSnapshot,
    "RadialProfileValueFilter": RadialProfileValueFilter,
    "RequiredColumnsTransformer": RequiredColumnsTransformer,
    "SimpleRadialProfileTransformer": SimpleRadialProfileTransformer,
    "SNRFilter": SNRFilter,
    "SNRTransformer": SNRTransformer,
    "SpecimenValidityFilter": SpecimenValidityFilter,
}

_DISABLED = object()


def transformer_registry() -> dict[str, type]:
    """Return the explicit transformer registry available to YAML pipelines."""
    return dict(TRANSFORMER_REGISTRY)


def build_pipeline_from_config(config: dict[str, Any]) -> Pipeline:
    """Build a sklearn Pipeline from `pipeline.steps` in a loaded config."""
    steps = build_pipeline_steps_from_config(config)
    return Pipeline(steps)


def build_pipeline_steps_from_config(config: dict[str, Any]) -> list[tuple[str, Any]]:
    """Build `(name, transformer)` tuples from `pipeline.steps`."""
    pipeline_config = config.get("pipeline", {})
    step_specs = pipeline_config.get("steps", [])
    if not step_specs:
        raise ValueError("Preprocessing config requires pipeline.steps.")
    steps = []
    seen_names: set[str] = set()
    for index, spec in enumerate(step_specs, start=1):
        enabled = _resolve_refs(spec.get("enabled", True), config)
        if enabled is False:
            continue
        name = str(spec.get("name") or f"step_{index}")
        if name in seen_names:
            raise ValueError(f"Duplicate pipeline step name: {name!r}.")
        seen_names.add(name)
        transformer_name = spec.get("transformer")
        if not transformer_name:
            raise ValueError(f"Pipeline step {name!r} is missing transformer.")
        if transformer_name not in TRANSFORMER_REGISTRY:
            raise ValueError(f"Unknown pipeline transformer: {transformer_name!r}.")
        params = _resolve_refs(deepcopy(spec.get("params", {})), config)
        steps.append((name, TRANSFORMER_REGISTRY[transformer_name](**params)))
    if not steps:
        raise ValueError("Preprocessing config has no enabled pipeline steps.")
    return steps


def _resolve_refs(value: Any, config: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if "enabled" in value:
            enabled = _resolve_refs(value["enabled"], config)
            if enabled is False:
                return _DISABLED
        if set(value) == {"$concat"}:
            concatenated: list[Any] = []
            for item in _resolve_refs(value["$concat"], config):
                if isinstance(item, list):
                    concatenated.extend(item)
                else:
                    concatenated.append(item)
            return concatenated
        if set(value) == {"$ref"}:
            return _config_ref(config, str(value["$ref"]))
        resolved = {
            key: _resolve_refs(item, config)
            for key, item in value.items()
            if key != "enabled"
        }
        return {key: item for key, item in resolved.items() if item is not _DISABLED}
    if isinstance(value, list):
        resolved_items = [_resolve_refs(item, config) for item in value]
        return [item for item in resolved_items if item is not _DISABLED]
    return value


def _config_ref(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    traversed: list[str] = []
    try:
        for part in path.split("."):
            traversed.append(part)
            if isinstance(current, list):
                current = current[int(part)]
            else:
                current = current[part]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        location = ".".join(traversed)
        raise ValueError(
            f"Invalid config ref {path!r} at {location!r}."
        ) from exc
    return deepcopy(current)
