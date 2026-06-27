# Radial Profile Snapshots

`RadialProfileSnapshot` is a small sklearn-style transformer for debugging and
validation of preprocessing pipelines.

It does not calculate or modify data.

It copies the current q/profile arrays into stage-specific columns.

## Purpose

Use snapshots when you need to see how `radial_profile_data` changes through the
pipeline.

This is useful for:

```text
debugging
validation
plotting intermediate stages
checking that normalization changed only the intended profile
checking which curves survived SNR filtering
```

The output DataFrame becomes wider. That is expected.

## API

```python
from xrd_preprocessing import RadialProfileSnapshot

snapshot = RadialProfileSnapshot("after_integration")
out = snapshot.fit_transform(df)
```

Input columns:

```text
q_range
radial_profile_data
```

Output columns:

```text
q_range_after_integration
radial_profile_data_after_integration
```

## Pipeline Use

```python
from sklearn.pipeline import Pipeline
from xrd_preprocessing import (
    AzimuthalIntegration,
    FaultyPixelDetector,
    PatientSpecimenValidityFilter,
    QRangeValueNormalizer,
    RadialProfileSnapshot,
    SNRFilter,
    SNRTransformer,
)

save_pipeline_stages = True

pipeline = Pipeline(
    [
        ("faulty_pixels", FaultyPixelDetector()),
        ("integrate", AzimuthalIntegration(...)),
        (
            "snapshot_after_integration",
            RadialProfileSnapshot("after_integration", enabled=save_pipeline_stages),
        ),
        ("snr", SNRTransformer(snr_method="poisson")),
        (
            "snapshot_after_snr",
            RadialProfileSnapshot("after_snr", enabled=save_pipeline_stages),
        ),
        ("snr_filter", SNRFilter(min_snr_db=20.0)),
        (
            "clinical_validity",
            PatientSpecimenValidityFilter(
                min_measurements_per_specimen=2,
                min_specimens_per_patient=1,
            ),
        ),
        (
            "snapshot_after_clinical_validity",
            RadialProfileSnapshot(
                "after_clinical_validity",
                enabled=save_pipeline_stages,
            ),
        ),
        (
            "normalize",
            QRangeValueNormalizer(
                q_min=6.7,
                q_max=7.1,
                statistic="median",
                save_initial_data=save_pipeline_stages,
            ),
        ),
        (
            "snapshot_after_normalization",
            RadialProfileSnapshot("after_normalization", enabled=save_pipeline_stages),
        ),
    ]
)
```

## Saved Columns

Typical saved columns:

```text
q_range_after_integration
radial_profile_data_after_integration
q_range_after_snr
radial_profile_data_after_snr
q_range_after_clinical_validity
radial_profile_data_after_clinical_validity
radial_profile_data_raw
q_range_after_normalization
radial_profile_data_after_normalization
```

`radial_profile_data_raw` is created by:

```python
QRangeValueNormalizer(save_initial_data=True)
```

It stores the profile before normalization.

## Disable Snapshots

To keep the same pipeline but stop saving stage columns:

```python
RadialProfileSnapshot("after_snr", enabled=False)
```

When disabled, the transformer returns the DataFrame unchanged.

## Custom Column Names

If needed:

```python
RadialProfileSnapshot(
    "after_custom_step",
    q_column="q_range",
    profile_column="radial_profile_data",
)
```

The output names are:

```text
<q_column>_<stage>
<profile_column>_<stage>
```
