# Product GFRM Preprocessing Pipeline

This is the canonical RAW GFRM preprocessing pipeline for product work.

The purpose is to transform RAW Bruker GFRM detector frames into normalized
1D radial profiles ready for product-specific analysis.

```text
h5_to_df
-> PatientFilter
-> FaultyPixelDetector
-> AzimuthalIntegration
-> SNRTransformer
-> SNRFilter
-> QRangeNormalizer
-> product-specific analysis
```

For debugging or validation, insert `RadialProfileSnapshot` after pipeline
stages. It expands the DataFrame with saved q/profile columns.

Snapshot details are documented in [`snapshots.md`](snapshots.md).

## 1. H5 To DataFrame

What happens:

```text
container v0.3 is opened
RAW GFRM artifact is resolved
GFRM is converted to EOS photon image
row metadata is collected into a DataFrame
```

Why:

```text
product preprocessing must start from traceable RAW detector files
stored NumPy arrays are treated as derived products, not source of truth
```

Input:

```text
container v0.3
mandatory RAW GFRM artifacts
PONI text
sample_thickness_mm metadata
```

Code:

```python
calibration_df, measurement_df = h5_to_df(
    "session.nxs.h5",
    raw_root="path/to/gfrm/files",
)
```

Output used by the next steps:

```text
measurement_data          photon image from GFRM
ponifile                  PONI text
sample_thickness_mm       sample thickness in mm
```

No GFRM means no product preprocessing.

## 2. Patient Filter

What happens:

```text
rows are filtered by one metadata column at a time
```

Why:

```text
Aramis/Bremen products process defined clinical cohorts
metadata filtering must be explicit and reproducible
```

Use one-column filters for clinical metadata:

```python
PatientFilter("diagnosis", op="in", values=["BENIGN", "CANCER"])
```

Several metadata filters are composed as several pipeline steps.

## 3. Faulty Pixel Detector

What happens:

```text
each photon image is checked for invalid or saturated pixels
a row-specific pyFAI mask is created
beam-zone pixels are excluded from faulty statistics
```

Why:

```text
bad detector pixels can create spikes after azimuthal integration
the mask must be frame-local because GFRM conversion can create row-specific defects
```

```python
FaultyPixelDetector(
    local_hot_min_value=500.0,
    exclude_beam_center_radius=0.04,
)
```

Rules:

```text
NaN/inf -> exclude
< 0     -> exclude
> 500   -> exclude
0..500  -> keep
```

Beam zone is excluded from all faulty outputs.

Key output:

```text
pyfai_faulty_pixel_mask
faulty_pixel_reason_map
faulty_pixel_reason_counts
```

## 4. Azimuthal Integration

What happens:

```text
2D detector image is integrated into a 1D q profile
PONI provides detector geometry
row-specific faulty-pixel mask is passed into pyFAI
thickness correction adjusts detector distance
Poisson sigma is calculated for SNR
```

Why:

```text
products work with 1D radial profiles, not raw 2D detector images
Poisson sigma is required for physical SNR filtering
thick samples require thickness correction
```

```python
AzimuthalIntegration(
    npt=900,
    calibration_mode="poni",
    mask_column="pyfai_faulty_pixel_mask",
    error_model="poisson",
    thickness_reference_mm=<explicit float>,
)
```

Required columns:

```text
measurement_data
ponifile
sample_thickness_mm
pyfai_faulty_pixel_mask
```

Set q range before integration:

```python
df["interpolation_q_range"] = [(2.0, 23.0)] * len(df)
```

The pyFAI integrator is geometry from PONI. The mask is passed per row.

Output:

```text
q_range
radial_profile_data
radial_profile_sigma
calculated_distance
thickness_adjusted_distance_m
```

## 5. Poisson SNR

What happens:

```text
profile-level SNR is calculated from radial_profile_data and radial_profile_sigma
```

Why:

```text
profiles with insufficient counting-statistics quality should not enter analysis
SNR is calculated after azimuthal integration because sigma is per q bin
```

```python
SNRTransformer(snr_method="poisson")
```

Formula:

```text
snr_q = abs(radial_profile_data(q)) / radial_profile_sigma(q)
snr_linear = sqrt(mean(snr_q^2))
snr_db = 20 * log10(snr_linear)
```

Output:

```text
noise_std
snr_linear
snr_db
snr_method_used
```

## 6. SNR Filter

What happens:

```text
rows with snr_db < 20 or missing SNR are removed
```

Why:

```text
downstream product features should be calculated only on profiles passing QC
```

```python
SNRFilter(min_snr_db=20.0)
```

Rule:

```text
keep finite snr_db >= 20
drop NaN or snr_db < 20
```

## 7. Q Range Normalization

What happens:

```text
remaining radial profiles are divided by area in q = 6.7..7.1 nm^-1
```

Why:

```text
normalization puts profiles on a comparable scale before product feature extraction
```

```python
QRangeNormalizer(q_min=6.7, q_max=7.1)
```

Default behavior:

```text
radial_profile_data is overwritten by normalized intensity
```

Use `save_initial_data=True` to keep `radial_profile_data_raw`.

## Saving Stage Profiles

Use this when you need to inspect how `radial_profile_data` changes through the
pipeline.

```python
from sklearn.pipeline import Pipeline
from xrd_preprocessing import (
    AzimuthalIntegration,
    FaultyPixelDetector,
    QRangeNormalizer,
    RadialProfileSnapshot,
    SNRFilter,
    SNRTransformer,
)

save_pipeline_stages = True

pipeline = Pipeline(
    [
        ("faulty_pixels", FaultyPixelDetector()),
        (
            "integrate",
            AzimuthalIntegration(
                npt=900,
                calibration_mode="poni",
                mask_column="pyfai_faulty_pixel_mask",
                error_model="poisson",
                thickness_reference_mm=11.0,
            ),
        ),
        (
            "snapshot_after_integration",
            RadialProfileSnapshot(
                "after_integration",
                enabled=save_pipeline_stages,
            ),
        ),
        ("snr", SNRTransformer(snr_method="poisson")),
        (
            "snapshot_after_snr",
            RadialProfileSnapshot("after_snr", enabled=save_pipeline_stages),
        ),
        ("snr_filter", SNRFilter(min_snr_db=20.0)),
        (
            "snapshot_after_snr_filter",
            RadialProfileSnapshot(
                "after_snr_filter",
                enabled=save_pipeline_stages,
            ),
        ),
        (
            "normalize",
            QRangeNormalizer(
                q_min=6.7,
                q_max=7.1,
                save_initial_data=save_pipeline_stages,
            ),
        ),
        (
            "snapshot_after_normalization",
            RadialProfileSnapshot(
                "after_normalization",
                enabled=save_pipeline_stages,
            ),
        ),
    ]
)
```

Example saved columns:

```text
q_range_after_integration
radial_profile_data_after_integration
q_range_after_snr
radial_profile_data_after_snr
q_range_after_snr_filter
radial_profile_data_after_snr_filter
radial_profile_data_raw
q_range_after_normalization
radial_profile_data_after_normalization
```

Set `save_pipeline_stages=False` to keep the same pipeline shape without
expanding the DataFrame.
