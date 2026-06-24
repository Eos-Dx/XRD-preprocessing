# Product GFRM Preprocessing Pipeline

This is the canonical RAW GFRM preprocessing pipeline for product work.

The purpose is to transform RAW Bruker GFRM detector frames into normalized
1D radial profiles ready for product-specific analysis.

```text
H5 container
-> H5SessionFilter(product/user supplied attrs)
-> h5_to_df
-> h5_to_df thickness exclusion
-> ColumnValueFilter(optional DataFrame metadata audit)
-> MetadataFilter(optional product/user supplied metadata)
-> FaultyPixelDetector
-> AzimuthalIntegration
-> SNRTransformer
-> SNRFilter
-> PatientSpecimenValidityFilter
-> QRangeNormalizer
-> product-specific analysis
```

For debugging or validation, insert `RadialProfileSnapshot` after pipeline
stages. It expands the DataFrame with saved q/profile columns.

Snapshot details are documented in [`snapshots.md`](snapshots.md).

## 1. H5 To DataFrame

What happens:

```text
container v0.3 attrs are inspected
H5 session filters select sessions before frame loading
selected session is opened
RAW GFRM artifact or embedded raw_file is resolved
measurement_data is created
row metadata is collected into DataFrames
```

For EOSCAN backfill combined archives, `data_preference="gfrm"` reads embedded
`measurements/*/raw_file` vendor bytes and runs `gfrm_to_photons`. The embedded
`/raw/data` matrix is the Fabio-decoded ADU frame and is diagnostic, not product
input.

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

Combined archive input:

```text
xrd-session-archive
calib_*/sample_* session groups
embedded measurement raw_file vendor bytes
embedded raw/data ADU matrices for diagnostics
```

Code:

```python
calibration_df, measurement_df = h5_to_df(
    "session.nxs.h5",
    raw_root="path/to/gfrm/files",
)
```

Combined archive with product/user supplied filters:

```python
from xrd_preprocessing import H5SessionFilter, h5_to_df

calibration_df, measurement_df = h5_to_df(
    "data/product-aramis-data/combined_archive.h5",
    data_preference="gfrm",
    drop_missing_sample_thickness=True,
    h5_filters=[
        H5SessionFilter("calibration_quality_status", op="==", value="accepted"),
    ],
    session_category="SAMPLE",
    set_category="SAMPLE",
)
```

Output used by the next steps:

```text
measurement_data          photon image from GFRM
ponifile                  PONI text
sample_thickness_mm       sample thickness in mm
```

No GFRM means no product preprocessing.

No sample thickness means no product integration:

```text
one measurement point has one sample thickness
sample thickness is needed to correct effective detector distance
effective detector distance controls the real q positions
without thickness, azimuthal integration maps signal to incorrect q values
```

AGBH/reference thickness must also be controlled:

```text
AGBH/reference thickness can differ between calibration sessions
if present, it should come from H5 metadata as agbh_thickness_mm
if absent, add it to the H5/product metadata before product integration
do not silently reuse one reference thickness across sessions when it differs
product-specific AGBH/HBH reliability policy lives outside xrd_preprocessing
```

## Product Development Questions

```text
Quantify how sample-thickness error propagates into real q-position error during
azimuthal integration, and define allowable thickness-error bounds for product
preprocessing.

Quantify how X-ray beam position / beam-center error propagates into real
q-position error during azimuthal integration, and define allowable beam-center
error bounds for product preprocessing.
```

Detailed questions for product-development iteration are in
[`product_development_questions.md`](product_development_questions.md).

## 2. Early Metadata Filters

What happens:

```text
H5-level filters select/drop sessions before detector frames are loaded
DataFrame-level filters select/drop rows after h5_to_df
one metadata column is filtered per explicit pipeline step
```

Why:

```text
day/date, cohort, protocol, batch, and quality policy are product-owned inputs
metadata filtering must be explicit and reproducible
xrd_preprocessing only applies the supplied filter command
early filtering avoids loading/integrating rows the product already commanded to drop
```

Use H5-level filters for archive attrs:

```python
H5SessionFilter("calibration_quality_status", op="==", value="accepted")
```

Use H5-level product-supplied filters when metadata exists:

```python
H5SessionFilter("spectrum_status", op="==", value="accepted")
H5SessionFilter("product_selection_status", op="==", value="selected")
```

Do not encode product selection decisions in `xrd_preprocessing`:

```text
product decides which batches/sessions/specimens are selected
product writes that decision into JSON/H5 metadata or a reviewed manifest
xrd_preprocessing receives that decision as H5SessionFilter/ColumnValueFilter
```

If selected batches are provided externally, store them in product metadata as
explicit patient/specimen/batch selection fields. The library must not infer
product importance from free-text notes.

Use one-column DataFrame filters after `h5_to_df`:

```python
ColumnValueFilter("calibration_quality_status", op="==", value="accepted")
MetadataFilter("diagnosis", op="in", values=["BENIGN", "CANCER"])
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
    calibration_mode="poni",
    mask_column="pyfai_faulty_pixel_mask",
    error_model="poisson",
    thickness_reference_mm=<explicit float>,
)
```

If the H5/product DataFrame contains per-row AGBH/reference thickness:

```python
AzimuthalIntegration(
    calibration_mode="poni",
    mask_column="pyfai_faulty_pixel_mask",
    error_model="poisson",
    thickness_reference_column="agbh_thickness_mm",
)
```

Required columns:

```text
measurement_data
ponifile
sample_thickness_mm
pyfai_faulty_pixel_mask
```

Optional per-row reference column:

```text
agbh_thickness_mm
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

## 7. Patient / Specimen Validity Filter

What happens:

```text
after low-SNR rows are removed, rows are checked by patientId/specimenId
specimens with too few surviving measurements are removed
patients with too few valid specimens are removed
```

Why:

```text
replicate validity must be based on rows that survived signal-quality QC
a specimen with fewer than two valid measurements is not reliable for product features
```

```python
PatientSpecimenValidityFilter(
    min_measurements_per_specimen=2,
    min_specimens_per_patient=1,
)
```

Default rule:

```text
keep specimen if it has >= 2 surviving measurements
keep patient if it has >= 1 surviving specimen
```

To require both breast specimens:

```python
PatientSpecimenValidityFilter(min_specimens_per_patient=2)
```

Required columns:

```text
patientId
specimenId
```

`h5_to_df` creates these columns from container clinical metadata and raises an
error if they are missing.

## 8. Q Range Normalization

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
    ColumnValueFilter,
    FaultyPixelDetector,
    MetadataFilter,
    PatientSpecimenValidityFilter,
    QRangeNormalizer,
    RadialProfileSnapshot,
    SNRFilter,
    SNRTransformer,
)

save_pipeline_stages = True

pipeline = Pipeline(
    [
        (
            "calibration_quality",
            ColumnValueFilter("calibration_quality_status", op="==", value="accepted"),
        ),
        (
            "diagnosis",
            MetadataFilter("diagnosis", op="in", values=["BENIGN", "CANCER"]),
        ),
        ("faulty_pixels", FaultyPixelDetector()),
        (
            "integrate",
            AzimuthalIntegration(
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
q_range_after_clinical_validity
radial_profile_data_after_clinical_validity
radial_profile_data_raw
q_range_after_normalization
radial_profile_data_after_normalization
```

Set `save_pipeline_stages=False` to keep the same pipeline shape without
expanding the DataFrame.
