# Product GFRM Preprocessing Pipeline

This is the canonical RAW GFRM preprocessing pipeline for product work.

The purpose is to transform RAW Bruker GFRM detector frames into normalized
1D radial profiles ready for product-specific analysis.

```text
H5 container
-> H5SessionSelectorTransformer(product/user supplied attrs: date/status/PONI q/thickness coverage)
-> H5MeasurementSetAuditTransformer(optional metadata-only stage counts)
-> H5ToDataFrameTransformer
-> h5_to_df thickness exclusion
-> ProductColumnBuilder / ProductStatusGroupFilter(product label policy)
-> ColumnValueFilter / MetadataFilter(optional audit only)
-> FaultyPixelDetector
-> AzimuthalIntegration
-> SNRTransformer
-> SNRFilter
-> PatientSpecimenValidityFilter
-> QRangeValueNormalizer
-> RadialProfileValueFilter(optional product signal gate)
-> product-specific analysis
```

For debugging or validation, insert `RadialProfileSnapshot` after pipeline
stages. It expands the DataFrame with saved q/profile columns.

Snapshot details are documented in [`snapshots.md`](snapshots.md).

## 0. Transformer Contract

Product routes should be composed from sklearn-style transformers.

```text
input object -> transformer.fit_transform(input object) -> output object
```

Use direct functions such as `h5_to_df` only for low-level debugging or backward
compatibility. Product notebooks and scripts should prefer:

```python
from xrd_preprocessing import (
    H5MeasurementSetAuditTransformer,
    H5SessionSelectorTransformer,
    H5ToDataFrameTransformer,
    ProductColumnBuilder,
    ProductStatusGroupFilter,
    PairedGroupFilter,
    ConstantQRangeTransformer,
    DropColumnsTransformer,
)
```

The product repository owns YAML/JSON rules. `xrd_preprocessing` owns the
reusable movement primitives.

Implementation modules are split by responsibility:

```text
xrd_preprocessing.transformers.h5        H5 selection/audit/reader movement
xrd_preprocessing.transformers.metadata  product metadata and output movement
xrd_preprocessing.transformers.labels    status-group and paired-patient movement
xrd_preprocessing.transformers.profiles  synthetic profile movement for tests
```

H5 archive movement should use a manifest flow:

```text
H5 path
-> H5SessionSelectorTransformer
-> manifest with archive_path, all_session_df, selected session_df, h5_filters
-> H5MeasurementSetAuditTransformer(optional audit)
-> manifest with h5_stage_frames
-> H5ToDataFrameTransformer
-> measurement DataFrame
```

This keeps H5 filtering and stage statistics before detector frame loading.

Reusable preprocessing YAML template/contracts are packaged here:

```text
src/xrd_preprocessing/configs/preprocessing_config_template.yaml
src/xrd_preprocessing/configs/preprocessing_branch_config_template.yaml
```

Use `preprocessing_config_template.yaml` for a combined product contract that
keeps multiple branches in one file. Use
`preprocessing_branch_config_template.yaml` for the current Aramis style: one
concrete YAML per product branch with branch-specific rules under
`branch_settings`.

Product repositories should keep their concrete configs in the product repo,
then load and validate them with:

```python
from xrd_preprocessing import load_preprocessing_config

config = load_preprocessing_config("/path/to/product_preprocessing.yaml")
```

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
from xrd_preprocessing import (
    H5MeasurementSetAuditTransformer,
    H5SessionFilter,
    H5SessionSelectorTransformer,
    H5ToDataFrameTransformer,
)

selector = H5SessionSelectorTransformer(
    filters=[
        H5SessionFilter("started_at", op="date in", values=accepted_dates),
        H5SessionFilter("poni_q_max_nm_inv", op=">=", value=23.0),
        H5SessionFilter("h5_sample_all_sets_have_thickness", op="==", value=True),
    ],
    session_category="SAMPLE",
)
manifest = selector.fit_transform("session.nxs.h5")

audit = H5MeasurementSetAuditTransformer(
    stage_filters={"after_h5_filters": selector.filters},
    session_category="SAMPLE",
    set_category="SAMPLE",
)
manifest = audit.fit_transform(manifest)

reader = H5ToDataFrameTransformer(
    data_preference="gfrm",
    raw_root="path/to/gfrm/files",
    drop_missing_sample_thickness=True,
)
measurement_df = reader.fit_transform(manifest)
calibration_df = reader.calibration_df_
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

calibrant/reference thickness must also be controlled:

```text
calibrant/reference thickness can differ between calibration sessions
if present, it should come from H5 metadata as calibrant_thickness_mm
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

Use one-column DataFrame filters after `h5_to_df` only as audit guards or when
the required metadata was not available at H5-filter time:

```python
ColumnValueFilter("calibration_quality_status", op="==", value="accepted")
MetadataFilter("diagnosis", op="in", values=["BENIGN", "CANCER"])
```

Several metadata filters are composed as several pipeline steps.

## 3. H5-Level PONI Q-Range Product Filter

What happens:

```text
PONI geometry is loaded with pyFAI
wavelength/energy, detector distance, pixel size, and detector shape define q coverage
available q_max is estimated before azimuthal integration
sessions that cannot reach the product q_max are excluded before GFRM decode
```

Why:

```text
product features can require signal up to a fixed q value
longer detector distance can make the requested q range physically unavailable
integrating outside available geometry gives an invalid product profile
```

Example for Aramis-style q coverage:

```python
from xrd_preprocessing import H5SessionFilter, h5_to_df

_, measurement_df = h5_to_df(
    "combined_archive.h5",
    data_preference="gfrm",
    h5_filters=[
        H5SessionFilter("poni_q_max_nm_inv", op=">=", value=23.0),
    ],
    session_category="SAMPLE",
    set_category="SAMPLE",
)
```

`list_h5_sessions` exposes conservative session-level PONI q coverage:

```text
poni_q_max_nm_inv = minimum q_max across SAMPLE sets in the session
```

Output:

```text
poni_q_min_nm_inv
poni_q_max_nm_inv
poni_q_max_nm_inv_max
poni_calculated_distance_m
h5_sample_set_count
h5_sample_poni_count
```

## 3a. Optional AgBH Monochromaticity QC

Product code can score day/session beam monochromaticity from integrated AgBH
calibration profiles before selecting product measurements for those days.

```python
from xrd_preprocessing import AgBHMonochromaticityQualityControl

agbh_qc = AgBHMonochromaticityQualityControl(
    id_column="session_uid",
    max_score=0.1,
)
scored_agbh_df = agbh_qc.fit_transform(agbh_df)

h5_filters = [agbh_qc.selection_.h5_id_filter(column="linked_agbh_session_uid")]
```

By default the scorer builds an internal median AgBH baseline from all fit rows;
a product may pass `reference_df` only if it has a controlled baseline. This
computes quality metadata and exposes accepted AgBH IDs for strict H5 session
filtering. Product code decides how accepted/review/rejected AgBH sessions map
to measurement inclusion. Date fallback exists, but ID-based filtering is
preferred when product H5 metadata has shared linkage. Details:
[`agbh_monochromaticity.md`](agbh_monochromaticity.md).

## 4. Faulty Pixel Detector

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

## 5. Azimuthal Integration

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

If the H5/product DataFrame contains per-row calibrant/reference thickness:

```python
AzimuthalIntegration(
    calibration_mode="poni",
    mask_column="pyfai_faulty_pixel_mask",
    error_model="poisson",
    thickness_reference_column="calibrant_thickness_mm",
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
calibrant_thickness_mm
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

## 6. Poisson SNR

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

## 7. SNR Filter

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

## 8. Patient / Specimen Validity Filter

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

## 9. Q Range Normalization

What happens:

```text
remaining radial profiles are divided by a value statistic in q = 6.7..7.1 nm^-1
```

Why:

```text
normalization puts profiles on a comparable scale before product feature extraction
```

```python
QRangeValueNormalizer(q_min=6.7, q_max=7.1, statistic="median")
```

Default behavior:

```text
radial_profile_data is overwritten by normalized intensity
```

Use `save_initial_data=True` to keep `radial_profile_data_raw`.

## 10. Radial Profile Value Filter

What happens:

```text
target q is provided by product/user
nearest q pixel is found in q_range
radial_profile_data at that q pixel is compared with threshold
rows failing the threshold are excluded
```

Why:

```text
after normalization, empty/air-like measurements can still survive earlier QC
some products require non-trivial signal at a known q position
this is an explicit product signal gate, not a diagnostic claim
```

Example:

```python
from xrd_preprocessing import RadialProfileValueFilter

signal_gate = RadialProfileValueFilter(
    q_value_nm_inv=14.0,
    threshold=2.0,
    op=">",
)

df = signal_gate.fit_transform(df)
```

Output:

```text
radial_profile_nearest_q_nm_inv
radial_profile_value_at_q
radial_profile_q_delta_nm_inv
radial_profile_value_pass
```

## Saving Stage Profiles

Use this when you need to inspect how `radial_profile_data` changes through the
pipeline.

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
        (
            "integrate",
            AzimuthalIntegration(
                calibration_mode="poni",
                mask_column="pyfai_faulty_pixel_mask",
                error_model="poisson",
                thickness_reference_column="calibrant_thickness_mm",
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
            QRangeValueNormalizer(
                q_min=6.7,
                q_max=7.1,
                statistic="median",
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
