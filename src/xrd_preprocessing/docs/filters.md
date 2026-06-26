# Reusable Filters

Filters are sklearn-style transformers.

They take a DataFrame and return a filtered DataFrame.

## Column Value Filter

All row filters are based on one column.

Example:

```python
from xrd_preprocessing import ColumnValueFilter

diagnosis_filter = ColumnValueFilter(
    "diagnosis",
    op="in",
    values=["BENIGN", "CANCER"],
)

df = diagnosis_filter.fit_transform(df)
```

Supported operations:

```text
in
not_in
==
!=
>
>=
<
<=
between
date>=
date<=
date_after
date_before
date_between
contains
isna
notna
```

## Metadata / Patient Filter

`MetadataFilter` and `PatientFilter` are aliases of `ColumnValueFilter`.

They do not have default columns or default values.

You must define the column and rule explicitly:

```python
from xrd_preprocessing import PatientFilter

patient_filter = PatientFilter(
    "diagnosis",
    op="in",
    values=["BENIGN", "CANCER"],
)
```

For several columns, compose several one-column filters in a pipeline:

```python
from sklearn.pipeline import Pipeline
from xrd_preprocessing import ColumnValueFilter, MetadataFilter

pipeline = Pipeline(
    [
        ("diagnosis", MetadataFilter("diagnosis", values=["BENIGN", "CANCER"])),
        ("scan_type", ColumnValueFilter("scan_type", op="==", value="water")),
    ]
)
```

This keeps one column = one filter.

## Patient / Specimen Validity Filter

Clinical product DataFrames must contain:

```text
patientId
specimenId
```

`h5_to_df` creates these standard columns from container clinical names and
raises an error when measurement rows do not have them.

Use `SpecimenValidityFilter` when the dataset is specimen-level and patient
pairing must not participate in the validity rule:

```python
from xrd_preprocessing import SpecimenValidityFilter

validity = SpecimenValidityFilter(
    min_measurements_per_specimen=1,
)

df = validity.fit_transform(df)
```

This is the preferred one-to-many pattern: keep/discard rows by `specimenId`
measurement count; keep `patientId` only as metadata for later leakage control.

Use `PatientSpecimenValidityFilter` when patient context matters:

```python
from xrd_preprocessing import PatientSpecimenValidityFilter

validity = PatientSpecimenValidityFilter(
    min_measurements_per_specimen=2,
    min_specimens_per_patient=1,
)

df = validity.fit_transform(df)
```

Default rule:

```text
keep specimen only if it has >= 2 measurements
keep patient only if it has >= 1 valid specimen
```

This filter runs after `SNRFilter`, because replicate validity must count only
measurements that survived signal-quality filtering.

To require two valid specimens per patient:

```python
validity = PatientSpecimenValidityFilter(min_specimens_per_patient=2)
```

The filter adds:

```text
specimen_measurement_count
patient_valid_specimen_count
patient_specimen_valid
patient_specimen_validity_reason
```

Reasons:

```text
valid
missing_patient_or_specimen_id
specimen_measurements_below_minimum
patient_specimens_below_minimum
```

## PONI Q-Range Filter

Prefer H5-level `H5SessionFilter("poni_q_max_nm_inv", op=">=", value=...)`
when reading v0.3 archives, because that runs before GFRM decode.

Use `PoniQRangeFilter` after `h5_to_df` only when PONI q coverage was not
available at H5-filter time.

The filter reads `ponifile`, loads the PONI geometry with pyFAI, estimates the
available detector q range without integrating the image, and keeps only rows
where:

```text
poni_q_max_nm_inv >= required_q_max_nm_inv
```

Example:

```python
from xrd_preprocessing import PoniQRangeFilter

q_filter = PoniQRangeFilter(
    required_q_max_nm_inv=23.0,
    thickness_adjustment=True,
    thickness_reference_column="calibrant_thickness_mm",
)

df = q_filter.fit_transform(df)
```

The filter adds:

```text
poni_q_min_nm_inv
poni_q_max_nm_inv
poni_calculated_distance_m
poni_q_range_pass
```

If thickness correction will be used during `AzimuthalIntegration`, use the same
thickness settings in `PoniQRangeFilter`. Otherwise the q-range decision and
the integration geometry can disagree.

## SNR Filter

Use `SNRFilter` after `SNRTransformer`.

`SNRFilter` is the SNR-specific alias/wrapper around the same column-value
filter idea.

Default:

```python
from xrd_preprocessing import SNRFilter

snr_filter = SNRFilter(min_snr_db=20.0)
```

Default SNR column is `snr_db`.

Rule:

```text
keep finite snr_column >= min_snr_db
drop NaN and low-SNR rows
```

## Radial Profile Value Filter

Use `RadialProfileValueFilter` after integration or after q-range normalization
when a product requires signal above/below a threshold near a specific q value.

The filter finds the nearest q point in `q_range`, reads the matching
`radial_profile_data` value, and applies a threshold.

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

The filter adds:

```text
radial_profile_nearest_q_nm_inv
radial_profile_value_at_q
radial_profile_q_delta_nm_inv
radial_profile_value_pass
```

Optional `max_q_delta_nm_inv` can reject rows where the nearest q point is too
far from the requested q value.

## Standard Product Order

```text
H5SessionFilter(product/user supplied attrs: date/status/PONI q/thickness)
h5_to_df
ColumnValueFilter / MetadataFilter(optional audit only)
PoniQRangeFilter(optional fallback when H5 PONI q filter was unavailable)
FaultyPixelDetector
AzimuthalIntegration(error_model="poisson")
SNRTransformer(snr_method="poisson")
SNRFilter(min_snr_db=20.0)
PatientSpecimenValidityFilter
QRangeNormalizer(q_min=6.7, q_max=7.1)
RadialProfileValueFilter(optional product signal gate)
product-specific analysis
```

AgBH monochromaticity filters are available for product-owned day/session
quality metadata:

```python
from xrd_preprocessing import AgBHMonochromaticityQualityControl
```
