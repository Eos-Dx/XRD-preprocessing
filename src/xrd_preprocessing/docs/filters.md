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

Use `PatientSpecimenValidityFilter` for group-level replicate checks:

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

## Standard Product Order

```text
h5_to_df
ColumnValueFilter(date cutoff)
PatientFilter / MetadataFilter(diagnosis cohort)
FaultyPixelDetector
AzimuthalIntegration(error_model="poisson")
SNRTransformer(snr_method="poisson")
SNRFilter(min_snr_db=20.0)
PatientSpecimenValidityFilter
QRangeNormalizer(q_min=6.7, q_max=7.1)
product-specific analysis
```
