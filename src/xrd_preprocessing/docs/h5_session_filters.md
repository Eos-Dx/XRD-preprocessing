# H5 Session Filters

H5 session filters run before `h5_to_df` loads detector frames.

Use them for large `xrd-session-archive` containers when product/user commands
can be applied from session attrs:

```text
combined_archive.h5
-> list_h5_sessions reads H5 attrs only
-> H5SessionFilter selects session groups
-> h5_to_df copies/opens selected sessions only
-> measurement_data arrays are loaded
-> calibration_df, measurement_df are created
```

This is different from `ColumnValueFilter`:

```text
H5SessionFilter
    input: H5 attrs
    moment: before frame loading
    use: explicit filters on session metadata

ColumnValueFilter
    input: DataFrame column
    moment: after h5_to_df
    use: row filters after arrays and metadata are materialized
```

## Minimal API

```python
from xrd_preprocessing import H5SessionFilter, filter_h5_sessions, h5_to_df

archive = "data/product-aramis-data/combined_archive.h5"

selected_sessions = filter_h5_sessions(
    archive,
    [
        H5SessionFilter("demo_quality_status", op="==", value="accepted"),
    ],
    session_category="SAMPLE",
    max_sessions=5,
)

calibration_df, measurement_df = h5_to_df(
    archive,
    data_preference="gfrm",
    drop_missing_sample_thickness=True,
    h5_filters=[
        H5SessionFilter("demo_quality_status", op="==", value="accepted"),
    ],
    session_category="SAMPLE",
    set_category="SAMPLE",
    max_sessions=5,
)
```

## Product-Owned Protocol Filters

`xrd_preprocessing` does not decide which protocol, batch, patient, specimen, or
measurement a product uses.

The product owns:

```text
clinical cohort
day/date selection
K-alpha/K-beta policy
AGBH/HBH reliability policy
chromatic-quality policy
batch/session/specimen selection
label policy
```

This library only applies explicit filters supplied by the product or user.
Product metadata should live outside the library:

```text
product JSON/H5 metadata/reviewed manifest -> H5SessionFilter -> h5_to_df
```

Example product-owned H5 metadata fields:

```text
spectrum_status
protocol_status
calibration_quality_status
product_selection_status
product_protocol_version
product_batch_id
patientId
specimenId
```

Example H5-level filter when product metadata is present:

```python
h5_filters = [
    H5SessionFilter("calibration_quality_status", op="==", value="accepted"),
    H5SessionFilter("product_selection_status", op="==", value="selected"),
]
```

Example H5-level measurement-day filter when product supplies selected dates:

```python
h5_filters = [
    H5SessionFilter(
        "started_at",
        op="date in",
        values=["2026-01-02", "2026-01-09", "2026-01-16"],
    ),
]
```

`date in` and `date not in` compare calendar dates. H5 timestamps with time,
for example `2026-01-02 10:00:00`, match user dates like `2026-01-02`.

If a collaborator provides selected batches, encode them as metadata, not as
free-text notes. Example:

```text
product_selection_status = selected for selected patientId/specimenId/session rows
```

If those fields are absent from the H5 container, the product may add them before
dataset construction or pass an external manifest into its own pipeline.

## Supported Ops

```text
==
!=
in
not in
contains
startswith
endswith
isna
notna
>
>=
<
<=
date>
date>=
date<
date<=
date in
date_in
date not in
date_not_in
between
date_between
```

## Conversion Moment

`h5_to_df` is the conversion boundary.

Before this boundary:

```text
H5 file is inspected
session attrs are read
session groups are selected
no detector frame arrays are loaded
```

After this boundary:

```text
selected session is opened through eosdx-container reader
external .gfrm or embedded raw_file is converted through gfrm_to_photons
measurement_data is the GFRM photon image
one row per selected set is written to DataFrame
DataFrame filters can run
```

Measurement-level thickness exclusion can also run inside `h5_to_df`:

```python
h5_to_df(..., drop_missing_sample_thickness=True)
```

This rejects measurement sets without positive numeric `sample_thickness_mm`,
`sample_thickness`, or `thickness_raw_mm`.

For EOSCAN backfill product mode:

```text
data_preference="gfrm"
measurement_data = gfrm_to_photons(measurements/*/raw_file)
raw_file is original GFRM vendor bytes
/raw/data is Fabio-decoded ADU diagnostic data
```

For product GFRM mode:

```text
data_preference="gfrm"
measurement_data = gfrm_to_photons(resolved .gfrm path)
measurement_data = gfrm_to_photons(embedded raw_file fallback)
```

## Demo

```bash
PYTHONPATH=src:/Users/sad/dev/container/src \
python examples/h5_session_filter_demo.py \
  --input /Users/sad/dev/eos_play/jupyter_notebooks/Clinical_trials/data/product-aramis-data/combined_archive.h5 \
  --output /tmp/tiny_h5_filter_demo.h5
```

The demo:

```text
copies five selected SAMPLE sessions into a tiny archive
adds demo_quality_status attrs to copied sessions
prints session counts before and after the H5 attr filter
runs h5_to_df only on selected sessions
writes scalar preview CSV next to the tiny archive
```
