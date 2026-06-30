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

## Transformer API

Use transformer objects when building product pipelines or notebooks:

```python
from xrd_preprocessing import (
    H5MeasurementSetAuditTransformer,
    H5SessionFilter,
    H5SessionSelectorTransformer,
    H5ToDataFrameTransformer,
)

selector = H5SessionSelectorTransformer(
    filters=[
        H5SessionFilter("session_uid", op="not_in", values=excluded_session_ids),
        H5SessionFilter("poni_q_max_nm_inv", op=">=", value=23.0),
        H5SessionFilter("h5_sample_all_sets_have_thickness", op="==", value=True),
    ],
    session_category="SAMPLE",
)
manifest = selector.fit_transform(archive)

audit = H5MeasurementSetAuditTransformer(
    stage_filters={"after_h5_filters": selector.filters},
    session_category="SAMPLE",
    set_category="SAMPLE",
)
manifest = audit.fit_transform(manifest)

reader = H5ToDataFrameTransformer(data_preference="gfrm")
measurement_df = reader.fit_transform(manifest)
```

The selector output is a manifest:

```text
archive_path
all_session_df
session_df
h5_filters
```

The audit transformer adds `h5_stage_frames` for count plots and logs. It does
not load detector arrays.

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
specimen_status
```

Example H5-level filter when product metadata is present:

```python
h5_filters = [
    H5SessionFilter("calibration_quality_status", op="==", value="accepted"),
    H5SessionFilter("product_selection_status", op="==", value="selected"),
]
```

Example H5-level diagnosis/status filter:

```python
h5_filters = [
    H5SessionFilter("specimen_status", op="in", values=["BENIGN", "CANCER"]),
]
```

For v0.3 containers, `list_h5_sessions` exposes selected `session/sample`
metadata, including `specimen_status`, `biopsy`, `side`, `patientId`, and
`specimenId`, so these filters run before GFRM decode.

## H5-Level PONI Q Coverage

`list_h5_sessions` also inspects SAMPLE set `artifacts/poni` without loading
detector frames.

It exposes conservative session-level q coverage:

```text
poni_q_min_nm_inv
poni_q_max_nm_inv
poni_q_max_nm_inv_max
poni_calculated_distance_m
h5_sample_set_count
h5_sample_poni_count
h5_sample_all_sets_have_poni
h5_sample_thickness_count
h5_sample_all_sets_have_thickness
sample_thickness_mm_min
sample_thickness_mm_max
```

`poni_q_max_nm_inv` is the minimum q max across SAMPLE sets in the session, so:

```python
h5_filters = [
    H5SessionFilter("poni_q_max_nm_inv", op=">=", value=23.0),
]
```

keeps only sessions whose SAMPLE measurements can physically cover the product
q range before GFRM decode.

Likewise, sample-thickness availability can be checked before `h5_to_df`:

```python
h5_filters = [
    H5SessionFilter("h5_sample_all_sets_have_thickness", op="==", value=True),
]
```

Keep `drop_missing_sample_thickness=True` in `h5_to_df` as a safety net when
measurement sets can still be mixed inside a selected session.

Calibrant thickness metadata can also be checked before `h5_to_df`:

```python
from xrd_preprocessing import calibrant_thickness_h5_filters

h5_filters = [
    *calibrant_thickness_h5_filters(min_mm=10.0, max_mm=40.0),
]
```

This requires `calibrant_thickness_mm` in H5 metadata and rejects values outside
the current product safety range before GFRM decode.

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

Fallback filter example for product QC:

```python
H5SessionFilter(
    column="linked_agbh_session_uid",
    op="not in",
    values=rejected_session_ids,
    fallback={
        "column": "started_at",
        "op": "date not in",
        "values": rejected_dates,
    },
)
```

The fallback is used only when the primary column is absent from the H5 session
metadata table. Prefer session ID exclusions when the session-link column exists;
one calendar date can contain multiple calibration sessions.

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

Measurement-level thickness exclusion can also run inside `h5_to_df` as a
last safety check:

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
