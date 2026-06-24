# AgBH Monochromaticity QC

This module scores monochromaticity from integrated AgBH calibration profiles.

It does not decide which product days are used. Product code supplies the
threshold and any day/session selection policy.

There is no required fixed etalon. By default the scorer builds an internal
baseline as the median normalized AgBH profile from all rows passed to `fit`.
If a product has a controlled baseline profile, it can pass `reference_df`, but
that is optional.

## Meaning

```text
K-beta has higher energy than K-alpha
in K-alpha q coordinates, K-beta appears at smaller q
K-beta cannot create a right-side shoulder
right-side residual is used only as symmetric control
```

## Metric

```text
normalize measured AgBH profile
build median AgBH baseline from fit rows, unless reference_df is supplied
fit baseline by scale + offset + linear background
q_shift = 0
residual = measured - fitted_baseline
score = positive left-window residual area - positive right-control residual area
score is divided by total window width
lower score means more monochromatic
```

The score is a QC ranking metric, not a physical K-beta fraction.

Default threshold:

```text
max_score = 0.1
```

Product may use a stricter threshold, for example:

```text
max_score = 0.0075
```

## API

Preferred QC structure:

```python
from xrd_preprocessing import (
    AgBHMonochromaticityQualityControl,
    h5_to_df,
)

qc = AgBHMonochromaticityQualityControl(
    id_column="session_uid",
    max_score=0.1,
)

scored_agbh_df = qc.fit_transform(agbh_df)

h5_filters = [
    qc.selection_.h5_id_filter(column="linked_agbh_session_uid"),
]

calibration_df, measurement_df = h5_to_df(
    "combined_archive.h5",
    h5_filters=h5_filters,
)
```

This is the preferred path because the accepted AgBH IDs are explicit. It avoids
selecting by calendar date when several technical states may exist on the same
day.

Optional controlled baseline:

```python
scorer = AgBHMonochromaticityScorer(
    reference_df=controlled_agbh_baseline_df,
    max_score=0.1,
)
```

Date fallback, when no shared AgBH/session ID exists in the product H5:

```python
h5_filters = [
    qc.selection_.h5_date_filter(column="started_at"),
]
```

Date fallback is less strict than ID filtering. Use it only when the product has
no reliable AgBH/session linkage in H5 metadata.

Compact manifest for review or product JSON/H5 metadata:

```python
manifest_df = qc.selection_.manifest_columns()
```

Expected input columns:

```text
q_range
radial_profile_data
```

Output columns:

```text
agbh_monochromaticity_score
agbh_monochromaticity_pass
agbh_monochromaticity_status
agbh_monochromaticity_max_score
agbh_kbeta_left_positive_area
agbh_kbeta_right_control_positive_area
agbh_kbeta_left_net_area
agbh_kbeta_n_windows
agbh_kbeta_window_orders
agbh_kbeta_peak_window_details
agbh_baseline_fit_scale
agbh_baseline_fit_offset
agbh_baseline_fit_linear_background
```

Use this result as product-owned quality metadata, for example by writing an
accepted/review/rejected day or session status into product JSON/H5 metadata.

## Pipeline Role

```text
AgBH technical frames
-> integrate AgBH frames
-> AgBHMonochromaticityQualityControl
-> accepted AgBH/session IDs or reviewed manifest
-> H5SessionFilter for product measurement container
-> usual product preprocessing
```

This is pre-product-selection QC. It is separate from sample-level SNR,
faulty-pixel, thickness, and patient/specimen QC.
