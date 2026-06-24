# H5 To DataFrame

`h5_to_df` reads selected Eos-Dx container v0.3 sessions into DataFrames.

It uses the maintained `eosdx-container` reader:

```python
from container import open_container
```

Standalone `xrd-session` files are opened directly. Combined
`xrd-session-archive` files are handled by scanning session attrs first,
copying each selected grafted session group to a temporary standalone session
file, then using the same reader.

Product rule for original GFRM mode:

```text
container v0.3 must contain RAW GFRM artifacts
measurement_data is produced by gfrm_to_photons
stored NumPy arrays are not the product source of truth
```

EOSCAN backfill archive product rule:

```text
measurement_data is produced from measurements/*/raw_file through gfrm_to_photons
raw_file contains original vendor GFRM bytes
/raw/data is the Fabio-decoded ADU matrix and is diagnostic, not product input
```

## API

```python
from xrd_preprocessing import H5SessionFilter, filter_h5_sessions, h5_to_df

calibration_df, measurement_df = h5_to_df(
    "session.nxs.h5",
    raw_root="path/to/raw-gfrm-artifacts",
)
```

For a combined archive with product/user supplied H5 filters:

```python
calibration_df, measurement_df = h5_to_df(
    "data/product-aramis-data/combined_archive.h5",
    data_preference="gfrm",
    drop_missing_sample_thickness=True,
    h5_filters=[
        H5SessionFilter("calibration_quality_status", op="==", value="accepted"),
    ],
    session_category="SAMPLE",
    set_category="SAMPLE",
    max_sessions=10,
)
```

To inspect archive metadata without loading detector arrays:

```python
selected_sessions = filter_h5_sessions(
    "data/product-aramis-data/combined_archive.h5",
    [H5SessionFilter("calibration_quality_status", op="==", value="accepted")],
    session_category="SAMPLE",
)
```

Details are documented in [`h5_session_filters.md`](h5_session_filters.md).

Supported container identity:

```text
schema_version = 0.3
format = xrd-session
format = xrd-session-archive
```

Strict product defaults:

```text
data_preference = gfrm
convert_gfrm = True
require_clinical_ids = True
```

If a measurement set does not resolve to a `.gfrm` artifact, `h5_to_df` raises
`FileNotFoundError`.

Backfill diagnostic ADU mode:

```text
data_preference = raw
measurement_data_source = container_raw_data
```

Use this only to inspect the Fabio-decoded ADU matrix stored in `/raw/data`.
Product preprocessing should use `data_preference="gfrm"` so embedded
`measurements/*/raw_file` bytes go through `gfrm_to_photons`.

Thickness filter:

```text
drop_missing_sample_thickness = True
```

This excludes measurement sets when none of these fields contains a positive
numeric value:

```text
sample_thickness_mm
sample_thickness
thickness_raw_mm
```

Thickness command boundary:

```text
one measurement point has one sample thickness
one row in measurement_df represents one measured point/set
if that row has no sample thickness, its q axis cannot be corrected
xrd_preprocessing drops it only when drop_missing_sample_thickness=True
product decides whether to drop, flag, or stop before azimuthal integration
```

AGBH/reference thickness:

```text
AGBH/reference thickness can differ between calibration sessions
this value should be stored in the H5 container metadata
preferred standardized column: agbh_thickness_mm
if it is absent from H5, add it before product preprocessing
product-specific AGBH/HBH reliability policy lives outside xrd_preprocessing
```

Product rule:

```text
do not infer AGBH thickness from free-text paths in product runs
store explicit agbh_thickness_mm in H5/product metadata
```

K-beta/protocol metadata boundary:

```text
product owns day/date, K-alpha/K-beta, protocol, batch, quality, and cohort policy
xrd_preprocessing only applies explicit H5SessionFilter/ColumnValueFilter commands
example product-owned H5 fields: spectrum_status, protocol_status,
calibration_quality_status, product_selection_status, product_batch_id,
patientId, specimenId
```

Product metadata belongs outside `xrd_preprocessing`:

```text
controlled product JSON
H5 metadata
reviewed product manifest
```

If K-beta/batch metadata is absent from H5, the product may add it before
dataset construction or use an explicit reviewed patientId/specimenId manifest.

The filter runs inside `h5_to_df` before the row is returned to
`measurement_df`. The count is stored in:

```python
measurement_df.attrs["dropped_missing_sample_thickness"]
```

Reason:

```text
sample thickness is required to correct the effective detector distance
effective detector distance defines the real q positions in azimuthal integration
without thickness, the integrated intensity would be mapped onto wrong q values
```

## Data Source

Product output:

```text
measurement_data_source = gfrm_to_photons
measurement_data_source = embedded_raw_file_gfrm_to_photons
measurement_data        = photon image from GFRM
gfrm_path               = resolved GFRM path
gfrm_conversion_metadata
```

Backfill diagnostic ADU output:

```text
measurement_data_source = container_raw_data
measurement_data        = /sets/<set>/raw/data
raw_data                = /sets/<set>/raw/data
gfrm_path               = original file_path metadata
```

Stored arrays are preserved as decoded products:

```text
raw_data
processed_data
```

They are not used as product `measurement_data`.

## Key Columns

Common output columns:

```text
source_file
id
meas_name
patientId
specimenId
metadata
processing_config
detector_measurements
ponifile
measurement_data
measurement_data_source
raw_data
processed_data
gfrm_data
gfrm_path
gfrm_conversion_metadata
archive_group
archive_session_name
archive_session_path
calibration_session_uid
```

Clinical ID mapping:

```text
patientId  <- /session/sample/patient_name
specimenId <- /session/sample/name
```

If measurement rows do not contain valid `patientId` and `specimenId`,
`h5_to_df` raises `ValueError`.

## Conversion Boundary

Before DataFrame creation:

```text
H5 session attrs are listed
H5SessionFilter is applied
archive session groups are selected
detector frames are not loaded
```

During `h5_to_df`:

```text
selected session is opened through eosdx-container
external .gfrm or embedded raw_file is converted through gfrm_to_photons
one row per selected set becomes calibration_df or measurement_df
```

After `h5_to_df`:

```text
ColumnValueFilter and MetadataFilter operate on DataFrame rows
```

If integrated data already exists in the container, it is also exposed:

```text
q_range
radial_profile_data
radial_profile_sigma
integration_*
```

The product preprocessing path still starts from RAW GFRM.
