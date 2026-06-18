# H5 To DataFrame

`h5_to_df` reads Eos-Dx container v0.3 into DataFrames.

Product rule:

```text
container v0.3 must contain RAW GFRM artifacts
measurement_data is produced by gfrm_to_photons
stored NumPy arrays are not the product source of truth
```

## API

```python
from xrd_preprocessing import h5_to_df

calibration_df, measurement_df = h5_to_df(
    "session.nxs.h5",
    raw_root="path/to/raw-gfrm-artifacts",
)
```

Supported container identity:

```text
schema_version = 0.3
format = xrd-session
```

Strict product defaults:

```text
data_preference = gfrm
convert_gfrm = True
require_clinical_ids = True
```

If a measurement set does not resolve to a `.gfrm` artifact, `h5_to_df` raises
`FileNotFoundError`.

## Data Source

Product output:

```text
measurement_data_source = gfrm_to_photons
measurement_data        = photon image from GFRM
gfrm_path               = resolved GFRM path
gfrm_conversion_metadata
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
```

Clinical ID mapping:

```text
patientId  <- /session/sample/patient_name
specimenId <- /session/sample/name
```

If measurement rows do not contain valid `patientId` and `specimenId`,
`h5_to_df` raises `ValueError`.

If integrated data already exists in the container, it is also exposed:

```text
q_range
radial_profile_data
radial_profile_sigma
integration_*
```

The product preprocessing path still starts from RAW GFRM.
