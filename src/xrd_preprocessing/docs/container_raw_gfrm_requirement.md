# Container v0.3 Raw GFRM Requirement

Request for `Eos-Dx/container` / EoScan producer.

## Requirement

Session H5 containers must preserve original Bruker `.gfrm` files as mandatory
RAW artifacts.

Decoded NumPy arrays may be stored as derived products, but they must not be the
only source of detector data.

## Required Container Behavior

- Store or bundle every original `.gfrm` file for each detector measurement.
- Keep an explicit H5 pointer from each measurement to its `.gfrm` artifact.
- Keep decoded arrays, if present, labelled as decoded/derived products.
- Preserve conversion provenance:
  - source `.gfrm` path or artifact id;
  - `baseline_adu`;
  - `gain_adu_per_photon`;
  - conversion formula: `(ADU - baseline_adu) / gain_adu_per_photon`;
  - `negative_pixel_count`;
  - any invalid-row masking, currently row `511`.

## Consumer Expectation

`xrd-preprocessing` treats RAW GFRM as the required product input:

1. Resolve the measurement `.gfrm` artifact.
2. Convert GFRM to photon NumPy data inside `xrd_preprocessing.gfrm`.
3. Use this converted image as `measurement_data`.
4. Preserve `/raw/data` and `/processed/data` only as decoded products.

## Current Local API

```python
from xrd_preprocessing import h5_to_df

calib_df, meas_df = h5_to_df(
    "session.nxs.h5",
    raw_root="path/to/raw-gfrm-artifacts",
)
```

Expected `measurement_data_source` values:

- `gfrm_to_photons`

If no `.gfrm` artifact can be resolved, product preprocessing raises an error.

## Rationale

Faulty-pixel search and future preprocessing must be traceable to original RAW
detector files. Baseline-subtracted negative photon values stay negative in the
raw conversion path. The detector then excludes negative, NaN/inf, and saturated
pixels and writes both a pyFAI mask and a reason-coded map.
