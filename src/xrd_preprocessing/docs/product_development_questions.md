# Product Development Questions

These questions are for product-development iterations that use this package.

They are not validation claims.

## Geometry Error To Q-Position Error

### Sample Thickness

Question:

```text
How does sample-thickness measurement error propagate into real q-position error?
```

Why it matters:

```text
sample thickness changes the effective detector distance
effective detector distance is used to calculate real q positions
missing or incorrect thickness shifts the q axis
shifted q positions make the integrated profile incorrect for product use
```

Required follow-up:

```text
estimate dq/d(thickness)
simulate representative thickness errors
define maximum acceptable thickness error
decide whether thickness uncertainty should become a QC metric
```

Metadata requirement:

```text
one measurement point has one sample thickness
sample thickness must be present in H5/product metadata for every product row
missing sample thickness means the q axis cannot be corrected
```

### AGBH / Reference Thickness

Question:

```text
Can AGBH/HBH/reference thickness differ between calibration sessions, and is it
stored in product-owned H5 metadata for every product row?
```

Why it matters:

```text
sample-thickness correction is relative to calibrant/reference thickness
wrong reference thickness shifts the effective detector distance
effective detector distance sets real q positions
```

Required follow-up:

```text
standardize H5 field name as calibrant_thickness_mm
add calibrant_thickness_mm to H5/product metadata if absent
use AzimuthalIntegration(thickness_reference_column="calibrant_thickness_mm")
define product command for rows where required reference thickness is missing:
drop, flag, or stop
```

Current product-data requirement:

```text
do not rely on free-text file paths for product integration
write explicit calibrant_thickness_mm into H5/product metadata
verify every selected measurement row has both sample_thickness_mm and
calibrant_thickness_mm before AzimuthalIntegration
```

### X-Ray Beam Position

Question:

```text
How does X-ray beam position / beam-center error propagate into real q-position
error?
```

Why it matters:

```text
beam-center position defines detector pixel radius from direct beam
pixel radius is converted into scattering angle
scattering angle is converted into q
beam-center error shifts q positions and can distort product features
```

Required follow-up:

```text
estimate dq/d(beam-center position)
simulate representative beam-center errors in x and y
compare effect across low-q and high-q regions
define maximum acceptable beam-center error
decide whether calibration drift should become a QC metric
```

## Product-Owned Protocol / Spectrum Selection

### K-Beta Leakage

Question:

```text
Which sessions/batches does the product select for applications that require a
specific X-ray spectrum policy?
```

Metadata source:

```text
controlled product JSON
H5 metadata
reviewed product manifest
```

Why it matters:

```text
product may require K-alpha-only scattering for a defined application
K-beta leakage changes the measured signal
selection policy belongs to the product, not to xrd_preprocessing
xrd_preprocessing only applies product-supplied filters
```

Possible H5/product metadata:

```text
spectrum_status
protocol_status
calibration_quality_status
product_protocol_version
product_selection_status
product_batch_id
patientId
specimenId
```

Required follow-up:

```text
product defines which patientId/specimenId/session/day rows are selected
write product selection metadata into H5/product metadata or external manifest
pass explicit H5-level filters before h5_to_df materializes detector frames
define product command for unknown K-beta status: keep, drop, flag, or stop
```

### AGBH/HBH Reliability

Question:

```text
Which product-owned metric describes AGBH/HBH measurement reliability and
chromatic-quality status for a session/day?
```

Why it matters:

```text
product may compute data quality from AGBH/HBH measurements
product may mark a day/session as accepted, review, or rejected
xrd_preprocessing only applies the resulting explicit metadata filter
```

Required follow-up:

```text
product defines reliability metric name and allowed values
product writes reliability status into H5 metadata or a reviewed manifest
product decides keep/drop/flag/stop behavior for each reliability state
```

## Product Decision Needed

For each product, define controlled inputs supplied to preprocessing:

```text
maximum sample-thickness error
maximum beam-center x/y error
maximum detector-distance error
required calibration recency
required spectrum/protocol rule
required AGBH/HBH reliability rule
selected patientId/specimenId/session manifest
command for rows/specimens where geometry uncertainty is too high:
keep, drop, flag, or stop
```
