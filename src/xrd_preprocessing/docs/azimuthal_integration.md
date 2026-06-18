# Azimuthal Integration

`AzimuthalIntegration` is the simple one-detector pyFAI integration step.

No detector stitching is done here.

Detector geometry must come from PONI/pyFAI metadata. The code must not branch
on detector image shape such as `512 x 768`.

For `calibration_mode="poni"`, the PONI text stored in the DataFrame is written
to a temporary `.poni` file and loaded with `pyFAI.load()`.

If pyFAI does not recognize the detector name but the PONI contains
`Detector_config`, the loader falls back to pyFAI generic `Detector` while
preserving `Detector_config`. This is not detector-shape logic. Geometry still
comes from the PONI fields:

```text
Detector_config.pixel1
Detector_config.pixel2
Distance
Poni1
Poni2
Rot1
Rot2
Rot3
Wavelength
```

## Product Rule

In the GFRM workflow each measurement gets its own faulty-pixel mask.

Reason:

```text
GFRM ADU -> photon estimate can produce frame-dependent invalid pixels.
Lower photon signal can make more pixels fail the simple faulty-pixel rules.
Therefore the mask is measurement-specific.
```

Even when all rows use the same PONI calibration, integration is executed row by
row with that row's mask.

## Principle

pyFAI separates geometry from the integration call.

Geometry:

```text
PONI text -> pyFAI AzimuthalIntegrator
```

Per-measurement data:

```text
2D image
row-specific faulty-pixel mask
q range
azimuthal range
```

The `AzimuthalIntegrator` represents detector geometry. It can be reused for
rows with identical PONI text.

The faulty-pixel mask is not part of detector geometry. It is passed to pyFAI
for each integration call:

```python
ai.integrate1d(image, npt, mask=row_mask, error_model="poisson")
```

Therefore this is valid and expected:

```text
same PONI text
same cached pyFAI integrator
different measurement images
different row masks
different integrated profiles
```

## Minimal Pipeline Step

```python
from xrd_preprocessing import AzimuthalIntegration

integrator = AzimuthalIntegration(
    npt=900,
    calibration_mode="poni",
    column="measurement_data",
    mask_column="pyfai_faulty_pixel_mask",
    error_model="poisson",
    thickness_adjustment=True,
    thickness_reference_mm=11.0,
    sample_thickness_column="sample_thickness_mm",
)

out = integrator.fit_transform(df)
```

Input columns:

```text
measurement_data           2D detector image
ponifile                   PONI text
sample_thickness_mm        sample thickness in mm
interpolation_q_range      optional tuple, for example (2.0, 23.0)
azimuthal_range            optional azimuth range
pyfai_faulty_pixel_mask    optional uint8 mask, 1 = exclude
```

Output columns:

```text
q_range                         q positions
radial_profile_data             integrated intensity
radial_profile_sigma            pyFAI sigma, if requested by error_model
calculated_distance             pyFAI distance after optional thickness correction
azimuthal_mask_source           none, static, or mask column name
azimuthal_mask_pixels           number of excluded pixels
azimuthal_npt                   radial bins
azimuthal_npt_azimuthal         angular bins for 2D, otherwise None
azimuthal_mode                  1D or 2D
thickness_adjustment_applied    True/False
thickness_adjustment_reliable   True/False
thickness_adjustment_warning    warning text if correction was not applied
sample_thickness_mm             parsed float thickness
thickness_reference_mm          reference thickness in mm
thickness_adjusted_distance_m   corrected pyFAI distance
```

Use `error_model="poisson"` when the next step is `SNRTransformer`.

## Error Model For SNR

For product preprocessing use:

```python
AzimuthalIntegration(error_model="poisson")
```

pyFAI then returns `radial_profile_sigma` together with the integrated
intensity.

That sigma is the per-q-bin uncertainty used by `SNRTransformer`:

```text
snr_q = abs(radial_profile_data(q)) / radial_profile_sigma(q)
snr_linear = sqrt(mean(snr_q^2))
snr_db = 20 * log10(snr_linear)
```

Without `error_model="poisson"`, `radial_profile_sigma` can be missing and SNR
will be marked as `poisson_missing_sigma`.

The sigma is calculated along the integrated q profile because SNR is applied
after azimuthal integration, not on the raw 2D detector image.

## Mask Convention

The mask follows pyFAI convention:

```text
0 = keep pixel
1 = exclude pixel
```

For the current GFRM workflow, the mask normally comes from
`FaultyPixelDetector`.

The mask is not global. It belongs to one measurement row.

## Q Range

Set q range before integration:

```python
df["interpolation_q_range"] = [(2.0, 23.0)] * len(df)
```

The integrator passes this value directly to pyFAI as `radial_range`.

## 1D And 2D

1D integration:

```python
AzimuthalIntegration(mode="1D", npt=900)
```

2D integration:

```python
AzimuthalIntegration(mode="2D", npt=900, npt_azimuthal=360)
```

`npt` controls radial bins.

`npt_azimuthal` controls angular bins.

## Thickness Correction

Thickness correction is required for this project.

Reason:

```text
For thick samples, scattering does not come from the front surface only.
The effective scattering plane is approximated as the sample midpoint.
If calibration was done at a reference thickness, the detector distance must be
shifted by half of the thickness difference.
```

Formula:

```text
adjusted_distance_m =
    poni_distance_m - 0.5 * (sample_thickness_mm - thickness_reference_mm) * 1e-3
```

Example:

```text
PONI Distance = 0.100 m
sample_thickness_mm = 25.0
thickness_reference_mm = 11.0

adjusted_distance_m = 0.100 - 0.5 * (25.0 - 11.0) * 1e-3
                    = 0.093 m
```

`sample_thickness_mm` must be present in the input DataFrame.

`thickness_reference_mm` must be passed explicitly as a float.

If `sample_thickness_mm` or `thickness_reference_mm` is missing, product
azimuthal integration raises `ValueError`.

If `require_thickness_adjustment=False` is used outside the product path, missing
or invalid thickness can be marked as:

```text
thickness_adjustment_applied = False
thickness_adjustment_reliable = False
thickness_adjustment_warning = reason
```

Rows with `thickness_adjustment_reliable=False` should be treated as unreliable
for thick-sample analysis.
