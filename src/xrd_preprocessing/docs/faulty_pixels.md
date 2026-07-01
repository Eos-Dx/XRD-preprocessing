# Faulty Pixel Detection

`FaultyPixelDetector` is intentionally simple for GFRM detector frames.

It is not tuned for the Water example dataset. The Water scans are only an
example used to inspect behavior on real Bruker GFRM data.

The goal is narrow:

```text
build a frame-local pyFAI mask for pixels that can corrupt azimuthal integration
```

The detector does not infer a permanent detector map from one frame.

It does not try to classify detector physics beyond the current frame.

## Rules

```text
NaN/inf       -> faulty
image < 0     -> faulty
image > 500   -> faulty
0 <= image <= 500 -> OK
```

Zero values are not faulty.

## Minimal API

```python
from xrd_preprocessing import FaultyPixelDetector

detector = FaultyPixelDetector(
    bright_pixel_min_value=500.0,
    exclude_beam_center_radius=0.04,
)

out = detector.fit_transform(df)
```

Input DataFrame:

```text
measurement_data   2D photon image
ponifile           PONI text, used for beam-center exclusion
```

## Faulty-Pixel Search

There is no `3x3`, no `7x7`, no global z-score, no median filter, no MAD, and
no iterative sliding window.

The bright-pixel part of the faulty mask rule is absolute:

```text
pixel is masked when finite and non-negative and pixel > bright_pixel_min_value
```

Default:

```text
bright_pixel_min_value = 500.0
```

This threshold is deliberately high for the GFRM photon-converted detector
data. We are trying to remove pixels that can create sharp spikes after
azimuthal integration, not low or moderate signal variation.

Grouped bright pixels are handled naturally. If several neighboring pixels are
above `500`, every one of them is masked.

## Beam-Center Exclusion From PONI

Beam-center exclusion requires explicit opt-in:

```text
exclude_beam_center_radius = None
```

The detector reads beam geometry from the PONI text:

```text
Poni1
Poni2
Detector_config.pixel1
Detector_config.pixel2
```

Beam center in detector pixels:

```text
center_y = round(Poni1 / pixel1)
center_x = round(Poni2 / pixel2)
```

Excluded beam radius:

```text
radius_pixels = exclude_beam_center_radius * max(image.shape)
```

For a 512 x 768 detector, `0.04` is about `31` pixels.

Why exclude it:

```text
the primary-beam / beamstop region is not informative sample scattering
it can contain direct-beam tail, beamstop shadow, saturation, and geometry artifacts
marking it as faulty would confuse detector-defect statistics
pyFAI integration should not use it as evidence for real diffraction signal
```

Why `0.04`:

```text
small enough to preserve almost all detector area
large enough to cover the immediate beam-center artifact zone
expressed as detector-size fraction so it scales with image shape
configurable if a different detector/protocol needs a different exclusion zone
```

The beam zone is removed from `faulty_pixel_mask`. It is not counted as
detector damage.

## Output Columns

```text
faulty_pixel_mask           Nx2 coordinates, invalid + bright faulty
```

## Statistics

Use a separate stats function:

```python
from xrd_preprocessing import faulty_pixel_statistics

out = FaultyPixelDetector().fit_transform(df)
stats = faulty_pixel_statistics(out)
```

Returned fields:

```text
n_images
faulty_pixels_per_row
total_faulty_pixels
```
