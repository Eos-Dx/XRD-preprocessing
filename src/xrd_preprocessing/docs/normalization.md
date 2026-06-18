# Q Range Normalization

`QRangeNormalizer` normalizes 1D XRD profiles by integrated intensity inside a
fixed q range.

Default range:

```text
6.7 <= q <= 7.1 nm^-1
```

## Formula

For each row:

```text
area = integral(I(q), q=6.7..7.1)
I_norm(q) = I(q) / area
```

The integral is calculated with the trapezoidal rule.

The whole profile is divided by the band area, not only the band itself.

## Minimal API

```python
from xrd_preprocessing import QRangeNormalizer

normalizer = QRangeNormalizer(
    q_min=6.7,
    q_max=7.1,
)

out = normalizer.fit_transform(df)
```

Input columns:

```text
q_range                q positions
radial_profile_data    intensity
```

Output columns:

```text
radial_profile_data            normalized intensity
q_range_normalization_area     area used as denominator
q_range_normalization_min      6.7
q_range_normalization_max      7.1
```

Raw `radial_profile_data` is overwritten by default.

Parameter defaults:

```text
output_column = None      means overwrite radial_profile_data
save_initial_data = False
initial_column = radial_profile_data_raw
```

To keep the input profile:

```python
QRangeNormalizer(save_initial_data=True)
```

Then the raw input profile is stored in:

```text
radial_profile_data_raw
```

To write normalized data into a separate column:

```python
QRangeNormalizer(output_column="radial_profile_data_norm")
```

## Failure Cases

The transformer raises `ValueError` when:

```text
q_range column is missing
radial_profile_data column is missing
q/intensity have fewer than 2 points
normalization range has fewer than 2 finite points
integral is zero, NaN, or infinite
```
