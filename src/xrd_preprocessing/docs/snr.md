# Signal-to-Noise Ratio

Product SNR uses only Poisson uncertainty.

Use it after azimuthal integration with pyFAI sigma:

```python
AzimuthalIntegration(error_model="poisson")
SNRTransformer(snr_method="poisson")
SNRFilter(min_snr_db=20.0)
```

Input columns:

```text
q_range
radial_profile_data
radial_profile_sigma   from pyFAI error_model="poisson"
```

Output columns:

```text
noise_std
snr_linear
snr_db
snr_method_used      poisson
```

## Formula

For each q point:

```text
snr_q = abs(I(q)) / sigma(q)
```

For the whole profile:

```text
snr_linear = RMS(snr_q)
snr_db = 20 * log10(snr_linear)
```

This is the kept method because sigma comes from counting statistics during
azimuthal integration.

No smoothing is used. No residual profile is used. SNR is calculated only from
integrated intensity and pyFAI Poisson sigma.

If sigma is missing, SNR raises an error.

If sigma is invalid, SNR raises an error.

If profile intensity has fewer than two points, SNR raises an error.

## Filtering

`SNRFilter` keeps only rows with finite SNR at or above the threshold.

Default:

```python
SNRFilter(min_snr_db=20.0)
```

Rule:

```text
keep row when snr_db >= 20 dB
drop row when snr_db is NaN or snr_db < 20 dB
```

`SNRFilter` is a thin `ColumnValueFilter` alias. It does not write SNR-specific
columns.

It does not create:

```text
snr_pass
snr_min_db
```

`drop=False` is not supported because `SNRFilter` is only a named
`ColumnValueFilter` for `snr_db >= min_snr_db`.

Use a separate stats function when audit output is needed:

```python
from xrd_preprocessing import SNRFilter, snr_filter_statistics

filtered = SNRFilter(min_snr_db=20.0).fit_transform(df)
stats = snr_filter_statistics(
    before_df=df,
    after_df=filtered,
    snr_column="snr_db",
    min_snr_db=20.0,
)
```

## Pipeline Order

```text
GFRM -> photons
FaultyPixelDetector, optional
AzimuthalIntegration(error_model="poisson")
SNRTransformer(snr_method="poisson")
SNRFilter(min_snr_db=20.0)
QRangeValueNormalizer(q_min=6.7, q_max=7.1, statistic="median")
```
