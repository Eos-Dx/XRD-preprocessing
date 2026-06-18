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
snr_method_used      poisson, poisson_missing_sigma, or poisson_invalid_sigma
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

If sigma is missing, SNR is NaN and `snr_method_used = poisson_missing_sigma`.

If sigma is invalid, SNR is NaN and `snr_method_used = poisson_invalid_sigma`.

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

After `transform`, `SNRFilter.stats_` stores:

```text
rows_in
rows_pass
rows_fail
min_snr_db
max_snr_db
failed_ids, when sample_id column exists
```

## Pipeline Order

```text
GFRM -> photons
FaultyPixelDetector, optional
AzimuthalIntegration(error_model="poisson")
SNRTransformer(snr_method="poisson")
SNRFilter(min_snr_db=20.0)
QRangeNormalizer(q_min=6.7, q_max=7.1)
```
