# XRD Preprocessing Docs

Start here:

```text
pipeline.md                      full preprocessing order and product boundary
h5_to_df.md                      H5 container reader and DataFrame materialization
h5_session_filters.md            H5-level filtering before detector arrays load
gfrm_converter.md                Bruker GFRM decode and EOS photon conversion
container_raw_gfrm_requirement.md RAW GFRM requirement for H5 producers
faulty_pixels.md                 detector mask creation
azimuthal_integration.md         pyFAI integration, PONI, thickness correction
snr.md                           Poisson SNR calculation and SNRFilter
normalization.md                 q-range normalization
filters.md                       reusable DataFrame row filters
agbh_monochromaticity.md         AgBH monochromaticity QC
snapshots.md                     optional radial-profile pipeline snapshots
product_development_questions.md open product/research questions
```

## Boundary

`xrd_preprocessing` does not decide product inclusion policy.

Product repositories decide:

```text
cohort
dates
batches
K-alpha/K-beta policy
AgBH/HBH quality policy
labels
model target
```

`xrd_preprocessing` applies explicit reader, filter, integration, SNR, and
normalization commands supplied by product code or user code.

## Statistics

Transformers may keep internal `stats_` for sklearn-style inspection.

For reusable audit reports prefer explicit statistics helpers:

```text
faulty_pixel_statistics(df)
gfrm_photon_statistics(photons, metadata)
snr_filter_statistics(before_df, after_df)
agbh_filter_statistics(before_df, after_df)
h5_filter_statistics(before_df, after_df)
```

## Beta API Notes

Current beta API uses explicit names only:

```text
SNRTransformer(snr_method="poisson")
calculate_snr(..., snr_method="poisson")
read_gfrm_as_photons(save=False)
save_gfrm_as_npy(...) for explicit materialization
gfrm_photon_statistics(...) for GFRM photon diagnostics
trapz_compat(...) in xrd_preprocessing.numeric
```
