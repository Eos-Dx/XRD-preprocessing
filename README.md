# XRD-preprocessing

Lightweight preprocessing toolkit extracted from `xrd-analysis`.

Core scope:

- simple faulty-pixel detection and mask creation
- simple one-detector azimuthal integration via `pyFAI`
- q-range profile normalization
- Poisson-only signal-to-noise ratio in dB from pyFAI sigma
- AgBH monochromaticity QC scoring from calibration profiles
- reusable metadata and SNR row filters

Standard product preprocessing order:

```text
H5 container
-> H5SessionFilter(product/user supplied attrs)
-> h5_to_df(drop_missing_sample_thickness=True)
-> ColumnValueFilter(optional DataFrame metadata audit)
-> MetadataFilter(optional product/user supplied metadata)
-> FaultyPixelDetector
-> AzimuthalIntegration(error_model="poisson", thickness_reference_mm=<explicit float>)
-> SNRTransformer(snr_method="poisson")
-> SNRFilter(min_snr_db=20.0)
-> PatientSpecimenValidityFilter
-> QRangeNormalizer(q_min=6.7, q_max=7.1)
```

Product preprocessing requires container v0.3 with original RAW GFRM artifacts.
`measurement_data` is produced from `gfrm_to_photons`; stored NumPy arrays are
not the product source of truth.

Thickness contract:

```text
one measurement point has one sample thickness
missing sample thickness means no correct azimuthal integration
AGBH/reference thickness can differ between calibration sessions
AGBH/reference thickness should be present in H5/product metadata as agbh_thickness_mm
AzimuthalIntegration can use thickness_reference_column="agbh_thickness_mm"
product-specific AGBH/HBH reliability policy lives outside xrd_preprocessing
```

Protocol/spectrum boundary:

```text
xrd_preprocessing does not decide which measurements a product uses
product owns days, cohort, protocol, K-alpha/K-beta, batch, quality, and label policy
xrd_preprocessing applies explicit filters supplied by product/user
product metadata lives outside the library in controlled JSON/H5 attrs/manifests
example product-owned fields: spectrum_status, protocol_status,
calibration_quality_status, product_selection_status, patientId, specimenId
```

Open product-development questions:

```text
How does sample-thickness measurement error propagate into real q-position error?
How does X-ray beam position / beam-center error propagate into real q-position error?
Which product-owned metadata fields will each product provide for protocol/batch selection?
Which product-owned metadata fields will represent AGBH/HBH measurement reliability?
What geometry-error bounds should become product preprocessing QC thresholds?
```

Question details:

```text
src/xrd_preprocessing/docs/product_development_questions.md
```

Environment:

```bash
conda env create -f environment.yml
conda activate eosproduct
pytest
```

Pip-only:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

Parity checks against local `xrd-analysis`:

```bash
PYTHONPATH=/Users/sad/dev/xrd-analysis/src:src pytest -q
```

Examples:

```bash
marimo run examples/faulty_pixel_detection_demo.py
```

Key docs:

```text
src/xrd_preprocessing/docs/pipeline.md
src/xrd_preprocessing/docs/h5_to_df.md
src/xrd_preprocessing/docs/h5_session_filters.md
src/xrd_preprocessing/docs/agbh_monochromaticity.md
src/xrd_preprocessing/docs/product_development_questions.md
src/xrd_preprocessing/docs/filters.md
src/xrd_preprocessing/docs/snapshots.md
src/xrd_preprocessing/docs/gfrm_converter.md
src/xrd_preprocessing/docs/container_raw_gfrm_requirement.md
src/xrd_preprocessing/docs/azimuthal_integration.md
src/xrd_preprocessing/docs/faulty_pixels.md
src/xrd_preprocessing/docs/snr.md
src/xrd_preprocessing/docs/normalization.md
```
