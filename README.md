# XRD-preprocessing

Lightweight preprocessing toolkit extracted from `xrd-analysis`.

Core scope:

- simple faulty-pixel detection and mask creation
- simple one-detector azimuthal integration via `pyFAI`
- q-range profile normalization
- Poisson-only signal-to-noise ratio in dB from pyFAI sigma
- reusable metadata and SNR row filters

Standard product preprocessing order:

```text
h5_to_df
-> ColumnValueFilter(date cutoff)
-> MetadataFilter(diagnosis cohort)
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
src/xrd_preprocessing/docs/snapshots.md
src/xrd_preprocessing/docs/azimuthal_integration.md
src/xrd_preprocessing/docs/faulty_pixels.md
src/xrd_preprocessing/docs/snr.md
src/xrd_preprocessing/docs/normalization.md
```
