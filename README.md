# XRD-preprocessing

Lightweight preprocessing toolkit extracted from `xrd-analysis`.

Core scope:

- simple faulty-pixel detection and mask creation
- simple one-detector azimuthal integration via `pyFAI`
- q-range profile normalization
- Poisson-only signal-to-noise ratio in dB from pyFAI sigma
- AgBH monochromaticity QC scoring from calibration profiles
- reusable metadata and SNR row filters
- sklearn-style product transformers for H5 reading, product metadata columns,
  paired-group filters, q-range metadata, and final column selection
- reusable preprocessing YAML template/contract, loader API, and YAML pipeline
  builder

Standard product preprocessing order:

```text
H5 container
-> H5SessionSelectorTransformer(product/user supplied attrs: date/status/PONI q/thickness availability)
-> H5MeasurementSetAuditTransformer(optional metadata-only stage counts)
-> H5ToDataFrameTransformer(drop_missing_sample_thickness=True)
-> ProductColumnBuilder / ProductStatusGroupFilter(product label policy)
-> FaultyPixelDetector
-> AzimuthalIntegration(error_model="poisson", thickness_reference_column="calibrant_thickness_mm")
-> SNRTransformer(snr_method="poisson")
-> SNRFilter(min_snr_db=20.0)
-> PatientSpecimenValidityFilter
-> QRangeValueNormalizer(q_min=6.7, q_max=7.1, statistic="median")
-> RadialProfileValueFilter(optional product signal gate)
```

Product preprocessing requires container v0.3 with original RAW GFRM artifacts.
`measurement_data` is produced from `gfrm_to_photons`; stored NumPy arrays are
not the product source of truth.

Detailed module docs start at:

```text
src/xrd_preprocessing/docs/README.md
```

Thickness contract:

```text
one measurement point has one sample thickness
missing sample thickness means no correct azimuthal integration
calibrant/reference thickness can differ between calibration sessions
calibrant/reference thickness should be present in H5/product metadata as calibrant_thickness_mm
AzimuthalIntegration can use thickness_reference_column="calibrant_thickness_mm"
product-specific AGBH/HBH reliability policy lives outside xrd_preprocessing
```

Transformer contract:

```text
v0.1.6-beta product movement should be expressed as transformers
every DataFrame-changing product step should support fit_transform
H5SessionSelectorTransformer returns a manifest with archive_path, all_session_df, selected session_df, and h5_filters
H5MeasurementSetAuditTransformer adds stage count DataFrames without loading detector arrays
H5ToDataFrameTransformer can consume that manifest and materialize only selected H5 sessions
functions remain available for low-level direct use and backward compatibility
XRD-preprocessing owns reusable YAML templates/contracts
product repositories own concrete YAML/JSON rules and compose XRD-preprocessing transformers
load_preprocessing_config(...) validates concrete configs against the reusable contract
build_pipeline_from_config(...) builds sklearn Pipeline from YAML pipeline.steps
```

Statistics contract:

```text
transformers may keep internal stats_ for sklearn-style inspection
human/product audit reports should prefer explicit statistics helpers
examples: faulty_pixel_statistics, snr_filter_statistics, agbh_filter_statistics,
gfrm_photon_statistics, h5_filter_statistics
```

YAML pipeline contract:

```yaml
pipeline:
  steps:
    - name: h5_to_df
      transformer: H5ToDataFrameTransformer
      params:
        data_preference:
          $ref: raw_data.source
    - name: keep_columns
      transformer: KeepColumnsTransformer
      params:
        columns:
          $concat:
            - $ref: metadata.output_columns
            - $ref: branch_settings.output_columns
```

Each `transformer` must be in the explicit XRD-preprocessing transformer
registry. Products can add, remove, disable, or reorder product steps in YAML
without hardcoding the route in product code. `$ref` resolves values from the
loaded config after `extends` is merged. `extends` can be one YAML file or an
ordered list of YAML files; later files override earlier files. `$concat`
appends resolved lists and scalars; use it when a shared base owns common
output columns and a branch adds only branch-specific columns. Any step or
nested parameter item can define `enabled: false` or
`enabled: {$ref: some.boolean.flag}`. Step names must be unique.

Transformer module layout:

```text
xrd_preprocessing.transformers.h5        H5 selection/audit/reader transformers
xrd_preprocessing.transformers.metadata  product columns, q-range, required columns, joblib output
xrd_preprocessing.transformers.labels    product status-group and paired-patient filters
xrd_preprocessing.transformers.profiles  lightweight synthetic radial-profile transformer
```

Bundled preprocessing template:

```text
src/xrd_preprocessing/configs/preprocessing_config_template.yaml
  commented combined multi-branch YAML template

src/xrd_preprocessing/configs/preprocessing_branch_config_template.yaml
  commented branch-specific YAML template used by current Aramis configs
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

H5-level filtering rule:

```text
If a filter can be computed from H5 attrs/metadata/PONI without loading frames,
apply it through H5SessionFilter before h5_to_df.
Examples: started_at, specimen_status, patientId/specimenId, side,
poni_q_max_nm_inv, biopsy, h5_sample_all_sets_have_thickness.
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
python examples/h5_session_filter_demo.py --required-q-max 23
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
