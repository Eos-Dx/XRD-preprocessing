"""Marimo product pipeline demo from RAW GFRM files."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import importlib
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline

    repo_root = (
        Path(__file__).resolve().parents[1]
        if "__file__" in globals()
        else Path.cwd()
    )
    src_path = repo_root / "src"
    examples_path = repo_root / "examples"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(examples_path) not in sys.path:
        sys.path.insert(0, str(examples_path))

    import product_pipeline_helpers as helpers

    helpers = importlib.reload(helpers)

    from xrd_preprocessing import (
        AzimuthalIntegration,
        FaultyPixelDetector,
        PatientSpecimenValidityFilter,
        QRangeNormalizer,
        RadialProfileSnapshot,
        SNRFilter,
        SNRTransformer,
    )

    return (
        AzimuthalIntegration,
        FaultyPixelDetector,
        PatientSpecimenValidityFilter,
        Pipeline,
        QRangeNormalizer,
        RadialProfileSnapshot,
        SNRFilter,
        SNRTransformer,
        helpers,
        np,
        pd,
        plt,
        repo_root,
    )


@app.cell
def _(mo):
    mo.md("""
    # Product GFRM preprocessing pipeline

    This notebook uses one product path only:

    ```text
    already H5-selected rows -> GFRM -> photons -> DataFrame with sample_thickness_mm
    -> FaultyPixelDetector
    -> AzimuthalIntegration, q = 2..23 nm^-1
    -> SNRTransformer
    -> SNRFilter
    -> PatientSpecimenValidityFilter
    -> QRangeNormalizer
    ```

    This is a loose-GFRM demonstration, not an H5-container loader. In product
    container workflows, date, diagnosis/status, PONI q coverage, and other
    metadata filters are applied with `H5SessionFilter` before `h5_to_df`.
    """)
    return


@app.cell
def _():
    thickness_reference_mm = 11.0
    npt = 100
    return npt, thickness_reference_mm


@app.cell
def _(helpers, repo_root, thickness_reference_mm):
    input_df = helpers.load_example_gfrm_dataframe(
        repo_root=repo_root,
        thickness_reference_mm=thickness_reference_mm,
    )
    input_df["measurementDate"] = "2026-06-08"
    input_df["diagnosis"] = "BENIGN"
    input_df["patientId"] = [
        f"demo_patient_{_idx // 6 + 1:02d}" for _idx in range(len(input_df))
    ]
    input_df["specimenId"] = [
        f"{_patient_id}_specimen_{(_idx // 3) % 2 + 1}"
        for _idx, _patient_id in enumerate(input_df["patientId"])
    ]
    return (input_df,)


@app.cell
def _(input_df, mo, thickness_reference_mm):
    _thickness_values = sorted(input_df["sample_thickness_mm"].unique().tolist())
    mo.md(
        f"""
        Step 1. Already H5-selected GFRM rows.

        Each Bruker GFRM frame was decoded with FabIO and converted to EOS
        photon estimates. The DataFrame already contains the required
        `sample_thickness_mm` column. This loose-GFRM example adds demo clinical
        metadata only for later patient/specimen validity grouping.

        Real product container workflows should apply date, diagnosis/status,
        and PONI q-coverage filters at H5 level before `h5_to_df`.

        ```text
        frames = {len(input_df)}
        q integration range = 2.0..23.0 nm^-1
        thickness_reference_mm = {thickness_reference_mm}
        sample_thickness_mm values = {_thickness_values}
        measurement_data source = RAW GFRM -> gfrm_to_photons
        grouping columns = patientId, specimenId
        ```
        """
    )
    return


@app.cell
def _(FaultyPixelDetector, input_df):
    faulty_detector = FaultyPixelDetector(
        local_hot_min_value=500.0,
        exclude_beam_center_radius=0.04,
    )
    faulty_df = faulty_detector.fit_transform(input_df)
    faulty_stats = faulty_detector.stats_
    return faulty_df, faulty_stats


@app.cell
def _(faulty_df, faulty_stats, helpers, mo):
    _fig = helpers.plot_faulty_counts(faulty_df)
    mo.vstack(
        [
            mo.md(
                f"""
                Step 2. Measurement-level faulty-pixel detection.

                NaN/inf, negative values, and values above 500 are excluded.
                Beam-zone pixels are not counted as faulty pixels.

                ```text
                total_faulty_pixels = {faulty_stats["total_faulty_pixels"]}
                total_invalid_pixels = {faulty_stats["total_invalid_pixels"]}
                total_suspected_hot_pixels = {faulty_stats["total_suspected_hot_pixels"]}
                beam_center_radius_frac = {faulty_stats["beam_center_radius_frac"]}
                ```
                """
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def _(AzimuthalIntegration, faulty_df, npt, thickness_reference_mm):
    integrator = AzimuthalIntegration(
        npt=npt,
        calibration_mode="poni",
        column="measurement_data",
        mask_column="pyfai_faulty_pixel_mask",
        error_model="poisson",
        thickness_reference_mm=thickness_reference_mm,
    )
    integrated_df = integrator.fit_transform(faulty_df)
    return (integrated_df,)


@app.cell
def _(helpers, integrated_df, mo):
    _fig = helpers.plot_profiles(
        integrated_df,
        title="After GFRM conversion, faulty pixels, and azimuthal integration",
    )
    mo.vstack(
        [
            mo.md("""
            Step 3. Azimuthal integration.

            The pyFAI integrator uses PONI geometry. The faulty-pixel mask is
            passed per row. Integration range is q = 2..23 nm^-1.
            """),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def _(SNRTransformer, integrated_df):
    snr_transformer = SNRTransformer(snr_method="poisson")
    snr_df = snr_transformer.fit_transform(integrated_df)
    return (snr_df,)


@app.cell
def _(helpers, mo, snr_df):
    _profile_fig = helpers.plot_profiles(
        snr_df,
        title="After Poisson SNR calculation",
        show_snr=True,
    )
    _snr_fig = helpers.plot_snr(snr_df)
    mo.vstack(
        [
            mo.md("""
            Step 4. Poisson SNR analysis.

            `radial_profile_sigma` from pyFAI is used to calculate SNR in dB.
            The profile legend and bar labels show SNR for every curve.
            """),
            mo.as_html(_profile_fig),
            mo.as_html(_snr_fig),
        ]
    )
    return


@app.cell
def _(SNRFilter, snr_df):
    snr_filter = SNRFilter(min_snr_db=20.0)
    snr_filtered_df = snr_filter.fit_transform(snr_df)
    snr_filter_stats = snr_filter.stats_
    return snr_filter_stats, snr_filtered_df


@app.cell
def _(helpers, mo, snr_filter_stats, snr_filtered_df):
    _fig = helpers.plot_profiles(
        snr_filtered_df,
        title="After SNRFilter, kept profiles",
        show_snr=True,
    )
    mo.vstack(
        [
            mo.md(
                f"""
                Step 5. SNR filtering.

                Curves with `snr_db < 20` or missing SNR are removed.

                ```text
                rows_in = {snr_filter_stats["rows_in"]}
                rows_pass = {snr_filter_stats["rows_pass"]}
                rows_fail = {snr_filter_stats["rows_fail"]}
                threshold = 20 dB
                ```
                """
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def _(PatientSpecimenValidityFilter, snr_filtered_df):
    clinical_validity_filter = PatientSpecimenValidityFilter(
        min_measurements_per_specimen=2,
        min_specimens_per_patient=1,
    )
    clinically_valid_df = clinical_validity_filter.fit_transform(snr_filtered_df)
    clinical_validity_stats = clinical_validity_filter.stats_
    return clinical_validity_stats, clinically_valid_df


@app.cell
def _(clinical_validity_stats, clinically_valid_df, helpers, mo):
    _fig = helpers.plot_profiles(
        clinically_valid_df,
        title="After patient/specimen validity filtering",
        show_snr=True,
    )
    mo.vstack(
        [
            mo.md(
                f"""
                Step 6. Patient/specimen validity filtering.

                This runs after SNR filtering, so specimen measurement counts
                are based only on profiles that survived signal-quality QC.

                ```text
                rows_in = {clinical_validity_stats["rows_in"]}
                rows_pass = {clinical_validity_stats["rows_pass"]}
                rows_fail = {clinical_validity_stats["rows_fail"]}
                patients_pass = {clinical_validity_stats["patients_pass"]}
                specimens_pass = {clinical_validity_stats["specimens_pass"]}
                ```
                """
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def _(QRangeNormalizer, clinically_valid_df):
    normalizer = QRangeNormalizer(
        q_min=6.7,
        q_max=7.1,
        save_initial_data=True,
    )
    normalized_df = normalizer.fit_transform(clinically_valid_df)
    return (normalized_df,)


@app.cell
def _(helpers, mo, normalized_df):
    _fig = helpers.plot_profiles(
        normalized_df,
        title="After QRangeNormalizer, normalized profiles",
        show_snr=True,
    )
    mo.vstack(
        [
            mo.md("""
            Step 7. Q-range normalization.

            Remaining curves are normalized by the integrated area in
            q = 6.7..7.1 nm^-1 and plotted together.
            """),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def _(mo, normalized_df):
    mo.md(
        f"""
        Final product-preprocessed DataFrame:

        ```text
        rows = {len(normalized_df)}
        columns include:
        pyfai_faulty_pixel_mask
        q_range
        radial_profile_data
        radial_profile_sigma
        snr_db
        q_range_normalization_area
        radial_profile_data_raw
        ```
        """
    )
    return


@app.cell
def _(
    AzimuthalIntegration,
    FaultyPixelDetector,
    PatientSpecimenValidityFilter,
    Pipeline,
    QRangeNormalizer,
    RadialProfileSnapshot,
    SNRFilter,
    SNRTransformer,
    input_df,
    npt,
    thickness_reference_mm,
):
    save_pipeline_stages = True
    product_pipeline = Pipeline(
        [
            (
                "faulty_pixels",
                FaultyPixelDetector(
                    local_hot_min_value=500.0,
                    exclude_beam_center_radius=0.04,
                ),
            ),
            (
                "integrate",
                AzimuthalIntegration(
                    npt=npt,
                    calibration_mode="poni",
                    column="measurement_data",
                    mask_column="pyfai_faulty_pixel_mask",
                    error_model="poisson",
                    thickness_reference_mm=thickness_reference_mm,
                ),
            ),
            (
                "snapshot_after_integration",
                RadialProfileSnapshot(
                    "after_integration",
                    enabled=save_pipeline_stages,
                ),
            ),
            ("snr", SNRTransformer(snr_method="poisson")),
            (
                "snapshot_after_snr",
                RadialProfileSnapshot("after_snr", enabled=save_pipeline_stages),
            ),
            ("snr_filter", SNRFilter(min_snr_db=20.0)),
            (
                "clinical_validity",
                PatientSpecimenValidityFilter(
                    min_measurements_per_specimen=2,
                    min_specimens_per_patient=1,
                ),
            ),
            (
                "snapshot_after_clinical_validity",
                RadialProfileSnapshot(
                    "after_clinical_validity",
                    enabled=save_pipeline_stages,
                ),
            ),
            (
                "normalize",
                QRangeNormalizer(
                    q_min=6.7,
                    q_max=7.1,
                    save_initial_data=save_pipeline_stages,
                ),
            ),
            (
                "snapshot_after_normalization",
                RadialProfileSnapshot(
                    "after_normalization",
                    enabled=save_pipeline_stages,
                ),
            ),
        ]
    )
    full_pipeline_df = product_pipeline.fit_transform(input_df)
    full_pipeline_stats = product_pipeline.named_steps["snr_filter"].stats_
    full_pipeline_clinical_stats = product_pipeline.named_steps[
        "clinical_validity"
    ].stats_
    return (
        full_pipeline_clinical_stats,
        full_pipeline_df,
        full_pipeline_stats,
        product_pipeline,
        save_pipeline_stages,
    )


@app.cell
def _(
    full_pipeline_clinical_stats,
    full_pipeline_df,
    full_pipeline_stats,
    helpers,
    mo,
    save_pipeline_stages,
):
    _snapshot_columns = [
        _column
        for _column in full_pipeline_df.columns
        if _column.startswith("radial_profile_data_")
        or _column.startswith("q_range_")
    ]
    _fig = helpers.plot_profiles(
        full_pipeline_df,
        title="Full Pipeline.fit_transform output",
        show_snr=True,
    )
    mo.vstack(
        [
            mo.md(
                f"""
                Full product pipeline.

                One input DataFrame goes into `Pipeline.fit_transform`.
                One output DataFrame comes out.

                ```text
                save_pipeline_stages = {save_pipeline_stages}
                rows before SNRFilter = {full_pipeline_stats["rows_in"]}
                rows removed by SNRFilter = {full_pipeline_stats["rows_fail"]}
                rows before clinical validity = {full_pipeline_clinical_stats["rows_in"]}
                rows removed by clinical validity = {full_pipeline_clinical_stats["rows_fail"]}
                output rows = {len(full_pipeline_df)}
                saved stage columns = {_snapshot_columns}
                ```
                """
            ),
            mo.as_html(_fig),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
