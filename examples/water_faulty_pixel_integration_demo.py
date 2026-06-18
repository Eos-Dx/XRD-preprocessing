"""Marimo demo comparing azimuthal integration with and without faulty masks."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import sys
    from pathlib import Path

    import marimo as mo
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
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from xrd_preprocessing import (
        AzimuthalIntegration,
        FaultyPixelDetector,
        QRangeNormalizer,
        SNRFilter,
        SNRTransformer,
        extract_gfrm_archive,
        gfrm_to_photons,
    )

    return (
        AzimuthalIntegration,
        FaultyPixelDetector,
        Path,
        Pipeline,
        QRangeNormalizer,
        SNRFilter,
        SNRTransformer,
        extract_gfrm_archive,
        gfrm_to_photons,
        mo,
        np,
        pd,
        plt,
        re,
        repo_root,
    )


@app.cell
def _(mo):
    mo.md("""
    # Water GFRM faulty-pixel integration check

    All Water-folder GFRM frames are converted with standard code, then
    integrated twice:

    ```text
    no_faulty_pipeline:   AzimuthalIntegration -> SNRTransformer -> SNRFilter -> QRangeNormalizer
    with_faulty_pipeline: FaultyPixelDetector -> AzimuthalIntegration -> SNRTransformer -> SNRFilter -> QRangeNormalizer
    ```

    The second pipeline passes the PyFAI mask produced by faulty-pixel detection
    into the same simple one-image azimuthal integration applied row by row.
    Both pipelines calculate Poisson SNR after azimuthal integration, keep
    profiles with SNR >= 20 dB, then normalize by q = 6.7-7.1 nm^-1.
    """)
    return


@app.cell
def _():
    thickness_reference_mm = 11.0
    return thickness_reference_mm


@app.cell
def _(extract_gfrm_archive, gfrm_to_photons, pd, re, repo_root, thickness_reference_mm):
    data_dir = repo_root / "examples" / "data"
    archive_path = data_dir / "gfrm_measurements.tar.gz"
    extracted_root = extract_gfrm_archive(archive_path, data_dir / "_gfrm_measurements")
    water_gfrm_paths = sorted(
        (extracted_root / "GFRM_measurements" / "Water").glob("*/*.gfrm")
    )
    poni_text = (data_dir / "poni_coords_for_all.poni").read_text()

    rows = []
    for _gfrm_path in water_gfrm_paths:
        _image, _metadata = gfrm_to_photons(_gfrm_path)
        _match = re.search(r"Water_(\d+)mm", _gfrm_path.parent.name)
        _sample_thickness_mm = (
            float(_match.group(1)) if _match is not None else thickness_reference_mm
        )
        rows.append(
            {
                "sample_id": _gfrm_path.parent.name,
                "measurement_data": _image,
                "ponifile": poni_text,
                "sample_thickness_mm": _sample_thickness_mm,
                "interpolation_q_range": (2.0, 23.0),
                "azimuthal_range": None,
                "gfrm_path": str(_gfrm_path),
                "baseline_adu": _metadata["baseline_adu"],
                "gain_adu_per_photon": _metadata["gain_adu_per_photon"],
                "masked_row_511": _metadata["masked_row_511"],
            }
        )

    water_df = pd.DataFrame(rows)
    return archive_path, water_df


@app.cell
def _(archive_path, mo, water_df):
    _baseline_values = sorted(water_df["baseline_adu"].unique().tolist())
    _gain_values = sorted(water_df["gain_adu_per_photon"].unique().tolist())
    mo.md(
        f"""
        Loaded `{len(water_df)}` Water-folder GFRM frames from:

        `{archive_path}`

        Conversion:

        ```text
        baseline_adu_values = {_baseline_values}
        gain_adu_per_photon_values = {_gain_values}
        masked_row_511 = {water_df["masked_row_511"].all()}
        q_range = 2.0-23.0 nm^-1
        ```
        """
    )
    return


@app.cell
def _(
    AzimuthalIntegration,
    FaultyPixelDetector,
    Pipeline,
    QRangeNormalizer,
    SNRFilter,
    SNRTransformer,
    thickness_reference_mm,
):
    no_faulty_pipeline = Pipeline(
        [
            (
                "integrate",
                AzimuthalIntegration(
                    npt=900,
                    calibration_mode="poni",
                    column="measurement_data",
                    error_model="poisson",
                    thickness_reference_mm=thickness_reference_mm,
                ),
            ),
            (
                "snr",
                SNRTransformer(snr_method="poisson"),
            ),
            (
                "snr_filter",
                SNRFilter(min_snr_db=20.0),
            ),
            (
                "normalize",
                QRangeNormalizer(q_min=6.7, q_max=7.1),
            ),
        ]
    )

    with_faulty_pipeline = Pipeline(
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
                    npt=900,
                    calibration_mode="poni",
                    column="measurement_data",
                    mask_column="pyfai_faulty_pixel_mask",
                    error_model="poisson",
                    thickness_reference_mm=thickness_reference_mm,
                ),
            ),
            (
                "snr",
                SNRTransformer(snr_method="poisson"),
            ),
            (
                "snr_filter",
                SNRFilter(min_snr_db=20.0),
            ),
            (
                "normalize",
                QRangeNormalizer(q_min=6.7, q_max=7.1),
            ),
        ]
    )
    return no_faulty_pipeline, with_faulty_pipeline


@app.cell
def _(no_faulty_pipeline, water_df, with_faulty_pipeline):
    no_faulty_df = no_faulty_pipeline.fit_transform(water_df)
    with_faulty_df = with_faulty_pipeline.fit_transform(water_df)
    return no_faulty_df, with_faulty_df


@app.cell
def _(
    mo,
    no_faulty_df,
    no_faulty_pipeline,
    water_df,
    with_faulty_df,
    with_faulty_pipeline,
):
    _no_stats = no_faulty_pipeline.named_steps["snr_filter"].stats_
    _with_stats = with_faulty_pipeline.named_steps["snr_filter"].stats_
    mo.md(
        f"""
        SNR filter:

        ```text
        method = poisson
        threshold = 20 dB
        input_frames = {len(water_df)}
        no_mask_frames_after_snr = {len(no_faulty_df)}
        with_mask_frames_after_snr = {len(with_faulty_df)}
        no_mask_snr_db_min_max_before_filter = ({_no_stats["min_snr_db"]:.6g}, {_no_stats["max_snr_db"]:.6g})
        with_mask_snr_db_min_max_before_filter = ({_with_stats["min_snr_db"]:.6g}, {_with_stats["max_snr_db"]:.6g})
        no_mask_failed_ids = {_no_stats["failed_ids"]}
        with_mask_failed_ids = {_with_stats["failed_ids"]}
        ```
        """
    )
    return


@app.cell
def _(no_faulty_df, np, plt, with_faulty_df):
    _fig, _ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    _with_by_sample = {
        _row.sample_id: float(_row.snr_db)
        for _row in with_faulty_df.itertuples(index=False)
    }
    _labels = []
    _no_values = []
    _with_values = []
    for _row in no_faulty_df.itertuples(index=False):
        if _row.sample_id in _with_by_sample:
            _labels.append(str(_row.sample_id).replace("20260608_", ""))
            _no_values.append(float(_row.snr_db))
            _with_values.append(_with_by_sample[_row.sample_id])
    _x = np.arange(len(_labels))
    _ax.bar(_x - 0.18, _no_values, width=0.36, label="without mask")
    _ax.bar(_x + 0.18, _with_values, width=0.36, label="with mask")
    _ax.axhline(20, color="0.2", linewidth=0.8)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_labels, rotation=45, ha="right", fontsize=7)
    _ax.set_ylabel("SNR, dB")
    _ax.set_title("Poisson SNR after azimuthal integration")
    _ax.grid(True, axis="y", alpha=0.25)
    _ax.legend(fontsize=8)
    _fig
    return


@app.cell
def _(np, water_df, with_faulty_df):
    _reason_totals = {}
    for _counts in with_faulty_df["faulty_pixel_reason_counts"]:
        for _key, _value in _counts.items():
            _reason_totals[_key] = _reason_totals.get(_key, 0) + int(_value)

    faulty_summary = {
        "frames": int(len(with_faulty_df)),
        "faulty_pixels": int(sum(len(_mask) for _mask in with_faulty_df["faulty_pixel_mask"])),
        "invalid_pixels": int(sum(len(_mask) for _mask in with_faulty_df["invalid_pixel_mask"])),
        "suspected_hot_pixels": int(
            sum(len(_mask) for _mask in with_faulty_df["suspected_hot_pixel_mask"])
        ),
        "negative_pixels_before_beam_exclusion": int(
            sum(np.sum(_image < 0) for _image in water_df["measurement_data"])
        ),
        "nan_or_inf_pixels_before_beam_exclusion": int(
            sum(np.sum(~np.isfinite(_image)) for _image in water_df["measurement_data"])
        ),
        "reason_counts": _reason_totals,
    }
    return (faulty_summary,)


@app.cell
def _(faulty_summary, mo):
    mo.md(
        f"""
        Faulty-pixel summary:

        ```text
        frames = {faulty_summary["frames"]}
        faulty_pixels = {faulty_summary["faulty_pixels"]}
        invalid_pixels = {faulty_summary["invalid_pixels"]}
        suspected_hot_pixels = {faulty_summary["suspected_hot_pixels"]}
        negative_pixels_before_beam_exclusion = {faulty_summary["negative_pixels_before_beam_exclusion"]}
        nan_or_inf_pixels_before_beam_exclusion = {faulty_summary["nan_or_inf_pixels_before_beam_exclusion"]}
        reason_counts = {faulty_summary["reason_counts"]}
        ```
        """
    )
    return


@app.cell
def _(mo, no_faulty_df, np, with_faulty_df):
    _no_area = np.asarray(no_faulty_df["q_range_normalization_area"], dtype=float)
    _with_area = np.asarray(with_faulty_df["q_range_normalization_area"], dtype=float)
    mo.md(
        f"""
        Normalization:

        ```text
        q_range_normalization = 6.7-7.1 nm^-1
        no_mask_area_min_max = ({np.nanmin(_no_area):.6g}, {np.nanmax(_no_area):.6g})
        with_mask_area_min_max = ({np.nanmin(_with_area):.6g}, {np.nanmax(_with_area):.6g})
        ```
        """
    )
    return


@app.cell
def _(no_faulty_df, np, plt, with_faulty_df):
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 4.8),
        constrained_layout=True,
    )

    for _row in no_faulty_df.itertuples(index=False):
        _q_no = np.asarray(_row.q_range)
        _i_no = np.asarray(_row.radial_profile_data)
        _label = _row.sample_id.replace("20260608_", "")
        _axes[0].plot(_q_no, _i_no, linewidth=1.0, alpha=0.85, label=_label)

    _axes[0].set_title("Without faulty-pixel mask")
    _axes[0].set_xlabel("q-range")
    _axes[0].set_ylabel("intensity")
    _axes[0].grid(True, alpha=0.25)

    for _row in with_faulty_df.itertuples(index=False):
        _q_with = np.asarray(_row.q_range)
        _i_with = np.asarray(_row.radial_profile_data)
        _label = _row.sample_id.replace("20260608_", "")
        _axes[1].plot(_q_with, _i_with, linewidth=1.0, alpha=0.85, label=_label)

    _axes[1].set_title("With faulty-pixel mask")
    _axes[1].set_xlabel("q-range")
    _axes[1].set_ylabel("intensity")
    _axes[1].grid(True, alpha=0.25)
    _axes[1].legend(fontsize=6, loc="best")

    _fig.suptitle("Azimuthal integration of all Water-folder frames")
    _fig
    return


@app.cell
def _(no_faulty_df, np, plt, with_faulty_df):
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 4.8),
        constrained_layout=True,
    )

    for _row in no_faulty_df.itertuples(index=False):
        _q_no = np.asarray(_row.q_range)
        _i_no = np.asarray(_row.radial_profile_data_norm)
        _label = _row.sample_id.replace("20260608_", "")
        _axes[0].plot(_q_no, _i_no, linewidth=1.0, alpha=0.85, label=_label)

    _axes[0].set_title("Normalized without faulty-pixel mask")
    _axes[0].set_xlabel("q-range")
    _axes[0].set_ylabel("normalized intensity")
    _axes[0].grid(True, alpha=0.25)

    for _row in with_faulty_df.itertuples(index=False):
        _q_with = np.asarray(_row.q_range)
        _i_with = np.asarray(_row.radial_profile_data_norm)
        _label = _row.sample_id.replace("20260608_", "")
        _axes[1].plot(_q_with, _i_with, linewidth=1.0, alpha=0.85, label=_label)

    _axes[1].set_title("Normalized with faulty-pixel mask")
    _axes[1].set_xlabel("q-range")
    _axes[1].set_ylabel("normalized intensity")
    _axes[1].grid(True, alpha=0.25)
    _axes[1].legend(fontsize=6, loc="best")

    _fig.suptitle("Q-range-normalized profiles, q = 6.7-7.1 nm^-1")
    _fig
    return


@app.cell
def _(no_faulty_df, np, plt, with_faulty_df):
    _fig, _ax = plt.subplots(figsize=(11, 4), constrained_layout=True)

    for _no_row, _with_row in zip(
        no_faulty_df.itertuples(index=False),
        with_faulty_df.itertuples(index=False),
    ):
        _q = np.asarray(_no_row.q_range)
        _i_no = np.asarray(_no_row.radial_profile_data)
        _i_with = np.asarray(_with_row.radial_profile_data)
        _delta = _i_with - _i_no
        _label = _no_row.sample_id.replace("20260608_", "")
        _ax.plot(_q, _delta, linewidth=0.9, alpha=0.85, label=_label)

    _ax.axhline(0, color="0.2", linewidth=0.8)
    _ax.set_title("Difference: with mask minus without mask")
    _ax.set_xlabel("q-range")
    _ax.set_ylabel("intensity delta")
    _ax.grid(True, alpha=0.25)
    _ax.legend(fontsize=6, loc="best")
    _fig
    return


@app.cell
def _(no_faulty_df, np, plt, with_faulty_df):
    _fig, _ax = plt.subplots(figsize=(11, 4), constrained_layout=True)

    for _no_row, _with_row in zip(
        no_faulty_df.itertuples(index=False),
        with_faulty_df.itertuples(index=False),
    ):
        _q = np.asarray(_no_row.q_range)
        _i_no = np.asarray(_no_row.radial_profile_data_norm)
        _i_with = np.asarray(_with_row.radial_profile_data_norm)
        _delta = _i_with - _i_no
        _label = _no_row.sample_id.replace("20260608_", "")
        _ax.plot(_q, _delta, linewidth=0.9, alpha=0.85, label=_label)

    _ax.axhline(0, color="0.2", linewidth=0.8)
    _ax.set_title("Normalized difference: with mask minus without mask")
    _ax.set_xlabel("q-range")
    _ax.set_ylabel("normalized intensity delta")
    _ax.grid(True, alpha=0.25)
    _ax.legend(fontsize=6, loc="best")
    _fig
    return


@app.cell
def _(np, plt, water_df, with_faulty_df):
    _hot_counts = [len(_mask) for _mask in with_faulty_df["suspected_hot_pixel_mask"]]
    _example_idx = int(np.argmax(_hot_counts))
    _image = water_df["measurement_data"].iloc[_example_idx]
    _sample_id = water_df["sample_id"].iloc[_example_idx]
    _invalid_pixels = with_faulty_df["invalid_pixel_mask"].iloc[_example_idx]
    _hot_pixels = with_faulty_df["suspected_hot_pixel_mask"].iloc[_example_idx]
    _mask = with_faulty_df["pyfai_faulty_pixel_mask"].iloc[_example_idx]
    _invalid_mask = np.zeros_like(_mask, dtype=np.uint8)
    _hot_mask = np.zeros_like(_mask, dtype=np.uint8)
    for _y, _x in _invalid_pixels:
        _invalid_mask[int(_y), int(_x)] = 1
    for _y, _x in _hot_pixels:
        _hot_mask[int(_y), int(_x)] = 1
    _suppressed = np.array(_image, copy=True)
    _suppressed[_mask.astype(bool)] = np.nan

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=5,
        figsize=(15, 4.2),
        constrained_layout=True,
    )
    _vmax = np.nanpercentile(_image, 99.5)

    _im0 = _axes[0].imshow(_image, cmap="inferno", vmin=0, vmax=_vmax)
    _axes[0].set_title("Converted image")
    _axes[0].set_xlabel("column")
    _axes[0].set_ylabel("row")
    _fig.colorbar(_im0, ax=_axes[0], fraction=0.046, pad=0.04)

    _axes[1].imshow(_invalid_mask, cmap="gray_r", vmin=0, vmax=1)
    _axes[1].set_title("Invalid mask")
    _axes[1].set_xlabel("column")
    _axes[1].set_ylabel("row")

    _axes[2].imshow(_hot_mask, cmap="gray_r", vmin=0, vmax=1)
    _axes[2].set_title("Suspected hot mask")
    _axes[2].set_xlabel("column")
    _axes[2].set_ylabel("row")

    _axes[3].imshow(_mask, cmap="gray_r", vmin=0, vmax=1)
    _axes[3].set_title("Combined PyFAI mask")
    _axes[3].set_xlabel("column")
    _axes[3].set_ylabel("row")

    _im2 = _axes[4].imshow(_suppressed, cmap="inferno", vmin=0, vmax=_vmax)
    _axes[4].set_title("Masked pixels hidden")
    _axes[4].set_xlabel("column")
    _axes[4].set_ylabel("row")
    _fig.colorbar(_im2, ax=_axes[4], fraction=0.046, pad=0.04)

    _fig.suptitle(f"Mask decomposition for {_sample_id}")
    _fig
    return


@app.cell
def _(np, plt, with_faulty_df):
    _n_items = len(with_faulty_df)
    _ncols = 4
    _nrows = int(np.ceil(_n_items / _ncols))
    _fig, _axes = plt.subplots(
        _nrows,
        _ncols,
        figsize=(13, 2.8 * _nrows),
        constrained_layout=True,
    )
    _axes = np.asarray(_axes).ravel()

    for _ax, _row in zip(_axes, with_faulty_df.itertuples(index=False)):
        _mask = np.asarray(_row.pyfai_faulty_pixel_mask)
        _hot_count = len(_row.suspected_hot_pixel_mask)
        _invalid_count = len(_row.invalid_pixel_mask)
        _label = _row.sample_id.replace("20260608_", "")
        _ax.imshow(_mask, cmap="gray_r", vmin=0, vmax=1)
        _ax.set_title(f"{_label}\ninvalid={_invalid_count}, hot={_hot_count}", fontsize=8)
        _ax.set_xticks([])
        _ax.set_yticks([])

    for _ax in _axes[_n_items:]:
        _ax.axis("off")

    _fig.suptitle("Combined PyFAI masks for all Water-folder frames")
    _fig
    return


if __name__ == "__main__":
    app.run()
