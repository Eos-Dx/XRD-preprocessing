"""Marimo demo explaining Bruker GFRM raw ADU and baseline subtraction."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    repo_root = (
        Path(__file__).resolve().parents[1]
        if "__file__" in globals()
        else Path.cwd()
    )
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from xrd_preprocessing.gfrm import (
        extract_gfrm_archive,
        gfrm_conversion_metadata,
        read_gfrm_adu,
    )

    return (
        extract_gfrm_archive,
        gfrm_conversion_metadata,
        mo,
        np,
        pd,
        plt,
        read_gfrm_adu,
        repo_root,
    )


@app.cell
def _(mo):
    mo.md("""
    # GFRM converter demo

    This notebook loads all Water GFRM scans from the local archive.

    Displayed stages:

    ```text
    raw_counts = ADU
    baseline_subtracted_counts = ADU - baseline_adu
    photons = (ADU - baseline_adu) / gain_adu_per_photon
    ```

    The requested plots stop before photon division.
    """)
    return


@app.cell
def _(
    extract_gfrm_archive,
    gfrm_conversion_metadata,
    pd,
    read_gfrm_adu,
    repo_root,
):
    data_dir = repo_root / "examples" / "data"
    archive_path = data_dir / "gfrm_measurements.tar.gz"
    extracted_root = extract_gfrm_archive(archive_path, data_dir / "_gfrm_measurements")
    water_paths = sorted(
        (extracted_root / "GFRM_measurements" / "Water").glob("*/*.gfrm")
    )

    rows = []
    for _path in water_paths:
        _adu = read_gfrm_adu(_path)
        _metadata = gfrm_conversion_metadata(_path)
        _baseline = float(_metadata["baseline_adu"])
        rows.append(
            {
                "sample_id": _path.parent.name,
                "gfrm_path": str(_path),
                "baseline_adu": _baseline,
                "gain_adu_per_photon": float(_metadata["gain_adu_per_photon"]),
                "raw_adu": _adu,
                "adu_minus_baseline": _adu.astype(float) - _baseline,
            }
        )

    scans_df = pd.DataFrame(rows)
    return archive_path, scans_df


@app.cell
def _(archive_path, mo, scans_df):
    _baseline_values = sorted(scans_df["baseline_adu"].unique().tolist())
    mo.md(
        f"""
        Loaded `{len(scans_df)}` Water GFRM scans from:

        `{archive_path}`

        Baseline ADU values found in headers:

        ```text
        {_baseline_values}
        ```
        """
    )
    return


@app.cell
def _(np, scans_df):
    summary_table = scans_df.assign(
        raw_min=lambda _df: _df["raw_adu"].map(np.min),
        raw_median=lambda _df: _df["raw_adu"].map(np.median),
        raw_max=lambda _df: _df["raw_adu"].map(np.max),
        minus64_min=lambda _df: _df["adu_minus_baseline"].map(np.min),
        minus64_median=lambda _df: _df["adu_minus_baseline"].map(np.median),
        minus64_max=lambda _df: _df["adu_minus_baseline"].map(np.max),
        minus64_negative=lambda _df: _df["adu_minus_baseline"].map(
            lambda _image: int(np.sum(_image < 0))
        ),
    )[
        [
            "sample_id",
            "baseline_adu",
            "gain_adu_per_photon",
            "raw_min",
            "raw_median",
            "raw_max",
            "minus64_min",
            "minus64_median",
            "minus64_max",
            "minus64_negative",
        ]
    ]
    return (summary_table,)


@app.cell
def _(summary_table):
    summary_table
    return


@app.cell
def _(mo, np, scans_df):
    _raw_values = np.concatenate(scans_df["raw_adu"].map(np.ravel))
    _subtracted_values = np.concatenate(scans_df["adu_minus_baseline"].map(np.ravel))
    _raw_default_max = float(np.percentile(_raw_values, 99.9))
    _sub_default_min = float(np.percentile(_subtracted_values, 0.1))
    _sub_default_max = float(np.percentile(_subtracted_values, 99.9))
    _raw_counts, _ = np.histogram(_raw_values, bins=np.linspace(0, _raw_default_max, 180))
    _sub_counts, _ = np.histogram(
        _subtracted_values,
        bins=np.linspace(_sub_default_min, _sub_default_max, 180),
    )

    raw_x_min_input = mo.ui.number(label="Raw x min", value=0.0, step=10.0)
    raw_x_max_input = mo.ui.number(
        label="Raw x max",
        value=round(_raw_default_max, 1),
        step=10.0,
    )
    raw_y_min_input = mo.ui.number(label="Raw y min", value=1.0, step=1.0)
    raw_y_max_input = mo.ui.number(
        label="Raw y max",
        value=float(max(_raw_counts.max(), 1)),
        step=100.0,
    )
    subtracted_x_min_input = mo.ui.number(
        label="ADU-64 x min",
        value=round(_sub_default_min, 1),
        step=10.0,
    )
    subtracted_x_max_input = mo.ui.number(
        label="ADU-64 x max",
        value=round(_sub_default_max, 1),
        step=10.0,
    )
    subtracted_y_min_input = mo.ui.number(label="ADU-64 y min", value=1.0, step=1.0)
    subtracted_y_max_input = mo.ui.number(
        label="ADU-64 y max",
        value=float(max(_sub_counts.max(), 1)),
        step=100.0,
    )

    mo.vstack(
        [
            mo.md("## Histogram axes"),
            mo.hstack(
                [raw_x_min_input, raw_x_max_input, raw_y_min_input, raw_y_max_input]
            ),
            mo.hstack(
                [
                    subtracted_x_min_input,
                    subtracted_x_max_input,
                    subtracted_y_min_input,
                    subtracted_y_max_input,
                ]
            ),
        ]
    )
    return (
        raw_x_max_input,
        raw_x_min_input,
        raw_y_max_input,
        raw_y_min_input,
        subtracted_x_max_input,
        subtracted_x_min_input,
        subtracted_y_max_input,
        subtracted_y_min_input,
    )


@app.cell
def _(
    raw_x_max_input,
    raw_x_min_input,
    raw_y_max_input,
    raw_y_min_input,
    subtracted_x_max_input,
    subtracted_x_min_input,
    subtracted_y_max_input,
    subtracted_y_min_input,
):
    raw_x_min = float(raw_x_min_input.value)
    raw_x_max = float(raw_x_max_input.value)
    raw_y_min = max(float(raw_y_min_input.value), 1e-9)
    raw_y_max = max(float(raw_y_max_input.value), raw_y_min * 10)
    subtracted_x_min = float(subtracted_x_min_input.value)
    subtracted_x_max = float(subtracted_x_max_input.value)
    subtracted_y_min = max(float(subtracted_y_min_input.value), 1e-9)
    subtracted_y_max = max(float(subtracted_y_max_input.value), subtracted_y_min * 10)
    return (
        raw_x_max,
        raw_x_min,
        raw_y_max,
        raw_y_min,
        subtracted_x_max,
        subtracted_x_min,
        subtracted_y_max,
        subtracted_y_min,
    )


@app.cell
def _(np, plt, scans_df):
    _n_scans = len(scans_df)
    _ncols = 4
    _nrows = int(np.ceil(_n_scans / _ncols))
    _fig, _axes = plt.subplots(
        _nrows,
        _ncols,
        figsize=(13, 2.8 * _nrows),
        constrained_layout=True,
    )
    _axes = np.asarray(_axes).ravel()
    _vmax = np.percentile(np.concatenate(scans_df["raw_adu"].map(np.ravel)), 99.5)

    for _ax, _row in zip(_axes, scans_df.itertuples(index=False)):
        _im = _ax.imshow(_row.raw_adu, cmap="inferno", vmin=0, vmax=_vmax)
        _ax.set_title(_row.sample_id.replace("20260608_", ""), fontsize=8)
        _ax.set_xticks([])
        _ax.set_yticks([])

    for _ax in _axes[_n_scans:]:
        _ax.axis("off")

    _fig.colorbar(_im, ax=_axes[:_n_scans], fraction=0.02, pad=0.01)
    _fig.suptitle("Raw GFRM ADU images, no baseline subtraction")
    _fig
    return


@app.cell
def _(
    np,
    plt,
    raw_x_max,
    raw_x_min,
    raw_y_max,
    raw_y_min,
    scans_df,
    subtracted_x_max,
    subtracted_x_min,
    subtracted_y_max,
    subtracted_y_min,
):
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 5),
        constrained_layout=True,
    )

    _raw_bins = np.linspace(raw_x_min, raw_x_max, 180)
    _sub_bins = np.linspace(subtracted_x_min, subtracted_x_max, 180)

    for _row in scans_df.itertuples(index=False):
        _label = _row.sample_id.replace("20260608_", "").replace("_Main", "")
        _axes[0].hist(
            _row.raw_adu.ravel(),
            bins=_raw_bins,
            histtype="step",
            linewidth=1,
            alpha=0.8,
            label=_label,
        )
        _axes[1].hist(
            _row.adu_minus_baseline.ravel(),
            bins=_sub_bins,
            histtype="step",
            linewidth=1,
            alpha=0.8,
            label=_label,
        )

    _axes[0].set_title("Raw ADU counts, all Water scans")
    _axes[0].set_xlabel("ADU counts")
    _axes[0].set_ylabel("pixels")
    _axes[0].set_yscale("log")
    _axes[0].set_xlim(raw_x_min, raw_x_max)
    _axes[0].set_ylim(raw_y_min, raw_y_max)

    _axes[1].axvline(0, color="red", linewidth=1)
    _axes[1].set_title("ADU - baseline, no photon division")
    _axes[1].set_xlabel("ADU - baseline_adu")
    _axes[1].set_ylabel("pixels")
    _axes[1].set_yscale("log")
    _axes[1].set_xlim(subtracted_x_min, subtracted_x_max)
    _axes[1].set_ylim(subtracted_y_min, subtracted_y_max)

    _axes[1].legend(fontsize=6, ncols=1, loc="upper right")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    Important behavior:

    - raw GFRM counts are shown before subtracting `64`;
    - baseline-subtracted counts are shown as `ADU - baseline_adu`;
    - values below zero are preserved;
    - photon division is not applied in these histograms.
    """)
    return


if __name__ == "__main__":
    app.run()
