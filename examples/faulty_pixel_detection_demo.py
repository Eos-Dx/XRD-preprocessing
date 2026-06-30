"""Marimo demo for row-wise faulty pixel detection."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.patches as mpatches
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

    from xrd_preprocessing.faulty_pixels import FaultyPixelDetector, create_mask
    from xrd_preprocessing.gfrm import extract_gfrm_archive, gfrm_to_photons

    return FaultyPixelDetector, Path, create_mask, extract_gfrm_archive, gfrm_to_photons, mo, mpatches, np, pd, plt, repo_root


@app.cell
def _(mo):
    mo.md("""
    # Faulty pixel detection demo

    This notebook loads example Bruker GFRM measurements from the
    local compressed archive.
    Each DataFrame row contains one 2D image and its own PONI metadata.
    We run `FaultyPixelDetector`, suppress the detected pixels, visualize
    the per-row fault map, and compare predictions with known or baseline
    faulty-pixel coordinates.
    """)
    return


@app.cell
def _(Path, extract_gfrm_archive, gfrm_to_photons, np, pd, repo_root):
    real_base = repo_root / "examples" / "data"
    gfrm_archive_path = real_base / "gfrm_measurements.tar.gz"
    extracted_gfrm_root = real_base / "_gfrm_measurements"
    real_poni_path = real_base / "poni_coords_for_all.poni"
    baseline_pixels_path = real_base / "baseline_faulty_pixels.npy"
    baseline_mask_path = real_base / "baseline_faulty_pixel_mask.npy"
    real_paths = [
        real_poni_path,
        baseline_pixels_path,
        baseline_mask_path,
    ]

    if not all(path.exists() for path in real_paths) or not gfrm_archive_path.exists():
        missing = [str(path) for path in real_paths if not path.exists()]
        if not gfrm_archive_path.exists():
            missing.append(str(gfrm_archive_path))
        raise FileNotFoundError("Missing required example data: " + ", ".join(missing))

    extracted_gfrm_root = extract_gfrm_archive(
        gfrm_archive_path,
        extracted_gfrm_root,
    )
    water_gfrm_paths = sorted(
        (extracted_gfrm_root / "GFRM_measurements" / "Water").glob("*/*.gfrm")
    )

    baseline_pixels = np.asarray(np.load(baseline_pixels_path), dtype=int)
    baseline_mask = np.asarray(np.load(baseline_mask_path), dtype=bool)
    baseline_from_mask = set(zip(*np.where(baseline_mask)))
    baseline_from_pixels = {tuple(pixel) for pixel in baseline_pixels.tolist()}
    baseline_faults = baseline_from_pixels | baseline_from_mask
    poni_text = real_poni_path.read_text()

    rows = []
    for _gfrm_path in water_gfrm_paths:
        _image, _metadata = gfrm_to_photons(_gfrm_path)
        _has_baseline = _gfrm_path.name == "20260608_112438_Water_20mm_Main.gfrm"
        rows.append(
            {
                "sample_id": _gfrm_path.parent.name,
                "measurement_data": _image.astype(float),
                "ponifile": poni_text,
                "known_faulty_pixels": baseline_faults if _has_baseline else set(),
                "truth_source": (
                    "baseline_faulty_pixels.npy + mask"
                    if _has_baseline
                    else "no baseline mask"
                ),
                "gfrm_path": str(_metadata["source_path"]),
                "baseline_adu": _metadata["baseline_adu"],
                "gain_adu_per_photon": _metadata["gain_adu_per_photon"],
                "negative_pixel_count": _metadata["negative_pixel_count"],
            }
        )

    df = pd.DataFrame(rows)
    beam_zone_radius_frac = 0.04
    real_data_status = (
        f"loaded {len(df)} example GFRM frames from {gfrm_archive_path}"
    )
    return beam_zone_radius_frac, df, real_data_status


@app.cell
def _(beam_zone_radius_frac, df, mo, real_data_status):
    mo.md(
        f"""
        The generated DataFrame has **{len(df)} rows**. Each row has one image
        and one PONI string.

        Real data status: `{real_data_status}`

        Beam-center radius drawn on the heatmaps:
        `{beam_zone_radius_frac}` of the larger image dimension.
        """
    )
    return


@app.cell
def _(FaultyPixelDetector, beam_zone_radius_frac, df):
    detector = FaultyPixelDetector(
        detect_negative_pixels=True,
        detect_bright_pixels=True,
        bright_pixel_min_value=500.0,
        exclude_beam_center_radius=beam_zone_radius_frac,
    )
    detected_df = detector.fit_transform(df)
    return detected_df, detector


@app.cell
def _(detected_df, np):
    def suppress_faulty_pixels(
        image: np.ndarray,
        faulty_pixels: np.ndarray,
        window: int = 1,
    ) -> np.ndarray:
        """Replace detected faulty pixels with a local finite median."""
        corrected = image.copy()
        global_median = float(np.nanmedian(image[np.isfinite(image)]))

        for _y, _x in faulty_pixels:
            y0 = max(0, int(_y) - window)
            y1 = min(image.shape[0], int(_y) + window + 1)
            x0 = max(0, int(_x) - window)
            x1 = min(image.shape[1], int(_x) + window + 1)
            patch = corrected[y0:y1, x0:x1]
            valid = patch[np.isfinite(patch) & (patch > 1e-8)]
            corrected[int(_y), int(_x)] = (
                float(np.median(valid)) if valid.size else global_median
            )

        return corrected

    corrected_images = [
        suppress_faulty_pixels(_row["measurement_data"], _row["faulty_pixel_mask"])
        for _, _row in detected_df.iterrows()
    ]
    return (corrected_images,)


@app.cell
def _(detected_df, np, pd):
    def score_prediction(row) -> dict[str, float | int | str]:
        """Compare predicted faulty pixels with injected ground truth."""
        predicted = {tuple(pixel) for pixel in row["faulty_pixel_mask"].tolist()}
        truth = set(row["known_faulty_pixels"])
        has_truth = bool(truth)

        true_positive = len(predicted & truth)
        false_positive = len(predicted - truth)
        false_negative = len(truth - predicted)

        precision = true_positive / len(predicted) if has_truth and predicted else np.nan
        recall = true_positive / len(truth) if has_truth else np.nan
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if has_truth and precision + recall > 0
            else np.nan
        )
        return {
            "sample_id": row["sample_id"],
            "truth_source": row["truth_source"],
            "known_faulty": len(truth),
            "predicted_faulty": len(predicted),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    score_table = pd.DataFrame(
        [score_prediction(row) for _, row in detected_df.iterrows()]
    )
    metric_columns = ["precision", "recall", "f1"]
    score_table[metric_columns] = score_table[metric_columns].round(3)
    return (score_table,)


@app.cell
def _(score_table):
    score_table
    return


@app.cell
def _(detected_df, pd):
    _rows = []
    for _, _row in detected_df.iterrows():
        _faulty_count = int(len(_row["faulty_pixel_mask"]))
        _total_pixels = int(_row["measurement_data"].size)
        _rows.append(
            {
                "sample_id": _row["sample_id"],
                "total_pixels": _total_pixels,
                "faulty_pixels": _faulty_count,
                "faulty_fraction": round(_faulty_count / _total_pixels, 6),
            }
        )

    faulty_stats_table = pd.DataFrame(_rows)
    return (faulty_stats_table,)


@app.cell
def _(faulty_stats_table):
    faulty_stats_table
    return


@app.cell
def _(faulty_reason_columns, faulty_stats_table, plt):
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4.2),
        constrained_layout=True,
    )

    _samples = faulty_stats_table["sample_id"].to_numpy()
    _axes[0].bar(_samples, faulty_stats_table["faulty_pixels"].to_numpy())
    _axes[0].set_title("Faulty pixels per image")
    _axes[0].set_ylabel("pixels")
    _axes[0].tick_params(axis="x", rotation=25)

    _bottom = None
    for _reason in faulty_reason_columns:
        _values = faulty_stats_table[_reason].to_numpy()
        _axes[1].bar(_samples, _values, bottom=_bottom, label=_reason)
        _bottom = _values if _bottom is None else _bottom + _values
    _axes[1].set_title("Faulty pixel reason histogram")
    _axes[1].set_ylabel("pixels")
    _axes[1].tick_params(axis="x", rotation=25)
    _axes[1].legend(fontsize=8)

    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    All rows are read from archived `.gfrm` files and converted to photon
    NumPy data inside `xrd_preprocessing.gfrm`. The Water 20mm row is compared
    against the provided baseline faulty-pixel files. Other GFRM rows
    have no baseline mask, so score metrics are left as NaN. NaN, negative
    values, and values above 500
    pixels are treated as faulty. Zero and locally low pixels are not treated as
    faulty by default because they can be valid XRD intensities.
    """)
    return


@app.cell
def _(
    beam_zone_radius_frac,
    corrected_images,
    create_mask,
    detected_df,
    detector,
    mpatches,
    np,
    plt,
):
    def draw_beam_zone(ax, row, image_shape):
        """Draw the PONI beam-center radius on an image axis."""
        center = detector._get_beam_center_pixels(row.get("ponifile"))
        if center is None:
            return

        center_y, center_x = center
        radius = beam_zone_radius_frac * max(image_shape)
        circle = mpatches.Circle(
            (center_x, center_y),
            radius,
            fill=False,
            edgecolor="deepskyblue",
            linewidth=1.2,
            linestyle="--",
            label="beam radius",
        )
        ax.add_patch(circle)

    fig, axes = plt.subplots(
        nrows=len(detected_df),
        ncols=3,
        figsize=(13.5, 4.2 * len(detected_df)),
        constrained_layout=True,
    )
    if len(detected_df) == 1:
        axes = np.asarray([axes])

    for _row_idx, (_, _row) in enumerate(detected_df.iterrows()):
        original = _row["measurement_data"]
        corrected = corrected_images[_row_idx]
        known = np.array(sorted(_row["known_faulty_pixels"]), dtype=int)
        predicted = _row["faulty_pixel_mask"]
        predicted_mask = create_mask(predicted, original.shape)
        vmax = np.nanpercentile(original, 99.5)

        left = axes[_row_idx, 0]
        middle = axes[_row_idx, 1]
        right = axes[_row_idx, 2]

        im_left = left.imshow(original, cmap="inferno", vmin=0, vmax=vmax)
        left.set_title(f"{_row['sample_id']} - original")
        left.set_xlabel("column")
        left.set_ylabel("row")
        if known.size:
            left.scatter(
                known[:, 1],
                known[:, 0],
                marker="s",
                s=16 if len(known) < 200 else 3,
                facecolors="none",
                edgecolors="cyan",
                linewidths=0.8,
                label="known faults",
            )
        draw_beam_zone(left, _row, original.shape)
        left.legend(loc="upper right", fontsize=8)

        middle.imshow(predicted_mask, cmap="gray_r", vmin=0, vmax=1)
        middle.set_title(f"{_row['sample_id']} - detected mask")
        middle.set_xlabel("column")
        middle.set_ylabel("row")
        draw_beam_zone(middle, _row, original.shape)

        im_right = right.imshow(corrected, cmap="inferno", vmin=0, vmax=vmax)
        right.set_title(f"{_row['sample_id']} - detected pixels suppressed")
        right.set_xlabel("column")
        right.set_ylabel("row")
        if predicted.size:
            right.scatter(
                predicted[:, 1],
                predicted[:, 0],
                marker="x",
                s=20,
                c="lime",
                linewidths=0.8,
                label="detected faults",
            )
        draw_beam_zone(right, _row, original.shape)
        right.legend(loc="upper right", fontsize=8)

        fig.colorbar(im_left, ax=left, fraction=0.046, pad=0.04)
        fig.colorbar(im_right, ax=right, fraction=0.046, pad=0.04)

    fig
    return


@app.cell
def _(detector, mo):
    mo.md(
        f"""
        Detector stats:

        ```python
        {detector.stats_}
        ```

        The left heatmaps show the original detector images with known or
        baseline faulty pixels outlined in cyan. The middle panels show the
        one detector mask. The right heatmaps show the same images after
        replacing detected faulty pixels by a local median estimate.

        The dashed blue circle marks the PONI beam-center radius zone. It is
        also excluded from detection by `exclude_beam_center_radius`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
