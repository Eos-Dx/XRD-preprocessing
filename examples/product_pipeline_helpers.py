from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xrd_preprocessing import extract_gfrm_archive, gfrm_to_photons


def infer_sample_thickness_mm(sample_id: str, reference_mm: float) -> float:
    match = re.search(r"Water_(\d+)mm", sample_id)
    return float(match.group(1)) if match is not None else float(reference_mm)


def load_example_gfrm_dataframe(
    *,
    repo_root: Path,
    thickness_reference_mm: float,
) -> pd.DataFrame:
    data_dir = repo_root / "examples" / "data"
    archive_path = data_dir / "gfrm_measurements.tar.gz"
    extracted_root = extract_gfrm_archive(archive_path, data_dir / "_gfrm_measurements")
    gfrm_paths = sorted((extracted_root / "GFRM_measurements" / "Water").glob("*/*.gfrm"))
    poni_text = (data_dir / "poni_coords_for_all.poni").read_text()

    rows = []
    for path in gfrm_paths:
        image, metadata = gfrm_to_photons(path)
        sample_id = path.parent.name
        rows.append(
            {
                "sample_id": sample_id,
                "gfrm_path": str(path),
                "measurement_data": image,
                "ponifile": poni_text,
                "sample_thickness_mm": infer_sample_thickness_mm(
                    sample_id,
                    thickness_reference_mm,
                ),
                "interpolation_q_range": (2.0, 23.0),
                "azimuthal_range": None,
                "baseline_adu": metadata["baseline_adu"],
                "gain_adu_per_photon": metadata["gain_adu_per_photon"],
                "masked_row_511": metadata["masked_row_511"],
            }
        )
    return pd.DataFrame(rows)


def plot_profiles(
    df: pd.DataFrame,
    *,
    title: str,
    y_column: str = "radial_profile_data",
    show_snr: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for row in df.itertuples(index=False):
        label = str(row.sample_id)
        if show_snr and "snr_db" in df.columns:
            label = f"{label} | SNR {float(row.snr_db):.1f} dB"
        ax.plot(row.q_range, getattr(row, y_column), linewidth=0.9, label=label)
    ax.set_title(title)
    ax.set_xlabel("q, nm^-1")
    ax.set_ylabel(y_column)
    ax.grid(True, alpha=0.25)
    if len(df) <= 12:
        ax.legend(fontsize=7, loc="best")
    return fig


def plot_faulty_counts(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    labels = [str(value).replace("20260608_", "") for value in df["sample_id"]]
    faulty = [len(value) for value in df["faulty_pixel_mask"]]
    x = np.arange(len(labels))
    ax.bar(x, faulty, width=0.45, label="faulty")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("pixels")
    ax.set_title("Faulty-pixel counts")
    ax.legend(fontsize=8)
    return fig


def plot_snr(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    labels = [str(value).replace("20260608_", "") for value in df["sample_id"]]
    x = np.arange(len(labels))
    values = df["snr_db"].astype(float).to_numpy()
    ax.bar(x, values)
    for xpos, value in zip(x, values):
        ax.text(xpos, value, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(20.0, color="0.2", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("SNR, dB")
    ax.set_title("Poisson SNR before filtering")
    return fig
