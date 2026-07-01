"""Q-range normalization utilities for radial profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._compat import BaseEstimator, TransformerMixin
from .numeric import trapz_compat


def _validate_q_profile(
    q: np.ndarray,
    intensity: np.ndarray,
    *,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    n = min(len(q), len(intensity))
    if n < min_points:
        word = "one" if min_points == 1 else "two" if min_points == 2 else str(min_points)
        noun = "point" if min_points == 1 else "points"
        raise ValueError(
            f"q and intensity must contain at least {word} {noun}."
        )
    q_valid = q[:n]
    intensity_valid = intensity[:n]
    finite = np.isfinite(q_valid) & np.isfinite(intensity_valid)
    return q_valid[finite], intensity_valid[finite]


def _q_window_values(
    q: np.ndarray,
    intensity: np.ndarray,
    *,
    q_min: float,
    q_max: float,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    q_valid, intensity_valid = _validate_q_profile(
        q,
        intensity,
        min_points=min_points,
    )

    q_lo = min(float(q_min), float(q_max))
    q_hi = max(float(q_min), float(q_max))
    band = (q_valid >= q_lo) & (q_valid <= q_hi)
    if int(np.sum(band)) < min_points:
        raise ValueError(
            f"Normalization q range [{q_lo}, {q_hi}] has <{min_points} points."
        )
    order = np.argsort(q_valid[band])
    return q_valid[band][order], intensity_valid[band][order]


def _check_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def _save_initial_profile(
    frame: pd.DataFrame,
    *,
    intensity_column: str,
    initial_column: str,
) -> None:
    frame[initial_column] = [
        np.asarray(value).copy() for value in frame[intensity_column]
    ]


def normalize_profile_by_q_range(
    q: np.ndarray,
    intensity: np.ndarray,
    *,
    q_min: float = 6.7,
    q_max: float = 7.1,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Normalize one 1D profile by its area inside a q range."""
    q_band, intensity_band = _q_window_values(
        q,
        intensity,
        q_min=q_min,
        q_max=q_max,
        min_points=2,
    )
    area = trapz_compat(intensity_band, q_band)
    if not np.isfinite(area) or abs(area) <= eps:
        raise ValueError(f"Invalid normalization area: {area!r}.")

    return np.asarray(intensity, dtype=float) / area, float(area)


def normalize_profile_by_q_range_value(
    q: np.ndarray,
    intensity: np.ndarray,
    *,
    q_min: float = 6.7,
    q_max: float = 7.1,
    statistic: str = "median",
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Normalize one 1D profile by a value statistic inside a q range."""
    _q_band, values = _q_window_values(
        q,
        intensity,
        q_min=q_min,
        q_max=q_max,
        min_points=1,
    )
    statistic_name = str(statistic).lower()
    if statistic_name == "median":
        scale = float(np.median(values))
    elif statistic_name == "mean":
        scale = float(np.mean(values))
    elif statistic_name == "min":
        scale = float(np.min(values))
    elif statistic_name == "max":
        scale = float(np.max(values))
    else:
        raise ValueError(f"Unsupported q-range value statistic: {statistic!r}.")

    if not np.isfinite(scale) or abs(scale) <= eps:
        raise ValueError(f"Invalid normalization value: {scale!r}.")

    return np.asarray(intensity, dtype=float) / scale, scale


@dataclass
class QRangeNormalizer(TransformerMixin, BaseEstimator):
    """Normalize 1D XRD profiles by integrated intensity in a q range."""

    q_column: str = "q_range"
    intensity_column: str = "radial_profile_data"
    output_column: str | None = None
    save_initial_data: bool = False
    initial_column: str = "radial_profile_data_raw"
    scale_column: str = "q_range_normalization_area"
    q_min: float = 6.7
    q_max: float = 7.1
    add_metadata_columns: bool = False

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_columns(X, (self.q_column, self.intensity_column))

        out = X.copy()
        normalized = []
        scales = []
        target_column = self.output_column or self.intensity_column

        if self.save_initial_data:
            _save_initial_profile(
                out,
                intensity_column=self.intensity_column,
                initial_column=self.initial_column,
            )

        for _, row in out.iterrows():
            profile, area = normalize_profile_by_q_range(
                row[self.q_column],
                row[self.intensity_column],
                q_min=self.q_min,
                q_max=self.q_max,
            )
            normalized.append(profile)
            scales.append(area)

        out[target_column] = normalized
        if self.add_metadata_columns:
            out[self.scale_column] = scales
            out["q_range_normalization_min"] = float(min(self.q_min, self.q_max))
            out["q_range_normalization_max"] = float(max(self.q_min, self.q_max))
        return out


@dataclass
class QRangeValueNormalizer(TransformerMixin, BaseEstimator):
    """Normalize 1D XRD profiles by a value in a q range."""

    q_column: str = "q_range"
    intensity_column: str = "radial_profile_data"
    output_column: str | None = None
    save_initial_data: bool = False
    initial_column: str = "radial_profile_data_raw"
    scale_column: str = "q_range_normalization_value"
    statistic_column: str = "q_range_normalization_statistic"
    q_min: float = 6.7
    q_max: float = 7.1
    statistic: str = "median"
    add_metadata_columns: bool = False

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_columns(X, (self.q_column, self.intensity_column))

        out = X.copy()
        normalized = []
        scales = []
        target_column = self.output_column or self.intensity_column

        if self.save_initial_data:
            _save_initial_profile(
                out,
                intensity_column=self.intensity_column,
                initial_column=self.initial_column,
            )

        for _, row in out.iterrows():
            profile, scale = normalize_profile_by_q_range_value(
                row[self.q_column],
                row[self.intensity_column],
                q_min=self.q_min,
                q_max=self.q_max,
                statistic=self.statistic,
            )
            normalized.append(profile)
            scales.append(scale)

        out[target_column] = normalized
        if self.add_metadata_columns:
            out[self.scale_column] = scales
            out[self.statistic_column] = str(self.statistic).lower()
            out["q_range_normalization_min"] = float(min(self.q_min, self.q_max))
            out["q_range_normalization_max"] = float(max(self.q_min, self.q_max))
        return out
