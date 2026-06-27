"""Q-range normalization utilities for radial profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._compat import BaseEstimator, TransformerMixin
from .snr import _trapz_compat


def normalize_profile_by_q_range(
    q: np.ndarray,
    intensity: np.ndarray,
    *,
    q_min: float = 6.7,
    q_max: float = 7.1,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Normalize one 1D profile by its area inside a q range."""
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    n = min(len(q), len(intensity))
    if n < 2:
        raise ValueError("q and intensity must contain at least two points.")

    q_valid = q[:n]
    intensity_valid = intensity[:n]
    finite = np.isfinite(q_valid) & np.isfinite(intensity_valid)
    q_valid = q_valid[finite]
    intensity_valid = intensity_valid[finite]

    q_lo = min(float(q_min), float(q_max))
    q_hi = max(float(q_min), float(q_max))
    band = (q_valid >= q_lo) & (q_valid <= q_hi)
    if int(np.sum(band)) < 2:
        raise ValueError(f"Normalization q range [{q_lo}, {q_hi}] has <2 points.")

    order = np.argsort(q_valid[band])
    area = _trapz_compat(intensity_valid[band][order], q_valid[band][order])
    if not np.isfinite(area) or abs(area) <= eps:
        raise ValueError(f"Invalid normalization area: {area!r}.")

    return intensity / area, float(area)


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
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    n = min(len(q), len(intensity))
    if n < 1:
        raise ValueError("q and intensity must contain at least one point.")

    q_valid = q[:n]
    intensity_valid = intensity[:n]
    finite = np.isfinite(q_valid) & np.isfinite(intensity_valid)
    q_valid = q_valid[finite]
    intensity_valid = intensity_valid[finite]

    q_lo = min(float(q_min), float(q_max))
    q_hi = max(float(q_min), float(q_max))
    band = (q_valid >= q_lo) & (q_valid <= q_hi)
    if int(np.sum(band)) < 1:
        raise ValueError(f"Normalization q range [{q_lo}, {q_hi}] has <1 point.")

    values = intensity_valid[band]
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

    return intensity / scale, scale


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

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in (self.q_column, self.intensity_column) if column not in X.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        out = X.copy()
        normalized = []
        scales = []
        target_column = self.output_column or self.intensity_column

        if self.save_initial_data:
            out[self.initial_column] = [np.asarray(value).copy() for value in out[self.intensity_column]]

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

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in (self.q_column, self.intensity_column) if column not in X.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        out = X.copy()
        normalized = []
        scales = []
        target_column = self.output_column or self.intensity_column

        if self.save_initial_data:
            out[self.initial_column] = [np.asarray(value).copy() for value in out[self.intensity_column]]

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
        out[self.scale_column] = scales
        out[self.statistic_column] = str(self.statistic).lower()
        out["q_range_normalization_min"] = float(min(self.q_min, self.q_max))
        out["q_range_normalization_max"] = float(max(self.q_min, self.q_max))
        return out
