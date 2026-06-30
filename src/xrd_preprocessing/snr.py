"""Signal-to-noise ratio calculations and transformers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._compat import BaseEstimator, TransformerMixin


def _trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate ``y`` over ``x`` across NumPy versions."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2 or x.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _calculate_poisson_snr(
    intensity: np.ndarray,
    sigma: np.ndarray | None,
    *,
    eps: float = 1e-12,
) -> dict[str, float | str]:
    """Calculate profile SNR from pyFAI Poisson sigma."""
    if sigma is None:
        return {
            "noise_std": np.nan,
            "snr_linear": np.nan,
            "snr_db": np.nan,
            "method": "poisson_missing_sigma",
        }

    intensity = np.asarray(intensity, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.asarray([float(sigma)], dtype=float)

    n = min(len(intensity), len(sigma))
    if n < 2:
        return {
            "noise_std": np.nan,
            "snr_linear": np.nan,
            "snr_db": np.nan,
            "method": "poisson_invalid_sigma",
        }

    intensity = intensity[:n]
    sigma = sigma[:n]
    valid = np.isfinite(intensity) & np.isfinite(sigma) & (sigma > 0)
    if int(np.sum(valid)) < 2:
        return {
            "noise_std": np.nan,
            "snr_linear": np.nan,
            "snr_db": np.nan,
            "method": "poisson_invalid_sigma",
        }

    # Product SNR uses pyFAI Poisson sigma from azimuthal integration.
    # snr_q is an amplitude ratio, so decibels use 20 * log10(...).
    snr_q = np.abs(intensity[valid]) / (sigma[valid] + eps)
    snr_linear = float(np.sqrt(np.mean(np.square(snr_q))))
    snr_db = float(20.0 * np.log10(snr_linear + eps))
    noise_std = float(np.sqrt(np.mean(np.square(sigma[valid]))))
    return {
        "noise_std": noise_std,
        "snr_linear": snr_linear,
        "snr_db": snr_db,
        "method": "poisson",
    }


def calculate_snr(
    q: np.ndarray,
    intensity: np.ndarray,
    *,
    sigma: np.ndarray | None = None,
    method: str = "poisson",
) -> dict[str, Any]:
    """Calculate Poisson SNR for one integrated XRD profile."""
    _ = q
    out = SNRTransformer(snr_method=method).transform(
        pd.DataFrame(
            {
                "radial_profile_data": [intensity],
                "radial_profile_sigma": [sigma],
            }
        )
    )
    row = out.iloc[0]
    return {
        "snr_db": row["snr_db"],
        "snr_linear": row["snr_linear"],
        "noise_std": row["noise_std"],
        "method": row["snr_method_used"],
    }


class SNRTransformer(TransformerMixin, BaseEstimator):
    """Add Poisson signal-to-noise metrics to integrated XRD profiles."""

    def __init__(
        self,
        y_column: str = "radial_profile_data",
        sigma_column: str = "radial_profile_sigma",
        snr_method: str = "poisson",
        method: str | None = None,
    ) -> None:
        """Store transformer configuration."""
        self.y_column = y_column
        self.sigma_column = sigma_column
        self.snr_method = str(method or snr_method).strip().lower()
        if self.snr_method != "poisson":
            raise ValueError("snr_method must be 'poisson'.")

    def fit(self, X: pd.DataFrame, y=None):
        """No-op fit method required by the scikit-learn pipeline API."""
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with Poisson SNR columns."""
        df = X.copy()
        df["noise_std"] = np.nan
        df["snr_linear"] = np.nan
        df["snr_db"] = np.nan
        df["snr_method_used"] = None

        for i, row in df.iterrows():
            intensity = np.asarray(row.get(self.y_column), dtype=float)
            if intensity.ndim == 0 or len(intensity) < 2:
                continue

            result = _calculate_poisson_snr(
                intensity,
                row.get(self.sigma_column, None),
            )
            df.at[i, "noise_std"] = result["noise_std"]
            df.at[i, "snr_linear"] = result["snr_linear"]
            df.at[i, "snr_db"] = result["snr_db"]
            df.at[i, "snr_method_used"] = result["method"]

        return df
