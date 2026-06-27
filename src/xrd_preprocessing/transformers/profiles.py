"""Profile transformers used by lightweight/synthetic pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._compat import BaseEstimator, TransformerMixin


class SimpleRadialProfileTransformer(TransformerMixin, BaseEstimator):
    """Create simple radial profiles from 2D arrays without PONI geometry.

    This is useful for synthetic pipeline tests. Product data with real geometry
    should use ``AzimuthalIntegration``.
    """

    def __init__(
        self,
        *,
        npt: int = 100,
        q_min: float = 2.0,
        q_max: float = 23.0,
        data_column: str = "measurement_data",
        q_column: str = "q_range",
        profile_column: str = "radial_profile_data",
        sigma_column: str = "radial_profile_sigma",
        thickness_adjustment_applied: bool = True,
        sample_thickness_column: str = "sample_thickness_mm",
        thickness_reference_column: str = "calibrant_thickness_mm",
    ) -> None:
        self.npt = int(npt)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.data_column = data_column
        self.q_column = q_column
        self.profile_column = profile_column
        self.sigma_column = sigma_column
        self.thickness_adjustment_applied = bool(thickness_adjustment_applied)
        self.sample_thickness_column = sample_thickness_column
        self.thickness_reference_column = thickness_reference_column
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None):
        _ = X
        _ = y
        return self

    def _radial_profile(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yy, xx = np.indices(image.shape)
        center_y = (image.shape[0] - 1) / 2.0
        center_x = (image.shape[1] - 1) / 2.0
        radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        bins = np.linspace(0, float(radius.max()) + 1e-9, self.npt + 1)
        index = np.clip(np.digitize(radius.ravel(), bins) - 1, 0, self.npt - 1)
        sums = np.bincount(index, weights=image.ravel(), minlength=self.npt)
        counts = np.bincount(index, minlength=self.npt)
        profile = sums / np.maximum(counts, 1)
        q = np.linspace(self.q_min, self.q_max, self.npt)
        sigma = np.sqrt(np.maximum(np.abs(profile), 1.0))
        return q, profile, sigma

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.data_column not in X.columns:
            raise KeyError(f"Missing data column: {self.data_column}")
        out = X.copy()
        q_values = []
        profile_values = []
        sigma_values = []
        for image in out[self.data_column]:
            q, profile, sigma = self._radial_profile(np.asarray(image, dtype=float))
            q_values.append(q)
            profile_values.append(profile)
            sigma_values.append(sigma)
        out[self.q_column] = q_values
        out[self.profile_column] = profile_values
        out[self.sigma_column] = sigma_values
        out["thickness_correction_applied"] = self.thickness_adjustment_applied
        out["thickness_sample_column"] = self.sample_thickness_column
        out["thickness_reference_column"] = self.thickness_reference_column
        self.stats_ = {
            "filter_type": "simple_radial_profile",
            "rows": int(len(out)),
            "npt": self.npt,
            "q_min": self.q_min,
            "q_max": self.q_max,
        }
        return out
