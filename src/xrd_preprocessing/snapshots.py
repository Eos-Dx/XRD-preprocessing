from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._compat import BaseEstimator, TransformerMixin


@dataclass
class RadialProfileSnapshot(TransformerMixin, BaseEstimator):
    """Optionally save q/profile arrays at a named pipeline stage."""

    stage: str
    enabled: bool = True
    q_column: str = "q_range"
    profile_column: str = "radial_profile_data"

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if not self.enabled:
            return out
        missing = [
            column
            for column in (self.q_column, self.profile_column)
            if column not in out.columns
        ]
        if missing:
            raise ValueError(f"Missing snapshot column(s): {', '.join(missing)}")

        q_output = f"{self.q_column}_{self.stage}"
        profile_output = f"{self.profile_column}_{self.stage}"
        out[q_output] = [np.asarray(value).copy() for value in out[self.q_column]]
        out[profile_output] = [
            np.asarray(value).copy() for value in out[self.profile_column]
        ]
        return out
