from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._compat import BaseEstimator, TransformerMixin


class ColumnValueFilter(TransformerMixin, BaseEstimator):
    """Reusable row filter based on one DataFrame column."""

    def __init__(
        self,
        column: str,
        *,
        op: str = "in",
        value: Any = None,
        values: Sequence[Any] | None = None,
        lower: Any = None,
        upper: Any = None,
        keep_na: bool = False,
        reset_index: bool = True,
    ) -> None:
        self.column = column
        self.op = str(op)
        self.value = value
        self.values = list(values) if values is not None else None
        self.lower = lower
        self.upper = upper
        self.keep_na = bool(keep_na)
        self.reset_index = bool(reset_index)
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def _build_mask(self, series: pd.Series) -> pd.Series:
        op = self.op.lower()
        if op == "in":
            if self.values is None:
                raise ValueError("values must be provided for op='in'.")
            return series.isin(self.values)
        if op == "not_in":
            if self.values is None:
                raise ValueError("values must be provided for op='not_in'.")
            return ~series.isin(self.values)
        if op in {"==", "eq"}:
            return series.eq(self.value)
        if op in {"!=", "ne"}:
            return series.ne(self.value)
        if op in {">", ">=", "<", "<="}:
            numeric = pd.to_numeric(series, errors="coerce")
            threshold = float(self.value)
            if op == ">":
                return numeric.gt(threshold)
            if op == ">=":
                return numeric.ge(threshold)
            if op == "<":
                return numeric.lt(threshold)
            return numeric.le(threshold)
        if op == "between":
            if self.lower is None or self.upper is None:
                raise ValueError("lower and upper must be provided for op='between'.")
            numeric = pd.to_numeric(series, errors="coerce")
            return numeric.between(float(self.lower), float(self.upper), inclusive="both")
        if op == "contains":
            return series.fillna("").astype(str).str.contains(str(self.value), regex=True, na=False)
        if op == "isna":
            return series.isna()
        if op == "notna":
            return series.notna()
        raise ValueError(f"Unsupported column filter op: {self.op}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.column not in X.columns:
            raise KeyError(f"Column '{self.column}' not found in DataFrame.")
        out = X.copy()
        mask = self._build_mask(out[self.column])
        if self.keep_na:
            mask = mask | out[self.column].isna()
        filtered = out.loc[mask].copy()
        if self.reset_index:
            filtered.reset_index(drop=True, inplace=True)
        self.stats_ = {
            "filter_type": "column_value",
            "filter_column": self.column,
            "filter_op": self.op,
            "filter_value": self.value,
            "filter_values": self.values,
            "filter_lower": self.lower,
            "filter_upper": self.upper,
            "rows_in": int(len(out)),
            "rows_pass": int(np.sum(mask)),
            "rows_fail": int(len(out) - np.sum(mask)),
        }
        return filtered


class MetadataFilter(ColumnValueFilter):
    """ColumnValueFilter alias for metadata columns."""

    def __init__(
        self,
        column: str,
        *,
        op: str = "in",
        value: Any = None,
        values: Sequence[Any] | None = None,
        lower: Any = None,
        upper: Any = None,
        keep_na: bool = False,
        reset_index: bool = True,
    ) -> None:
        super().__init__(
            column,
            op=op,
            value=value,
            values=values,
            lower=lower,
            upper=upper,
            keep_na=keep_na,
            reset_index=reset_index,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = super().transform(X)
        self.stats_["filter_type"] = "metadata"
        return out


class PatientFilter(MetadataFilter):
    """MetadataFilter alias for patient/sample selection stages."""


class SNRFilter(ColumnValueFilter):
    """ColumnValueFilter alias for scalar SNR in dB."""

    def __init__(
        self,
        snr_column: str = "snr_db",
        min_snr_db: float = 20.0,
        pass_column: str = "snr_pass",
        drop: bool = True,
        reset_index: bool = False,
    ) -> None:
        super().__init__(
            snr_column,
            op=">=",
            value=float(min_snr_db),
            keep_na=False,
            reset_index=reset_index,
        )
        self.snr_column = snr_column
        self.min_snr_db = float(min_snr_db)
        self.pass_column = pass_column
        self.drop = bool(drop)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.snr_column not in X.columns:
            raise KeyError(f"Column '{self.snr_column}' not found in DataFrame.")
        out = X.copy()
        snr = pd.to_numeric(out[self.snr_column], errors="coerce")
        passed = np.isfinite(snr) & (snr >= self.min_snr_db)
        out[self.pass_column] = passed.to_numpy(dtype=bool)
        out["snr_min_db"] = self.min_snr_db
        finite = snr[np.isfinite(snr)]
        failed_ids = (
            out.loc[~out[self.pass_column], "sample_id"].astype(str).tolist()
            if "sample_id" in out.columns
            else []
        )
        self.stats_ = {
            "filter_type": "snr",
            "filter_column": self.snr_column,
            "filter_op": ">=",
            "filter_value": self.min_snr_db,
            "rows_in": int(len(out)),
            "rows_pass": int(np.sum(passed)),
            "rows_fail": int(len(out) - np.sum(passed)),
            "min_snr_db": float(np.nanmin(finite)) if len(finite) else np.nan,
            "max_snr_db": float(np.nanmax(finite)) if len(finite) else np.nan,
            "failed_ids": failed_ids,
        }
        if self.drop:
            out = out.loc[out[self.pass_column]].copy()
            if self.reset_index:
                out.reset_index(drop=True, inplace=True)
        return out
