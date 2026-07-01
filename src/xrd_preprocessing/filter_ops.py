"""Shared scalar and Series filter operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def filter_values(value: Any = None, values: Sequence[Any] | set[Any] | None = None):
    if values is not None:
        return list(values)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, set | tuple | list):
        return list(value)
    return [value]


def date_values(values: Sequence[Any]) -> set[Any]:
    timestamps = pd.to_datetime(list(values), errors="coerce")
    if pd.isna(timestamps).any():
        invalid = [
            value
            for value, timestamp in zip(values, timestamps, strict=False)
            if pd.isna(timestamp)
        ]
        raise ValueError(f"Invalid date filter values: {invalid!r}.")
    return set(timestamps.date)


def build_filter_mask(
    series: pd.Series,
    *,
    op: str,
    value: Any = None,
    values: Sequence[Any] | set[Any] | None = None,
    lower: Any = None,
    upper: Any = None,
    contains_regex: bool = False,
) -> pd.Series:
    op = str(op).lower()
    if op in {"==", "eq"}:
        return series == value
    if op in {"!=", "ne"}:
        return series != value
    if op == "in":
        if values is None and value is None:
            raise ValueError("values must be provided for op='in'.")
        return series.isin(filter_values(value=value, values=values))
    if op in {"not in", "not_in"}:
        if values is None and value is None:
            raise ValueError(f"values must be provided for op={op!r}.")
        return ~series.isin(filter_values(value=value, values=values))
    if op == "contains":
        return series.astype(str).str.contains(
            str(value),
            regex=bool(contains_regex),
            na=False,
        )
    if op == "startswith":
        return series.astype(str).str.startswith(str(value), na=False)
    if op == "endswith":
        return series.astype(str).str.endswith(str(value), na=False)
    if op == "isna":
        return series.isna()
    if op == "notna":
        return series.notna()
    if op in {">", ">=", "<", "<="}:
        numeric = pd.to_numeric(series, errors="coerce")
        threshold = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(threshold):
            raise ValueError(f"Invalid numeric filter value: {value!r}.")
        if op == ">":
            return numeric > threshold
        if op == ">=":
            return numeric >= threshold
        if op == "<":
            return numeric < threshold
        return numeric <= threshold
    if op in {"date>", "date>=", "date<", "date<=", "date_after", "date_before"}:
        timestamps = pd.to_datetime(series, errors="coerce")
        threshold = pd.to_datetime(value, errors="coerce")
        if pd.isna(threshold):
            raise ValueError(f"Invalid date filter value: {value!r}.")
        if op in {"date>", "date_after"}:
            return timestamps > threshold
        if op == "date>=":
            return timestamps >= threshold
        if op in {"date<", "date_before"}:
            return timestamps < threshold
        return timestamps <= threshold
    if op in {"date_after_or_equal"}:
        return build_filter_mask(series, op="date>=", value=value)
    if op in {"date_before_or_equal"}:
        return build_filter_mask(series, op="date<=", value=value)
    if op in {"date in", "date_in", "date not in", "date_not_in"}:
        selected_values = filter_values(value=value, values=values)
        if not selected_values:
            raise ValueError(f"{op!r} requires value or values.")
        allowed_dates = date_values(selected_values)
        mask = pd.to_datetime(series, errors="coerce").dt.date.isin(allowed_dates)
        if op in {"date not in", "date_not_in"}:
            return ~mask
        return mask
    if op in {"between", "date_between"}:
        if lower is None or upper is None:
            selected_values = filter_values(value=value, values=values)
            if len(selected_values) != 2:
                raise ValueError(f"{op!r} requires lower/upper or two values.")
            lower, upper = selected_values
        if op == "date_between":
            selected_series = pd.to_datetime(series, errors="coerce")
            lower = pd.to_datetime(lower, errors="coerce")
            upper = pd.to_datetime(upper, errors="coerce")
            if pd.isna(lower) or pd.isna(upper):
                raise ValueError("Invalid date_between lower/upper value.")
            return (selected_series >= lower) & (selected_series <= upper)
        numeric = pd.to_numeric(series, errors="coerce")
        lower = pd.to_numeric(pd.Series([lower]), errors="coerce").iloc[0]
        upper = pd.to_numeric(pd.Series([upper]), errors="coerce").iloc[0]
        if pd.isna(lower) or pd.isna(upper):
            raise ValueError("Invalid between lower/upper value.")
        return (numeric >= lower) & (numeric <= upper)
    raise ValueError(f"Unsupported filter op: {op!r}.")
