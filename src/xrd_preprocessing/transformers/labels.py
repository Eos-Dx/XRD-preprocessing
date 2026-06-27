"""Product label and paired specimen transformers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from .._compat import BaseEstimator, TransformerMixin


class ProductStatusGroupFilter(TransformerMixin, BaseEstimator):
    """Filter rows by a product status-group column."""

    def __init__(
        self,
        allowed: Sequence[str],
        *,
        group_column: str = "product_status_group",
        reset_index: bool = True,
    ) -> None:
        self.allowed = tuple(allowed)
        self.group_column = group_column
        self.reset_index = reset_index
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.group_column not in X.columns:
            raise KeyError(f"Missing product status-group column: {self.group_column}")
        out = X.copy()
        allowed = {value.upper() for value in self.allowed}
        groups = out[self.group_column].fillna("").astype(str).str.upper()
        mask = groups.isin(allowed)
        filtered = out.loc[mask].copy()
        if self.reset_index:
            filtered.reset_index(drop=True, inplace=True)
        counts_pass = (
            filtered[self.group_column]
            .fillna("")
            .astype(str)
            .str.upper()
            .value_counts(dropna=False)
            .to_dict()
        )
        rows_fail = int(len(out) - mask.sum())
        self.stats_ = {
            "filter_type": "product_status_group",
            "group_column": self.group_column,
            "allowed": sorted(allowed),
            "rows_in": int(len(out)),
            "rows_pass": int(mask.sum()),
            "rows_fail": rows_fail,
            "rows_dropped": rows_fail,
            "counts_in": groups.value_counts(dropna=False).to_dict(),
            "counts_pass": counts_pass,
            "before_counts": groups.value_counts(dropna=False).to_dict(),
            "after_counts": counts_pass,
        }
        return filtered


class PairedGroupFilter(TransformerMixin, BaseEstimator):
    """Keep patients/specimen pairs whose grouped labels match allowed pairs."""

    def __init__(
        self,
        *,
        patient_column: str = "patientId",
        specimen_column: str = "specimenId",
        group_column: str = "product_status_group",
        allowed_pairs: Sequence[Sequence[str]] = (
            ("BENIGN", "CANCER"),
            ("BENIGN", "NORMAL"),
            ("CANCER", "NORMAL"),
        ),
        output_column: str = "one_to_one_pair_type",
        reset_index: bool = True,
    ) -> None:
        self.patient_column = patient_column
        self.specimen_column = specimen_column
        self.group_column = group_column
        self.allowed_pairs = tuple(tuple(pair) for pair in allowed_pairs)
        self.output_column = output_column
        self.reset_index = reset_index
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [
            column
            for column in (self.patient_column, self.specimen_column, self.group_column)
            if column not in X.columns
        ]
        if missing:
            raise KeyError(f"Missing paired-group columns: {missing}")
        out = X.copy()
        specimen_groups = (
            out[[self.patient_column, self.specimen_column, self.group_column]]
            .dropna(
                subset=[self.patient_column, self.specimen_column, self.group_column]
            )
            .drop_duplicates()
            .groupby(
                [self.patient_column, self.specimen_column],
                dropna=False,
                as_index=False,
            )
            .agg(**{self.group_column: (self.group_column, "first")})
        )
        allowed = {
            tuple(sorted(str(item).upper() for item in pair))
            for pair in self.allowed_pairs
        }
        patient_pairs: dict[Any, tuple[str, ...]] = {}
        for patient_id, rows in specimen_groups.groupby(
            self.patient_column, dropna=False
        ):
            patient_pairs[patient_id] = tuple(
                sorted(rows[self.group_column].astype(str).str.upper())
            )
        valid_patients = [
            patient_id
            for patient_id, groups in patient_pairs.items()
            if len(groups) == 2 and groups in allowed
        ]
        pair_labels = {
            patient_id: "__".join(patient_pairs[patient_id])
            for patient_id in valid_patients
        }
        filtered = out.loc[out[self.patient_column].isin(valid_patients)].copy()
        filtered[self.output_column] = filtered[self.patient_column].map(pair_labels)
        if self.reset_index:
            filtered.reset_index(drop=True, inplace=True)
        before_counts = {
            "__".join(groups) if groups else "NA": int(count)
            for groups, count in pd.Series(patient_pairs).value_counts().items()
        }
        after_pair_counts = (
            filtered[self.output_column].value_counts(dropna=False).to_dict()
        )
        rows_fail = int(len(out) - len(filtered))
        self.stats_ = {
            "filter_type": "paired_group",
            "patient_column": self.patient_column,
            "specimen_column": self.specimen_column,
            "group_column": self.group_column,
            "allowed_pairs": ["__".join(pair) for pair in self.allowed_pairs],
            "rows_in": int(len(out)),
            "rows_pass": int(len(filtered)),
            "rows_fail": rows_fail,
            "rows_dropped": rows_fail,
            "patients_in": int(out[self.patient_column].nunique()),
            "patients_pass": int(filtered[self.patient_column].nunique()),
            "specimens_in": int(
                out[[self.patient_column, self.specimen_column]]
                .drop_duplicates()
                .shape[0]
            ),
            "specimens_pass": int(
                filtered[[self.patient_column, self.specimen_column]]
                .drop_duplicates()
                .shape[0]
            ),
            "before_pair_counts": before_counts,
            "after_pair_counts": after_pair_counts,
            "after_pair_row_counts": after_pair_counts,
        }
        return filtered
