from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._compat import BaseEstimator, TransformerMixin
from .h5 import H5SessionFilter, h5_to_df


class H5ToDataFrameTransformer(TransformerMixin, BaseEstimator):
    """Read an H5 container into measurement rows as a sklearn transformer."""

    def __init__(
        self,
        *,
        data_preference: str = "gfrm",
        raw_root: str | Path | None = None,
        convert_gfrm: bool = True,
        require_clinical_ids: bool = True,
        drop_missing_sample_thickness: bool = False,
        h5_filters: Sequence[H5SessionFilter | dict[str, Any]] | None = None,
        max_sessions: int | None = None,
        session_category: str | Sequence[str] | set[str] | None = None,
        session_started_at_min: str | pd.Timestamp | None = None,
        set_category: str | Sequence[str] | set[str] | None = None,
        reader: Callable[..., tuple[pd.DataFrame, pd.DataFrame]] = h5_to_df,
    ) -> None:
        self.data_preference = data_preference
        self.raw_root = raw_root
        self.convert_gfrm = convert_gfrm
        self.require_clinical_ids = require_clinical_ids
        self.drop_missing_sample_thickness = drop_missing_sample_thickness
        self.h5_filters = h5_filters
        self.max_sessions = max_sessions
        self.session_category = session_category
        self.session_started_at_min = session_started_at_min
        self.set_category = set_category
        self.reader = reader
        self.calibration_df_: pd.DataFrame | None = None
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: str | Path, y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: str | Path) -> pd.DataFrame:
        calibration_df, measurement_df = self.reader(
            X,
            data_preference=self.data_preference,
            raw_root=self.raw_root,
            convert_gfrm=self.convert_gfrm,
            require_clinical_ids=self.require_clinical_ids,
            drop_missing_sample_thickness=self.drop_missing_sample_thickness,
            h5_filters=self.h5_filters,
            max_sessions=self.max_sessions,
            session_category=self.session_category,
            session_started_at_min=self.session_started_at_min,
            set_category=self.set_category,
        )
        self.calibration_df_ = calibration_df
        self.stats_ = {
            "filter_type": "h5_to_dataframe",
            "calibration_rows": int(len(calibration_df)),
            "measurement_rows": int(len(measurement_df)),
            "dropped_missing_sample_thickness": int(
                measurement_df.attrs.get("dropped_missing_sample_thickness", 0)
            ),
        }
        return measurement_df


class ProductColumnBuilder(TransformerMixin, BaseEstimator):
    """Create standard product metadata columns from H5 metadata columns."""

    def __init__(
        self,
        *,
        status_column: str = "specimen_status",
        date_column: str = "measurementDate",
        date_source_column: str = "started_at",
        sample_thickness_column: str = "sample_thickness_mm",
        sample_thickness_sources: Sequence[str] = (
            "sample_thickness_mm",
            "sample_thickness",
            "thickness_raw_mm",
        ),
        calibrant_thickness_column: str = "calibrant_thickness_mm",
        calibrant_thickness_sources: Sequence[str] = (
            "calibrant_thickness_mm",
            "agbh_thickness_mm",
        ),
        group_column: str = "product_status_group",
        diagnosis_column: str = "product_diagnosis",
        patient_diagnosis_column: str = "patient_product_diagnosis",
        patient_column: str = "patientId",
        cancer_values: Sequence[str] = ("CANCER", "ATYPICAL", "PRE_CANCEROUS"),
        benign_values: Sequence[str] = ("BENIGN",),
        normal_values: Sequence[str] = ("NORMAL",),
    ) -> None:
        self.status_column = status_column
        self.date_column = date_column
        self.date_source_column = date_source_column
        self.sample_thickness_column = sample_thickness_column
        self.sample_thickness_sources = tuple(sample_thickness_sources)
        self.calibrant_thickness_column = calibrant_thickness_column
        self.calibrant_thickness_sources = tuple(calibrant_thickness_sources)
        self.group_column = group_column
        self.diagnosis_column = diagnosis_column
        self.patient_diagnosis_column = patient_diagnosis_column
        self.patient_column = patient_column
        self.cancer_values = tuple(cancer_values)
        self.benign_values = tuple(benign_values)
        self.normal_values = tuple(normal_values)
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None):
        _ = X
        _ = y
        return self

    def _status_values(self, df: pd.DataFrame) -> pd.Series:
        if self.status_column not in df.columns:
            return pd.Series(["NA"] * len(df), index=df.index, dtype="object")
        values = df[self.status_column].fillna("NA").astype(str).str.strip().str.upper()
        return values.replace("", "NA")

    def _group_values(self, df: pd.DataFrame) -> pd.Series:
        status = self._status_values(df)
        cancer = {value.upper() for value in self.cancer_values}
        benign = {value.upper() for value in self.benign_values}
        normal = {value.upper() for value in self.normal_values}
        grouped = pd.Series("EXCLUDE", index=df.index, dtype="object")
        grouped.loc[status.isin(benign)] = "BENIGN"
        grouped.loc[status.isin(cancer)] = "CANCER"
        grouped.loc[status.isin(normal)] = "NORMAL"
        return grouped

    def _first_numeric_source(
        self,
        df: pd.DataFrame,
        sources: Sequence[str],
    ) -> pd.Series:
        present = [column for column in sources if column in df.columns]
        if not present:
            return pd.Series(np.nan, index=df.index, dtype="float64")
        numeric_sources = [pd.to_numeric(df[column], errors="coerce") for column in present]
        return pd.concat(numeric_sources, axis=1).bfill(axis=1).iloc[:, 0]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.date_column not in out.columns:
            out[self.date_column] = out.get(self.date_source_column)
        out[self.sample_thickness_column] = self._first_numeric_source(
            out,
            self.sample_thickness_sources,
        )
        out[self.calibrant_thickness_column] = self._first_numeric_source(
            out,
            self.calibrant_thickness_sources,
        )
        out[self.group_column] = self._group_values(out)
        out[self.diagnosis_column] = out[self.group_column].where(
            out[self.group_column].isin(["BENIGN", "CANCER"])
        )
        if self.patient_column in out.columns:
            patient_groups = (
                out[[self.patient_column, self.group_column]]
                .dropna(subset=[self.patient_column])
                .groupby(self.patient_column, dropna=False)[self.group_column]
                .apply(lambda values: sorted(set(values.astype(str))))
            )
            patient_dx: dict[Any, str] = {}
            for patient_id, groups in patient_groups.items():
                if "CANCER" in groups:
                    patient_dx[patient_id] = "CANCER"
                elif "BENIGN" in groups:
                    patient_dx[patient_id] = "BENIGN"
                elif "NORMAL" in groups:
                    patient_dx[patient_id] = "NORMAL"
                else:
                    patient_dx[patient_id] = "EXCLUDE"
            out[self.patient_diagnosis_column] = out[self.patient_column].map(patient_dx)
        self.stats_ = {
            "filter_type": "product_column_builder",
            "rows": int(len(out)),
            "status_counts": self._status_values(out).value_counts(dropna=False).to_dict(),
            "group_counts": out[self.group_column].value_counts(dropna=False).to_dict(),
        }
        return out


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
        self.stats_ = {
            "filter_type": "product_status_group",
            "group_column": self.group_column,
            "allowed": sorted(allowed),
            "rows_in": int(len(out)),
            "rows_pass": int(mask.sum()),
            "rows_fail": int(len(out) - mask.sum()),
            "counts_in": groups.value_counts(dropna=False).to_dict(),
            "counts_pass": filtered[self.group_column]
            .fillna("")
            .astype(str)
            .str.upper()
            .value_counts(dropna=False)
            .to_dict(),
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
            .dropna(subset=[self.patient_column, self.specimen_column, self.group_column])
            .drop_duplicates()
            .groupby([self.patient_column, self.specimen_column], dropna=False, as_index=False)
            .agg(**{self.group_column: (self.group_column, "first")})
        )
        allowed = {
            tuple(sorted(str(item).upper() for item in pair))
            for pair in self.allowed_pairs
        }
        patient_pairs: dict[Any, tuple[str, ...]] = {}
        for patient_id, rows in specimen_groups.groupby(self.patient_column, dropna=False):
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
        self.stats_ = {
            "filter_type": "paired_group",
            "patient_column": self.patient_column,
            "specimen_column": self.specimen_column,
            "group_column": self.group_column,
            "allowed_pairs": ["__".join(pair) for pair in self.allowed_pairs],
            "rows_in": int(len(out)),
            "rows_pass": int(len(filtered)),
            "rows_fail": int(len(out) - len(filtered)),
            "patients_in": int(out[self.patient_column].nunique()),
            "patients_pass": int(filtered[self.patient_column].nunique()),
            "specimens_in": int(
                out[[self.patient_column, self.specimen_column]].drop_duplicates().shape[0]
            ),
            "specimens_pass": int(
                filtered[[self.patient_column, self.specimen_column]]
                .drop_duplicates()
                .shape[0]
            ),
            "before_pair_counts": before_counts,
            "after_pair_counts": filtered[self.output_column]
            .value_counts(dropna=False)
            .to_dict(),
        }
        return filtered


class ConstantQRangeTransformer(TransformerMixin, BaseEstimator):
    """Attach the same interpolation q range to every row."""

    def __init__(
        self,
        *,
        q_min: float = 2.0,
        q_max: float = 23.0,
        output_column: str = "interpolation_q_range",
    ) -> None:
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.output_column = output_column
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out[self.output_column] = [(self.q_min, self.q_max)] * len(out)
        self.stats_ = {
            "filter_type": "constant_q_range",
            "rows": int(len(out)),
            "q_min": self.q_min,
            "q_max": self.q_max,
            "output_column": self.output_column,
        }
        return out


class DropColumnsTransformer(TransformerMixin, BaseEstimator):
    """Drop configured columns and expose the actually dropped columns."""

    def __init__(
        self,
        columns: Sequence[str],
        *,
        errors: str = "ignore",
    ) -> None:
        self.columns = tuple(columns)
        self.errors = errors
        self.dropped_columns_: list[str] = []
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.dropped_columns_ = [column for column in self.columns if column in X.columns]
        out = X.drop(columns=self.dropped_columns_, errors=self.errors).copy()
        self.stats_ = {
            "filter_type": "drop_columns",
            "rows": int(len(out)),
            "requested_columns": list(self.columns),
            "dropped_columns": list(self.dropped_columns_),
        }
        return out


class RequiredColumnsTransformer(TransformerMixin, BaseEstimator):
    """Fail if required columns are missing or invalid."""

    def __init__(
        self,
        columns: Sequence[str] | Mapping[str, tuple[float | None, float | None]],
    ) -> None:
        self.columns = columns
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if isinstance(self.columns, Mapping):
            missing = [column for column in self.columns if column not in out.columns]
            if missing:
                raise KeyError(f"Missing required columns: {missing}")
            for column, bounds in self.columns.items():
                lower, upper = bounds
                values = pd.to_numeric(out[column], errors="coerce")
                valid = np.isfinite(values)
                if lower is not None:
                    valid &= values >= float(lower)
                if upper is not None:
                    valid &= values <= float(upper)
                if not bool(valid.all()):
                    raise ValueError(f"Invalid values in required column: {column}")
        else:
            missing = [column for column in self.columns if column not in out.columns]
            if missing:
                raise KeyError(f"Missing required columns: {missing}")
        self.stats_ = {
            "filter_type": "required_columns",
            "rows": int(len(out)),
            "columns": list(self.columns),
        }
        return out
