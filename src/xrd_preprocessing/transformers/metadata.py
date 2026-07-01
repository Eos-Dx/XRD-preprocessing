"""Metadata, q-range, column, and output transformers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._compat import BaseEstimator, TransformerMixin
from ..artifacts import save_preprocessing_artifact


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
        numeric_sources = [
            pd.to_numeric(df[column], errors="coerce") for column in present
        ]
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
            out[self.patient_diagnosis_column] = out[self.patient_column].map(
                patient_dx
            )
        self.stats_ = {
            "filter_type": "product_column_builder",
            "rows": int(len(out)),
            "status_counts": self._status_values(out)
            .value_counts(dropna=False)
            .to_dict(),
            "group_counts": out[self.group_column].value_counts(dropna=False).to_dict(),
        }
        return out


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
        self.dropped_columns_ = [
            column for column in self.columns if column in X.columns
        ]
        out = X.drop(columns=self.dropped_columns_, errors=self.errors).copy()
        self.stats_ = {
            "filter_type": "drop_columns",
            "rows": int(len(out)),
            "requested_columns": list(self.columns),
            "dropped_columns": list(self.dropped_columns_),
        }
        return out


class KeepColumnsTransformer(TransformerMixin, BaseEstimator):
    """Keep configured columns in order."""

    def __init__(self, columns: Sequence[str], *, errors: str = "raise") -> None:
        self.columns = tuple(columns)
        self.errors = errors
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.errors not in {"raise", "ignore"}:
            raise ValueError("errors must be 'raise' or 'ignore'.")
        missing = [column for column in self.columns if column not in X.columns]
        if missing and self.errors == "raise":
            raise KeyError(f"Missing kept output columns: {missing}")
        kept_columns = [column for column in self.columns if column in X.columns]
        out = X.loc[:, kept_columns].copy()
        self.stats_ = {
            "filter_type": "keep_columns",
            "rows": int(len(out)),
            "columns": kept_columns,
            "missing_columns": missing,
            "errors": self.errors,
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


class JoblibWriterTransformer(TransformerMixin, BaseEstimator):
    """Write a DataFrame or preprocessing artifact to joblib."""

    def __init__(
        self,
        output_path: str | Path | None = None,
        *,
        artifact: bool = False,
        preprocessing_config: dict[str, Any] | None = None,
        preprocessing_config_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_path = output_path
        self.artifact = artifact
        self.preprocessing_config = preprocessing_config
        self.preprocessing_config_text = preprocessing_config_text
        self.metadata = metadata
        self.output_path_: Path | None = None
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.output_path is not None:
            output_path = Path(self.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.artifact:
                save_preprocessing_artifact(
                    X,
                    output_path,
                    preprocessing_config=self.preprocessing_config,
                    preprocessing_config_text=self.preprocessing_config_text,
                    metadata=self.metadata,
                )
            else:
                import joblib

                joblib.dump(X, output_path)
            self.output_path_ = output_path
        self.stats_ = {
            "filter_type": "joblib_writer",
            "rows": int(len(X)),
            "artifact": bool(self.artifact),
            "output_path": str(self.output_path_) if self.output_path_ else None,
        }
        return X
