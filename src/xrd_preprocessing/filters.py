"""Reusable DataFrame filters for XRD product preprocessing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._compat import BaseEstimator, TransformerMixin
from .azimuthal import estimate_poni_q_range_nm_inv


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
        if op in {"date>=", "date_after_or_equal", "date_after"}:
            date_values = pd.to_datetime(series, errors="coerce")
            cutoff = pd.Timestamp(self.value)
            if op == "date_after":
                return date_values.gt(cutoff)
            return date_values.ge(cutoff)
        if op in {"date<=", "date_before_or_equal", "date_before"}:
            date_values = pd.to_datetime(series, errors="coerce")
            cutoff = pd.Timestamp(self.value)
            if op == "date_before":
                return date_values.lt(cutoff)
            return date_values.le(cutoff)
        if op == "date_between":
            if self.lower is None or self.upper is None:
                raise ValueError("lower and upper must be provided for op='date_between'.")
            date_values = pd.to_datetime(series, errors="coerce")
            return date_values.between(
                pd.Timestamp(self.lower),
                pd.Timestamp(self.upper),
                inclusive="both",
            )
        if op in {"date in", "date_in"}:
            if self.values is None:
                raise ValueError("values must be provided for op='date in'.")
            date_values = pd.to_datetime(series, errors="coerce").dt.date.astype(str)
            allowed_dates = {
                str(pd.Timestamp(value).date())
                for value in self.values
                if not pd.isna(pd.Timestamp(value))
            }
            return date_values.isin(allowed_dates)
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
    """Column-value filter for metadata columns."""

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
    """Metadata filter for patient/sample selection stages."""


class SpecimenValidityFilter(TransformerMixin, BaseEstimator):
    """Specimen-level replicate validity filter."""

    def __init__(
        self,
        specimen_column: str = "specimenId",
        *,
        min_measurements_per_specimen: int = 2,
        drop: bool = True,
        reset_index: bool = True,
        validity_column: str = "specimen_valid",
        reason_column: str = "specimen_validity_reason",
    ) -> None:
        self.specimen_column = specimen_column
        self.min_measurements_per_specimen = int(min_measurements_per_specimen)
        self.drop = bool(drop)
        self.reset_index = bool(reset_index)
        self.validity_column = validity_column
        self.reason_column = reason_column
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    @staticmethod
    def _present(series: pd.Series) -> pd.Series:
        return series.notna() & series.astype(str).str.strip().ne("")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.specimen_column not in X.columns:
            raise KeyError(f"Missing required specimen ID column: {self.specimen_column}")
        if self.min_measurements_per_specimen < 1:
            raise ValueError("min_measurements_per_specimen must be >= 1.")

        out = X.copy()
        specimen = out[self.specimen_column]
        has_specimen = self._present(specimen)
        valid_specimen_frame = out.loc[has_specimen, [self.specimen_column]]
        specimen_counts = valid_specimen_frame.groupby(
            self.specimen_column,
            sort=False,
            dropna=False,
        ).size()
        counts = [
            int(specimen_counts.get(specimen_value, 0))
            for specimen_value in specimen
        ]
        out["specimen_measurement_count"] = counts
        valid = has_specimen & (
            out["specimen_measurement_count"] >= self.min_measurements_per_specimen
        )
        out[self.validity_column] = valid.to_numpy(dtype=bool)
        out[self.reason_column] = "valid"
        out.loc[~has_specimen, self.reason_column] = "missing_specimen_id"
        out.loc[
            has_specimen
            & (out["specimen_measurement_count"] < self.min_measurements_per_specimen),
            self.reason_column,
        ] = "specimen_measurements_below_minimum"
        self.stats_ = {
            "filter_type": "specimen_validity",
            "specimen_column": self.specimen_column,
            "min_measurements_per_specimen": self.min_measurements_per_specimen,
            "rows_in": int(len(out)),
            "rows_pass": int(np.sum(valid)),
            "rows_fail": int(len(out) - np.sum(valid)),
            "specimens_in": int(valid_specimen_frame.drop_duplicates().shape[0]),
            "specimens_pass": int(out.loc[valid, self.specimen_column].nunique()),
        }
        if self.drop:
            out = out.loc[out[self.validity_column]].copy()
            if self.reset_index:
                out.reset_index(drop=True, inplace=True)
        return out


class PatientSpecimenValidityFilter(TransformerMixin, BaseEstimator):
    """Group-level validity filter for clinical patient/specimen replicates."""

    def __init__(
        self,
        patient_column: str = "patientId",
        specimen_column: str = "specimenId",
        *,
        min_measurements_per_specimen: int = 2,
        min_specimens_per_patient: int = 1,
        drop: bool = True,
        reset_index: bool = True,
        validity_column: str = "patient_specimen_valid",
        reason_column: str = "patient_specimen_validity_reason",
    ) -> None:
        self.patient_column = patient_column
        self.specimen_column = specimen_column
        self.min_measurements_per_specimen = int(min_measurements_per_specimen)
        self.min_specimens_per_patient = int(min_specimens_per_patient)
        self.drop = bool(drop)
        self.reset_index = bool(reset_index)
        self.validity_column = validity_column
        self.reason_column = reason_column
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    @staticmethod
    def _present(series: pd.Series) -> pd.Series:
        return series.notna() & series.astype(str).str.strip().ne("")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [
            column
            for column in (self.patient_column, self.specimen_column)
            if column not in X.columns
        ]
        if missing:
            raise KeyError(f"Missing required clinical ID columns: {missing}")
        if self.min_measurements_per_specimen < 1:
            raise ValueError("min_measurements_per_specimen must be >= 1.")
        if self.min_specimens_per_patient < 1:
            raise ValueError("min_specimens_per_patient must be >= 1.")

        out = X.copy()
        patient = out[self.patient_column]
        specimen = out[self.specimen_column]
        has_ids = self._present(patient) & self._present(specimen)

        valid_id_frame = out.loc[has_ids, [self.patient_column, self.specimen_column]]
        specimen_counts = valid_id_frame.groupby(
            [self.patient_column, self.specimen_column],
            sort=False,
            dropna=False,
        ).size()

        counts = [
            int(specimen_counts.get((patient_value, specimen_value), 0))
            for patient_value, specimen_value in zip(patient, specimen, strict=False)
        ]
        out["specimen_measurement_count"] = counts

        specimen_valid = has_ids & (
            out["specimen_measurement_count"] >= self.min_measurements_per_specimen
        )
        valid_specimens = out.loc[
            specimen_valid, [self.patient_column, self.specimen_column]
        ].drop_duplicates()
        patient_specimen_counts = valid_specimens.groupby(
            self.patient_column,
            sort=False,
            dropna=False,
        ).size()
        patient_counts = [
            int(patient_specimen_counts.get(patient_value, 0))
            for patient_value in patient
        ]
        out["patient_valid_specimen_count"] = patient_counts

        valid = specimen_valid & (
            out["patient_valid_specimen_count"] >= self.min_specimens_per_patient
        )
        out[self.validity_column] = valid.to_numpy(dtype=bool)
        out[self.reason_column] = "valid"
        out.loc[~has_ids, self.reason_column] = "missing_patient_or_specimen_id"
        out.loc[
            has_ids
            & (out["specimen_measurement_count"] < self.min_measurements_per_specimen),
            self.reason_column,
        ] = "specimen_measurements_below_minimum"
        out.loc[
            specimen_valid
            & (out["patient_valid_specimen_count"] < self.min_specimens_per_patient),
            self.reason_column,
        ] = "patient_specimens_below_minimum"

        self.stats_ = {
            "filter_type": "patient_specimen_validity",
            "patient_column": self.patient_column,
            "specimen_column": self.specimen_column,
            "min_measurements_per_specimen": self.min_measurements_per_specimen,
            "min_specimens_per_patient": self.min_specimens_per_patient,
            "rows_in": int(len(out)),
            "rows_pass": int(np.sum(valid)),
            "rows_fail": int(len(out) - np.sum(valid)),
            "patients_in": int(patient[has_ids].nunique()),
            "patients_pass": int(out.loc[valid, self.patient_column].nunique()),
            "specimens_in": int(valid_id_frame.drop_duplicates().shape[0]),
            "specimens_pass": int(valid_specimens.shape[0]),
        }
        if self.drop:
            out = out.loc[out[self.validity_column]].copy()
            if self.reset_index:
                out.reset_index(drop=True, inplace=True)
        return out


class PoniQRangeFilter(TransformerMixin, BaseEstimator):
    """Filter rows whose PONI geometry cannot cover the requested q range."""

    def __init__(
        self,
        *,
        required_q_max_nm_inv: float,
        poni_column: str = "ponifile",
        data_column: str | None = "measurement_data",
        shape_column: str | None = None,
        q_min_column: str = "poni_q_min_nm_inv",
        q_max_column: str = "poni_q_max_nm_inv",
        distance_column: str = "poni_calculated_distance_m",
        pass_column: str = "poni_q_range_pass",
        drop: bool = True,
        reset_index: bool = True,
        thickness_adjustment: bool = False,
        require_thickness_adjustment: bool = False,
        sample_thickness_column: str = "sample_thickness_mm",
        thickness_reference_mm: float | None = None,
        thickness_reference_column: str | None = None,
    ) -> None:
        self.required_q_max_nm_inv = float(required_q_max_nm_inv)
        self.poni_column = poni_column
        self.data_column = data_column
        self.shape_column = shape_column
        self.q_min_column = q_min_column
        self.q_max_column = q_max_column
        self.distance_column = distance_column
        self.pass_column = pass_column
        self.drop = bool(drop)
        self.reset_index = bool(reset_index)
        self.thickness_adjustment = bool(thickness_adjustment)
        self.require_thickness_adjustment = bool(require_thickness_adjustment)
        self.sample_thickness_column = sample_thickness_column
        self.thickness_reference_mm = thickness_reference_mm
        self.thickness_reference_column = thickness_reference_column
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def _row_shape(self, row: pd.Series) -> tuple[int, int] | None:
        if self.shape_column is not None:
            if self.shape_column not in row.index:
                raise KeyError(f"Column '{self.shape_column}' not found in DataFrame.")
            value = row[self.shape_column]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            shape = tuple(int(item) for item in value)
            if len(shape) != 2:
                raise ValueError(f"Shape column '{self.shape_column}' must contain 2 values.")
            return shape
        if self.data_column is not None and self.data_column in row.index:
            data = row[self.data_column]
            if hasattr(data, "shape") and len(data.shape) >= 2:
                return int(data.shape[0]), int(data.shape[1])
        return None

    def _reference_thickness_mm(self, row: pd.Series) -> float:
        if self.thickness_reference_column is not None:
            if self.thickness_reference_column not in row.index:
                raise KeyError(
                    f"Column '{self.thickness_reference_column}' not found in DataFrame."
                )
            reference = pd.to_numeric(
                pd.Series([row[self.thickness_reference_column]]),
                errors="coerce",
            ).iloc[0]
            if not np.isfinite(reference):
                raise ValueError(
                    "Invalid thickness reference value in column: "
                    f"{self.thickness_reference_column}"
                )
            return float(reference)
        if self.thickness_reference_mm is None:
            return 0.0
        return float(self.thickness_reference_mm)

    def _sample_thickness_mm(self, row: pd.Series) -> float | None:
        if not self.thickness_adjustment:
            return None
        if self.sample_thickness_column not in row.index:
            if self.require_thickness_adjustment:
                raise KeyError(
                    f"Column '{self.sample_thickness_column}' not found in DataFrame."
                )
            return None
        thickness = pd.to_numeric(
            pd.Series([row[self.sample_thickness_column]]),
            errors="coerce",
        ).iloc[0]
        if not np.isfinite(thickness):
            if self.require_thickness_adjustment:
                raise ValueError(
                    f"Invalid thickness value in column: {self.sample_thickness_column}"
                )
            return None
        return float(thickness)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.poni_column not in X.columns:
            raise KeyError(f"Column '{self.poni_column}' not found in DataFrame.")

        out = X.copy()
        q_min_values = []
        q_max_values = []
        distance_values = []
        passed_values = []
        for _row in out.itertuples(index=False):
            row = pd.Series(_row._asdict())
            q_min, q_max, distance_m = estimate_poni_q_range_nm_inv(
                str(row[self.poni_column]),
                shape=self._row_shape(row),
                sample_thickness_mm=self._sample_thickness_mm(row),
                thickness_reference_mm=self._reference_thickness_mm(row),
            )
            q_min_values.append(q_min)
            q_max_values.append(q_max)
            distance_values.append(distance_m)
            passed_values.append(q_max >= self.required_q_max_nm_inv)

        passed = np.asarray(passed_values, dtype=bool)
        out[self.q_min_column] = q_min_values
        out[self.q_max_column] = q_max_values
        out[self.distance_column] = distance_values
        out[self.pass_column] = passed
        failed_ids = (
            out.loc[~out[self.pass_column], "sample_id"].astype(str).tolist()
            if "sample_id" in out.columns
            else []
        )
        self.stats_ = {
            "filter_type": "poni_q_range",
            "required_q_max_nm_inv": self.required_q_max_nm_inv,
            "poni_column": self.poni_column,
            "q_max_column": self.q_max_column,
            "rows_in": int(len(out)),
            "rows_pass": int(np.sum(passed)),
            "rows_fail": int(len(out) - np.sum(passed)),
            "min_q_max_nm_inv": float(np.nanmin(q_max_values)) if q_max_values else np.nan,
            "max_q_max_nm_inv": float(np.nanmax(q_max_values)) if q_max_values else np.nan,
            "failed_ids": failed_ids,
            "thickness_adjustment": self.thickness_adjustment,
            "thickness_reference_mm": self.thickness_reference_mm,
            "thickness_reference_column": self.thickness_reference_column,
        }
        if self.drop:
            out = out.loc[out[self.pass_column]].copy()
            if self.reset_index:
                out.reset_index(drop=True, inplace=True)
        return out


class RadialProfileValueFilter(TransformerMixin, BaseEstimator):
    """Filter radial profiles by intensity near a target q value."""

    def __init__(
        self,
        *,
        q_value_nm_inv: float,
        threshold: float,
        op: str = ">",
        q_column: str = "q_range",
        profile_column: str = "radial_profile_data",
        pass_column: str = "radial_profile_value_pass",
        value_column: str = "radial_profile_value_at_q",
        nearest_q_column: str = "radial_profile_nearest_q_nm_inv",
        q_delta_column: str = "radial_profile_q_delta_nm_inv",
        max_q_delta_nm_inv: float | None = None,
        drop: bool = True,
        reset_index: bool = True,
    ) -> None:
        self.q_value_nm_inv = float(q_value_nm_inv)
        self.threshold = float(threshold)
        self.op = str(op)
        self.q_column = q_column
        self.profile_column = profile_column
        self.pass_column = pass_column
        self.value_column = value_column
        self.nearest_q_column = nearest_q_column
        self.q_delta_column = q_delta_column
        self.max_q_delta_nm_inv = (
            None if max_q_delta_nm_inv is None else float(max_q_delta_nm_inv)
        )
        self.drop = bool(drop)
        self.reset_index = bool(reset_index)
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def _compare(self, values: np.ndarray) -> np.ndarray:
        op = self.op.lower()
        if op in {">", "gt"}:
            return values > self.threshold
        if op in {">=", "ge"}:
            return values >= self.threshold
        if op in {"<", "lt"}:
            return values < self.threshold
        if op in {"<=", "le"}:
            return values <= self.threshold
        raise ValueError(f"Unsupported radial profile value filter op: {self.op}")

    def _nearest_value(self, q_values: Any, profile_values: Any) -> tuple[float, float, float]:
        q = np.asarray(q_values, dtype=float)
        profile = np.asarray(profile_values, dtype=float)
        if q.shape != profile.shape:
            raise ValueError(
                f"Columns '{self.q_column}' and '{self.profile_column}' must have same shape."
            )
        finite = np.isfinite(q) & np.isfinite(profile)
        if not np.any(finite):
            return np.nan, np.nan, np.nan
        q_finite = q[finite]
        profile_finite = profile[finite]
        nearest_index = int(np.argmin(np.abs(q_finite - self.q_value_nm_inv)))
        nearest_q = float(q_finite[nearest_index])
        value = float(profile_finite[nearest_index])
        delta = abs(nearest_q - self.q_value_nm_inv)
        return nearest_q, value, float(delta)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [
            column
            for column in (self.q_column, self.profile_column)
            if column not in X.columns
        ]
        if missing:
            raise KeyError(f"Missing required radial profile columns: {missing}")

        out = X.copy()
        nearest_q_values = []
        profile_values = []
        q_delta_values = []
        for q_values, radial_values in zip(
            out[self.q_column],
            out[self.profile_column],
            strict=False,
        ):
            nearest_q, value, delta = self._nearest_value(q_values, radial_values)
            nearest_q_values.append(nearest_q)
            profile_values.append(value)
            q_delta_values.append(delta)

        values = np.asarray(profile_values, dtype=float)
        q_deltas = np.asarray(q_delta_values, dtype=float)
        passed = np.isfinite(values) & self._compare(values)
        if self.max_q_delta_nm_inv is not None:
            passed &= np.isfinite(q_deltas) & (q_deltas <= self.max_q_delta_nm_inv)

        out[self.nearest_q_column] = nearest_q_values
        out[self.value_column] = profile_values
        out[self.q_delta_column] = q_delta_values
        out[self.pass_column] = passed

        failed_ids = (
            out.loc[~out[self.pass_column], "sample_id"].astype(str).tolist()
            if "sample_id" in out.columns
            else []
        )
        finite_values = values[np.isfinite(values)]
        self.stats_ = {
            "filter_type": "radial_profile_value",
            "q_value_nm_inv": self.q_value_nm_inv,
            "threshold": self.threshold,
            "op": self.op,
            "q_column": self.q_column,
            "profile_column": self.profile_column,
            "value_column": self.value_column,
            "max_q_delta_nm_inv": self.max_q_delta_nm_inv,
            "rows_in": int(len(out)),
            "rows_pass": int(np.sum(passed)),
            "rows_fail": int(len(out) - np.sum(passed)),
            "min_value_at_q": (
                float(np.nanmin(finite_values)) if len(finite_values) else np.nan
            ),
            "max_value_at_q": (
                float(np.nanmax(finite_values)) if len(finite_values) else np.nan
            ),
            "failed_ids": failed_ids,
        }
        if self.drop:
            out = out.loc[out[self.pass_column]].copy()
            if self.reset_index:
                out.reset_index(drop=True, inplace=True)
        return out


class SNRFilter(ColumnValueFilter):
    """Column-value filter for scalar SNR in dB."""

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
