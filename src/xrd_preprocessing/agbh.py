from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd

from ._compat import BaseEstimator, TransformerMixin
from .h5 import H5SessionFilter

AGBH_D_SPACING_NM = 5.838
AG_KBETA_TO_KALPHA_Q_RATIO = 0.886


def agbh_alpha_peaks(
    q_max: float,
    *,
    d_spacing_nm: float = AGBH_D_SPACING_NM,
) -> np.ndarray:
    """Return AgBH K-alpha peak positions up to ``q_max`` in nm^-1."""
    q_first = 2.0 * np.pi / float(d_spacing_nm)
    max_order = int(np.floor(float(q_max) / q_first))
    return q_first * np.arange(1, max_order + 1, dtype=float)


def _window_mask(grid: np.ndarray, center: float, half_width: float) -> np.ndarray:
    return (grid >= center - half_width) & (grid <= center + half_width)


def _profile_on_grid(q: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(q) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(q[mask])
    return np.interp(grid, q[mask][order], y[mask][order])


def _normalize_profile(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float)
    if not np.isfinite(out).any():
        return out
    out = out - np.nanmin(out)
    max_value = np.nanmax(out)
    if np.isfinite(max_value) and max_value > 0:
        out = out / max_value
    return out


def agbh_kbeta_windows(
    q_min: float,
    q_max: float,
    *,
    d_spacing_nm: float = AGBH_D_SPACING_NM,
    beta_ratio: float = AG_KBETA_TO_KALPHA_Q_RATIO,
    beta_half_width: float = 0.12,
    min_order: int = 3,
) -> list[dict[str, float | int]]:
    """Return left K-beta windows and right-side control windows for AgBH peaks."""
    windows: list[dict[str, float | int]] = []
    for order, alpha_q in enumerate(
        agbh_alpha_peaks(q_max, d_spacing_nm=d_spacing_nm),
        start=1,
    ):
        if order < min_order:
            continue
        beta_q = float(alpha_q * beta_ratio)
        right_control_q = float(alpha_q + (alpha_q - beta_q))
        if beta_q - beta_half_width < q_min:
            continue
        if right_control_q + beta_half_width > q_max:
            continue
        windows.append(
            {
                "order": int(order),
                "alpha_q": float(alpha_q),
                "beta_q": beta_q,
                "right_control_q": right_control_q,
            }
        )
    return windows


def _alpha_fit_mask(
    grid: np.ndarray,
    *,
    q_max: float,
    d_spacing_nm: float,
    left_width: float,
    right_width: float,
    min_order: int,
) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    for order, alpha_q in enumerate(
        agbh_alpha_peaks(q_max, d_spacing_nm=d_spacing_nm),
        start=1,
    ):
        if order >= min_order:
            mask |= (grid >= float(alpha_q) - left_width) & (
                grid <= float(alpha_q) + right_width
            )
    return mask


def _fit_baseline(
    y: np.ndarray,
    baseline_curve: np.ndarray,
    grid: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    valid = fit_mask & np.isfinite(baseline_curve) & np.isfinite(y)
    if int(np.sum(valid)) < 10:
        raise ValueError("Not enough finite AgBH points to fit baseline profile.")
    centered_q = grid - float(np.nanmean(grid))
    design = np.column_stack(
        [
            baseline_curve[valid],
            np.ones(int(np.sum(valid))),
            centered_q[valid],
        ]
    )
    scale, offset, slope = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    fitted = scale * baseline_curve + offset + slope * centered_q
    return fitted, float(scale), float(offset), float(slope)


@dataclass(frozen=True)
class AgBHMonochromaticityResult:
    """AgBH left-side K-beta residual score; lower means more monochromatic."""

    score: float
    passed: bool
    status: str
    left_positive_area: float
    right_control_positive_area: float
    net_area: float
    n_windows: int
    window_orders: str
    peak_window_details: str
    baseline_fit_scale: float
    baseline_fit_offset: float
    baseline_fit_linear_background: float
    max_score: float


@dataclass(frozen=True)
class AgBHMonochromaticitySelection:
    """Accepted/rejected AgBH QC rows and explicit H5 filter builders."""

    scored_df: pd.DataFrame
    id_column: str | None
    date_column: str | None
    score_column: str
    pass_column: str

    @property
    def accepted_df(self) -> pd.DataFrame:
        return self.scored_df.loc[self.scored_df[self.pass_column].astype(bool)].copy()

    @property
    def rejected_df(self) -> pd.DataFrame:
        return self.scored_df.loc[~self.scored_df[self.pass_column].astype(bool)].copy()

    @property
    def accepted_ids(self) -> list[Any]:
        if self.id_column is None:
            return []
        return _unique_nonempty_values(self.accepted_df[self.id_column])

    @property
    def rejected_ids(self) -> list[Any]:
        if self.id_column is None:
            return []
        return _unique_nonempty_values(self.rejected_df[self.id_column])

    @property
    def accepted_dates(self) -> list[str]:
        if self.date_column is None or self.date_column not in self.scored_df.columns:
            return []
        dates = pd.to_datetime(self.accepted_df[self.date_column], errors="coerce")
        return sorted({date.date().isoformat() for date in dates if not pd.isna(date)})

    def h5_id_filter(self, column: str | None = None) -> H5SessionFilter:
        """Return strict H5 ID filter from accepted AgBH IDs."""
        filter_column = column or self.id_column
        if filter_column is None:
            raise ValueError("id_column is required to build an H5 ID filter.")
        accepted_ids = self.accepted_ids
        if not accepted_ids:
            raise ValueError("No accepted AgBH IDs are available for H5 filtering.")
        return H5SessionFilter(filter_column, op="in", values=accepted_ids)

    def h5_date_filter(self, column: str = "started_at") -> H5SessionFilter:
        """Return date fallback H5 filter from accepted AgBH dates."""
        accepted_dates = self.accepted_dates
        if not accepted_dates:
            raise ValueError("No accepted AgBH dates are available for H5 filtering.")
        return H5SessionFilter(column, op="date in", values=accepted_dates)

    def manifest_columns(self) -> pd.DataFrame:
        """Return compact product manifest columns for persistence/review."""
        columns = [
            column
            for column in (
                self.id_column,
                self.date_column,
                self.score_column,
                self.pass_column,
                "agbh_monochromaticity_status",
                "agbh_monochromaticity_max_score",
                "agbh_kbeta_left_net_area",
                "agbh_kbeta_left_positive_area",
                "agbh_kbeta_right_control_positive_area",
                "agbh_kbeta_n_windows",
                "agbh_kbeta_window_orders",
            )
            if column is not None and column in self.scored_df.columns
        ]
        return self.scored_df.loc[:, list(dict.fromkeys(columns))].copy()


def _unique_nonempty_values(series: pd.Series) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for value in series:
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _infer_id_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns and _unique_nonempty_values(df[column]):
            return column
    return None


def calculate_agbh_monochromaticity(
    q: np.ndarray,
    intensity: np.ndarray,
    *,
    baseline_q: np.ndarray | None = None,
    baseline_intensity: np.ndarray | None = None,
    reference_q: np.ndarray | None = None,
    reference_intensity: np.ndarray | None = None,
    max_score: float = 0.1,
    q_min: float | None = None,
    q_max: float | None = None,
    q_points: int = 1800,
    d_spacing_nm: float = AGBH_D_SPACING_NM,
    beta_ratio: float = AG_KBETA_TO_KALPHA_Q_RATIO,
    beta_half_width: float = 0.12,
    min_order: int = 3,
    alpha_fit_left_width: float = 0.03,
    alpha_fit_right_width: float = 0.18,
) -> AgBHMonochromaticityResult:
    """Score AgBH monochromaticity from a measured profile and AgBH baseline.

    The score is a QC ranking metric, not a physical K-beta fraction. K-beta has
    higher energy than K-alpha, so in K-alpha q coordinates it appears at smaller
    q. The metric therefore counts only left-side positive residual area and
    subtracts symmetric right-side control area.
    """
    q = np.asarray(q, dtype=float)
    if baseline_q is None:
        baseline_q = reference_q
    if baseline_intensity is None:
        baseline_intensity = reference_intensity
    if baseline_q is None or baseline_intensity is None:
        raise ValueError("baseline_q and baseline_intensity must be provided.")
    baseline_q = np.asarray(baseline_q, dtype=float)
    if q_min is None:
        q_min = float(max(np.nanmin(q), np.nanmin(baseline_q)))
    if q_max is None:
        q_max = float(min(np.nanmax(q), np.nanmax(baseline_q)))
    if q_max <= q_min:
        raise ValueError("q_max must be greater than q_min.")
    grid = np.linspace(float(q_min), float(q_max), int(q_points))
    y = _normalize_profile(_profile_on_grid(q, intensity, grid))
    baseline_curve = _normalize_profile(
        _profile_on_grid(baseline_q, baseline_intensity, grid)
    )
    windows = agbh_kbeta_windows(
        float(q_min),
        float(q_max),
        d_spacing_nm=d_spacing_nm,
        beta_ratio=beta_ratio,
        beta_half_width=beta_half_width,
        min_order=min_order,
    )
    if not windows:
        raise ValueError("No AgBH K-beta windows fall inside the selected q range.")

    beta_mask = np.zeros_like(grid, dtype=bool)
    for window in windows:
        beta_mask |= _window_mask(grid, float(window["beta_q"]), beta_half_width)
    fit_mask = _alpha_fit_mask(
        grid,
        q_max=float(q_max),
        d_spacing_nm=d_spacing_nm,
        left_width=alpha_fit_left_width,
        right_width=alpha_fit_right_width,
        min_order=min_order,
    )
    fit_mask &= ~beta_mask
    fitted_baseline, scale, offset, slope = _fit_baseline(
        y,
        baseline_curve,
        grid,
        fit_mask,
    )
    residual = y - fitted_baseline
    left_area_sum = 0.0
    right_control_area_sum = 0.0
    peak_rows = []
    for window in windows:
        left_mask = _window_mask(grid, float(window["beta_q"]), beta_half_width)
        right_mask = _window_mask(
            grid,
            float(window["right_control_q"]),
            beta_half_width,
        )
        left_residual = np.clip(residual[left_mask], 0.0, None)
        right_residual = np.clip(residual[right_mask], 0.0, None)
        left_area = float(np.trapezoid(left_residual, grid[left_mask]))
        right_area = float(np.trapezoid(right_residual, grid[right_mask]))
        left_area_sum += left_area
        right_control_area_sum += right_area
        peak_rows.append(
            {
                "order": int(window["order"]),
                "alpha_q": float(window["alpha_q"]),
                "beta_q": float(window["beta_q"]),
                "right_control_q": float(window["right_control_q"]),
                "left_area": left_area,
                "right_control_area": right_area,
            }
        )
    net_area = max(left_area_sum - right_control_area_sum, 0.0)
    window_width_total = 2.0 * beta_half_width * max(1, len(windows))
    score = float(net_area / window_width_total)
    passed = bool(np.isfinite(score) and score <= float(max_score))
    return AgBHMonochromaticityResult(
        score=score,
        passed=passed,
        status="accepted" if passed else "rejected",
        left_positive_area=float(left_area_sum),
        right_control_positive_area=float(right_control_area_sum),
        net_area=float(net_area),
        n_windows=len(windows),
        window_orders=",".join(str(item["order"]) for item in windows),
        peak_window_details=json.dumps(peak_rows),
        baseline_fit_scale=float(scale),
        baseline_fit_offset=float(offset),
        baseline_fit_linear_background=float(slope),
        max_score=float(max_score),
    )


class AgBHMonochromaticityScorer(TransformerMixin, BaseEstimator):
    """Add AgBH monochromaticity QC score columns to integrated AgBH rows."""

    def __init__(
        self,
        reference_df: pd.DataFrame | None = None,
        *,
        q_column: str = "q_range",
        intensity_column: str = "radial_profile_data",
        score_column: str = "agbh_monochromaticity_score",
        pass_column: str = "agbh_monochromaticity_pass",
        status_column: str = "agbh_monochromaticity_status",
        max_score: float = 0.1,
        q_min: float | None = None,
        q_max: float | None = None,
        q_points: int = 1800,
        d_spacing_nm: float = AGBH_D_SPACING_NM,
        beta_ratio: float = AG_KBETA_TO_KALPHA_Q_RATIO,
        beta_half_width: float = 0.12,
        min_order: int = 3,
        alpha_fit_left_width: float = 0.03,
        alpha_fit_right_width: float = 0.18,
    ) -> None:
        self.reference_df = reference_df
        self.q_column = q_column
        self.intensity_column = intensity_column
        self.score_column = score_column
        self.pass_column = pass_column
        self.status_column = status_column
        self.max_score = float(max_score)
        self.q_min = q_min
        self.q_max = q_max
        self.q_points = int(q_points)
        self.d_spacing_nm = float(d_spacing_nm)
        self.beta_ratio = float(beta_ratio)
        self.beta_half_width = float(beta_half_width)
        self.min_order = int(min_order)
        self.alpha_fit_left_width = float(alpha_fit_left_width)
        self.alpha_fit_right_width = float(alpha_fit_right_width)

    def fit(self, X: pd.DataFrame, y=None):
        _ = y
        baseline_frame = self.reference_df if self.reference_df is not None else X
        if baseline_frame.empty:
            raise ValueError("AgBH baseline frame must contain at least one row.")
        for column in (self.q_column, self.intensity_column):
            if column not in baseline_frame.columns:
                raise KeyError(f"Column '{column}' not found in AgBH baseline frame.")
        q_arrays = [
            np.asarray(value, dtype=float) for value in baseline_frame[self.q_column]
        ]
        if self.q_min is None:
            q_min = max(float(np.nanmin(q)) for q in q_arrays)
        else:
            q_min = float(self.q_min)
        if self.q_max is None:
            q_max = min(float(np.nanmax(q)) for q in q_arrays)
        else:
            q_max = float(self.q_max)
        if q_max <= q_min:
            raise ValueError("Reference q range is empty.")
        grid = np.linspace(q_min, q_max, self.q_points)
        matrix = [
            _normalize_profile(
                _profile_on_grid(
                    np.asarray(row[self.q_column], dtype=float),
                    np.asarray(row[self.intensity_column], dtype=float),
                    grid,
                )
            )
            for _, row in baseline_frame.iterrows()
        ]
        self.baseline_q_ = grid
        self.baseline_intensity_ = np.nanmedian(np.asarray(matrix), axis=0)
        self.stats_ = {
            "baseline_rows": int(len(baseline_frame)),
            "baseline_source": "reference_df" if self.reference_df is not None else "fit_X",
            "q_min": q_min,
            "q_max": q_max,
            "q_points": int(self.q_points),
            "max_score": float(self.max_score),
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "baseline_q_"):
            raise RuntimeError("AgBHMonochromaticityScorer must be fitted before transform.")
        for column in (self.q_column, self.intensity_column):
            if column not in X.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame.")
        out = X.copy()
        results = [
            calculate_agbh_monochromaticity(
                np.asarray(row[self.q_column], dtype=float),
                np.asarray(row[self.intensity_column], dtype=float),
                baseline_q=self.baseline_q_,
                baseline_intensity=self.baseline_intensity_,
                max_score=self.max_score,
                q_min=float(self.stats_["q_min"]),
                q_max=float(self.stats_["q_max"]),
                q_points=self.q_points,
                d_spacing_nm=self.d_spacing_nm,
                beta_ratio=self.beta_ratio,
                beta_half_width=self.beta_half_width,
                min_order=self.min_order,
                alpha_fit_left_width=self.alpha_fit_left_width,
                alpha_fit_right_width=self.alpha_fit_right_width,
            )
            for _, row in out.iterrows()
        ]
        out[self.score_column] = [result.score for result in results]
        out[self.pass_column] = [result.passed for result in results]
        out[self.status_column] = [result.status for result in results]
        out["agbh_monochromaticity_max_score"] = [result.max_score for result in results]
        out["agbh_kbeta_left_positive_area"] = [
            result.left_positive_area for result in results
        ]
        out["agbh_kbeta_right_control_positive_area"] = [
            result.right_control_positive_area for result in results
        ]
        out["agbh_kbeta_left_net_area"] = [result.net_area for result in results]
        out["agbh_kbeta_n_windows"] = [result.n_windows for result in results]
        out["agbh_kbeta_window_orders"] = [result.window_orders for result in results]
        out["agbh_kbeta_peak_window_details"] = [
            result.peak_window_details for result in results
        ]
        out["agbh_baseline_fit_scale"] = [
            result.baseline_fit_scale for result in results
        ]
        out["agbh_baseline_fit_offset"] = [
            result.baseline_fit_offset for result in results
        ]
        out["agbh_baseline_fit_linear_background"] = [
            result.baseline_fit_linear_background for result in results
        ]
        return out


class AgBHMonochromaticityFilter(TransformerMixin, BaseEstimator):
    """Filter rows by AgBH monochromaticity score already computed by scorer."""

    def __init__(
        self,
        score_column: str = "agbh_monochromaticity_score",
        pass_column: str = "agbh_monochromaticity_pass",
        max_score: float = 0.1,
        *,
        drop: bool = True,
        reset_index: bool = True,
    ) -> None:
        self.score_column = score_column
        self.pass_column = pass_column
        self.max_score = float(max_score)
        self.drop = bool(drop)
        self.reset_index = bool(reset_index)
        self.stats_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.score_column not in X.columns:
            raise KeyError(f"Column '{self.score_column}' not found in DataFrame.")
        out = X.copy()
        score = pd.to_numeric(out[self.score_column], errors="coerce")
        passed = np.isfinite(score) & (score <= self.max_score)
        out[self.pass_column] = passed.to_numpy(dtype=bool)
        self.stats_ = {
            "filter_type": "agbh_monochromaticity",
            "filter_column": self.score_column,
            "filter_op": "<=",
            "filter_value": float(self.max_score),
            "rows_in": int(len(out)),
            "rows_pass": int(np.sum(passed)),
            "rows_fail": int(len(out) - np.sum(passed)),
        }
        if self.drop:
            out = out.loc[out[self.pass_column]].copy()
            if self.reset_index:
                out.reset_index(drop=True, inplace=True)
        return out


class AgBHMonochromaticityQualityControl(TransformerMixin, BaseEstimator):
    """Score AgBH rows and expose accepted IDs for later H5 session filtering."""

    def __init__(
        self,
        reference_df: pd.DataFrame | None = None,
        *,
        id_column: str | None = None,
        id_candidates: tuple[str, ...] = (
            "session_uid",
            "archive_session_path",
            "calibration_session_uid",
            "source_file",
            "file_name",
        ),
        date_column: str | None = "started_at",
        q_column: str = "q_range",
        intensity_column: str = "radial_profile_data",
        score_column: str = "agbh_monochromaticity_score",
        pass_column: str = "agbh_monochromaticity_pass",
        status_column: str = "agbh_monochromaticity_status",
        max_score: float = 0.1,
        q_min: float | None = None,
        q_max: float | None = None,
        q_points: int = 1800,
        d_spacing_nm: float = AGBH_D_SPACING_NM,
        beta_ratio: float = AG_KBETA_TO_KALPHA_Q_RATIO,
        beta_half_width: float = 0.12,
        min_order: int = 3,
        alpha_fit_left_width: float = 0.03,
        alpha_fit_right_width: float = 0.18,
    ) -> None:
        self.reference_df = reference_df
        self.id_column = id_column
        self.id_candidates = tuple(id_candidates)
        self.date_column = date_column
        self.q_column = q_column
        self.intensity_column = intensity_column
        self.score_column = score_column
        self.pass_column = pass_column
        self.status_column = status_column
        self.max_score = float(max_score)
        self.q_min = q_min
        self.q_max = q_max
        self.q_points = int(q_points)
        self.d_spacing_nm = float(d_spacing_nm)
        self.beta_ratio = float(beta_ratio)
        self.beta_half_width = float(beta_half_width)
        self.min_order = int(min_order)
        self.alpha_fit_left_width = float(alpha_fit_left_width)
        self.alpha_fit_right_width = float(alpha_fit_right_width)

    def fit(self, X: pd.DataFrame, y=None):
        _ = y
        self.scorer_ = AgBHMonochromaticityScorer(
            reference_df=self.reference_df,
            q_column=self.q_column,
            intensity_column=self.intensity_column,
            score_column=self.score_column,
            pass_column=self.pass_column,
            status_column=self.status_column,
            max_score=self.max_score,
            q_min=self.q_min,
            q_max=self.q_max,
            q_points=self.q_points,
            d_spacing_nm=self.d_spacing_nm,
            beta_ratio=self.beta_ratio,
            beta_half_width=self.beta_half_width,
            min_order=self.min_order,
            alpha_fit_left_width=self.alpha_fit_left_width,
            alpha_fit_right_width=self.alpha_fit_right_width,
        ).fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "scorer_"):
            raise RuntimeError(
                "AgBHMonochromaticityQualityControl must be fitted before transform."
            )
        scored = self.scorer_.transform(X)
        resolved_id_column = self.id_column
        if resolved_id_column is None:
            resolved_id_column = _infer_id_column(scored, self.id_candidates)
        if resolved_id_column is not None and resolved_id_column not in scored.columns:
            raise KeyError(f"Column '{resolved_id_column}' not found in DataFrame.")
        if self.date_column is not None and self.date_column not in scored.columns:
            date_column = None
        else:
            date_column = self.date_column
        self.selection_ = AgBHMonochromaticitySelection(
            scored_df=scored,
            id_column=resolved_id_column,
            date_column=date_column,
            score_column=self.score_column,
            pass_column=self.pass_column,
        )
        self.stats_ = {
            **self.scorer_.stats_,
            "id_column": resolved_id_column,
            "date_column": date_column,
            "rows_in": int(len(scored)),
            "rows_pass": int(scored[self.pass_column].sum()),
            "rows_fail": int(len(scored) - scored[self.pass_column].sum()),
        }
        return scored
