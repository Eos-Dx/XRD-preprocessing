from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import re
from typing import Any
import warnings

import numpy as np
import pandas as pd

from ._compat import BaseEstimator, TransformerMixin
from .faulty_pixels import create_mask


def _row_values(
    values: Any,
    *,
    n_rows: int,
    name: str,
) -> list[float] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise TypeError(f"{name} must be numeric or a sequence of numeric values.")
    if np.isscalar(values):
        numeric = float(values)
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must contain finite numeric values.")
        return [numeric] * n_rows
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence.")
    if len(array) != n_rows:
        raise ValueError(
            f"{name} length must match input rows: {len(array)} != {n_rows}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values.")
    return array.tolist()


def _pyfai():
    try:
        import pyFAI
        from pyFAI.detectors import Detector

        try:
            from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        except ImportError:
            from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
    except Exception as exc:
        raise ImportError("pyFAI is required for azimuthal integration") from exc
    return pyFAI, Detector, AzimuthalIntegrator


def _integrator_from_dataframe(
    pixel_size_m: float,
    center_col: float,
    center_row: float,
    wavelength_angstrom: float,
    distance_mm: float,
):
    _, Detector, AzimuthalIntegrator = _pyfai()
    detector = Detector(pixel_size_m, pixel_size_m)
    ai = AzimuthalIntegrator(detector=detector)
    ai.setFit2D(distance_mm, center_col, center_row, wavelength=wavelength_angstrom)
    return ai


@cache
def _integrator_from_poni_text(poni_text: str):
    import os
    import tempfile

    pyFAI, _, _ = _pyfai()
    fd, path = tempfile.mkstemp(suffix=".poni")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(poni_text)
        try:
            return pyFAI.load(path)
        except RuntimeError as exc:
            if "unknown" not in str(exc).lower() or "Detector_config:" not in poni_text:
                raise
            generic_poni_text = re.sub(
                r"^Detector:\s*.+$",
                "Detector: Detector",
                poni_text,
                count=1,
                flags=re.MULTILINE,
            )
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(generic_poni_text)
            return pyFAI.load(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _adjust_poni_distance(
    poni_text: str,
    sample_thickness_mm: float,
    reference_thickness_mm: float = 0.0,
) -> tuple[str, float | None]:
    anchor = "Distance:"
    pos = poni_text.find(anchor)
    if pos < 0:
        return poni_text, None
    start = pos + len(anchor)
    while start < len(poni_text) and poni_text[start] in " \t":
        start += 1
    end = poni_text.find("\n", start)
    if end < 0:
        end = len(poni_text)
    distance_m = float(poni_text[start:end])
    adjusted = distance_m - 0.5 * (
        float(sample_thickness_mm) - float(reference_thickness_mm)
    ) * 1e-3
    return poni_text[:start] + f"{adjusted:.6f}" + poni_text[end:], float(adjusted)


def estimate_poni_q_range_nm_inv(
    poni_text: str,
    *,
    shape: tuple[int, int] | None = None,
    sample_thickness_mm: float | None = None,
    thickness_reference_mm: float = 0.0,
) -> tuple[float, float, float]:
    """Estimate available q range from PONI geometry without integrating image data."""
    adjusted_poni = str(poni_text)
    if sample_thickness_mm is not None:
        adjusted_poni, _ = _adjust_poni_distance(
            adjusted_poni,
            sample_thickness_mm=float(sample_thickness_mm),
            reference_thickness_mm=float(thickness_reference_mm),
        )
    ai = _integrator_from_poni_text(adjusted_poni)
    q_array = ai.qArray(shape) if shape is not None else ai.qArray()
    finite_q = np.asarray(q_array, dtype=float)
    finite_q = finite_q[np.isfinite(finite_q)]
    if finite_q.size == 0:
        raise ValueError("PONI geometry produced no finite q values.")
    return float(np.nanmin(finite_q)), float(np.nanmax(finite_q)), float(ai.dist)


def _thickness_metadata(
    *,
    applied: bool,
    reliable: bool,
    warning_message: str,
    sample_thickness_mm: float | None,
    reference_thickness_mm: float | None,
    reference_source: str,
) -> dict[str, object]:
    return {
        "thickness_adjustment_applied": bool(applied),
        "thickness_adjustment_reliable": bool(reliable),
        "thickness_adjustment_warning": warning_message,
        "sample_thickness_mm": sample_thickness_mm,
        "thickness_reference_mm": (
            float(reference_thickness_mm) if reference_thickness_mm is not None else None
        ),
        "thickness_reference_source": reference_source,
    }


def _resolve_reference_thickness(
    row: pd.Series,
    *,
    thickness_reference_mm: float | None,
    thickness_reference_column: str | None,
) -> tuple[float | None, str]:
    if thickness_reference_column is None:
        return thickness_reference_mm, "constant"
    if thickness_reference_column not in row.index:
        raise ValueError(
            f"Missing required thickness reference column: {thickness_reference_column}"
        )
    try:
        row_reference_mm = float(row[thickness_reference_column])
    except (TypeError, ValueError):
        row_reference_mm = np.nan
    if not np.isfinite(row_reference_mm):
        raise ValueError(
            f"Invalid thickness reference value in column: {thickness_reference_column}"
        )
    return row_reference_mm, thickness_reference_column


def _resolve_thickness(
    row: pd.Series,
    *,
    thickness_adjustment: bool,
    require_thickness_adjustment: bool,
    sample_thickness_column: str,
    thickness_reference_mm: float | None,
    thickness_reference_column: str | None,
) -> tuple[float | None, dict[str, object]]:
    resolved_reference_mm, reference_source = _resolve_reference_thickness(
        row,
        thickness_reference_mm=thickness_reference_mm,
        thickness_reference_column=thickness_reference_column,
    )
    if not thickness_adjustment:
        message = (
            "Thickness adjustment disabled; azimuthal integration is unreliable for thick samples."
            if require_thickness_adjustment
            else ""
        )
        if require_thickness_adjustment:
            raise ValueError(message)
        if message:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        return None, _thickness_metadata(
            applied=False,
            reliable=not require_thickness_adjustment,
            warning_message=message,
            sample_thickness_mm=None,
            reference_thickness_mm=resolved_reference_mm,
            reference_source=reference_source,
        )

    if resolved_reference_mm is None:
        raise ValueError("thickness_reference_mm must be set explicitly.")

    if sample_thickness_column not in row.index:
        message = f"Missing required thickness column: {sample_thickness_column}"
        if require_thickness_adjustment:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return None, _thickness_metadata(
            applied=False,
            reliable=False,
            warning_message=message,
            sample_thickness_mm=None,
            reference_thickness_mm=resolved_reference_mm,
            reference_source=reference_source,
        )

    try:
        sample_thickness_mm = float(row[sample_thickness_column])
    except (TypeError, ValueError):
        sample_thickness_mm = np.nan

    if not np.isfinite(sample_thickness_mm):
        message = f"Invalid thickness value in column: {sample_thickness_column}"
        if require_thickness_adjustment:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return None, _thickness_metadata(
            applied=False,
            reliable=False,
            warning_message=message,
            sample_thickness_mm=None,
            reference_thickness_mm=resolved_reference_mm,
            reference_source=reference_source,
        )

    return sample_thickness_mm, _thickness_metadata(
        applied=True,
        reliable=True,
        warning_message="",
        sample_thickness_mm=float(sample_thickness_mm),
        reference_thickness_mm=resolved_reference_mm,
        reference_source=reference_source,
    )


def _perform_azimuthal_integration_with_metadata(
    row: pd.Series,
    *,
    column: str = "measurement_data",
    npt: int = 100,
    npt_azimuthal: int = 360,
    mask: np.ndarray | None = None,
    mask_column: str | None = None,
    mode: str = "1D",
    calibration_mode: str = "dataframe",
    error_model: str | None = None,
    thickness_adjustment: bool = True,
    require_thickness_adjustment: bool = True,
    thickness_reference_mm: float | None = None,
    thickness_reference_column: str | None = None,
    sample_thickness_column: str = "sample_thickness_mm",
):
    """Integrate one detector image and return profile plus row-level metadata."""
    data = np.asarray(row[column])
    row_mask = row.get(mask_column) if mask_column is not None else None
    mask_source = "none"
    if row_mask is not None:
        mask = np.asarray(row_mask)
        mask_source = mask_column or "row"
    elif mask is not None:
        mask = np.asarray(mask)
        mask_source = "static"
    mask_pixels = int(np.sum(mask.astype(bool))) if mask is not None else 0

    radial_range = row.get("interpolation_q_range")
    azimuth_range = row.get("azimuthal_range")
    sample_thickness, thickness_meta = _resolve_thickness(
        row,
        thickness_adjustment=thickness_adjustment,
        require_thickness_adjustment=require_thickness_adjustment,
        sample_thickness_column=sample_thickness_column,
        thickness_reference_mm=thickness_reference_mm,
        thickness_reference_column=thickness_reference_column,
    )

    if calibration_mode == "dataframe":
        sample_distance_mm = float(row["calculated_distance"]) * 1e3
        adjusted_distance_m = None
        if sample_thickness is not None:
            sample_distance_mm -= 0.5 * (
                sample_thickness - float(thickness_meta["thickness_reference_mm"])
            )
            adjusted_distance_m = sample_distance_mm * 1e-3
        ai = _integrator_from_dataframe(
            float(row["pixel_size"]) * 1e-6,
            float(row["center"][1]),
            float(row["center"][0]),
            float(row["wavelength"]) * 10.0,
            sample_distance_mm,
        )
        if adjusted_distance_m is not None:
            thickness_meta["thickness_adjusted_distance_m"] = float(adjusted_distance_m)
    elif calibration_mode == "poni":
        poni_text = str(row["ponifile"])
        if sample_thickness is not None:
            poni_text, adjusted_distance_m = _adjust_poni_distance(
                poni_text,
                sample_thickness,
                float(thickness_meta["thickness_reference_mm"]),
            )
            if adjusted_distance_m is None:
                message = "PONI Distance field missing; thickness adjustment was not applied."
                warnings.warn(message, RuntimeWarning, stacklevel=3)
                thickness_meta.update(
                    {
                        "thickness_adjustment_applied": False,
                        "thickness_adjustment_reliable": False,
                        "thickness_adjustment_warning": message,
                    }
                )
            else:
                thickness_meta["thickness_adjusted_distance_m"] = float(adjusted_distance_m)
        ai = _integrator_from_poni_text(poni_text)
    else:
        raise ValueError("calibration_mode must be 'dataframe' or 'poni'")

    if mode == "1D":
        result = ai.integrate1d(
            data,
            npt,
            radial_range=radial_range,
            azimuth_range=azimuth_range,
            mask=mask,
            error_model=error_model,
        )
        if isinstance(result, tuple):
            radial, intensity = result[0], result[1]
            sigma = result[2] if len(result) > 2 else None
        else:
            radial = result.radial
            intensity = result.intensity
            sigma = getattr(result, "sigma", None)
        metadata = {
            **thickness_meta,
            "azimuthal_mask_source": mask_source,
            "azimuthal_mask_pixels": mask_pixels,
            "azimuthal_npt": int(npt),
            "azimuthal_npt_azimuthal": None,
            "azimuthal_mode": "1D",
        }
        return radial, intensity, sigma, ai.dist, metadata

    if mode == "2D":
        result = ai.integrate2d(
            data,
            npt,
            int(npt_azimuthal),
            radial_range=radial_range,
            azimuth_range=azimuth_range,
            mask=mask,
            error_model=error_model,
        )
        if isinstance(result, tuple):
            intensity, radial, azimuthal = result[0], result[1], result[2]
        else:
            intensity = result.intensity
            radial = result.radial
            azimuthal = result.azimuthal
        metadata = {
            **thickness_meta,
            "azimuthal_mask_source": mask_source,
            "azimuthal_mask_pixels": mask_pixels,
            "azimuthal_npt": int(npt),
            "azimuthal_npt_azimuthal": int(npt_azimuthal),
            "azimuthal_mode": "2D",
        }
        return radial, intensity, azimuthal, ai.dist, metadata

    raise ValueError("mode must be '1D' or '2D'")


def perform_azimuthal_integration(
    row: pd.Series,
    *,
    column: str = "measurement_data",
    npt: int = 100,
    npt_azimuthal: int = 360,
    mask: np.ndarray | None = None,
    mask_column: str | None = None,
    mode: str = "1D",
    calibration_mode: str = "dataframe",
    error_model: str | None = None,
    thickness_adjustment: bool = True,
    require_thickness_adjustment: bool = True,
    thickness_reference_mm: float | None = None,
    thickness_reference_column: str | None = None,
    sample_thickness_column: str = "sample_thickness_mm",
):
    """Integrate one detector image into one 1D or 2D pyFAI profile."""
    radial, intensity, sigma_or_azimuthal, distance, _ = _perform_azimuthal_integration_with_metadata(
        row,
        column=column,
        npt=npt,
        npt_azimuthal=npt_azimuthal,
        mask=mask,
        mask_column=mask_column,
        mode=mode,
        calibration_mode=calibration_mode,
        error_model=error_model,
        thickness_adjustment=thickness_adjustment,
        require_thickness_adjustment=require_thickness_adjustment,
        thickness_reference_mm=thickness_reference_mm,
        thickness_reference_column=thickness_reference_column,
        sample_thickness_column=sample_thickness_column,
    )
    return radial, intensity, sigma_or_azimuthal, distance


@dataclass
class AzimuthalIntegration(TransformerMixin, BaseEstimator):
    """Sklearn transformer for simple row-wise pyFAI azimuthal integration."""

    column: str = "measurement_data"
    output_column: str = "radial_profile_data"
    q_range_column: str = "q_range"
    sigma_column: str = "radial_profile_sigma"
    npt: int = 100
    npt_azimuthal: int = 360
    mode: str = "1D"
    calibration_mode: str = "dataframe"
    faulty_pixels: list[tuple[int, int]] | None = None
    mask: np.ndarray | None = None
    mask_column: str | None = None
    error_model: str | None = None
    thickness_adjustment: bool = True
    require_thickness_adjustment: bool = True
    sample_thickness_mm: Any = None
    thickness_reference_mm: Any = None
    thickness_reference_column: str | None = None
    sample_thickness_column: str = "sample_thickness_mm"

    def fit(self, X: pd.DataFrame, y=None):
        _ = X
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        sample_thickness_column = self.sample_thickness_column
        sample_thickness_values = _row_values(
            self.sample_thickness_mm,
            n_rows=len(out),
            name="sample_thickness_mm",
        )
        if sample_thickness_values is not None:
            sample_thickness_column = "sample_thickness_mm"
            out[sample_thickness_column] = sample_thickness_values

        thickness_reference_mm = self.thickness_reference_mm
        thickness_reference_column = self.thickness_reference_column
        reference_values = _row_values(
            self.thickness_reference_mm,
            n_rows=len(out),
            name="thickness_reference_mm",
        )
        if reference_values is not None and not np.isscalar(self.thickness_reference_mm):
            thickness_reference_mm = None
            thickness_reference_column = "thickness_reference_mm"
            out[thickness_reference_column] = reference_values

        mask = self.mask
        if mask is None and self.faulty_pixels is not None:
            mask = create_mask(self.faulty_pixels, np.asarray(out[self.column].iloc[0]).shape)

        results = out.apply(
            lambda row: _perform_azimuthal_integration_with_metadata(
                row,
                column=self.column,
                npt=self.npt,
                npt_azimuthal=self.npt_azimuthal,
                mask=mask,
                mask_column=self.mask_column,
                mode=self.mode,
                calibration_mode=self.calibration_mode,
                error_model=self.error_model,
                thickness_adjustment=self.thickness_adjustment,
                require_thickness_adjustment=self.require_thickness_adjustment,
                thickness_reference_mm=thickness_reference_mm,
                thickness_reference_column=thickness_reference_column,
                sample_thickness_column=sample_thickness_column,
            ),
            axis=1,
        )

        out[self.q_range_column] = [item[0] for item in results]
        out[self.output_column] = [item[1] for item in results]
        if self.mode == "1D":
            out[self.sigma_column] = [item[2] for item in results]
        else:
            out["azimuthal_positions"] = [item[2] for item in results]
        out["calculated_distance"] = [item[3] for item in results]
        metadata = [item[4] for item in results]
        for key in metadata[0] if metadata else []:
            out[key] = [item.get(key) for item in metadata]
        return out
