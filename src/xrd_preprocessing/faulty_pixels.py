"""Minimal faulty-pixel detection for GFRM detector frames."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from ._compat import BaseEstimator, TransformerMixin

Pixel = tuple[int, int]

_FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _as_2d_image(value, context: str = "image") -> np.ndarray:
    image = np.asarray(value, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"{context} must be a 2D numeric image.")
    return image


def _pixel_array(pixels: set[Pixel]) -> np.ndarray:
    return np.array(sorted(pixels), dtype=int) if pixels else np.empty((0, 2), dtype=int)


def create_mask(
    faulty_pixels: Iterable[Pixel] | np.ndarray | None,
    size: tuple[int, int] = (256, 256),
) -> np.ndarray | None:
    """Create a pyFAI mask: 1 excludes a pixel, 0 keeps it."""
    if faulty_pixels is None:
        return None

    mask = np.zeros(size, dtype=np.uint8)
    for y, x in np.asarray(list(faulty_pixels), dtype=int).reshape(-1, 2):
        if 0 <= y < size[0] and 0 <= x < size[1]:
            mask[y, x] = 1
    return mask


def _find_image_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        _as_2d_image(df[preferred].dropna().iloc[0], preferred)
        return preferred

    for column in df.columns:
        series = df[column].dropna()
        if not len(series):
            continue
        try:
            if np.asarray(series.iloc[0]).ndim == 2:
                _as_2d_image(series.iloc[0], column)
                return column
        except (TypeError, ValueError):
            continue
    raise ValueError(f"No 2D numeric image column found in {list(df.columns)!r}.")


def _float_field(text: str, key: str) -> float | None:
    match = re.search(rf"^{key}:\s*([0-9.eE+-]+)", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def _beam_center_pixels(poni_text: str) -> tuple[int, int] | None:
    poni1 = _float_field(poni_text, "Poni1")
    poni2 = _float_field(poni_text, "Poni2")
    match = re.search(r"^Detector_config:\s*(.+)$", poni_text, re.MULTILINE)
    if poni1 is None or poni2 is None or match is None:
        return None
    pixel1 = _detector_config_float(match.group(1), "pixel1")
    pixel2 = _detector_config_float(match.group(1), "pixel2")
    if pixel1 is None or pixel2 is None:
        return None
    if pixel1 <= 0 or pixel2 <= 0:
        return None
    return int(round(poni1 / pixel1)), int(round(poni2 / pixel2))


def _detector_config_float(text: str, key: str) -> float | None:
    match = re.search(rf"['\"]?{key}['\"]?\s*:\s*({_FLOAT_RE})", text)
    return float(match.group(1)) if match else None


def _beam_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius_fraction: float,
) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    center_y, center_x = center
    radius = float(radius_fraction) * max(shape)
    return np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2) <= radius


def detect_faulty_pixels(image: np.ndarray, **kwargs) -> set[Pixel]:
    """Return one set of excluded pixel coordinates for one image."""
    return FaultyPixelDetector(**kwargs).detect(image)


def faulty_pixel_statistics(
    df: pd.DataFrame,
    *,
    mask_column: str = "faulty_pixel_mask",
) -> dict[str, object]:
    """Return faulty-pixel audit statistics without mutating the DataFrame."""
    if mask_column not in df.columns:
        raise KeyError(f"Column '{mask_column}' not found in DataFrame.")
    counts = [int(len(mask)) for mask in df[mask_column]]
    return {
        "mask_column": mask_column,
        "n_images": int(len(df)),
        "faulty_pixels_per_row": counts,
        "total_faulty_pixels": int(sum(counts)),
    }


class FaultyPixelDetector(TransformerMixin, BaseEstimator):
    """Frame-local detector for GFRM preprocessing.

    The detector is intentionally simple. It does not estimate a permanent
    detector map from one image. It only marks pixels that are invalid for the
    current frame or bright enough to create spikes after azimuthal integration.
    """

    def __init__(
        self,
        image_column: str = "measurement_data",
        negative_threshold: float = 0.0,
        detect_negative_pixels: bool = True,
        detect_bright_pixels: bool = True,
        bright_pixel_min_value: float = 500.0,
        exclude_beam_center_radius: float | None = None,
        poni_column: str = "ponifile",
        mask_column: str = "faulty_pixel_mask",
    ):
        self.image_column = image_column
        self.negative_threshold = float(negative_threshold)
        self.detect_negative_pixels = bool(detect_negative_pixels)
        self.detect_bright_pixels = bool(detect_bright_pixels)
        self.bright_pixel_min_value = float(bright_pixel_min_value)
        self.exclude_beam_center_radius = exclude_beam_center_radius
        self.poni_column = poni_column
        self.mask_column = mask_column

    def fit(self, df: pd.DataFrame, y=None):
        _ = df
        _ = y
        self.is_fitted_ = True
        return self

    def _get_beam_center_pixels(self, poni_text: str) -> tuple[int, int] | None:
        return _beam_center_pixels(str(poni_text))

    def _invalid_pixels(self, image: np.ndarray) -> set[Pixel]:
        mask = ~np.isfinite(image)
        if self.detect_negative_pixels:
            mask |= image < self.negative_threshold
        return {(int(y), int(x)) for y, x in zip(*np.where(mask))}

    def _bright_pixels(self, image: np.ndarray) -> set[Pixel]:
        if not self.detect_bright_pixels:
            return set()

        bright = (
            np.isfinite(image)
            & (image >= self.negative_threshold)
            & (image > self.bright_pixel_min_value)
        )
        return {(int(y), int(x)) for y, x in zip(*np.where(bright))}

    def _detect_mask(
        self,
        image: np.ndarray,
        exclude_mask: np.ndarray | None = None,
    ) -> set[Pixel]:
        image = _as_2d_image(image)
        invalid = self._invalid_pixels(image)
        bright = self._bright_pixels(image) - invalid
        pixels = invalid | bright
        if exclude_mask is not None:
            pixels = {(y, x) for y, x in pixels if not exclude_mask[y, x]}
        return pixels

    def detect(self, image: np.ndarray) -> set[Pixel]:
        return self._detect_mask(image)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.image_column not in out.columns:
            raise KeyError(f"Column '{self.image_column}' not found in DataFrame.")
        image_column = self.image_column

        faulty_masks: list[np.ndarray] = []

        for _, row in out.iterrows():
            image = _as_2d_image(row[image_column], image_column)
            exclude_mask = None
            if (
                self.exclude_beam_center_radius is not None
                and self.exclude_beam_center_radius > 0
                and self.poni_column in out.columns
            ):
                center = _beam_center_pixels(str(row[self.poni_column]))
                if center is not None:
                    exclude_mask = _beam_mask(
                        image.shape,
                        center,
                        self.exclude_beam_center_radius,
                    )
            faulty = self._detect_mask(image, exclude_mask=exclude_mask)

            faulty_mask = _pixel_array(faulty)
            faulty_masks.append(faulty_mask)

        out[self.mask_column] = faulty_masks
        return out
